"""Streamlit web app.  Run:  streamlit run app.py"""

import numpy as np
import pandas as pd
import streamlit as st

import model as m

st.set_page_config(page_title="Breast Cancer Risk Predictor", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Training model (first run only, ~20 s)...")
def get_artifact():
    if m.DEFAULT_MODEL_PATH.exists():
        return m.load()
    artifact = m.train()
    try:
        m.save(artifact)
    except OSError:
        pass  # read-only filesystem (e.g. some hosts): keep the model in memory
    return artifact


artifact = get_artifact()
features = artifact["feature_names"]
stats = artifact["feature_stats"]


def fill_with(values: dict):
    for f in features:
        st.session_state[f] = float(values[f])


def load_random_example():
    X_test = artifact["X_test"]
    i = np.random.randint(len(X_test))
    fill_with(X_test.iloc[i].to_dict())
    actual = artifact["target_names"][int(artifact["y_test"].iloc[i])]
    st.session_state["example_note"] = f"Loaded test-set patient #{X_test.index[i]} (actual: **{actual}**)"


def load_averages():
    fill_with(stats["mean"].to_dict())
    st.session_state["example_note"] = "Loaded dataset averages"


if features[0] not in st.session_state:
    load_averages()

st.title("🩺 Breast Cancer Risk Predictor")
st.caption("Educational machine-learning demo on the Wisconsin Breast Cancer dataset. "
           "**Not medical advice.**")

tab_predict, tab_model = st.tabs(["Predict", "Model performance"])

with tab_predict:
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("🎲 Random test patient", on_click=load_random_example, width="stretch")
    c2.button("↺ Reset to averages", on_click=load_averages, width="stretch")
    if note := st.session_state.get("example_note"):
        st.info(note)

    groups = {
        "Mean values": [f for f in features if f.startswith("mean ")],
        "Standard error": [f for f in features if f.endswith(" error")],
        "Worst values": [f for f in features if f.startswith("worst ")],
    }
    for group_tab, (label, names) in zip(st.tabs(list(groups)), groups.items()):
        with group_tab:
            cols = st.columns(3)
            for i, f in enumerate(names):
                lo, hi = float(stats.loc[f, "min"]), float(stats.loc[f, "max"])
                cols[i % 3].number_input(f, min_value=0.0, max_value=hi * 2,
                                         step=(hi - lo) / 100 or 0.01, format="%.4f", key=f)

    result = m.predict(artifact, {f: st.session_state[f] for f in features})
    st.divider()
    left, right = st.columns(2)
    if result["label"] == "malignant":
        left.error(f"### Prediction: Malignant")
    else:
        left.success(f"### Prediction: Benign")
    right.metric("Probability malignant", f"{result['probability_malignant']:.1%}")
    right.progress(result["probability_malignant"])

with tab_model:
    st.subheader(f"Best model: {artifact['model_name']}")
    mt = artifact["metrics"]
    cols = st.columns(5)
    for col, (name, key) in zip(cols, [("Accuracy", "accuracy"), ("Precision", "precision_malignant"),
                                        ("Recall", "recall_malignant"), ("F1", "f1_malignant"),
                                        ("ROC-AUC", "roc_auc")]):
        col.metric(name, f"{mt[key]:.3f}")
    st.caption("Precision / recall / F1 are for the *malignant* class, measured on a held-out 20% test set.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Model comparison (5-fold CV ROC-AUC)**")
        st.dataframe(artifact["comparison"].style.format({"cv_roc_auc_mean": "{:.4f}",
                                                          "cv_roc_auc_std": "{:.4f}"}),
                     hide_index=True, width="stretch")
        st.markdown("**Confusion matrix (test set)**")
        names = artifact["target_names"]
        st.dataframe(pd.DataFrame(mt["confusion_matrix"],
                                  index=[f"actual {n}" for n in names],
                                  columns=[f"predicted {n}" for n in names]),
                     width="stretch")
    with right:
        if not artifact["importance"].empty:
            st.markdown("**Top 10 most important features**")
            st.bar_chart(artifact["importance"].head(10), horizontal=True)
