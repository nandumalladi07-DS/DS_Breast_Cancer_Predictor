"""Core ML logic: load data, compare models, train the best one, save/load it.

Dataset: scikit-learn's built-in Wisconsin Breast Cancer dataset (569 samples,
30 numeric features). Target: 0 = malignant, 1 = benign.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "model.joblib"


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return features X, target y and the class names ["malignant", "benign"]."""
    data = load_breast_cancer(as_frame=True)
    return data.data, data.target, list(data.target_names)


def candidate_models() -> dict[str, Pipeline]:
    """The models we compare. Each one is scaled first so they compete fairly."""
    return {
        "Logistic Regression": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000))]
        ),
        "Random Forest": Pipeline(
            [("scaler", StandardScaler()),
             ("clf", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE))]
        ),
        "SVM (RBF)": Pipeline(
            [("scaler", StandardScaler()),
             ("clf", CalibratedClassifierCV(SVC(), ensemble=False))]
        ),
        "Gradient Boosting": Pipeline(
            [("scaler", StandardScaler()),
             ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]
        ),
    }


def compare_models(X: pd.DataFrame, y: pd.Series, folds: int = 5) -> pd.DataFrame:
    """5-fold cross-validated ROC-AUC for every candidate, best first."""
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in candidate_models().items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        rows.append({"model": name, "cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()})
    return pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Hold-out metrics. 'Positive' class here is malignant (label 0), the one we must not miss."""
    pred = model.predict(X_test)
    proba_malignant = model.predict_proba(X_test)[:, 0]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision_malignant": float(precision_score(y_test, pred, pos_label=0)),
        "recall_malignant": float(recall_score(y_test, pred, pos_label=0)),
        "f1_malignant": float(f1_score(y_test, pred, pos_label=0)),
        "roc_auc": float(roc_auc_score(y_test == 0, proba_malignant)),
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }


def feature_importance(model: Pipeline, feature_names: list[str]) -> pd.Series:
    """Importance per feature (tree importances or |coefficients|), largest first."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
    else:  # e.g. RBF SVM has no built-in importance
        return pd.Series(dtype=float)
    return pd.Series(values, index=feature_names).sort_values(ascending=False)


def train(test_size: float = 0.2) -> dict:
    """Full pipeline. Returns an artifact dict holding the fitted model and its report."""
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    comparison = compare_models(X_train, y_train)
    best_name = comparison.loc[0, "model"]
    best_model = candidate_models()[best_name].fit(X_train, y_train)

    return {
        "model": best_model,
        "model_name": best_name,
        "feature_names": list(X.columns),
        "target_names": target_names,
        "comparison": comparison,
        "metrics": evaluate(best_model, X_test, y_test),
        "importance": feature_importance(best_model, list(X.columns)),
        "feature_stats": X.describe().T[["min", "mean", "max"]],
        "X_test": X_test,
        "y_test": y_test,
    }


def save(artifact: dict, path: Path = DEFAULT_MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load(path: Path = DEFAULT_MODEL_PATH) -> dict:
    return joblib.load(path)


def predict(artifact: dict, features: dict[str, float]) -> dict:
    """Predict for one patient. `features` maps every feature name to a value."""
    missing = [f for f in artifact["feature_names"] if f not in features]
    if missing:
        raise ValueError(f"Missing features: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    row = pd.DataFrame([[features[f] for f in artifact["feature_names"]]],
                       columns=artifact["feature_names"])
    proba = artifact["model"].predict_proba(row)[0]
    label = int(np.argmax(proba))
    return {
        "label": artifact["target_names"][label],
        "probability_malignant": float(proba[0]),
        "probability_benign": float(proba[1]),
    }
