"""Train the model from the command line and write reports.

Usage:
    python train.py
Outputs:
    models/model.joblib            the trained best model
    reports/metrics.json           hold-out test metrics + model comparison
    reports/model_comparison.png   cross-validated ROC-AUC per model
    reports/confusion_matrix.png   test-set confusion matrix
    reports/feature_importance.png top 10 features (if the model provides them)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt

import model as m

REPORTS = Path(__file__).parent / "reports"


def plot_comparison(comparison, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(comparison["model"], comparison["cv_roc_auc_mean"],
            xerr=comparison["cv_roc_auc_std"], color="#4C78A8")
    ax.invert_yaxis()
    ax.set_xlim(0.9, 1.0)
    ax.set_xlabel("Cross-validated ROC-AUC (higher is better)")
    ax.set_title("Model comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(cm, labels, path):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(importance, path):
    top = importance.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(top.index, top.values, color="#4C78A8")
    ax.set_title("Top 10 features")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    print("Comparing models with 5-fold cross-validation...")
    artifact = m.train()
    print(artifact["comparison"].to_string(index=False))
    print(f"\nBest model: {artifact['model_name']}")
    for k, v in artifact["metrics"].items():
        if k != "confusion_matrix":
            print(f"  {k:22s} {v:.4f}")

    model_path = m.save(artifact)
    print(f"\nSaved model -> {model_path}")

    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump({
            "best_model": artifact["model_name"],
            "test_metrics": artifact["metrics"],
            "comparison": artifact["comparison"].to_dict(orient="records"),
        }, f, indent=2)
    plot_comparison(artifact["comparison"], REPORTS / "model_comparison.png")
    plot_confusion(artifact["metrics"]["confusion_matrix"], artifact["target_names"],
                   REPORTS / "confusion_matrix.png")
    if not artifact["importance"].empty:
        plot_importance(artifact["importance"], REPORTS / "feature_importance.png")
    print(f"Saved reports -> {REPORTS}/")


if __name__ == "__main__":
    main()
