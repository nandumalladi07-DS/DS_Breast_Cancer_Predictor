import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model as m  # noqa: E402


@pytest.fixture(scope="module")
def artifact():
    # Training takes a few seconds, so do it once for all tests.
    return m.train()


def test_dataset_shape():
    X, y, names = m.load_data()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    assert names == ["malignant", "benign"]


def test_all_candidates_are_compared(artifact):
    assert set(artifact["comparison"]["model"]) == set(m.candidate_models())
    # sorted best first
    scores = artifact["comparison"]["cv_roc_auc_mean"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_best_model_is_accurate(artifact):
    metrics = artifact["metrics"]
    assert metrics["accuracy"] > 0.93
    assert metrics["roc_auc"] > 0.97
    assert metrics["recall_malignant"] > 0.88


def test_predict_returns_valid_probabilities(artifact):
    row = artifact["X_test"].iloc[0].to_dict()
    result = m.predict(artifact, row)
    assert result["label"] in ("malignant", "benign")
    assert result["probability_malignant"] + result["probability_benign"] == pytest.approx(1.0)


def test_predict_matches_known_cases(artifact):
    X, y = artifact["X_test"], artifact["y_test"]
    correct = sum(
        m.predict(artifact, X.iloc[i].to_dict())["label"] == artifact["target_names"][y.iloc[i]]
        for i in range(len(X))
    )
    assert correct / len(X) == pytest.approx(artifact["metrics"]["accuracy"])


def test_predict_rejects_missing_features(artifact):
    with pytest.raises(ValueError, match="Missing features"):
        m.predict(artifact, {"mean radius": 10.0})


def test_save_and_load_roundtrip(artifact, tmp_path):
    path = m.save(artifact, tmp_path / "model.joblib")
    loaded = m.load(path)
    row = artifact["X_test"].iloc[3].to_dict()
    assert m.predict(loaded, row) == m.predict(artifact, row)
