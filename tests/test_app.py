"""Smoke test: the Streamlit app runs end-to-end without a browser."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_loads_and_predicts():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert any("Breast Cancer Risk Predictor" in t.value for t in at.title)
    assert at.metric[0].label == "Probability malignant"


def test_random_patient_button():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button[0].click().run()
    assert not at.exception
    assert any("test-set patient" in i.value for i in at.info)
