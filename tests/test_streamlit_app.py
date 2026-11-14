"""Exercise the three Streamlit sections without starting a browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.data_validation import FEATURE_COLUMNS, validate_student
from src.predict import ModelInferenceError


APP_PATH = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_overview_action_and_assessment_form(monkeypatch):
    submitted = []

    def record_prediction(student):
        submitted.append(student)
        return {
            "prediction": "Low",
            "probabilities": {"Low": 0.7, "Medium": 0.2, "High": 0.1},
        }

    monkeypatch.setattr("src.predict.predict_student", record_prediction)
    app = AppTest.from_file(APP_PATH).run()
    assert not app.exception
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["page"] == "Risk assessment"
    assert len(app.number_input) + len(app.selectbox) == len(FEATURE_COLUMNS)

    app.button[0].click().run()
    assert not app.exception
    assert set(submitted[0]) == set(FEATURE_COLUMNS)
    assert validate_student(submitted[0]) == submitted[0]
    assert len(app.get("progress")) == 3


def test_performance_uses_saved_reports(tmp_path):
    from app.streamlit_app import read_metrics, read_table

    assert read_metrics(tmp_path) == {}
    assert read_table(tmp_path / "missing.csv") is None
    app = AppTest.from_file(APP_PATH).run()
    app.segmented_control[0].set_value("Model performance").run()
    assert not app.exception
    assert [item.label for item in app.metric] == [
        "Accuracy", "Macro precision", "Macro recall", "Macro F1-score"
    ]
    assert len(app.dataframe) == 3
    assert app.metric[0].value == f"{read_metrics()['accuracy']:.1%}"


def test_missing_model_shows_actionable_error(monkeypatch):
    def missing_model(student):
        raise ModelInferenceError("Model not found")

    monkeypatch.setattr("src.predict.predict_student", missing_model)
    app = AppTest.from_file(APP_PATH).run()
    app.segmented_control[0].set_value("Risk assessment").run()
    app.button[0].click().run()
    assert not app.exception
    assert "models/best_model.joblib" in app.error[0].value
    assert len(app.get("progress")) == 0


def test_default_form_predicts_with_saved_model():
    app = AppTest.from_file(APP_PATH).run()
    app.segmented_control[0].set_value("Risk assessment").run()
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["assessment_result"]["prediction"] in {
        "Low", "Medium", "High"
    }
    assert len(app.get("progress")) == 3
