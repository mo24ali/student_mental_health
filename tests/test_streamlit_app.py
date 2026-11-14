"""Exercise the Streamlit dashboard without starting a browser."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.data_validation import FEATURE_COLUMNS, validate_student
from src.predict import ModelInferenceError, predict_student


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
    assert [item.label for item in app.metric] == [
        "Test accuracy", "Macro F1-score", "Dataset records", "Predictive features"
    ]
    assert app.metric[2].value == "2,000"
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["page"] == "Risk Prediction"
    assert len(app.number_input) + len(app.selectbox) == len(FEATURE_COLUMNS)

    app.button[0].click().run()
    assert not app.exception
    assert set(submitted[0]) == set(FEATURE_COLUMNS)
    assert validate_student(submitted[0]) == submitted[0]
    assert len(app.get("progress")) == 3
    assert "Low Risk: 70.0%" in app.get("progress")[0].text
    assert "not a clinical diagnosis" in app.info[0].value


def test_performance_uses_saved_reports(tmp_path):
    from app.streamlit_app import read_metrics, read_table

    assert read_metrics(tmp_path) == {}
    assert read_table(tmp_path / "missing.csv") is None
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Model Performance").run()
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
    app.radio[0].set_value("Risk Prediction").run()
    app.button[0].click().run()
    assert not app.exception
    assert "models/best_model.joblib" in app.error[0].value
    assert len(app.get("progress")) == 0


def test_default_form_predicts_with_saved_model():
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Risk Prediction").run()
    inputs = {item.label: item.value for item in app.number_input}
    categories = {item.label: item.value for item in app.selectbox}
    student = {
        "Age": inputs["Age"], "Year_of_Study": inputs["Year of study"],
        "GPA": inputs["GPA"], "Study_Hours_Per_Week": inputs["Study hours per week"],
        "Attendance_Rate": inputs["Attendance rate (%)"],
        "Sleep_Hours": inputs["Sleep hours per day"],
        "Social_Media_Hours": inputs["Social media hours per day"],
        "Screen_Time_Hours": inputs["Screen time hours per day"],
        "Financial_Stress": inputs["Financial stress (0–10)"],
        "Family_Support": inputs["Family support (0–10)"],
        "Gender": categories["Gender"], "Part_Time_Job": categories["Part-time job"],
        "Physical_Activity": categories["Physical activity"],
        "Diet_Quality": categories["Diet quality"],
        "Previous_Mental_Health_History": categories["Previous mental health history"],
    }
    expected = predict_student(validate_student(student))
    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["assessment_result"]["prediction"] in {
        "Low", "Medium", "High"
    }
    assert app.session_state["assessment_result"] == expected
    assert len(app.get("progress")) == 3


def test_about_page_loads():
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("About the Project").run()
    assert not app.exception
    assert "About the Project" in [title.value for title in app.title]
