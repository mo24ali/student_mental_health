"""Student Wellbeing: a small interface for the saved research model."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Streamlit runs this file from app/; make src/ importable from any launch directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_validation import (  # noqa: E402
    CATEGORICAL_VALUES,
    FEATURE_COLUMNS,
    NUMERIC_RANGES,
    DataValidationError,
    validate_student,
)
from src.predict import ModelInferenceError, predict_student  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
RISK_COLORS = {
    "Low": "#527A62",
    "Medium": "#956A2B",
    "High": "#A05D4D",
}


def page_intro(kicker, title, description):
    """Draw the consistent heading shared by all three sections."""
    st.markdown(f'<p class="eyebrow">{kicker}</p>', unsafe_allow_html=True)
    st.title(title)
    st.write(description)
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)


def number_field(name, label, value, step, help_text=None):
    """Use validation bounds for every numeric form field."""
    minimum, maximum = NUMERIC_RANGES[name]
    if isinstance(value, float):
        minimum, maximum = float(minimum), float(maximum)
    return st.number_input(
        label, min_value=minimum, max_value=maximum,
        value=value, step=step, help=help_text,
    )


def category_field(name, label, options, default, help_text=None):
    """Display known categorical values as a select box."""
    assert set(options) == CATEGORICAL_VALUES[name]
    return st.selectbox(label, options, index=options.index(default), help=help_text)


def student_form():
    """Collect the exact feature dictionary expected by predict_student()."""
    with st.form("risk_assessment", border=False):
        student = {}

        st.subheader("Student profile")
        left, right = st.columns(2, gap="large")
        with left:
            student["Age"] = number_field("Age", "Age", 21, 1)
            student["Year_of_Study"] = number_field(
                "Year_of_Study", "Year of study", 2, 1,
                "Current year in your program.",
            )
        with right:
            student["Gender"] = category_field(
                "Gender", "Gender", ["Female", "Male", "Non-binary"], "Female"
            )
            student["GPA"] = number_field(
                "GPA", "GPA", 2.56, 0.01, "Grade point average on a 0–4 scale."
            )

        st.subheader("Academic habits")
        left, right = st.columns(2, gap="large")
        with left:
            student["Study_Hours_Per_Week"] = number_field(
                "Study_Hours_Per_Week", "Study hours per week", 19.8, 0.5
            )
            student["Part_Time_Job"] = category_field(
                "Part_Time_Job", "Part-time job", ["No", "Yes"], "No"
            )
        with right:
            student["Attendance_Rate"] = number_field(
                "Attendance_Rate", "Attendance rate (%)", 85.1, 0.5,
                "Percentage of classes attended.",
            )

        st.subheader("Lifestyle")
        left, right = st.columns(2, gap="large")
        with left:
            student["Sleep_Hours"] = number_field(
                "Sleep_Hours", "Sleep hours per day", 6.7, 0.1
            )
            student["Social_Media_Hours"] = number_field(
                "Social_Media_Hours", "Social media hours per day", 6.0, 0.1
            )
            student["Physical_Activity"] = category_field(
                "Physical_Activity", "Physical activity",
                ["Low", "Moderate", "High"], "Moderate"
            )
        with right:
            student["Screen_Time_Hours"] = number_field(
                "Screen_Time_Hours", "Screen time hours per day", 8.5, 0.1
            )
            student["Diet_Quality"] = category_field(
                "Diet_Quality", "Diet quality", ["Poor", "Average", "Good"],
                "Average", "General self-reported diet quality."
            )

        st.subheader("Personal context")
        left, right = st.columns(2, gap="large")
        with left:
            student["Financial_Stress"] = number_field(
                "Financial_Stress", "Financial stress (0–10)", 6.1, 0.1,
                "Higher numbers indicate more financial stress. Decimals are allowed.",
            )
            student["Previous_Mental_Health_History"] = category_field(
                "Previous_Mental_Health_History", "Previous mental health history",
                ["No", "Yes"], "No", "Self-reported history."
            )
        with right:
            student["Family_Support"] = number_field(
                "Family_Support", "Family support (0–10)", 6.7, 0.1,
                "Higher numbers indicate more perceived support.",
            )

        submitted = st.form_submit_button(
            "Run assessment", type="primary"
        )
    return student, submitted


def show_result(result):
    """Present a prediction as a model estimate, with restrained emphasis."""
    label = result["prediction"]
    color = RISK_COLORS[label]
    st.markdown("### Assessment result")
    st.markdown(
        f'<div class="risk-panel" style="border-left-color: {color}">'
        '<div class="eyebrow">MODEL ESTIMATE · RISK CATEGORY</div>'
        f'<div class="risk-label" style="color: {color}">{label}</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("Based on the last submitted information.")
    st.markdown("#### Estimated class probabilities")
    for category in ("Low", "Medium", "High"):
        probability = result["probabilities"].get(category, 0.0)
        name, value = st.columns([4, 1])
        name.write(category)
        value.write(f"{probability:.1%}")
        st.progress(probability, text=None)
    st.caption(
        "These percentages are model estimates across three categories; "
        "they are not clinically validated probabilities."
    )
    st.write(
        f"The model placed this entry in the **{label}** category based on "
        "patterns in its training data. This is not a diagnosis or a definitive "
        "assessment of mental health."
    )


def go_to_assessment():
    """Change pages before Streamlit creates the navigation widget."""
    st.session_state.page = "Risk assessment"


def overview_page():
    """Explain the research workflow without making clinical claims."""
    page_intro(
        "STUDENT WELLBEING  /  OVERVIEW",
        "Mental health risk assessment for students",
        "A simple way to explore how a trained model categorizes student "
        "information as Low, Medium, or High risk. Enter a student profile "
        "to see the model's estimate and how it distributes probability "
        "across the three categories.",
    )
    st.subheader("How it works")
    for number, title, detail in (
        ("01", "Enter student information", "Provide academic, lifestyle, and personal context."),
        ("02", "Run assessment", "Submit the profile to the saved prediction model."),
        ("03", "Review prediction", "Read the category and estimated class probabilities."),
    ):
        first, second = st.columns([1, 8], gap="small")
        first.markdown(f'<span class="step-number">{number}</span>', unsafe_allow_html=True)
        second.markdown(f"**{title}**  \n{detail}")
        st.markdown('<div class="step-rule"></div>', unsafe_allow_html=True)

    st.write("")
    st.button("Start an assessment", type="primary", on_click=go_to_assessment)
    st.caption(
        "Educational ML tool only. Its output is not a clinical diagnosis "
        "and should not replace evaluation by a qualified professional."
    )


def assessment_page():
    """Collect, validate, and predict without retraining or preprocessing here."""
    page_intro(
        "STUDENT WELLBEING  /  ASSESSMENT",
        "Risk assessment",
        "Complete the student profile below. Values are used only for this prediction.",
    )
    student, submitted = student_form()
    if submitted:
        st.session_state.pop("assessment_result", None)
        try:
            validated = validate_student(student)
            st.session_state.assessment_result = predict_student(validated)
        except DataValidationError as error:
            st.error(f"Please check the form: {error}")
        except ModelInferenceError:
            st.error(
                "The saved model could not be used right now. "
                "Check that models/best_model.joblib exists and can be loaded."
            )
    if "assessment_result" in st.session_state:
        show_result(st.session_state.assessment_result)


def read_table(path):
    """Read a report table if it exists, otherwise return None."""
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, index_col=0)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.warning(f"Could not read {path.name}: {error}")
        return None


def read_metrics(metrics_dir=METRICS_DIR):
    """Use saved test metrics, falling back to the classification report."""
    path = metrics_dir / "test_metrics.json"
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as error:
            st.warning(f"Could not read {path.name}: {error}")
    report = read_table(metrics_dir / "classification_report.csv")
    if (report is not None and {"accuracy", "macro avg"}.issubset(report.index)
            and {"precision", "recall", "f1-score"}.issubset(report.columns)):
        return {
            "accuracy": float(report.loc["accuracy", "f1-score"]),
            "precision_macro": float(report.loc["macro avg", "precision"]),
            "recall_macro": float(report.loc["macro avg", "recall"]),
            "f1_macro": float(report.loc["macro avg", "f1-score"]),
        }
    return {}


def performance_page():
    """Show only persisted evaluation results, without fabricated insights."""
    page_intro(
        "STUDENT WELLBEING  /  MODEL PERFORMANCE",
        "Model performance",
        "Results from the saved evaluation reports. The held-out test set "
        "is used for the final model assessment.",
    )
    metrics = read_metrics()
    names = (
        ("Accuracy", "accuracy"), ("Macro precision", "precision_macro"),
        ("Macro recall", "recall_macro"), ("Macro F1-score", "f1_macro"),
    )
    if metrics:
        columns = st.columns(4)
        for column, (name, key) in zip(columns, names):
            if key in metrics:
                column.metric(name, f"{metrics[key]:.1%}")
        st.caption(
            "Accuracy is the share of correct predictions. Macro precision, "
            "recall, and F1 average each class equally; F1 balances "
            "precision and recall."
        )
    else:
        st.info("Test metrics are not available yet.")

    st.markdown("### Classification report")
    report = read_table(METRICS_DIR / "classification_report.csv")
    if report is not None:
        st.dataframe(report.round(3))
        st.caption("Precision, recall, F1-score, and support by class, plus overall averages.")
    else:
        st.info("The classification report is not available yet.")

    st.markdown("### Confusion matrix")
    matrix = read_table(METRICS_DIR / "confusion_matrix.csv")
    if matrix is not None:
        st.dataframe(matrix)
        st.caption("Rows are actual categories; columns are predicted categories.")
    else:
        st.info("The confusion matrix is not available yet.")

    st.markdown("### Model comparison")
    comparison = METRICS_DIR / "model_comparison.csv"
    if comparison.is_file():
        try:
            table = pd.read_csv(comparison)
            st.dataframe(table.round(3), hide_index=True)
            st.caption("Baseline cross-validation scores on the training set; F1 Std is fold-to-fold variation.")
        except (OSError, ValueError, pd.errors.ParserError) as error:
            st.warning(f"Could not read {comparison.name}: {error}")
    else:
        st.info("Model comparison results are not available yet.")


def main():
    """Configure the page and route between the three sections."""
    st.set_page_config(
        page_title="Student Wellbeing",
        layout="centered", initial_sidebar_state="collapsed",
    )
    st.markdown(
        """<style>
        html, body, [class*="css"] { font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        .stApp { color: #202B2A; }
        .block-container { max-width: 960px; padding-top: 2.8rem; padding-bottom: 4rem; }
        h1, h2, h3 { letter-spacing: -0.025em; font-weight: 550; }
        h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); line-height: 1.18; margin-bottom: 0.8rem; }
        h3 { margin-top: 2.1rem; }
        p { line-height: 1.65; }
        .eyebrow, .step-number { color: #347C78; font-size: .76rem; font-weight: 650; letter-spacing: .11em; }
        .section-rule { border-top: 1px solid #E3E8E4; margin: 1.8rem 0 2.2rem; }
        .step-rule { border-top: 1px solid #E3E8E4; margin: .65rem 0 1rem; }
        .risk-panel { background: #FFFFFF; border: 1px solid #E3E8E4; border-left: 4px solid; padding: 1.3rem 1.6rem; border-radius: .3rem; }
        .risk-label { font-size: 1.85rem; font-weight: 580; letter-spacing: -.02em; margin-top: .45rem; }
        [data-testid="stForm"] { border-top: 0; }
        @media (max-width: 640px) { .block-container { padding-top: 1.5rem; } }
        </style>""",
        unsafe_allow_html=True,
    )
    masthead, label = st.columns([4, 2], vertical_alignment="center")
    masthead.markdown("**Student Wellbeing**")
    label.caption("STUDENT MENTAL HEALTH · RESEARCH TOOL")
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.session_state.setdefault("page", "Overview")
    st.segmented_control(
        "Navigate", ["Overview", "Risk assessment", "Model performance"],
        key="page", label_visibility="collapsed",
    )
    st.write("")
    if st.session_state.page == "Risk assessment":
        assessment_page()
    elif st.session_state.page == "Model performance":
        performance_page()
    else:
        overview_page()


if __name__ == "__main__":
    main()
