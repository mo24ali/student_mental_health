
"""Inference utilities for Student Mental Health Risk Prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data_validation import (
    FEATURE_COLUMNS,
    readable_labels,
    validate_dataframe,
    validate_student,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "best_model.joblib"
)


class ModelInferenceError(RuntimeError):
    """Raised when model loading or inference fails."""


def load_model(model_path=MODEL_PATH):
    """
    Load the fitted preprocessing + classifier pipeline.

    Only load trusted joblib files.
    """

    model_path = Path(model_path)
    if not model_path.is_file():
        raise ModelInferenceError(
            f"Model not found: {model_path}"
        )

    try:
        model = joblib.load(model_path)
    except Exception as error:
        raise ModelInferenceError(
            f"Could not load model: {error}"
        ) from error

    if not hasattr(model, "predict"):
        raise ModelInferenceError(
            "Saved object does not support predict()."
        )

    if not hasattr(model, "predict_proba"):
        raise ModelInferenceError(
            "Saved model does not support predict_proba()."
        )

    if not hasattr(model, "classes_"):
        raise ModelInferenceError(
            "Saved model has no classes_ attribute."
        )

    # Verify that preprocessing is included.
    if not hasattr(model, "named_steps"):
        raise ModelInferenceError(
            "Expected a fitted sklearn Pipeline."
        )

    if "preprocessor" not in model.named_steps:
        raise ModelInferenceError(
            "Saved pipeline has no 'preprocessor' step."
        )

    return model


def predict_student(
    student_data: dict[str, Any],
    model_path=MODEL_PATH,
) -> dict[str, Any]:
    """
    Predict the risk category for one student.

    Returns
    -------
    dict
        prediction: predicted class
        probabilities: probability for each class
        confidence: highest predicted probability
    """

    validated = validate_student(
        student_data,
        allow_missing=True,
    )

    student_df = pd.DataFrame(
        [validated],
        columns=FEATURE_COLUMNS,
    )

    model = load_model(model_path)

    try:
        prediction = readable_labels(model.predict(student_df))[0]
        probabilities = model.predict_proba(
            student_df
        )[0]
    except Exception as error:
        raise ModelInferenceError(
            f"Prediction failed: {error}"
        ) from error

    class_probabilities = {
        str(label): float(probability)
        for label, probability in zip(
            readable_labels(model.classes_),
            probabilities,
        )
    }

    return {
        "prediction": str(prediction),
        "probabilities": class_probabilities,
        "confidence": float(max(probabilities)),
    }


def predict_batch(
    students: pd.DataFrame,
    model_path=MODEL_PATH,
) -> pd.DataFrame:
    """
    Predict risk categories for multiple students.

    Returns a DataFrame containing predictions
    and class probabilities.
    """

    validated = validate_dataframe(
        students,
        allow_missing=True,
    )

    model = load_model(model_path)

    try:
        predictions = readable_labels(model.predict(validated))
        probabilities = model.predict_proba(validated)
    except Exception as error:
        raise ModelInferenceError(
            f"Batch prediction failed: {error}"
        ) from error

    results = pd.DataFrame(
        {"Predicted_Risk": predictions},
        index=validated.index,
    )

    for index, label in enumerate(readable_labels(model.classes_)):
        results[f"Probability_{label}"] = (
            probabilities[:, index]
        )

    results["Confidence"] = probabilities.max(axis=1)

    return results


def get_model_info(model_path=MODEL_PATH) -> dict[str, Any]:
    """Return basic information about the saved model."""

    model = load_model(model_path)

    return {
        "model_path": str(model_path),
        "model_type": type(model).__name__,
        "classes": readable_labels(model.classes_),
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS.copy(),
    }
