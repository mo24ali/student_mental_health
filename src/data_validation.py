
"""Input validation for Student Mental Health Risk Prediction."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# --------------------------------------------------
# 1. Feature schema
# --------------------------------------------------

NUMERIC_RANGES = {
    "Age": (15, 100),
    "Year_of_Study": (1, 10),
    "Sleep_Hours": (0, 24),
    "Study_Hours_Per_Week": (0, 168),
    "GPA": (0, 4),
    "Social_Media_Hours": (0, 24),
    "Screen_Time_Hours": (0, 24),
    "Financial_Stress": (0, 10),
    "Family_Support": (0, 10),
    "Attendance_Rate": (0, 100),
}

CATEGORICAL_VALUES = {
    "Gender": {"Male", "Female", "Non-binary"},
    "Physical_Activity": {"Low", "Moderate", "High"},
    "Part_Time_Job": {"Yes", "No"},
    "Diet_Quality": {"Poor", "Average", "Good"},
    "Previous_Mental_Health_History": {"Yes", "No"},
}

# Integer-valued features
INTEGER_FEATURES = {
    "Age",
    "Year_of_Study",
}

# Expected feature order
FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Physical_Activity",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
    "Part_Time_Job",
    "Diet_Quality",
    "Previous_Mental_Health_History",
]


# --------------------------------------------------
# 2. Custom validation exception
# --------------------------------------------------

class DataValidationError(ValueError):
    """Raised when student data fails validation."""


# --------------------------------------------------
# 3. Validate one student
# --------------------------------------------------

def validate_student(
    student: dict[str, Any],
    allow_missing: bool = False,
) -> dict[str, Any]:
    """
    Validate one student record.

    Parameters
    ----------
    student : dict
        Student features.
    allow_missing : bool
        If True, missing values are allowed and converted
        to NaN for the fitted preprocessing pipeline.

    Returns
    -------
    dict
        Validated student features in expected column order.

    Raises
    ------
    DataValidationError
        If the input violates the schema.
    """

    if not isinstance(student, dict):
        raise DataValidationError(
            "Student input must be a dictionary."
        )

    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in student
    ]

    extra_columns = [
        col for col in student
        if col not in FEATURE_COLUMNS
    ]

    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {missing_columns}"
        )

    if extra_columns:
        raise DataValidationError(
            f"Unexpected columns: {extra_columns}"
        )

    validated = {}

    for column in FEATURE_COLUMNS:
        value = student[column]

        # Handle missing values
        if value is None or (
            isinstance(value, str) and not value.strip()
        ):
            missing = True
        else:
            try:
                missing = bool(pd.isna(value))
            except (TypeError, ValueError):
                missing = False

        if missing:
            if not allow_missing:
                raise DataValidationError(
                    f"{column}: value is required."
                )

            validated[column] = float("nan")
            continue

        # Validate numerical features
        if column in NUMERIC_RANGES:

            if isinstance(value, bool):
                raise DataValidationError(
                    f"{column}: boolean is not a number."
                )

            try:
                numeric_value = float(value)
            except (TypeError, ValueError, OverflowError):
                raise DataValidationError(
                    f"{column}: expected a numeric value."
                ) from None

            if not math.isfinite(numeric_value):
                raise DataValidationError(
                    f"{column}: value must be finite."
                )

            minimum, maximum = NUMERIC_RANGES[column]

            if not minimum <= numeric_value <= maximum:
                raise DataValidationError(
                    f"{column}: expected a value between "
                    f"{minimum} and {maximum}."
                )

            if column in INTEGER_FEATURES:
                if not numeric_value.is_integer():
                    raise DataValidationError(
                        f"{column}: expected a whole number."
                    )
                validated[column] = int(numeric_value)
            else:
                validated[column] = numeric_value

        # Validate categorical features
        elif column in CATEGORICAL_VALUES:

            if not isinstance(value, str):
                raise DataValidationError(
                    f"{column}: expected a string."
                )

            value = value.strip()

            allowed = CATEGORICAL_VALUES[column]

            if value not in allowed:
                raise DataValidationError(
                    f"{column}: invalid value {value!r}. "
                    f"Allowed values: {sorted(allowed)}"
                )

            validated[column] = value

    return validated


# --------------------------------------------------
# 4. Validate a DataFrame
# --------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """
    Validate a batch of student records.

    Returns a validated DataFrame with the expected
    feature names and column order.
    """

    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise DataValidationError(
            "Input DataFrame cannot be empty."
        )

    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    extra_columns = [
        col for col in df.columns
        if col not in FEATURE_COLUMNS
    ]

    if missing_columns or extra_columns:
        raise DataValidationError(
            f"Missing columns: {missing_columns}; "
            f"unexpected columns: {extra_columns}"
        )

    if not df.columns.is_unique:
        raise DataValidationError(
            "Duplicate column names are not allowed."
        )

    validated_rows = []

    for position, (_, row) in enumerate(df.iterrows()):
        try:
            validated_rows.append(
                validate_student(
                    row.to_dict(),
                    allow_missing=allow_missing,
                )
            )
        except DataValidationError as error:
            raise DataValidationError(
                f"Row {position}: {error}"
            ) from error

    return pd.DataFrame(
        validated_rows,
        columns=FEATURE_COLUMNS,
        index=df.index,
    )


# --------------------------------------------------
# 5. Validate a training dataset
# --------------------------------------------------

TARGET_COLUMN = "Mental_Health_Risk"
TARGET_CLASSES = {"Low", "Medium", "High"}


def validate_training_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate a labeled dataset.

    Student_ID is removed because it is an identifier,
    not a predictive feature.

    Missing feature values are allowed because the
    preprocessing pipeline handles imputation.
    """

    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame.")
    if not df.columns.is_unique:
        raise DataValidationError("Duplicate column names are not allowed.")
    if TARGET_COLUMN not in df.columns:
        raise DataValidationError(
            f"Missing target column: {TARGET_COLUMN}"
        )

    labels = df[TARGET_COLUMN]

    if labels.isna().any():
        raise DataValidationError(
            "Target column contains missing values."
        )

    invalid_labels = set(labels.unique()) - TARGET_CLASSES

    if invalid_labels:
        raise DataValidationError(
            f"Invalid target labels: {invalid_labels}"
        )

    features = df.drop(
        columns=[TARGET_COLUMN, "Student_ID"],
        errors="ignore",
    )

    validated_features = validate_dataframe(
        features,
        allow_missing=True,
    )

    validated_features[TARGET_COLUMN] = labels

    return validated_features


def readable_labels(labels):
    """Decode XGBoost's integer labels using the notebook's sorted class order."""
    return [sorted(TARGET_CLASSES)[int(label)] if not isinstance(label, str)
            else label for label in labels]
