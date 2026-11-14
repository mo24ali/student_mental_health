
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

from src.data_validation import TARGET_COLUMN, validate_training_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw"
    / "student_mental_health.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


NUMERIC_FEATURES = [
    "Age",
    "Year_of_Study",
    "Sleep_Hours",
    "Study_Hours_Per_Week",
    "GPA",
    "Social_Media_Hours",
    "Screen_Time_Hours",
    "Financial_Stress",
    "Family_Support",
    "Attendance_Rate",
]

NOMINAL_FEATURES = [
    "Gender",
    "Part_Time_Job",
    "Previous_Mental_Health_History",
]

ORDINAL_FEATURES = [
    "Physical_Activity",
    "Diet_Quality",
]


def load_data(path=RAW_DATA_PATH):
    """Load and validate the raw dataset."""
    df = pd.read_csv(path)
    return validate_training_data(df)


def split_data(df):
    """Split the dataset into train and test sets."""

    X = df.drop(columns=[TARGET_COLUMN, "Student_ID"], errors="ignore")
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )


def build_preprocessor():
    """Build preprocessing steps for each feature type."""

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                ["Low", "Moderate", "High"],
                ["Poor", "Average", "Good"],
            ],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("nom", nominal_pipeline, NOMINAL_FEATURES),
        ("ord", ordinal_pipeline, ORDINAL_FEATURES),
    ])


def load_processed(path):
    """Load one saved, untransformed split for training or evaluation."""
    df = validate_training_data(pd.read_csv(path))
    return split_features(df)


def split_features(df):
    """Separate validated features and target."""
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def prepare_data(raw_path=RAW_DATA_PATH, processed_dir=PROCESSED_DIR):
    """Validate, split and save the dataset."""

    df = load_data(raw_path)

    X_train, X_test, y_train, y_test = split_data(df)

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_df.to_csv(
        processed_dir / "train.csv",
        index=False,
    )

    test_df.to_csv(
        processed_dir / "test.csv",
        index=False,
    )

    print("Data preparation completed.")
    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_data()
