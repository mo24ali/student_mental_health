"""Small checks for the dataset schema and fitted-pipeline contract."""

import numpy as np
import pandas as pd
import pytest

from src.data_validation import DataValidationError, readable_labels, validate_student, validate_training_data
from src.predict import predict_batch, predict_student
from src.preprocessing import build_preprocessor, load_data, split_data


def test_real_data_schema_and_split():
    data = load_data()
    X_train, X_test, y_train, y_test = split_data(data)
    assert data.shape == (2000, 16)
    assert X_train.shape == (1600, 15)
    assert X_test.shape == (400, 15)
    assert "Student_ID" not in X_train
    assert y_train.value_counts().to_dict() == {"High": 544, "Medium": 528, "Low": 528}
    assert y_test.value_counts().to_dict() == {"High": 136, "Medium": 132, "Low": 132}


def test_validation_of_decimal_missing_and_categories():
    student = load_data().iloc[0].drop("Mental_Health_Risk").to_dict()
    student.update(Gender="Non-binary", Financial_Stress=5.4, Diet_Quality=None)
    assert np.isnan(validate_student(student, allow_missing=True)["Diet_Quality"])
    with pytest.raises(DataValidationError, match="Diet_Quality"):
        validate_student(student)
    with pytest.raises(DataValidationError, match="Gender"):
        validate_student({**student, "Gender": "Other"}, allow_missing=True)
    with pytest.raises(DataValidationError, match="target"):
        validate_training_data(load_data().drop(columns="Mental_Health_Risk"))


def test_preprocessor_and_saved_inference(tmp_path):
    X_train, _, y_train, _ = split_data(load_data())
    transformed = build_preprocessor().fit_transform(X_train)
    assert transformed.shape == (1600, 19)
    assert np.isfinite(transformed).all()
    # Fit and serialize a small representative pipeline without touching the notebook model.
    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    model = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]).fit(X_train, y_train)
    path = tmp_path / "model.joblib"
    dump(model, path)
    student = X_train.iloc[0].to_dict()
    student["Study_Hours_Per_Week"] = None  # notebook inference example
    result = predict_student(student, model_path=path)
    assert result["prediction"] in {"Low", "Medium", "High"}
    assert sum(result["probabilities"].values()) == pytest.approx(1)
    batch = predict_batch(pd.DataFrame([student, student]), model_path=path)
    assert batch.shape == (2, 5)
    assert batch.Predicted_Risk.tolist() == [result["prediction"]] * 2
    assert readable_labels([0, 1, 2]) == ["High", "Low", "Medium"]


def test_xgboost_encoded_targets_return_named_probabilities(tmp_path):
    from joblib import dump
    from sklearn.preprocessing import LabelEncoder

    from src.train import candidate_models

    X_train, _, y_train, _ = split_data(load_data())
    model = candidate_models()["XGBoost"]
    model.set_params(classifier__n_estimators=2)
    model.fit(X_train.iloc[:150], LabelEncoder().fit_transform(y_train.iloc[:150]))
    path = tmp_path / "xgb.joblib"
    dump(model, path)
    result = predict_student(X_train.iloc[150].to_dict(), model_path=path)
    assert result["prediction"] in {"Low", "Medium", "High"}
    assert set(result["probabilities"]) == {"Low", "Medium", "High"}
    assert sum(result["probabilities"].values()) == pytest.approx(1)
