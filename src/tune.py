"""Tune notebook search spaces and select the best training-CV model."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

from src.preprocessing import PROCESSED_DIR, PROJECT_ROOT, load_processed
from src.train import CV, RANDOM_STATE, candidate_models


MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
TUNING_PATH = PROJECT_ROOT / "reports" / "metrics" / "tuning_results.csv"
METADATA_PATH = PROJECT_ROOT / "reports" / "best_model_metadata.json"


PARAMETERS = {
    "XGBoost": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [2, 3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 3, 5, 10],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    },
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__class_weight": [None, "balanced"],
    },
}


def tune_models(
    train_path=PROCESSED_DIR / "train.csv",
    model_path=MODEL_PATH,
    tuning_path=TUNING_PATH,
    metadata_path=METADATA_PATH,
):
    """Fit 10 randomized candidates per model and save the best full pipeline."""
    X_train, y_train = load_processed(train_path)
    encoded_y = LabelEncoder().fit_transform(y_train)
    results = []
    searches = {}
    pipelines = candidate_models()
    # Notebook search order also fixes the random sampling for each search.
    for name in ("XGBoost", "Random Forest", "Logistic Regression"):
        search = RandomizedSearchCV(
            estimator=pipelines[name],
            param_distributions=PARAMETERS[name],
            n_iter=10,
            scoring="f1_macro",
            cv=CV,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, encoded_y if name == "XGBoost" else y_train)
        searches[name] = search
        results.append({
            "Model": name,
            "Best CV Macro F1": float(search.best_score_),
            "Best Parameters": str(search.best_params_),
        })

    comparison = pd.DataFrame(results).sort_values("Best CV Macro F1", ascending=False)
    best_name = comparison.iloc[0]["Model"]
    best_model = searches[best_name].best_estimator_

    # Write only after all searches succeed; callers can direct outputs to a staging area.
    model_path, tuning_path, metadata_path = map(
        Path, (model_path, tuning_path, metadata_path)
    )
    for path in (model_path, tuning_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)
    comparison.to_csv(tuning_path, index=False)
    metadata_path.write_text(json.dumps({
        "model_name": best_name,
        "cv_macro_f1": float(searches[best_name].best_score_),
        "classes": ["Low", "Medium", "High"],
        "best_params": searches[best_name].best_params_,
    }, indent=4) + "\n")
    return comparison


if __name__ == "__main__":
    print(tune_models().to_string(index=False))
