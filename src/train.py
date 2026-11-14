"""Compare the three notebook classifiers using training-only cross-validation."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.preprocessing import PROCESSED_DIR, PROJECT_ROOT, build_preprocessor, load_processed


RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
COMPARISON_PATH = PROJECT_ROOT / "reports" / "metrics" / "model_comparison.csv"


def candidate_models():
    """Return pipelines with the same baseline settings as the experiments notebook."""
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob", num_class=3, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
            random_state=RANDOM_STATE, learning_rate=0.1,
            n_estimators=100, n_jobs=1,
        ),
    }
    return {
        name: Pipeline([("preprocessor", build_preprocessor()), ("classifier", classifier)])
        for name, classifier in classifiers.items()
    }


def train_models(train_path=PROCESSED_DIR / "train.csv", comparison_path=COMPARISON_PATH):
    """Cross-validate candidates on training data and save their comparison."""
    X_train, y_train = load_processed(train_path)
    encoded_y = LabelEncoder().fit_transform(y_train)
    rows = []
    scoring = {
        "accuracy": "accuracy", "precision_macro": "precision_macro",
        "recall_macro": "recall_macro", "f1_macro": "f1_macro",
    }
    for name, pipeline in candidate_models().items():
        target = encoded_y if name == "XGBoost" else y_train
        scores = cross_validate(pipeline, X_train, target, cv=CV, scoring=scoring)
        rows.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision_macro"].mean(),
            "Recall": scores["test_recall_macro"].mean(),
            "F1 Macro": scores["test_f1_macro"].mean(),
            "F1 Std": scores["test_f1_macro"].std(),
        })

    comparison = pd.DataFrame(rows).sort_values("F1 Macro", ascending=False)
    comparison_path = Path(comparison_path)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(comparison_path, index=False)
    return comparison


if __name__ == "__main__":
    print(train_models().to_string(index=False))
