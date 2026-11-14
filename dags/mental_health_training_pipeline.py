"""Manual, run-isolated training of the student mental health classifier."""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow.sdk import dag, get_current_context, task


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"


def run_directory():
    """Use the same private directory for every task in one DAG run."""
    run_id = get_current_context()["run_id"]
    name = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    artifacts = Path(os.environ.get("AIRFLOW_ARTIFACTS_DIR", PROJECT_ROOT / ".airflow" / "artifacts"))
    return artifacts / name


@dag(
    dag_id="student_mental_health_ml",
    schedule=None,
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    max_active_runs=1,
    tags=["ml", "education"],
)
def mental_health_training_pipeline():
    @task
    def validate_data():
        from src.preprocessing import load_data

        data = load_data(RAW_DATA)
        logging.info("Validated %s labeled records", len(data))

    @task
    def preprocess_data():
        from src.preprocessing import prepare_data

        output = run_directory() / "processed"
        prepare_data(raw_path=RAW_DATA, processed_dir=output)
        logging.info("Saved train/test split in %s", output)

    @task
    def train_models():
        from src.train import train_models as compare_models

        output = run_directory()
        comparison = compare_models(
            train_path=output / "processed" / "train.csv",
            comparison_path=output / "reports" / "metrics" / "model_comparison.csv",
        )
        logging.info("Compared %s models", len(comparison))

    @task
    def tune_model():
        from src.tune import tune_models

        output = run_directory()
        results = tune_models(
            train_path=output / "processed" / "train.csv",
            model_path=output / "models" / "best_model.joblib",
            tuning_path=output / "reports" / "metrics" / "tuning_results.csv",
            metadata_path=output / "reports" / "best_model_metadata.json",
        )
        logging.info("Selected %s", results.iloc[0]["Model"])

    @task
    def evaluate_model():
        from src.evaluate import evaluate_model as evaluate

        output = run_directory()
        metrics = evaluate(
            test_path=output / "processed" / "test.csv",
            model_path=output / "models" / "best_model.joblib",
            metrics_dir=output / "reports" / "metrics",
            figures_dir=output / "reports" / "figures",
            metadata_path=output / "reports" / "best_model_metadata.json",
        )
        logging.info("Held-out test macro F1: %.4f", metrics["f1_macro"])

    @task
    def verify_artifacts():
        from src.predict import load_model

        output = run_directory()
        files = [
            "processed/train.csv",
            "processed/test.csv",
            "models/best_model.joblib",
            "reports/best_model_metadata.json",
            "reports/metrics/model_comparison.csv",
            "reports/metrics/tuning_results.csv",
            "reports/metrics/test_metrics.json",
            "reports/metrics/classification_report.csv",
            "reports/metrics/confusion_matrix.csv",
            "reports/figures/confusion_matrix.png",
        ]
        missing = [name for name in files if not (output / name).is_file() or
                   (output / name).stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"Missing or empty artifacts: {missing}")
        load_model(output / "models" / "best_model.joblib")
        metrics = json.loads((output / "reports/metrics/test_metrics.json").read_text())
        if not 0 <= metrics["f1_macro"] <= 1:
            raise ValueError("Invalid test macro F1")
        logging.info("Verified staged model and reports at %s", output)

    validation = validate_data()
    preprocessing = preprocess_data()
    training = train_models()
    tuning = tune_model()
    evaluation = evaluate_model()
    verification = verify_artifacts()
    validation >> preprocessing >> training >> tuning >> evaluation >> verification


mental_health_training_pipeline()
