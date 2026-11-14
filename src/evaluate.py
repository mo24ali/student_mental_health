"""Evaluate the selected pipeline once on the held-out test split."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)

from src.data_validation import readable_labels
from src.preprocessing import PROCESSED_DIR, PROJECT_ROOT, load_processed
from src.tune import METADATA_PATH, MODEL_PATH


METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
LABELS = ["Low", "Medium", "High"]


def evaluate_model(
    test_path=PROCESSED_DIR / "test.csv",
    model_path=MODEL_PATH,
    metrics_dir=METRICS_DIR,
    figures_dir=FIGURES_DIR,
    metadata_path=METADATA_PATH,
):
    """Score the saved pipeline and write notebook-compatible evaluation files."""
    X_test, y_test = load_processed(test_path)
    model = joblib.load(model_path)
    predictions = readable_labels(model.predict(X_test))
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro")),
        "recall_macro": float(recall_score(y_test, predictions, average="macro")),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }
    report = classification_report(
        y_test, predictions, labels=LABELS, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=LABELS)
    metrics_dir, figures_dir, metadata_path = map(
        Path, (metrics_dir, figures_dir, metadata_path)
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).T.to_csv(metrics_dir / "classification_report.csv")
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(
        metrics_dir / "confusion_matrix.csv"
    )
    (metrics_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=4) + "\n")

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=LABELS)
    display.plot(cmap="Blues", values_format="d")
    title = "Confusion Matrix"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        title += f" - {metadata['model_name']}"
        metadata.update({
            "test_macro_f1": metrics["f1_macro"],
            "test_accuracy": metrics["accuracy"],
            "test_samples": len(y_test),
        })
        metadata_path.write_text(json.dumps(metadata, indent=4) + "\n")
    plt.title(title)
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    return metrics


if __name__ == "__main__":
    print(evaluate_model())
