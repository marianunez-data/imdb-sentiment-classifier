"""Drift monitoring for IMDB sentiment classifier.

Simulates production drift by splitting the test set into reference
and current halves, then compares prediction distributions and
classification performance using Evidently AI.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.core.datasets import BinaryClassification
from evidently.presets import ClassificationPreset, DataDriftPreset

from src.config import PROJECT_ROOT, get_config
from src.data.cleaner import clean_data
from src.data.loader import load_raw_data
from src.features.text_processing import normalize_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_drift_report() -> dict:
    """Generate drift report comparing reference vs current test data.

    Splits the test set 50/50 into reference (stable baseline) and
    current (simulated production) halves. Compares prediction
    probability distributions and classification performance.

    Returns:
        Drift summary dictionary with metrics and detection status.
    """
    config = get_config()

    logger.info("loading_data_for_drift")
    raw_df = load_raw_data(config)
    df = clean_data(raw_df, config)

    test_df = df[
        df[config.data.split_column] == config.data.test_label
    ].copy()
    test_df = test_df.reset_index(drop=True)

    logger.info("loading_champion_model", path=config.api.model_path)
    model_path = PROJECT_ROOT / config.api.model_path
    model = joblib.load(model_path)

    logger.info("generating_predictions", n_reviews=len(test_df))
    normalized_texts = test_df[config.data.text_column].apply(normalize_text)
    probas = model.predict_proba(normalized_texts)[:, 1]
    predictions = (probas >= 0.5).astype(int)

    test_df["prediction"] = predictions
    test_df["probability"] = probas
    test_df["review_length"] = test_df[config.data.text_column].str.len()

    split_idx = len(test_df) // 2
    reference_df = test_df.iloc[:split_idx].copy()
    current_df = test_df.iloc[split_idx:].copy()

    logger.info(
        "split_test_data",
        n_reference=len(reference_df),
        n_current=len(current_df),
    )

    data_definition = DataDefinition(
        classification=[
            BinaryClassification(
                target=config.data.target_column,
                prediction_labels="prediction",
                prediction_probas="probability",
                pos_label=1,
            ),
        ],
    )

    columns_for_drift = ["probability", "review_length"]
    reference_subset = reference_df[
        columns_for_drift + [config.data.target_column, "prediction"]
    ]
    current_subset = current_df[
        columns_for_drift + [config.data.target_column, "prediction"]
    ]

    ref_dataset = Dataset.from_pandas(
        reference_subset,
        data_definition=data_definition,
    )
    cur_dataset = Dataset.from_pandas(
        current_subset,
        data_definition=data_definition,
    )

    logger.info("running_drift_report")
    drift_report = Report(
        [
            DataDriftPreset(columns=columns_for_drift),
            ClassificationPreset(),
        ],
    )
    snapshot = drift_report.run(
        current_data=cur_dataset,
        reference_data=ref_dataset,
        name="IMDB Sentiment Drift Report",
    )

    reports_dir = PROJECT_ROOT / config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = reports_dir / "drift_report.html"
    snapshot.save_html(str(html_path))
    logger.info("saved_html_report", path=str(html_path))

    mean_prob_ref = float(np.mean(reference_df["probability"].values))
    mean_prob_cur = float(np.mean(current_df["probability"].values))
    prob_diff = abs(mean_prob_ref - mean_prob_cur)
    drift_detected = prob_diff > 0.05

    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    drift_summary = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "n_reference": len(reference_df),
        "n_current": len(current_df),
        "drift_detected": drift_detected,
        "drift_score": round(prob_diff, 6),
        "mean_prob_reference": round(mean_prob_ref, 6),
        "mean_prob_current": round(mean_prob_cur, 6),
    }

    metrics_dir = PROJECT_ROOT / config.paths.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / f"drift_summary_{today}.json"
    with open(json_path, "w") as f:
        json.dump(drift_summary, f, indent=2)
    logger.info("saved_drift_summary", path=str(json_path))

    mlflow.set_tracking_uri(str(PROJECT_ROOT / config.mlflow.tracking_uri))
    mlflow.set_experiment(config.mlflow.experiment_name)
    with mlflow.start_run(run_name=f"drift_monitoring_{today}"):
        mlflow.log_metric("drift_score", drift_summary["drift_score"])
        mlflow.log_metric(
            "mean_prob_reference",
            drift_summary["mean_prob_reference"],
        )
        mlflow.log_metric(
            "mean_prob_current",
            drift_summary["mean_prob_current"],
        )
        mlflow.log_metric(
            "drift_detected",
            int(drift_summary["drift_detected"]),
        )
        mlflow.log_artifact(str(json_path))
        mlflow.log_artifact(str(html_path))
    logger.info("logged_drift_to_mlflow", drift_detected=drift_detected)

    logger.info(
        "drift_report_complete",
        drift_detected=drift_detected,
        drift_score=drift_summary["drift_score"],
    )
    return drift_summary


if __name__ == "__main__":
    summary = generate_drift_report()
    print(json.dumps(summary, indent=2))
