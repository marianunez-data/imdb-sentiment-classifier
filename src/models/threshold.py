"""Business-driven threshold optimization.

Sweeps thresholds on cross-validation predictions to find the
value that maximizes the configured metric (F1 by default).
"""

import json
from datetime import datetime, timezone

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.config import AppConfig, PROJECT_ROOT, get_config
from src.data.cleaner import clean_data
from src.data.loader import load_raw_data
from src.features.text_processing import normalize_text
from src.utils.logger import get_logger

matplotlib.use("Agg")

logger = get_logger(__name__)

METRIC_FUNCTIONS = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
}


def optimize_threshold(
    config: AppConfig,
    model_name: str = "lr_tuned",
) -> dict:
    """Find optimal decision threshold on training CV predictions.

    Args:
        config: Application configuration.
        model_name: Name of the model pipeline to optimize (without _pipeline.joblib).

    Returns:
        Dictionary with optimal threshold and metrics at that threshold.
    """
    raw_df = load_raw_data(config)
    clean_df = clean_data(raw_df, config)

    train_mask = clean_df[config.data.split_column] == config.data.train_label
    train_df = clean_df[train_mask]

    x_text = train_df[config.data.text_column].apply(normalize_text).values
    y_train = train_df[config.data.target_column].values

    # Load model
    models_dir = PROJECT_ROOT / config.paths.models_dir
    model_path = models_dir / f"{model_name}_pipeline.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    cal_model = joblib.load(model_path)
    logger.info("model_loaded_for_threshold", model=model_name, path=str(model_path))

    logger.info(
        "threshold_optimization_started",
        model=model_name,
        range=config.threshold.search_range,
        step=config.threshold.step,
        optimize_for=config.threshold.optimize_for,
    )

    # Get CV probabilities
    skf = StratifiedKFold(
        n_splits=config.calibration.cv_folds,
        shuffle=True,
        random_state=config.project.random_seed,
    )
    cv_probs = cross_val_predict(
        cal_model, x_text, y_train, cv=skf, method="predict_proba"
    )[:, 1]

    # Sweep thresholds
    thresholds = np.arange(
        config.threshold.search_range[0],
        config.threshold.search_range[1] + config.threshold.step,
        config.threshold.step,
    )

    sweep_results = []
    for t in thresholds:
        y_pred_t = (cv_probs >= t).astype(int)
        # Skip thresholds that produce all-same predictions
        if len(np.unique(y_pred_t)) < 2:
            continue
        sweep_results.append({
            "threshold": round(float(t), 4),
            "f1": f1_score(y_train, y_pred_t),
            "precision": precision_score(y_train, y_pred_t),
            "recall": recall_score(y_train, y_pred_t),
        })

    # Find optimal
    optimize_metric = config.threshold.optimize_for
    best_result = max(sweep_results, key=lambda r: r[optimize_metric])

    logger.info(
        "optimal_threshold_found",
        threshold=best_result["threshold"],
        f1=round(best_result["f1"], 4),
        precision=round(best_result["precision"], 4),
        recall=round(best_result["recall"], 4),
    )

    # Default threshold comparison
    default_pred = (cv_probs >= 0.5).astype(int)
    default_f1 = f1_score(y_train, default_pred)
    logger.info(
        "default_threshold_comparison",
        default_f1=round(float(default_f1), 4),
        optimized_f1=round(best_result["f1"], 4),
        improvement=round(best_result["f1"] - float(default_f1), 4),
    )

    # Log to MLflow
    mlflow.set_tracking_uri(str(PROJECT_ROOT / config.mlflow.tracking_uri))
    mlflow.set_experiment(config.mlflow.experiment_name)
    with mlflow.start_run(run_name="threshold_optimization"):
        mlflow.log_param("optimize_for", optimize_metric)
        mlflow.log_param("optimal_threshold", best_result["threshold"])
        mlflow.log_metric("optimal_f1", best_result["f1"])
        mlflow.log_metric("optimal_precision", best_result["precision"])
        mlflow.log_metric("optimal_recall", best_result["recall"])
        mlflow.log_metric("default_f1", float(default_f1))

    # Generate plot
    _plot_threshold_sweep(sweep_results, best_result, config)

    # Save results
    metrics_dir = PROJECT_ROOT / config.paths.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    output = {
        "optimal_threshold": best_result["threshold"],
        "optimal_f1": round(best_result["f1"], 4),
        "optimal_precision": round(best_result["precision"], 4),
        "optimal_recall": round(best_result["recall"], 4),
        "default_f1": round(float(default_f1), 4),
        "improvement_over_default": round(
            best_result["f1"] - float(default_f1), 4
        ),
        "date": today,
    }
    metrics_path = metrics_dir / f"threshold_results_{today}.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("threshold_metrics_saved", path=str(metrics_path))
    return output


def _plot_threshold_sweep(
    sweep_results: list[dict],
    best_result: dict,
    config: AppConfig,
) -> None:
    """Plot F1, precision, recall vs threshold.

    Args:
        sweep_results: List of dicts with threshold and metrics.
        best_result: Best result dict.
        config: Application configuration.
    """
    thresholds = [r["threshold"] for r in sweep_results]
    f1_scores = [r["f1"] for r in sweep_results]
    precisions = [r["precision"] for r in sweep_results]
    recalls = [r["recall"] for r in sweep_results]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(thresholds, f1_scores, "b-", linewidth=2, label="F1")
    ax.plot(thresholds, precisions, "g--", linewidth=1.5, label="Precision")
    ax.plot(thresholds, recalls, "r--", linewidth=1.5, label="Recall")

    # Mark optimal
    ax.axvline(
        best_result["threshold"],
        color="black",
        linestyle=":",
        alpha=0.7,
        label=f"Optimal t={best_result['threshold']}",
    )
    ax.scatter(
        [best_result["threshold"]], [best_result["f1"]],
        color="black", s=100, zorder=5,
    )

    # Mark default
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5, label="Default t=0.50")

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Threshold Optimization — F1, Precision, Recall vs Threshold",
        fontsize=14,
    )
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(0.0, 1.0)

    reports_dir = PROJECT_ROOT / config.paths.reports_dir
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    fig_path = reports_dir / f"eval_threshold_optimization_{today}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("threshold_plot_saved", path=str(fig_path))


if __name__ == "__main__":
    import sys

    cfg = get_config()
    model_name = sys.argv[1] if len(sys.argv) > 1 else "lr_tuned"
    optimize_threshold(cfg, model_name=model_name)
