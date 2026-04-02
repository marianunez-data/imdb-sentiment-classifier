"""Probability calibration for the tuned LGBM model.

Compares isotonic and sigmoid calibration methods using Brier score.
Generates reliability diagram (calibration curve).
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, f1_score

from src.config import AppConfig, PROJECT_ROOT, get_config
from src.data.cleaner import clean_data
from src.data.loader import load_raw_data
from src.features.text_processing import normalize_text
from src.utils.logger import get_logger

matplotlib.use("Agg")

logger = get_logger(__name__)


def calibrate_model(config: AppConfig) -> dict:
    """Apply calibration to tuned LGBM and compare methods.

    Args:
        config: Application configuration.

    Returns:
        Dictionary with calibration results and best method.
    """
    raw_df = load_raw_data(config)
    clean_df = clean_data(raw_df, config)

    train_mask = clean_df[config.data.split_column] == config.data.train_label
    train_df = clean_df[train_mask]

    x_text = train_df[config.data.text_column].apply(normalize_text).values
    y_train = train_df[config.data.target_column].values

    # Load tuned LGBM pipeline
    models_dir = PROJECT_ROOT / config.paths.models_dir
    tuned_path = models_dir / "lgbm_tuned_pipeline.joblib"
    tuned_pipeline = joblib.load(tuned_path)

    logger.info("calibration_started", methods=config.calibration.methods)

    mlflow.set_tracking_uri(str(PROJECT_ROOT / config.mlflow.tracking_uri))
    mlflow.set_experiment(config.mlflow.experiment_name)

    # Get uncalibrated probabilities via cross-validation
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    skf = StratifiedKFold(
        n_splits=config.calibration.cv_folds,
        shuffle=True,
        random_state=config.project.random_seed,
    )
    uncalibrated_probs = cross_val_predict(
        tuned_pipeline, x_text, y_train, cv=skf, method="predict_proba"
    )[:, 1]
    uncalibrated_brier = brier_score_loss(y_train, uncalibrated_probs)

    results = {
        "uncalibrated": {
            "brier_score": round(float(uncalibrated_brier), 6),
        }
    }

    calibrated_models = {}

    for method in config.calibration.methods:
        with mlflow.start_run(run_name=f"calibration_{method}"):
            # Get out-of-fold predictions for fair Brier comparison
            cal_probs_cv = cross_val_predict(
                CalibratedClassifierCV(
                    tuned_pipeline,
                    method=method,
                    cv=config.calibration.cv_folds,
                ),
                x_text, y_train, cv=skf, method="predict_proba",
            )[:, 1]
            cal_brier = brier_score_loss(y_train, cal_probs_cv)

            cal_pred_cv = (cal_probs_cv >= 0.5).astype(int)
            cal_f1 = f1_score(y_train, cal_pred_cv)

            # Also fit on full training data for the final saved model
            cal_clf = CalibratedClassifierCV(
                tuned_pipeline,
                method=method,
                cv=config.calibration.cv_folds,
            )
            cal_clf.fit(x_text, y_train)

            results[method] = {
                "brier_score_cv": round(float(cal_brier), 6),
                "cv_f1": round(float(cal_f1), 4),
            }
            calibrated_models[method] = cal_clf

            mlflow.log_param("calibration_method", method)
            mlflow.log_metric("brier_score_cv", cal_brier)
            mlflow.log_metric("cv_f1", cal_f1)

            logger.info(
                "calibration_result",
                method=method,
                brier_score_cv=round(float(cal_brier), 6),
                cv_f1=round(float(cal_f1), 4),
            )

    # Pick best method (lowest Brier score from CV)
    best_method = min(
        config.calibration.methods,
        key=lambda m: results[m]["brier_score_cv"],
    )
    best_model = calibrated_models[best_method]

    logger.info(
        "best_calibration",
        method=best_method,
        brier_score_cv=results[best_method]["brier_score_cv"],
        uncalibrated_brier=results["uncalibrated"]["brier_score"],
    )

    # Save calibrated model
    cal_path = models_dir / "lgbm_calibrated_pipeline.joblib"
    joblib.dump(best_model, cal_path)
    logger.info("calibrated_model_saved", path=str(cal_path))

    # Generate reliability diagram
    _plot_calibration_curve(
        y_train, uncalibrated_probs, calibrated_models, x_text, config
    )

    # Save metrics
    metrics_dir = PROJECT_ROOT / config.paths.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    results["best_method"] = best_method
    metrics_path = metrics_dir / f"calibration_results_{today}.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


def _plot_calibration_curve(
    y_true: np.ndarray,
    uncalibrated_probs: np.ndarray,
    calibrated_models: dict,
    x_text: np.ndarray,
    config: AppConfig,
) -> None:
    """Generate reliability diagram comparing calibration methods.

    Args:
        y_true: True labels.
        uncalibrated_probs: Probabilities before calibration.
        calibrated_models: Dict of method name -> calibrated model.
        x_text: Training texts for generating calibrated predictions.
        config: Application configuration.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    # Uncalibrated
    prob_true, prob_pred = calibration_curve(
        y_true, uncalibrated_probs, n_bins=10, strategy="uniform"
    )
    brier = brier_score_loss(y_true, uncalibrated_probs)
    ax.plot(
        prob_pred, prob_true, "s-",
        label=f"Uncalibrated (Brier={brier:.4f})",
    )

    # Calibrated curves
    colors = {"isotonic": "o-", "sigmoid": "^-"}
    for method_name, cal_model in calibrated_models.items():
        cal_probs = cal_model.predict_proba(x_text)[:, 1]
        cal_brier = brier_score_loss(y_true, cal_probs)
        pt, pp = calibration_curve(y_true, cal_probs, n_bins=10, strategy="uniform")
        ax.plot(
            pp, pt, colors.get(method_name, "d-"),
            label=f"{method_name.capitalize()} (Brier={cal_brier:.4f})",
        )

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title("Reliability Diagram — Calibration Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    reports_dir = PROJECT_ROOT / config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    fig_path = reports_dir / f"eval_calibration_curve_{today}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("calibration_plot_saved", path=str(fig_path))


if __name__ == "__main__":
    cfg = get_config()
    calibrate_model(cfg)
