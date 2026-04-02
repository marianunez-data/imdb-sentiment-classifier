"""Final one-time evaluation on the test set.

Evaluates ALL trained models on test data with bootstrap confidence
intervals and McNemar's statistical significance test.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import AppConfig, PROJECT_ROOT, get_config
from src.data.cleaner import clean_data
from src.data.loader import load_raw_data
from src.features.text_processing import lemmatize_spacy, normalize_text
from src.utils.logger import get_logger

matplotlib.use("Agg")

logger = get_logger(__name__)


def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: callable,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    use_prob: bool = False,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        metric_fn: Metric function (y_true, y_pred) or (y_true, y_prob).
        n_iterations: Number of bootstrap samples.
        confidence: Confidence level.
        seed: Random seed.
        use_prob: If True, pass y_prob instead of y_pred.

    Returns:
        Tuple of (point estimate, lower bound, upper bound).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_input = y_prob if use_prob else y_pred
    point = float(metric_fn(y_true, y_input))

    scores = []
    for _ in range(n_iterations):
        idx = rng.randint(0, n, size=n)
        bt_true = y_true[idx]
        bt_input = y_input[idx]
        # Skip degenerate samples
        if len(np.unique(bt_true)) < 2:
            continue
        scores.append(float(metric_fn(bt_true, bt_input)))

    alpha = (1 - confidence) / 2
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))
    return point, lower, upper


def _mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> tuple[float, float]:
    """McNemar's test for comparing two classifiers.

    Args:
        y_true: True labels.
        y_pred_a: Predictions from model A (champion).
        y_pred_b: Predictions from model B (baseline).

    Returns:
        Tuple of (chi2 statistic, p-value).
    """
    from scipy.stats import chi2

    correct_a = y_pred_a == y_true
    correct_b = y_pred_b == y_true

    # b: A right, B wrong; c: A wrong, B right
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return 0.0, 1.0

    # McNemar with continuity correction
    chi2_stat = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)

    return float(chi2_stat), float(p_value)


def evaluate_all_models(config: AppConfig) -> dict:
    """Run final evaluation on test set for all models.

    Args:
        config: Application configuration.

    Returns:
        Dictionary with all evaluation results.
    """
    raw_df = load_raw_data(config)
    clean_df = clean_data(raw_df, config)

    # Split data
    train_mask = clean_df[config.data.split_column] == config.data.train_label
    test_mask = clean_df[config.data.split_column] == config.data.test_label
    test_df = clean_df[test_mask]

    x_test_raw = test_df[config.data.text_column]
    y_test = test_df[config.data.target_column].values

    # Prepare text versions
    x_test_normalized = x_test_raw.apply(normalize_text).values

    # Lemmatize test data for spaCy model
    logger.info("lemmatizing_test_data")
    x_test_lemmatized = np.array(lemmatize_spacy(x_test_raw, config))

    models_dir = PROJECT_ROOT / config.paths.models_dir

    # Load threshold
    metrics_dir = PROJECT_ROOT / config.paths.metrics_dir
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    threshold_path = metrics_dir / f"threshold_results_{today}.json"
    optimal_threshold = 0.5
    if threshold_path.exists():
        with open(threshold_path) as f:
            threshold_data = json.load(f)
            optimal_threshold = threshold_data.get("optimal_threshold", 0.5)
    logger.info("using_threshold", threshold=optimal_threshold)

    # Define models to evaluate
    classical_models = {
        "dummy": (models_dir / "dummy_pipeline.joblib", x_test_normalized, 0.5),
        "logistic_regression": (
            models_dir / "logistic_regression_pipeline.joblib",
            x_test_normalized,
            0.5,
        ),
        "lr_spacy": (
            models_dir / "lr_spacy_pipeline.joblib",
            x_test_lemmatized,
            0.5,
        ),
        "lgbm": (models_dir / "lgbm_pipeline.joblib", x_test_normalized, 0.5),
        "lgbm_tuned": (
            models_dir / "lgbm_tuned_pipeline.joblib",
            x_test_normalized,
            0.5,
        ),
        "lgbm_calibrated": (
            models_dir / "lgbm_calibrated_pipeline.joblib",
            x_test_normalized,
            optimal_threshold,
        ),
        "lr_tuned": (
            models_dir / "lr_tuned_pipeline.joblib",
            x_test_normalized,
            0.5,
        ),
        "lr_tuned_calibrated": (
            models_dir / "lr_tuned_calibrated_pipeline.joblib",
            x_test_normalized,
            0.5,
        ),
        "logistic_regression_calibrated": (
            models_dir / "logistic_regression_calibrated_pipeline.joblib",
            x_test_normalized,
            0.5,
        ),
    }

    mlflow.set_tracking_uri(str(PROJECT_ROOT / config.mlflow.tracking_uri))
    mlflow.set_experiment(config.mlflow.experiment_name)

    all_results = {}
    all_predictions = {}
    all_probs = {}

    n_boot = config.evaluation.bootstrap_n_iterations
    conf = config.evaluation.confidence_level

    for name, (model_path, x_data, threshold) in classical_models.items():
        if not model_path.exists():
            logger.warning("model_not_found", model=name, path=str(model_path))
            continue

        pipeline = joblib.load(model_path)
        y_prob = pipeline.predict_proba(x_data)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        all_predictions[name] = y_pred
        all_probs[name] = y_prob

        # Bootstrap CIs
        f1_val, f1_lo, f1_hi = _bootstrap_ci(
            y_test, y_pred, y_prob, f1_score, n_boot, conf
        )
        auc_val, auc_lo, auc_hi = _bootstrap_ci(
            y_test, y_pred, y_prob, roc_auc_score, n_boot, conf, use_prob=True
        )
        prec_val = float(precision_score(y_test, y_pred))
        rec_val = float(recall_score(y_test, y_pred))
        acc_val = float(accuracy_score(y_test, y_pred))

        result = {
            "f1": round(f1_val, 4),
            "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
            "roc_auc": round(auc_val, 4),
            "roc_auc_ci": [round(auc_lo, 4), round(auc_hi, 4)],
            "precision": round(prec_val, 4),
            "recall": round(rec_val, 4),
            "accuracy": round(acc_val, 4),
            "threshold": threshold,
        }
        all_results[name] = result

        with mlflow.start_run(run_name=f"eval_{name}"):
            mlflow.log_param("model_name", name)
            mlflow.log_param("threshold", threshold)
            mlflow.log_metric("test_f1", f1_val)
            mlflow.log_metric("test_auc", auc_val)
            mlflow.log_metric("test_precision", prec_val)
            mlflow.log_metric("test_recall", rec_val)
            mlflow.log_metric("test_accuracy", acc_val)

        logger.info(
            "model_evaluated",
            model=name,
            f1=f"{f1_val:.4f} ({f1_lo:.4f}-{f1_hi:.4f})",
            roc_auc=f"{auc_val:.4f} ({auc_lo:.4f}-{auc_hi:.4f})",
            precision=round(prec_val, 4),
            recall=round(rec_val, 4),
        )

    # BERT evaluation
    bert_model_dir = models_dir / "distilbert-finetuned"
    if bert_model_dir.exists():
        logger.info("evaluating_bert")
        from src.bert.inference import BertPredictor

        bert = BertPredictor(config)
        x_test_raw_arr = x_test_raw.values

        y_prob_bert = bert.predict_proba(x_test_raw_arr)[:, 1]
        y_pred_bert = (y_prob_bert >= 0.5).astype(int)

        all_predictions["distilbert_lora"] = y_pred_bert
        all_probs["distilbert_lora"] = y_prob_bert

        f1_val, f1_lo, f1_hi = _bootstrap_ci(
            y_test, y_pred_bert, y_prob_bert, f1_score, n_boot, conf
        )
        auc_val, auc_lo, auc_hi = _bootstrap_ci(
            y_test, y_pred_bert, y_prob_bert, roc_auc_score,
            n_boot, conf, use_prob=True,
        )
        prec_val = float(precision_score(y_test, y_pred_bert))
        rec_val = float(recall_score(y_test, y_pred_bert))
        acc_val = float(accuracy_score(y_test, y_pred_bert))

        all_results["distilbert_lora"] = {
            "f1": round(f1_val, 4),
            "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
            "roc_auc": round(auc_val, 4),
            "roc_auc_ci": [round(auc_lo, 4), round(auc_hi, 4)],
            "precision": round(prec_val, 4),
            "recall": round(rec_val, 4),
            "accuracy": round(acc_val, 4),
            "threshold": 0.5,
        }

        with mlflow.start_run(run_name="eval_distilbert_lora"):
            mlflow.log_param("model_name", "distilbert_lora")
            mlflow.log_metric("test_f1", f1_val)
            mlflow.log_metric("test_auc", auc_val)
            mlflow.log_metric("test_precision", prec_val)
            mlflow.log_metric("test_recall", rec_val)
            mlflow.log_metric("test_accuracy", acc_val)

        logger.info(
            "bert_evaluated",
            f1=f"{f1_val:.4f} ({f1_lo:.4f}-{f1_hi:.4f})",
            roc_auc=f"{auc_val:.4f} ({auc_lo:.4f}-{auc_hi:.4f})",
        )
    else:
        logger.warning("bert_model_not_found", path=str(bert_model_dir))

    # Determine champion
    champion_name = max(all_results, key=lambda k: all_results[k]["f1"])
    all_results["champion"] = champion_name
    logger.info(
        "champion_declared",
        model=champion_name,
        f1=all_results[champion_name]["f1"],
    )

    # McNemar's test: champion vs baseline
    if "dummy" in all_predictions and champion_name in all_predictions:
        chi2_stat, p_value = _mcnemar_test(
            y_test, all_predictions[champion_name], all_predictions["dummy"]
        )
        all_results["mcnemar_vs_baseline"] = {
            "chi2": round(chi2_stat, 2),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
        }
        logger.info(
            "mcnemar_test",
            champion=champion_name,
            baseline="dummy",
            chi2=round(chi2_stat, 2),
            p_value=round(p_value, 6),
        )

    # Generate plots
    _plot_roc_comparison(y_test, all_probs, config)
    _plot_confusion_matrices(y_test, all_predictions, config)

    # Save results
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results_path = metrics_dir / f"eval_results_{today}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info("evaluation_complete", path=str(results_path))
    return all_results


def _plot_roc_comparison(
    y_test: np.ndarray,
    all_probs: dict[str, np.ndarray],
    config: AppConfig,
) -> None:
    """Plot ROC curves for all models.

    Args:
        y_test: True labels.
        all_probs: Dictionary mapping model name to predicted probabilities.
        config: Application configuration.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for name, probs in all_probs.items():
        RocCurveDisplay.from_predictions(
            y_test, probs, name=name, ax=ax, alpha=0.8
        )

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_title("ROC Curve Comparison — All Models", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    reports_dir = PROJECT_ROOT / config.paths.reports_dir
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    fig_path = reports_dir / f"eval_roc_comparison_all_models_{today}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("roc_plot_saved", path=str(fig_path))


def _plot_confusion_matrices(
    y_test: np.ndarray,
    all_predictions: dict[str, np.ndarray],
    config: AppConfig,
) -> None:
    """Plot confusion matrices for key models.

    Args:
        y_test: True labels.
        all_predictions: Dictionary mapping model name to predictions.
        config: Application configuration.
    """
    # Only plot key models to avoid clutter
    key_models = [
        k for k in ["dummy", "lr_tuned_calibrated", "lgbm_calibrated", "distilbert_lora"]
        if k in all_predictions
    ]

    if not key_models:
        return

    n_models = len(key_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, key_models):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            all_predictions[name],
            ax=ax,
            cmap="Blues",
            display_labels=["Negative", "Positive"],
        )
        ax.set_title(name, fontsize=11)

    fig.suptitle("Confusion Matrix Comparison", fontsize=14, y=1.02)
    fig.tight_layout()

    reports_dir = PROJECT_ROOT / config.paths.reports_dir
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    fig_path = reports_dir / f"eval_confusion_matrices_{today}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("confusion_matrix_plot_saved", path=str(fig_path))


if __name__ == "__main__":
    cfg = get_config()
    evaluate_all_models(cfg)
