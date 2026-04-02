"""SHAP explanations for the champion model.

Generates global summary plot (top 20 features) and local
waterfall plots for representative examples.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import AppConfig, PROJECT_ROOT, get_config
from src.data.cleaner import clean_data
from src.data.loader import load_raw_data
from src.features.text_processing import normalize_text
from src.utils.logger import get_logger

matplotlib.use("Agg")

logger = get_logger(__name__)


def explain_champion(config: AppConfig, champion_name: str | None = None) -> None:
    """Generate SHAP explanations for the champion model.

    Args:
        config: Application configuration.
        champion_name: Name of the champion model to explain.
            If None, resolved dynamically from eval results.
    """
    if champion_name is None:
        metrics_dir = PROJECT_ROOT / config.paths.metrics_dir
        import glob

        eval_files = sorted(glob.glob(str(metrics_dir / "eval_results_*.json")))
        if eval_files:
            with open(eval_files[-1]) as f:
                eval_data = json.load(f)
            champion_name = eval_data.get("champion", "lr_tuned")
            logger.info("champion_resolved", name=champion_name, source=eval_files[-1])
        else:
            champion_name = "lr_tuned"
            logger.warning("no_eval_results_found", using_default=champion_name)

    raw_df = load_raw_data(config)
    clean_df = clean_data(raw_df, config)

    test_mask = clean_df[config.data.split_column] == config.data.test_label
    test_df = clean_df[test_mask]

    x_test_raw = test_df[config.data.text_column]
    y_test = test_df[config.data.target_column].values

    x_test_normalized = x_test_raw.apply(normalize_text).values

    models_dir = PROJECT_ROOT / config.paths.models_dir
    reports_dir = PROJECT_ROOT / config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    # Load champion model
    model_path = models_dir / f"{champion_name}_pipeline.joblib"
    if not model_path.exists():
        logger.warning("champion_not_found", path=str(model_path))
        return

    pipeline = joblib.load(model_path)

    # Extract vectorizer and classifier
    if hasattr(pipeline, 'calibrated_classifiers_'):
        base = pipeline.calibrated_classifiers_[0].estimator
        vectorizer = base.named_steps["tfidf"]
        classifier = base.named_steps["clf"]
    else:
        vectorizer = pipeline.named_steps["tfidf"]
        classifier = pipeline.named_steps["clf"]

    # Transform test data
    x_tfidf = vectorizer.transform(x_test_normalized)
    feature_names = np.array(vectorizer.get_feature_names_out())

    logger.info(
        "shap_started",
        champion=champion_name,
        n_features=len(feature_names),
        n_samples=x_tfidf.shape[0],
    )

    # Use appropriate explainer based on model type
    model_type = type(classifier).__name__

    if "LGBM" in model_type or "LightGBM" in model_type or "Booster" in model_type:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(x_tfidf)
        # For binary classification, TreeExplainer may return list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif "LogisticRegression" in model_type:
        explainer = shap.LinearExplainer(classifier, x_tfidf)
        shap_values = explainer.shap_values(x_tfidf)
    else:
        # Fallback: use KernelExplainer on a sample
        background = shap.sample(x_tfidf, 100)
        explainer = shap.KernelExplainer(
            classifier.predict_proba, background
        )
        shap_values = explainer.shap_values(x_tfidf[:500])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    logger.info("shap_values_computed", shape=str(np.array(shap_values).shape))

    # Global summary plot (top 20 features)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        x_tfidf,
        feature_names=feature_names,
        max_display=20,
        show=False,
        plot_size=None,
    )
    plt.title(f"SHAP Summary — {champion_name} (Top 20 Features)", fontsize=14)
    plt.tight_layout()
    summary_path = reports_dir / f"eval_shap_summary_{champion_name}_{today}.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close("all")
    logger.info("shap_summary_saved", path=str(summary_path))

    # Local waterfall plots for 3 representative examples
    probs = pipeline.predict_proba(x_test_normalized)[:, 1]

    # Find examples: strong positive, strong negative, borderline
    examples = {
        "strong_positive": int(np.argmax(probs)),
        "strong_negative": int(np.argmin(probs)),
        "borderline": int(np.argmin(np.abs(probs - 0.5))),
    }

    for label, idx in examples.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Get top features for this example
        sv = shap_values[idx]
        if hasattr(sv, "toarray"):
            sv = np.array(sv.toarray()).flatten()
        else:
            sv = np.array(sv).flatten()

        top_idx = np.argsort(np.abs(sv))[-15:]
        top_features = feature_names[top_idx]
        top_values = sv[top_idx]

        colors = ["#ff4444" if v < 0 else "#4488ff" for v in top_values]
        ax.barh(range(len(top_features)), top_values, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.set_xlabel("SHAP Value", fontsize=12)
        ax.set_title(
            f"SHAP Local — {label} (pred={probs[idx]:.3f}, actual={y_test[idx]})",
            fontsize=12,
        )
        ax.grid(alpha=0.3, axis="x")
        plt.tight_layout()

        local_path = reports_dir / f"eval_shap_local_{label}_{champion_name}_{today}.png"
        fig.savefig(local_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("shap_local_saved", example=label, path=str(local_path))

    # Save top features as JSON
    mean_abs_shap = np.abs(shap_values)
    if hasattr(mean_abs_shap, "toarray"):
        mean_abs_shap = np.array(mean_abs_shap.toarray())
    mean_abs_shap = np.array(mean_abs_shap).mean(axis=0).flatten()
    top_20_idx = np.argsort(mean_abs_shap)[-20:][::-1]

    top_features_dict = {
        feature_names[i]: round(float(mean_abs_shap[i]), 6)
        for i in top_20_idx
    }

    metrics_dir = PROJECT_ROOT / config.paths.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    features_path = metrics_dir / f"shap_top_features_{champion_name}_{today}.json"
    with open(features_path, "w") as f:
        json.dump(top_features_dict, f, indent=2)

    logger.info("shap_complete", n_top_features=20, path=str(features_path))


if __name__ == "__main__":
    cfg = get_config()
    explain_champion(cfg)
