"""Training orchestrator for classical models with MLflow tracking.

Trains all classical pipelines on training data ONLY.
Test set is NOT used here — that happens in evaluate.py.
"""

from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from src.config import AppConfig, PROJECT_ROOT, get_config
from src.data.cleaner import clean_data
from src.data.loader import load_raw_data
from src.features.text_processing import lemmatize_spacy, normalize_text
from src.models.pipelines import (
    build_dummy_pipeline,
    build_lgbm_pipeline,
    build_lr_pipeline,
    build_lr_spacy_pipeline,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_train_data(
    config: AppConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Load, clean, and split data. Return train texts, lemmatized texts, and labels.

    Args:
        config: Application configuration.

    Returns:
        Tuple of (normalized texts, lemmatized texts, labels) for training.
    """
    raw_df = load_raw_data(config)
    clean_df = clean_data(raw_df, config)

    train_mask = clean_df[config.data.split_column] == config.data.train_label
    train_df = clean_df[train_mask].copy()

    logger.info("train_data_prepared", rows=len(train_df))

    x_raw = train_df[config.data.text_column]
    y_train = train_df[config.data.target_column]

    # Normalize for classical models
    x_normalized = x_raw.apply(normalize_text)

    # Lemmatize for spaCy pipeline
    logger.info("starting_lemmatization")
    x_lemmatized = pd.Series(
        lemmatize_spacy(x_raw, config),
        index=x_raw.index,
    )

    return x_normalized, x_lemmatized, y_train


def train_all_classical(config: AppConfig) -> dict[str, Path]:
    """Train all classical pipelines and log to MLflow.

    Args:
        config: Application configuration.

    Returns:
        Dictionary mapping model name to saved model path.
    """
    np.random.seed(config.project.random_seed)

    models_dir = PROJECT_ROOT / config.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(str(PROJECT_ROOT / config.mlflow.tracking_uri))
    mlflow.set_experiment(config.mlflow.experiment_name)

    x_normalized, x_lemmatized, y_train = prepare_train_data(config)

    pipelines = {
        "dummy": (build_dummy_pipeline(config), x_normalized),
        "logistic_regression": (build_lr_pipeline(config), x_normalized),
        "lr_spacy": (build_lr_spacy_pipeline(config), x_lemmatized),
        "lgbm": (build_lgbm_pipeline(config), x_normalized),
    }

    saved_models: dict[str, Path] = {}
    results: list[dict] = []

    for name, (pipeline, x_train) in pipelines.items():
        logger.info("training_model", model=name)

        with mlflow.start_run(run_name=f"train_{name}"):
            pipeline.fit(x_train, y_train)

            # Train metrics for sanity check
            y_pred = pipeline.predict(x_train)
            train_f1 = f1_score(y_train, y_pred)

            # Get probabilities if available
            train_auc = 0.0
            if hasattr(pipeline, "predict_proba"):
                y_prob = pipeline.predict_proba(x_train)[:, 1]
                train_auc = roc_auc_score(y_train, y_prob)

            mlflow.log_param("model_name", name)
            mlflow.log_param("n_train_samples", len(y_train))
            mlflow.log_metric("train_f1", train_f1)
            mlflow.log_metric("train_auc", train_auc)

            # Serialize model
            model_path = models_dir / f"{name}_pipeline.joblib"
            joblib.dump(pipeline, model_path)
            saved_models[name] = model_path

            results.append({
                "model": name,
                "train_f1": round(train_f1, 4),
                "train_auc": round(train_auc, 4),
            })

            logger.info(
                "model_trained",
                model=name,
                train_f1=round(train_f1, 4),
                train_auc=round(train_auc, 4),
                path=str(model_path),
            )

    # Print comparison table
    logger.info("training_complete", n_models=len(results))
    for r in results:
        logger.info(
            "model_summary",
            model=r["model"],
            train_f1=r["train_f1"],
            train_auc=r["train_auc"],
        )

    return saved_models


if __name__ == "__main__":
    cfg = get_config()
    train_all_classical(cfg)
