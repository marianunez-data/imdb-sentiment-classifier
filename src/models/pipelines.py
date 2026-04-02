"""Sklearn pipeline definitions for IMDB sentiment classification.

Each model gets its OWN Pipeline with its OWN TfidfVectorizer instance.
No shared vectorizers — this prevents data leakage between models.
"""

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

from src.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _build_tfidf(config: AppConfig) -> TfidfVectorizer:
    """Create a fresh TfidfVectorizer from config.

    Args:
        config: Application configuration.

    Returns:
        New TfidfVectorizer instance.
    """
    tfidf_cfg = config.preprocessing.tfidf
    return TfidfVectorizer(
        min_df=tfidf_cfg.min_df,
        max_df=tfidf_cfg.max_df,
        ngram_range=tuple(tfidf_cfg.ngram_range),
        max_features=tfidf_cfg.max_features,
        sublinear_tf=tfidf_cfg.sublinear_tf,
    )


def build_dummy_pipeline(config: AppConfig) -> Pipeline:
    """Build baseline pipeline: TfidfVectorizer -> DummyClassifier.

    Args:
        config: Application configuration.

    Returns:
        Sklearn Pipeline with independent vectorizer.
    """
    pipeline = Pipeline([
        ("tfidf", _build_tfidf(config)),
        ("clf", DummyClassifier(
            strategy=config.models.baseline.strategy,
            random_state=config.project.random_seed,
        )),
    ])
    logger.info("pipeline_built", model="dummy", strategy=config.models.baseline.strategy)
    return pipeline


def build_lr_pipeline(config: AppConfig) -> Pipeline:
    """Build logistic regression pipeline: TfidfVectorizer -> LogisticRegression.

    Args:
        config: Application configuration.

    Returns:
        Sklearn Pipeline with independent vectorizer.
    """
    lr_cfg = config.models.logistic_regression
    pipeline = Pipeline([
        ("tfidf", _build_tfidf(config)),
        ("clf", LogisticRegression(
            solver=lr_cfg.solver,
            max_iter=lr_cfg.max_iter,
            C=lr_cfg.C,
            random_state=config.project.random_seed,
        )),
    ])
    logger.info("pipeline_built", model="logistic_regression", solver=lr_cfg.solver)
    return pipeline


def build_lr_spacy_pipeline(config: AppConfig) -> Pipeline:
    """Build logistic regression pipeline for pre-lemmatized text.

    Texts must be lemmatized with spaCy BEFORE entering this pipeline.
    Gets its OWN vectorizer because lemmatized text has different vocabulary.

    Args:
        config: Application configuration.

    Returns:
        Sklearn Pipeline with independent vectorizer.
    """
    lr_cfg = config.models.logistic_regression
    pipeline = Pipeline([
        ("tfidf", _build_tfidf(config)),
        ("clf", LogisticRegression(
            solver=lr_cfg.solver,
            max_iter=lr_cfg.max_iter,
            C=lr_cfg.C,
            random_state=config.project.random_seed,
        )),
    ])
    logger.info("pipeline_built", model="lr_spacy", solver=lr_cfg.solver)
    return pipeline


def build_lgbm_pipeline(config: AppConfig) -> Pipeline:
    """Build LightGBM pipeline: TfidfVectorizer -> LGBMClassifier.

    Args:
        config: Application configuration.

    Returns:
        Sklearn Pipeline with independent vectorizer.
    """
    lgbm_cfg = config.models.lgbm
    pipeline = Pipeline([
        ("tfidf", _build_tfidf(config)),
        ("clf", LGBMClassifier(
            n_estimators=lgbm_cfg.n_estimators,
            learning_rate=lgbm_cfg.learning_rate,
            num_leaves=lgbm_cfg.num_leaves,
            random_state=lgbm_cfg.random_state,
            verbose=-1,
        )),
    ])
    logger.info("pipeline_built", model="lgbm", n_estimators=lgbm_cfg.n_estimators)
    return pipeline


def build_lgbm_tuned_pipeline(
    config: AppConfig,
    best_params: dict,
) -> Pipeline:
    """Build LightGBM pipeline with Optuna-tuned hyperparameters.

    Args:
        config: Application configuration.
        best_params: Best hyperparameters from Optuna tuning.

    Returns:
        Sklearn Pipeline with independent vectorizer and tuned LGBM.
    """
    pipeline = Pipeline([
        ("tfidf", _build_tfidf(config)),
        ("clf", LGBMClassifier(
            verbose=-1,
            random_state=config.project.random_seed,
            **best_params,
        )),
    ])
    logger.info("pipeline_built", model="lgbm_tuned", params=best_params)
    return pipeline
