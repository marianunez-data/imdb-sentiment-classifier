"""Configuration module for IMDB sentiment classifier.

Loads and validates all project settings from configs/config.yaml
using Pydantic models for type safety.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, field_validator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str
    version: str
    random_seed: int
    description: str


class DataConfig(BaseModel):
    """Data loading and splitting configuration."""

    raw_path: str
    separator: str
    text_column: str
    target_column: str
    split_column: str
    train_label: str
    test_label: str
    min_review_length: int
    max_review_length: int


class TfidfConfig(BaseModel):
    """TF-IDF vectorizer configuration."""

    min_df: float
    max_df: float
    ngram_range: list[int]
    max_features: int
    sublinear_tf: bool = True

    @field_validator("ngram_range")
    @classmethod
    def validate_ngram_range(cls, v: list[int]) -> list[int]:
        """Ensure ngram_range has exactly 2 elements."""
        if len(v) != 2:
            raise ValueError("ngram_range must have exactly 2 elements")
        return v


class PreprocessingConfig(BaseModel):
    """Text preprocessing configuration."""

    spacy_model: str
    negation_words: list[str]
    tfidf: TfidfConfig


class BaselineConfig(BaseModel):
    """Baseline model configuration."""

    strategy: str


class LogisticRegressionConfig(BaseModel):
    """Logistic regression configuration."""

    solver: str
    max_iter: int
    C: float


class LgbmConfig(BaseModel):
    """LightGBM configuration."""

    n_estimators: int
    learning_rate: float
    num_leaves: int
    random_state: int


class ModelsConfig(BaseModel):
    """All model configurations."""

    baseline: BaselineConfig
    logistic_regression: LogisticRegressionConfig
    lgbm: LgbmConfig


class TuningConfig(BaseModel):
    """Optuna hyperparameter tuning configuration."""

    n_trials: int
    cv_folds: int
    scoring_metric: str
    timeout: int


class BertConfig(BaseModel):
    """DistilBERT fine-tuning configuration."""

    model_name: str
    max_length: int
    batch_size: int
    learning_rate: float
    epochs: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    warmup_ratio: float


class ThresholdConfig(BaseModel):
    """Threshold optimization configuration."""

    search_range: list[float]
    step: float
    optimize_for: str


class CalibrationConfig(BaseModel):
    """Probability calibration configuration."""

    methods: list[str]
    cv_folds: int


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    bootstrap_n_iterations: int
    confidence_level: float


class MlflowConfig(BaseModel):
    """MLflow tracking configuration."""

    tracking_uri: str
    experiment_name: str


class ApiConfig(BaseModel):
    """API serving configuration."""

    host: str
    port: int
    model_path: str


class MonitoringConfig(BaseModel):
    """Drift monitoring configuration."""

    reference_data_ratio: float


class PathsConfig(BaseModel):
    """File paths configuration."""

    models_dir: str
    reports_dir: str
    metrics_dir: str
    logs_dir: str


class AppConfig(BaseModel):
    """Root configuration container for the entire application."""

    project: ProjectConfig
    data: DataConfig
    preprocessing: PreprocessingConfig
    models: ModelsConfig
    tuning: TuningConfig
    bert: BertConfig
    threshold: ThresholdConfig
    calibration: CalibrationConfig
    evaluation: EvaluationConfig
    mlflow: MlflowConfig
    api: ApiConfig
    monitoring: MonitoringConfig
    paths: PathsConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file and return parsed dictionary.

    Args:
        path: Path to YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Load and validate configuration from config.yaml (singleton).

    Returns:
        Validated AppConfig instance.
    """
    raw = _load_yaml(CONFIG_PATH)
    config = AppConfig(**raw)

    data_path = PROJECT_ROOT / config.data.raw_path
    if not data_path.exists():
        import warnings

        warnings.warn(
            f"Data file not found at {data_path}. "
            "Download the dataset before running the pipeline.",
            stacklevel=2,
        )

    return config
