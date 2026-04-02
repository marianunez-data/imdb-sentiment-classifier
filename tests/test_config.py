"""Tests for configuration loading and validation."""

from src.config import AppConfig, get_config


def test_config_loads_successfully() -> None:
    """Config loads from config.yaml without errors."""
    config = get_config()
    assert isinstance(config, AppConfig)


def test_project_name() -> None:
    """Project name matches expected value."""
    config = get_config()
    assert config.project.name == "imdb-sentiment-classifier"


def test_all_sections_exist() -> None:
    """All required configuration sections are present."""
    config = get_config()
    assert config.project is not None
    assert config.data is not None
    assert config.preprocessing is not None
    assert config.models is not None
    assert config.tuning is not None
    assert config.bert is not None
    assert config.threshold is not None
    assert config.calibration is not None
    assert config.evaluation is not None
    assert config.mlflow is not None
    assert config.api is not None
    assert config.monitoring is not None
    assert config.paths is not None


def test_singleton_pattern() -> None:
    """get_config() returns the same instance on repeated calls."""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2


def test_random_seed_is_int() -> None:
    """Random seed is an integer."""
    config = get_config()
    assert isinstance(config.project.random_seed, int)


def test_tfidf_ngram_range() -> None:
    """TF-IDF ngram_range has exactly 2 elements."""
    config = get_config()
    assert len(config.preprocessing.tfidf.ngram_range) == 2


def test_negation_words_present() -> None:
    """Negation words list is non-empty."""
    config = get_config()
    assert len(config.preprocessing.negation_words) > 0
    assert "not" in config.preprocessing.negation_words


def test_data_paths() -> None:
    """Data configuration has valid path settings."""
    config = get_config()
    assert config.data.raw_path.endswith(".tsv")
    assert config.data.separator == "\t"


def test_metrics_dir_exists() -> None:
    """Metrics directory path is configured."""
    config = get_config()
    assert hasattr(config.paths, "metrics_dir")
    assert config.paths.metrics_dir == "reports/metrics"
