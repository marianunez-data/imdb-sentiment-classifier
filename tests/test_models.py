"""Tests for model artifacts and text preprocessing."""

from pathlib import Path

import joblib

from src.config import PROJECT_ROOT, get_config
from src.features.text_processing import normalize_text

config = get_config()
MODELS_DIR = PROJECT_ROOT / config.paths.models_dir
CHAMPION_PATH = PROJECT_ROOT / config.api.model_path

EXPECTED_JOBLIB_FILES = [
    "dummy_pipeline.joblib",
    "lgbm_calibrated_pipeline.joblib",
    "lgbm_pipeline.joblib",
    "lgbm_tuned_pipeline.joblib",
    "logistic_regression_calibrated_pipeline.joblib",
    "logistic_regression_pipeline.joblib",
    "lr_spacy_pipeline.joblib",
    "lr_tuned_calibrated_pipeline.joblib",
    "lr_tuned_pipeline.joblib",
]


def test_champion_model_exists() -> None:
    """Champion model joblib file exists on disk."""
    assert CHAMPION_PATH.exists(), f"Missing: {CHAMPION_PATH}"


def test_champion_model_loads() -> None:
    """Champion model loads successfully with joblib."""
    model = joblib.load(CHAMPION_PATH)
    assert model is not None


def test_champion_model_predicts() -> None:
    """Champion model predict_proba returns shape (1, 2)."""
    model = joblib.load(CHAMPION_PATH)
    proba = model.predict_proba(["this is a test review"])
    assert proba.shape == (1, 2)


def test_champion_is_calibrated() -> None:
    """Champion model is a CalibratedClassifierCV."""
    model = joblib.load(CHAMPION_PATH)
    assert hasattr(model, "calibrated_classifiers_")


def test_all_models_exist() -> None:
    """All 9 expected joblib model files exist."""
    for filename in EXPECTED_JOBLIB_FILES:
        path = MODELS_DIR / filename
        assert path.exists(), f"Missing model: {filename}"


def test_distilbert_exists() -> None:
    """DistilBERT fine-tuned directory exists."""
    distilbert_dir = MODELS_DIR / "distilbert-finetuned"
    assert distilbert_dir.is_dir(), "Missing: models/distilbert-finetuned/"


def test_normalize_text() -> None:
    """normalize_text lowercases and removes non-alpha characters."""
    result = normalize_text("Hello World!")
    assert result == "hello world"


def test_normalize_text_removes_html() -> None:
    """normalize_text strips HTML special characters, leaving alpha residue."""
    result = normalize_text("<br />test")
    assert ">" not in result
    assert "<" not in result
    assert "/" not in result
    assert "test" in result
