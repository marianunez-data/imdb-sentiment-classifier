"""Data loading module for IMDB reviews dataset.

Handles TSV ingestion, type conversions, and initial data inspection.
"""

from pathlib import Path

import pandas as pd

from src.config import AppConfig, PROJECT_ROOT, get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(config: AppConfig) -> pd.DataFrame:
    """Load raw IMDB reviews from TSV file.

    Args:
        config: Application configuration.

    Returns:
        Raw DataFrame with type fixes applied.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    data_path = PROJECT_ROOT / config.data.raw_path

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info("loading_data", path=str(data_path))

    df = pd.read_csv(
        data_path,
        sep=config.data.separator,
        dtype={"votes": "Int64"},
    )

    # Fix '\\N' sentinel values in numeric columns
    for col in ["end_year", "runtime_minutes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].replace("\\N", pd.NA), errors="coerce"
            )

    logger.info(
        "data_loaded",
        rows=len(df),
        columns=len(df.columns),
        column_names=list(df.columns),
    )

    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    if not cols_with_nulls.empty:
        logger.info(
            "null_counts",
            **{col: int(count) for col, count in cols_with_nulls.items()},
        )

    logger.info(
        "dtypes",
        **{col: str(dtype) for col, dtype in df.dtypes.items()},
    )

    return df


if __name__ == "__main__":
    cfg = get_config()
    df = load_raw_data(cfg)
    logger.info(
        "dataset_summary",
        shape=df.shape,
        train_count=int((df[cfg.data.split_column] == cfg.data.train_label).sum()),
        test_count=int((df[cfg.data.split_column] == cfg.data.test_label).sum()),
    )
