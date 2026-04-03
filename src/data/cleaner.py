"""Data cleaning module for IMDB reviews dataset.

Handles deduplication, null imputation, and length filtering.
"""

import pandas as pd

from src.config import AppConfig, get_config
from src.data.loader import load_raw_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame, config: AppConfig) -> pd.DataFrame:
    """Clean raw IMDB reviews data.

    Args:
        df: Raw DataFrame from loader.
        config: Application configuration.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    initial_rows = len(df)
    logger.info("cleaning_started", initial_rows=initial_rows)

    # Drop exact duplicate reviews
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[config.data.text_column], keep="first").copy()
    dupes_removed = before_dedup - len(df)
    logger.info("duplicates_removed", count=dupes_removed)

    # Handle nulls in numeric columns
    if "average_rating" in df.columns:
        null_rating = int(df["average_rating"].isnull().sum())
        if null_rating > 0:
            median_rating = df["average_rating"].median()
            df["average_rating"] = df["average_rating"].fillna(median_rating)
            logger.info(
                "filled_nulls",
                column="average_rating",
                count=null_rating,
                fill_value=float(median_rating),
            )

    if "votes" in df.columns:
        null_votes = int(df["votes"].isnull().sum())
        if null_votes > 0:
            median_votes = df["votes"].median()
            df["votes"] = df["votes"].fillna(median_votes)
            logger.info(
                "filled_nulls",
                column="votes",
                count=null_votes,
                fill_value=float(median_votes),
            )

    # Filter out reviews shorter than minimum length
    review_lengths = df[config.data.text_column].str.len()
    short_mask = review_lengths < config.data.min_review_length
    short_count = int(short_mask.sum())
    if short_count > 0:
        df = df[~short_mask]
        logger.info(
            "short_reviews_removed",
            count=short_count,
            min_length=config.data.min_review_length,
        )

    # Validate no nulls in critical columns
    critical_cols = [
        config.data.text_column,
        config.data.target_column,
        config.data.split_column,
    ]
    for col in critical_cols:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            logger.warning("critical_nulls_found", column=col, count=null_count)
            df = df.dropna(subset=[col])

    final_rows = len(df)
    logger.info(
        "cleaning_complete",
        initial_rows=initial_rows,
        final_rows=final_rows,
        rows_removed=initial_rows - final_rows,
    )

    return df.reset_index(drop=True)


if __name__ == "__main__":
    cfg = get_config()
    raw_df = load_raw_data(cfg)
    clean_df = clean_data(raw_df, cfg)
    logger.info(
        "clean_dataset_summary",
        shape=clean_df.shape,
        train_count=int(
            (clean_df[cfg.data.split_column] == cfg.data.train_label).sum()
        ),
        test_count=int(
            (clean_df[cfg.data.split_column] == cfg.data.test_label).sum()
        ),
        pos_ratio=round(
            float(clean_df[cfg.data.target_column].mean()), 4
        ),
    )
