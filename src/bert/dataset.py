"""HuggingFace Dataset adapter for IMDB reviews.

Converts pandas DataFrame to HuggingFace Dataset format
for use with the Trainer API.
"""

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from src.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_hf_dataset(
    df: pd.DataFrame,
    config: AppConfig,
) -> Dataset:
    """Convert pandas DataFrame to HuggingFace Dataset.

    Args:
        df: DataFrame with text and label columns.
        config: Application configuration.

    Returns:
        HuggingFace Dataset ready for tokenization.
    """
    hf_ds = Dataset.from_dict({
        "text": df[config.data.text_column].tolist(),
        "label": df[config.data.target_column].tolist(),
    })
    logger.info("hf_dataset_created", n_samples=len(hf_ds))
    return hf_ds


def tokenize_dataset(
    dataset: Dataset,
    config: AppConfig,
) -> Dataset:
    """Tokenize dataset using the model's tokenizer.

    Args:
        dataset: HuggingFace Dataset with 'text' column.
        config: Application configuration.

    Returns:
        Tokenized dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.bert.model_name)

    def tokenize_fn(examples: dict) -> dict:
        """Tokenize a batch of examples.

        Args:
            examples: Dictionary with 'text' key.

        Returns:
            Tokenized batch.
        """
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=config.bert.max_length,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, batch_size=1000)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    logger.info(
        "dataset_tokenized",
        n_samples=len(tokenized),
        max_length=config.bert.max_length,
    )
    return tokenized
