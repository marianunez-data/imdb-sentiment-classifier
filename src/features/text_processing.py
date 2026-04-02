"""Text preprocessing functions for classical and transformer models.

Three preprocessing paths:
- normalize_text: Basic normalization for classical models
- lemmatize_spacy: spaCy lemmatization for classical models
- normalize_text_bert: Minimal normalization for BERT (preserves semantics)
"""

import re

import pandas as pd
import spacy

from src.config import AppConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def normalize_text(text: str) -> str:
    """Basic text normalization for classical models.

    Lowercases, removes non-alphabetic characters, and collapses whitespace.

    Args:
        text: Raw review text.

    Returns:
        Normalized text string.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_spacy(
    texts: pd.Series,
    config: AppConfig,
    remove_stopwords: bool = True,
    preserve_negation: bool = True,
) -> list[str]:
    """Lemmatize texts with spaCy, preserving negation words.

    Args:
        texts: Series of raw review texts.
        config: Application configuration.
        remove_stopwords: Whether to remove stopwords.
        preserve_negation: Whether to keep negation words even if
            they are stopwords.

    Returns:
        List of lemmatized text strings.
    """
    nlp = spacy.load(
        config.preprocessing.spacy_model,
        disable=["parser", "ner"],
    )
    negation_set = set(config.preprocessing.negation_words)

    logger.info(
        "lemmatization_started",
        n_texts=len(texts),
        spacy_model=config.preprocessing.spacy_model,
    )

    results = []
    for doc in nlp.pipe(texts, batch_size=1000):
        tokens = []
        for token in doc:
            if token.is_space or token.is_punct:
                continue
            if remove_stopwords and token.is_stop:
                if preserve_negation and token.text.lower() in negation_set:
                    tokens.append(token.lemma_.lower())
                continue
            tokens.append(token.lemma_.lower())
        results.append(" ".join(tokens))

    logger.info("lemmatization_complete", n_texts=len(results))
    return results


def normalize_text_bert(text: str) -> str:
    """Minimal normalization for BERT models.

    Only lowercases, removes HTML tags, and collapses whitespace.
    BERT's tokenizer handles everything else.

    Args:
        text: Raw review text.

    Returns:
        Minimally normalized text for BERT tokenizer.
    """
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
