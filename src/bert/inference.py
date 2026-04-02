"""BERT inference pipeline for serving predictions.

Loads the fine-tuned DistilBERT + LoRA model and provides
prediction functions compatible with the evaluation pipeline.
"""

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import AppConfig, PROJECT_ROOT
from src.features.text_processing import normalize_text_bert
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BertPredictor:
    """Wrapper for DistilBERT + LoRA inference.

    Provides sklearn-compatible predict() and predict_proba() methods
    so it can be used interchangeably with classical pipelines.

    Args:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize predictor by loading fine-tuned model.

        Args:
            config: Application configuration.
        """
        model_dir = PROJECT_ROOT / config.paths.models_dir / "distilbert-finetuned"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.max_length = config.bert.max_length
        self.batch_size = config.bert.batch_size * 2

        # Load base model + LoRA weights
        base_model = AutoModelForSequenceClassification.from_pretrained(
            config.bert.model_name,
            num_labels=2,
        )
        self.model = PeftModel.from_pretrained(base_model, str(model_dir))
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            "bert_predictor_loaded",
            device=str(self.device),
            model_dir=str(model_dir),
        )

    def predict_proba(self, texts: np.ndarray) -> np.ndarray:
        """Predict probabilities for a batch of texts.

        Args:
            texts: Array of raw text strings.

        Returns:
            Array of shape (n_samples, 2) with class probabilities.
        """
        all_probs = []

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i: i + self.batch_size]

            # Normalize for BERT
            normalized = [normalize_text_bert(t) for t in batch_texts]

            encodings = self.tokenizer(
                normalized,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**encodings)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)

    def predict(self, texts: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary labels for a batch of texts.

        Args:
            texts: Array of raw text strings.
            threshold: Decision threshold for positive class.

        Returns:
            Array of binary predictions.
        """
        probs = self.predict_proba(texts)
        return (probs[:, 1] >= threshold).astype(int)
