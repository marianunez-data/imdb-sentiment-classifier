"""DistilBERT fine-tuning with LoRA (PEFT) on GPU.

Trains on the FULL training dataset, not a sample.
Uses HuggingFace Trainer API with PEFT for parameter-efficient fine-tuning.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import AppConfig, PROJECT_ROOT, get_config
from src.bert.dataset import create_hf_dataset, tokenize_dataset
from src.data.cleaner import clean_data
from src.data.loader import load_raw_data
from src.features.text_processing import normalize_text_bert
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_metrics(eval_pred: tuple) -> dict[str, float]:
    """Compute metrics for HuggingFace Trainer.

    Args:
        eval_pred: Tuple of (logits, labels) from Trainer.

    Returns:
        Dictionary with f1 and accuracy.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "f1": f1_score(labels, predictions),
        "accuracy": accuracy_score(labels, predictions),
    }


def fine_tune_bert(config: AppConfig) -> Path:
    """Fine-tune DistilBERT with LoRA on IMDB training data.

    Args:
        config: Application configuration.

    Returns:
        Path to saved fine-tuned model directory.

    Raises:
        RuntimeError: If CUDA GPU is not available.
    """
    # Verify GPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required for BERT fine-tuning. "
            f"torch.cuda.is_available() = {torch.cuda.is_available()}"
        )

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info("gpu_detected", name=gpu_name, memory_gb=round(gpu_mem, 1))

    # Load and prepare data
    raw_df = load_raw_data(config)
    clean_df = clean_data(raw_df, config)

    train_mask = clean_df[config.data.split_column] == config.data.train_label
    train_df = clean_df[train_mask].copy()

    # Minimal BERT normalization
    train_df[config.data.text_column] = train_df[config.data.text_column].apply(
        normalize_text_bert
    )

    logger.info("bert_training_data", n_samples=len(train_df))

    # Create HuggingFace dataset
    train_dataset = create_hf_dataset(train_df, config)

    # Cast label to ClassLabel for stratified split
    from datasets import ClassLabel

    train_dataset = train_dataset.cast_column(
        "label", ClassLabel(names=["negative", "positive"])
    )

    # Split off a small eval set for monitoring (5% of training)
    split = train_dataset.train_test_split(
        test_size=0.05,
        seed=config.project.random_seed,
        stratify_by_column="label",
    )
    train_split = split["train"]
    eval_split = split["test"]

    # Tokenize
    train_tokenized = tokenize_dataset(train_split, config)
    eval_tokenized = tokenize_dataset(eval_split, config)

    # Load base model
    model = AutoModelForSequenceClassification.from_pretrained(
        config.bert.model_name,
        num_labels=2,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.bert.lora_r,
        lora_alpha=config.bert.lora_alpha,
        lora_dropout=config.bert.lora_dropout,
        target_modules=["q_lin", "v_lin"],
    )
    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "lora_applied",
        trainable_params=trainable_params,
        total_params=total_params,
        trainable_pct=round(100 * trainable_params / total_params, 2),
    )

    # Training arguments
    output_dir = PROJECT_ROOT / config.paths.models_dir / "distilbert-finetuned"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.bert.epochs,
        per_device_train_batch_size=config.bert.batch_size,
        per_device_eval_batch_size=config.bert.batch_size * 2,
        learning_rate=config.bert.learning_rate,
        warmup_ratio=config.bert.warmup_ratio,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=config.project.random_seed,
        fp16=True,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        compute_metrics=_compute_metrics,
    )

    # MLflow tracking
    mlflow.set_tracking_uri(str(PROJECT_ROOT / config.mlflow.tracking_uri))
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run(run_name="distilbert_lora"):
        mlflow.log_param("model_name", config.bert.model_name)
        mlflow.log_param("lora_r", config.bert.lora_r)
        mlflow.log_param("lora_alpha", config.bert.lora_alpha)
        mlflow.log_param("epochs", config.bert.epochs)
        mlflow.log_param("batch_size", config.bert.batch_size)
        mlflow.log_param("learning_rate", config.bert.learning_rate)
        mlflow.log_param("trainable_params", trainable_params)
        mlflow.log_param("n_train_samples", len(train_tokenized))

        logger.info("bert_training_started")
        trainer.train()

        # Extract eval metrics from training log history
        # (avoids MLflow callback conflict with standalone evaluate())
        eval_results = {}
        for entry in reversed(trainer.state.log_history):
            if "eval_f1" in entry:
                eval_results = entry
                break

        mlflow.log_metric("eval_f1", eval_results.get("eval_f1", 0.0))
        mlflow.log_metric("eval_accuracy", eval_results.get("eval_accuracy", 0.0))
        mlflow.log_metric("eval_loss", eval_results.get("eval_loss", 0.0))

        logger.info("bert_training_complete", eval_results=eval_results)

    # Save model and tokenizer
    model.save_pretrained(str(output_dir))
    tokenizer = AutoTokenizer.from_pretrained(config.bert.model_name)
    tokenizer.save_pretrained(str(output_dir))

    logger.info("bert_model_saved", path=str(output_dir))

    # Save metrics
    metrics_dir = PROJECT_ROOT / config.paths.metrics_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    metrics = {
        "model": config.bert.model_name,
        "lora_r": config.bert.lora_r,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "eval_f1": round(eval_results.get("eval_f1", 0.0), 4),
        "eval_accuracy": round(eval_results.get("eval_accuracy", 0.0), 4),
        "n_train_samples": len(train_tokenized),
        "date": today,
    }
    metrics_path = metrics_dir / f"bert_training_{today}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return output_dir


if __name__ == "__main__":
    cfg = get_config()
    fine_tune_bert(cfg)
