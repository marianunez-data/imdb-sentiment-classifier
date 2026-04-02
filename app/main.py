"""FastAPI inference endpoint for IMDB sentiment classifier.

Serves the champion LR calibrated model with SHAP explanations
and confidence-based routing for human-in-the-loop workflows.
"""

from datetime import datetime, timezone

import joblib
import numpy as np
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import PROJECT_ROOT, get_config
from src.features.text_processing import normalize_text

config = get_config()

app = FastAPI(
    title="IMDB Sentiment Classifier",
    version=config.project.version,
    description=config.project.description,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = PROJECT_ROOT / config.api.model_path
model = joblib.load(MODEL_PATH)

base_pipeline = model.calibrated_classifiers_[0].estimator
vectorizer = base_pipeline.named_steps["tfidf"]
classifier = base_pipeline.named_steps["clf"]
explainer = shap.LinearExplainer(classifier, vectorizer.transform([""]))


class PredictRequest(BaseModel):
    """Input schema for sentiment prediction."""

    review: str = Field(..., min_length=1, description="Movie review text")


class ShapWord(BaseModel):
    """A word with its SHAP contribution score."""

    word: str
    score: float


class PredictResponse(BaseModel):
    """Output schema for sentiment prediction."""

    sentiment: str
    probability: float
    confidence_level: str
    routing_action: str
    top_positive_words: list[ShapWord]
    top_negative_words: list[ShapWord]


class HealthResponse(BaseModel):
    """Output schema for health check."""

    status: str
    model_name: str
    timestamp: str


def _get_routing(prob: float) -> tuple[str, str]:
    """Determine confidence level and routing action from probability.

    Args:
        prob: Predicted probability of positive class.

    Returns:
        Tuple of (confidence_level, routing_action).
    """
    if prob > 0.85 or prob < 0.15:
        return "high", "auto_classify"
    if prob > 0.60 or prob < 0.40:
        return "medium", "human_review"
    return "low", "escalate"


def _get_shap_words(
    text: str,
    top_n: int = 5,
) -> tuple[list[ShapWord], list[ShapWord]]:
    """Compute SHAP word contributions for a single review.

    Args:
        text: Normalized review text.
        top_n: Number of top words to return per direction.

    Returns:
        Tuple of (top_positive_words, top_negative_words).
    """
    x_tfidf = vectorizer.transform([text])
    shap_values = explainer.shap_values(x_tfidf)

    feature_names = np.array(vectorizer.get_feature_names_out())
    nonzero_indices = x_tfidf.nonzero()[1]

    if len(nonzero_indices) == 0:
        return [], []

    values = shap_values[0, nonzero_indices]
    names = feature_names[nonzero_indices]

    sorted_idx = np.argsort(values)

    positive_idx = sorted_idx[values[sorted_idx] > 0][-top_n:][::-1]
    negative_idx = sorted_idx[values[sorted_idx] < 0][:top_n]

    top_positive = [
        ShapWord(word=names[i], score=round(float(values[i]), 4))
        for i in positive_idx
    ]
    top_negative = [
        ShapWord(word=names[i], score=round(float(values[i]), 4))
        for i in negative_idx
    ]

    return top_positive, top_negative


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict sentiment for a movie review.

    Args:
        request: Review text input.

    Returns:
        Sentiment prediction with SHAP explanations and routing.
    """
    normalized = normalize_text(request.review)
    prob = float(model.predict_proba([normalized])[0, 1])
    sentiment = "positive" if prob >= 0.5 else "negative"
    confidence_level, routing_action = _get_routing(prob)
    top_positive, top_negative = _get_shap_words(normalized)

    return PredictResponse(
        sentiment=sentiment,
        probability=round(prob, 4),
        confidence_level=confidence_level,
        routing_action=routing_action,
        top_positive_words=top_positive,
        top_negative_words=top_negative,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Service status, model name, and current timestamp.
    """
    return HealthResponse(
        status="healthy",
        model_name=MODEL_PATH.stem,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


from mangum import Mangum  # noqa: E402

handler = Mangum(app)
