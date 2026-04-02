"""Tests for FastAPI inference endpoint."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

POSITIVE_REVIEW = (
    "This movie was absolutely fantastic! The acting was superb, "
    "the story was compelling, and the cinematography was breathtaking. "
    "One of the best films I have ever seen, truly a masterpiece."
)
NEGATIVE_REVIEW = (
    "This movie was absolutely terrible. The acting was awful, "
    "the plot made no sense, and the dialogue was painful to listen to. "
    "A complete waste of time, one of the worst films ever made."
)


def test_health_endpoint() -> None:
    """GET /health returns 200 with status healthy."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_positive() -> None:
    """Positive review returns positive sentiment with high probability."""
    response = client.post("/predict", json={"review": POSITIVE_REVIEW})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["probability"] > 0.7


def test_predict_negative() -> None:
    """Negative review returns negative sentiment with low probability."""
    response = client.post("/predict", json={"review": NEGATIVE_REVIEW})
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "negative"
    assert data["probability"] < 0.3


def test_predict_has_shap_words() -> None:
    """Prediction response includes SHAP word contribution lists."""
    response = client.post("/predict", json={"review": POSITIVE_REVIEW})
    data = response.json()
    assert "top_positive_words" in data
    assert "top_negative_words" in data
    assert isinstance(data["top_positive_words"], list)
    assert isinstance(data["top_negative_words"], list)


def test_predict_has_routing() -> None:
    """Prediction response includes confidence level and routing action."""
    response = client.post("/predict", json={"review": POSITIVE_REVIEW})
    data = response.json()
    assert "confidence_level" in data
    assert "routing_action" in data
    assert data["confidence_level"] in {"high", "medium", "low"}
    assert data["routing_action"] in {
        "auto_classify",
        "human_review",
        "escalate",
    }


def test_predict_empty_review() -> None:
    """Empty review string returns 422 validation error."""
    response = client.post("/predict", json={"review": ""})
    assert response.status_code == 422


def test_predict_routing_high_confidence() -> None:
    """Very positive review gets auto_classify routing."""
    response = client.post("/predict", json={"review": POSITIVE_REVIEW})
    data = response.json()
    assert data["routing_action"] == "auto_classify"
