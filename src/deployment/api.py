"""FastAPI application for sentiment analysis inference."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.sentiment_model import FinancialSentimentModel
from src.utils.config import load_env_variables
import yaml

# Load environment variables
load_env_variables()

# Prometheus metrics
REQUEST_COUNT = Counter(
    "sentiment_request_total", "Total sentiment analysis requests", ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "sentiment_request_latency_seconds", "Request latency in seconds", ["endpoint"]
)
PREDICTION_COUNT = Counter(
    "sentiment_prediction_total", "Total predictions by sentiment", ["sentiment"]
)

# Create FastAPI app
app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="Enterprise-grade sentiment analysis for financial text",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class TextInput(BaseModel):
    """Single text input."""

    text: str = Field(..., description="Text to analyze", min_length=1, max_length=5000)


class BatchTextInput(BaseModel):
    """Batch text input."""

    texts: List[str] = Field(..., description="List of texts to analyze", max_items=100)


class SentimentResponse(BaseModel):
    """Sentiment prediction response."""

    sentiment: str = Field(..., description="Predicted sentiment label")
    confidence: float = Field(..., description="Confidence score", ge=0.0, le=1.0)
    probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    model_version: Optional[str] = Field(None, description="Model version")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchSentimentResponse(BaseModel):
    """Batch sentiment prediction response."""

    predictions: List[SentimentResponse]
    total_count: int
    processing_time: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    timestamp: str


# Global model instance
model: Optional[FinancialSentimentModel] = None
model_config: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model, model_config

    try:
        # Load model configuration
        config_path = Path(__file__).parent.parent.parent / "configs" / "model" / "finbert.yaml"
        with open(config_path, "r") as f:
            model_config = yaml.safe_load(f)

        # Initialize and load model
        model = FinancialSentimentModel(model_config)

        # Try to load best model, fallback to pretrained
        best_model_path = Path("models") / "best_model"
        if best_model_path.exists():
            model.load_model(str(best_model_path))
            print("Loaded best trained model")
        else:
            model.load_model()
            print("Loaded pretrained model")

        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Financial Sentiment Analysis API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=SentimentResponse, tags=["Prediction"])
async def predict_sentiment(input_data: TextInput):
    """
    Predict sentiment for a single text.

    Args:
        input_data: Text input

    Returns:
        Sentiment prediction
    """
    start_time = time.time()

    try:
        if model is None:
            REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Make prediction
        result = model.predict_single(input_data.text)

        # Update metrics
        REQUEST_COUNT.labels(endpoint="/predict", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        PREDICTION_COUNT.labels(sentiment=result["sentiment"]).inc()

        # Format response
        response = SentimentResponse(
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_version="0.1.0",
            timestamp=datetime.now().isoformat(),
        )

        return response

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchSentimentResponse, tags=["Prediction"])
async def predict_sentiment_batch(input_data: BatchTextInput):
    """
    Predict sentiment for multiple texts.

    Args:
        input_data: Batch text input

    Returns:
        Batch sentiment predictions
    """
    start_time = time.time()

    try:
        if model is None:
            REQUEST_COUNT.labels(endpoint="/predict/batch", status="error").inc()
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Make predictions
        labels, confidences, probabilities = model.predict(input_data.texts)

        # Format responses
        predictions = []
        for i, text in enumerate(input_data.texts):
            probs_dict = {
                label: prob
                for label, prob in zip(model.id2label.values(), probabilities[i])
            }

            predictions.append(
                SentimentResponse(
                    sentiment=labels[i],
                    confidence=confidences[i],
                    probabilities=probs_dict,
                    model_version="0.1.0",
                    timestamp=datetime.now().isoformat(),
                )
            )

            # Update metrics
            PREDICTION_COUNT.labels(sentiment=labels[i]).inc()

        # Update metrics
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="success").inc()
        processing_time = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict/batch").observe(processing_time)

        return BatchSentimentResponse(
            predictions=predictions,
            total_count=len(predictions),
            processing_time=processing_time,
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict/batch", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return model.get_model_info()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
