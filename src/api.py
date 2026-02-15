"""
FastAPI service for phishing detection.
Provides REST API endpoints for prediction and health check.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import (
    MODEL_PATH, MODEL_VERSION, PROJECT_ROOT,
    METRICS_PATH, CONFUSION_MATRIX_PATH, THRESHOLD_METRICS_PATH,
    TOP_FEATURES_PHISHING_PATH, TOP_FEATURES_LEGIT_PATH
)
from src.train import load_model
from src.explain import explain_email
from src.utils import load_json, load_csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {}


def get_model():
    """Get cached model, loading if necessary."""
    if 'pipeline' not in _model_cache:
        logger.info("Loading model...")
        _model_cache['pipeline'] = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
    return _model_cache['pipeline']


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    # Startup: Load model
    logger.info("Starting up Phish Detector API...")
    try:
        get_model()
        logger.info("Model pre-loaded successfully")
    except FileNotFoundError:
        logger.warning("Model not found. Please train the model first.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Phish Detector API...")
    _model_cache.clear()


# Create FastAPI app
app = FastAPI(
    title="Phish Detector API",
    description="ML-powered phishing email detection service",
    version=MODEL_VERSION,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    subject: Optional[str] = Field(default="", description="Email subject")
    body: Optional[str] = Field(default="", description="Email body")
    text: Optional[str] = Field(default=None, description="Combined text (alternative to subject+body)")
    urls: Optional[str] = Field(default="", description="URLs in the email")
    
    class Config:
        json_schema_extra = {
            "example": {
                "subject": "URGENT: Verify your account",
                "body": "Click the link below to verify your identity.",
                "urls": "http://suspicious-site.xyz/verify"
            }
        }


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    label: str = Field(description="Prediction label: PHISHING or LEGIT")
    score: float = Field(description="Phishing probability score (0-1)")
    reasons: List[str] = Field(description="Human-readable explanation reasons")
    model_version: str = Field(description="Model version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "PHISHING",
                "score": 0.87,
                "reasons": [
                    "Contains URL with suspicious TLD - indicates phishing",
                    "Contains 3 phishing keyword(s) - indicates phishing"
                ],
                "model_version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    model_loaded: bool
    model_version: str


class HeuristicsResponse(BaseModel):
    """Extended response including heuristic features."""
    label: str
    score: float
    reasons: List[str]
    heuristics: dict
    model_version: str


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the service status and whether the model is loaded.
    """
    model_loaded = 'pipeline' in _model_cache
    
    if not model_loaded:
        try:
            get_model()
            model_loaded = True
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """
    Predict if an email is phishing.
    
    Accepts either:
    - `text`: Combined subject and body
    - `subject` + `body`: Separate fields (will be combined)
    
    Returns prediction label, confidence score, and explanation.
    """
    # Determine text input
    if request.text:
        text = request.text
    else:
        subject = request.subject or ""
        body = request.body or ""
        text = f"{subject}\n{body}"
    
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Email content is required. Provide 'text' or 'subject'/'body'."
        )
    
    urls = request.urls or ""
    
    try:
        pipeline = get_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please train the model first."
        )
    except Exception as e:
        logger.error(f"Model loading error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to load model."
        )
    
    try:
        result = explain_email(pipeline, text, urls)
        
        return PredictResponse(
            label=result['label'],
            score=result['score'],
            reasons=result['reasons'],
            model_version=MODEL_VERSION,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/detailed", response_model=HeuristicsResponse, tags=["Prediction"])
async def predict_detailed(request: PredictRequest):
    """
    Predict with detailed heuristics information.
    
    Same as /predict but includes all extracted heuristic features.
    """
    # Determine text input
    if request.text:
        text = request.text
    else:
        subject = request.subject or ""
        body = request.body or ""
        text = f"{subject}\n{body}"
    
    if not text.strip():
        raise HTTPException(
            status_code=400,
            detail="Email content is required."
        )
    
    urls = request.urls or ""
    
    try:
        pipeline = get_model()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not available."
        )
    
    try:
        result = explain_email(pipeline, text, urls)
        
        return HeuristicsResponse(
            label=result['label'],
            score=result['score'],
            reasons=result['reasons'],
            heuristics=result['heuristics'],
            model_version=MODEL_VERSION,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics", tags=["Metrics"])
async def get_metrics():
    """
    Get model evaluation metrics.
    
    Returns all metrics including confusion matrix, threshold analysis,
    and top features.
    """
    result = {
        "metrics": None,
        "confusion_matrix": None,
        "threshold_analysis": None,
        "top_features_phishing": None,
        "top_features_legit": None,
    }
    
    # Load metrics
    if METRICS_PATH.exists():
        try:
            result["metrics"] = load_json(METRICS_PATH)
        except Exception as e:
            logger.warning(f"Could not load metrics: {e}")
    
    # Load confusion matrix
    if CONFUSION_MATRIX_PATH.exists():
        try:
            df = load_csv(CONFUSION_MATRIX_PATH)
            result["confusion_matrix"] = df.to_dict(orient='records')
        except Exception as e:
            logger.warning(f"Could not load confusion matrix: {e}")
    
    # Load threshold analysis
    if THRESHOLD_METRICS_PATH.exists():
        try:
            df = load_csv(THRESHOLD_METRICS_PATH)
            result["threshold_analysis"] = df.to_dict(orient='records')
        except Exception as e:
            logger.warning(f"Could not load threshold analysis: {e}")
    
    # Load top features
    if TOP_FEATURES_PHISHING_PATH.exists():
        try:
            df = load_csv(TOP_FEATURES_PHISHING_PATH)
            result["top_features_phishing"] = df.head(20).to_dict(orient='records')
        except Exception as e:
            logger.warning(f"Could not load phishing features: {e}")
    
    if TOP_FEATURES_LEGIT_PATH.exists():
        try:
            df = load_csv(TOP_FEATURES_LEGIT_PATH)
            result["top_features_legit"] = df.head(20).to_dict(orient='records')
        except Exception as e:
            logger.warning(f"Could not load legit features: {e}")
    
    if result["metrics"] is None:
        raise HTTPException(
            status_code=404,
            detail="Metrics not found. Run evaluation first: python -m src.evaluate"
        )
    
    return result


@app.get("/api", tags=["Info"])
async def api_info():
    """API information endpoint."""
    return {
        "name": "Phish Detector API",
        "version": MODEL_VERSION,
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_detailed": "/predict/detailed",
            "docs": "/docs",
            "frontend": "/",
        }
    }


# Serve frontend
FRONTEND_DIR = PROJECT_ROOT / "frontend"


@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """Serve the frontend HTML page."""
    frontend_path = FRONTEND_DIR / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    return {
        "message": "Phish Detector API",
        "frontend": "Frontend not found. Place index.html in frontend/ directory.",
        "docs": "/docs"
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    
    uvicorn.run(
        "src.api:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
