from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import sys
from typing import List

# Add parent directory to path to import BC
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="Hate Speech Detection API",
    description="Ensemble-based hate speech detector with three-level classification",
    version="1.0.0"
)

# Load ensemble model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "binary_classifier_ensemble.joblib"
)

try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    raise


# Pydantic models for request/response
class PredictionRequest(BaseModel):
    texts: List[str]
    class Config:
        example = {
            "texts": ["This is a great day!", "You are a fucking idiot"]
        }


class PredictionResult(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    status: str
    predictions: List[PredictionResult]


@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "ok",
        "message": "Hate Speech Detection API is running",
        "endpoints": {
            "POST /predict": "Predict hate speech labels for multiple texts",
            "GET /health": "Health check",
            "GET /info": "Model information"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": model.device if hasattr(model, 'device') else "cpu"
    }


@app.get("/info")
async def info():
    """Get model configuration and information"""
    return {
        "model_type": "Ensemble Hate Speech Classifier",
        "version": "1.0.0",
        "components": {
            "lexical": {
                "type": "TF-IDF + Logistic Regression",
                "weight": model.lexical_weight
            },
            "semantic": {
                "type": "Sentence Transformers (all-MiniLM-L6-v2) + XGBoost",
                "weight": model.semantic_weight
            }
        },
        "classification_levels": {
            "SAFE": "probability <= 0.4",
            "MAYBE_HATE": "0.4 < probability <= 0.6",
            "HATE": "probability > 0.6"
        },
        "thresholds": {
            "low": model.low,
            "high": model.high
        },
        "device": model.device if hasattr(model, 'device') else "cpu",
        "dataset": "Balanced hate speech dataset"
    }


# @app.post("/predict", response_model=PredictionResponse)
# async def predict(request: PredictionRequest):
#     """
#     Predict hate speech labels for provided texts.
    
#     Returns three classification levels:
#     - SAFE: Not hate speech
#     - MAYBE_HATE: Uncertain
#     - HATE: Hate speech detected
#     """
#     try:
#         # Validate input
#         if not request.texts:
#             raise HTTPException(status_code=400, detail="No texts provided")
        
#         if len(request.texts) > 1000:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Maximum 1000 texts per request"
#             )
        
#         # Get predictions
#         predictions = model.predict_with_labels(request.texts)
        
#         # Format response
#         formatted_predictions = [
#             PredictionResult(
#                 label=pred['label'],
#                 confidence=pred['confidence']
#             )
#             for pred in predictions
#         ]
        
#         return PredictionResponse(
#             status="success",
#             predictions=formatted_predictions
#         )
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict hate speech labels for provided texts.

    """
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")

        if len(request.texts) > 1000:
            raise HTTPException(
                status_code=400,
                detail="Maximum 1000 texts per request"
            )

        raw_predictions = model.predict_with_labels(request.texts)

        formatted_predictions = []
        for pred in raw_predictions:
            label = pred["label"]
            confidence = float(pred["confidence"])

            if label == "HATE" and confidence < 0.6:
                label = "MAYBE_HATE"

            formatted_predictions.append(
                PredictionResult(
                    label=label,
                    confidence=confidence
                )
            )

        return PredictionResponse(
            status="success",
            predictions=formatted_predictions
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict-batch")
async def predict_batch(request: PredictionRequest):
    """
    Batch prediction endpoint (same as /predict but for larger batches)
    """
    return await predict(request)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)


