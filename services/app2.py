# hatespeech/services/app2.py
import os
import sys
from typing import List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from ensemble2 import HateSpeechEnsemble

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
ensemble = HateSpeechEnsemble(model_dir=MODEL_DIR)

app = FastAPI(
    title="Hate Speech Detection API (Ensemble v2)",
    version="2.1.0"
)

# ----------------------------
# Schemas
# ----------------------------
class PredictionRequest(BaseModel):
    texts: List[str]


class PredictionResult(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    status: str
    predictions: List[PredictionResult]


class PredictionDebugItem(BaseModel):
    text: str
    p_lr: float
    p_xgb_tfidf: float
    p_xgb_semantic: float
    flags: dict
    multiplier: float
    raw_ensemble_prob: float
    final_prob: float
    calibrated_prob: float | None = None


class PredictionDebugResponse(BaseModel):
    status: str
    details: List[PredictionDebugItem]


# ----------------------------
# Routes
# ----------------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Hate Speech Ensemble API (v2.1)"}


@app.get("/health")
async def health():
    return {"status": "healthy", "device": ensemble.encoder.device if hasattr(ensemble.encoder, 'device') else "cpu"}


@app.get("/info")
async def info():
    return {
        "weights": {
            "lr_tfidf": ensemble.w_lr,
            "xgb_tfidf": ensemble.w_xgb_tfidf,
            "xgb_semantic": ensemble.w_xgb_semantic
        },
        "thresholds": {
            "safe_below": ensemble.low_th,
            "hate_above": ensemble.high_th
        },
        "has_calibrator": ensemble.has_calibrator
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: PredictionRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    if len(req.texts) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 texts allowed")

    results = ensemble.predict(req.texts)
    return PredictionResponse(
        status="success",
        predictions=[PredictionResult(**r) for r in results]
    )


@app.post("/predict-batch", response_model=PredictionResponse)
async def predict_batch(req: PredictionRequest, batch_size: int = Query(64, ge=1, le=1000)):
    """
    Process input texts in batches to avoid memory spikes.
    """
    texts = req.texts or []
    if not texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    all_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        all_results.extend(ensemble.predict(batch))

    return PredictionResponse(status="success", predictions=[PredictionResult(**r) for r in all_results])


@app.post("/predict/debug", response_model=PredictionDebugResponse)
async def predict_debug(req: PredictionRequest):
    """
    Returns per-model probabilities, rule flags, multipliers, raw and calibrated probs.
    Useful for triage and active learning.
    """
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    if len(req.texts) > 500:
        raise HTTPException(status_code=400, detail="Max 500 texts for debug endpoint")

    details = ensemble.predict_debug(req.texts)
    # normalize fields to match schema
    normalized = []
    for d in details:
        normalized.append({
            "text": d["text"],
            "p_lr": d["p_lr"],
            "p_xgb_tfidf": d["p_xgb_tfidf"],
            "p_xgb_semantic": d["p_xgb_semantic"],
            "flags": d["flags"],
            "multiplier": d["multiplier"],
            "raw_ensemble_prob": d["raw_ensemble_prob"],
            "final_prob": d["final_prob"],
            "calibrated_prob": d["calibrated_prob"]
        })

    return PredictionDebugResponse(status="success", details=normalized)
