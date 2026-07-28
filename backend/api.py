"""FastAPI backend for serving FruitVision predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from fruitvision.config import CONFIG
from fruitvision.inference import predict, top_prediction
from fruitvision.utils.image import preprocess_upload

app = FastAPI(title="FruitVision API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict_fruit(file: UploadFile = File(...)) -> Dict[str, object]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    image_batch, processed = preprocess_upload(image_bytes, CONFIG)

    try:
        scores = predict(image_batch, CONFIG)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    label, confidence = top_prediction(scores)
    return {
        "prediction": label,
        "confidence": confidence,
        "scores": scores,
        "input_shape": list(processed.shape),
    }
