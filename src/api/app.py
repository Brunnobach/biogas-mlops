from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.build_features import build_features

app = FastAPI(
    title="Biogas MLOps API",
    description="API for predicting biogas production based on operational parameters.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("models/biogas_model.pkl")

# Load model at startup
if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None


class PredictionRequest(BaseModel):
    temperature: float = Field(..., ge=30, le=55, description="Digester temperature in °C")
    ph: float = Field(..., ge=6.0, le=8.5, description="Digester pH")
    organic_load: float = Field(..., ge=0.5, le=8.0, description="Organic load in kg VS/m³/day")
    hydraulic_retention_time: float = Field(..., ge=10, le=60, description="HRT in days")
    substrate_type: str = Field(..., description="One of: cattle_swine_poultry, energy_crops, food_waste, agricultural_residue")
    humidity: float = Field(..., ge=60, le=98, description="Humidity %")
    ambient_temperature: float = Field(..., ge=5, le=40, description="Ambient temperature in °C")
    previous_day_production: float = Field(..., ge=0, description="Previous day production in m³")
    month: int = Field(..., ge=1, le=12, description="Month of the year")


class PredictionResponse(BaseModel):
    predicted_biogas_production_m3: float
    confidence_interval_lower: float
    confidence_interval_upper: float


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train model first.")

    # Create DataFrame with single row
    input_df = build_features(
        __import__("pandas").DataFrame([request.model_dump()])
    )

    prediction = model.predict(input_df)[0]

    # Simple confidence interval based on model residuals (simplified)
    std_residual = 8.0
    lower = max(0, prediction - 1.96 * std_residual)
    upper = prediction + 1.96 * std_residual

    return PredictionResponse(
        predicted_biogas_production_m3=round(prediction, 2),
        confidence_interval_lower=round(lower, 2),
        confidence_interval_upper=round(upper, 2),
    )


@app.get("/")
def root():
    return {
        "message": "Biogas MLOps API",
        "docs": "/docs",
        "health": "/health",
    }
