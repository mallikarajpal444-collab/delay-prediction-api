from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Delay Prediction API")

# -----------------------------
# Load trained model
# -----------------------------
try:
    model = joblib.load("delay_model_clean.joblib")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Error loading model:", e)
    model = None

# -----------------------------
# Input Schema
# -----------------------------
class PredictionInput(BaseModel):
    distance_km: float
    route_congestion_score: float
    weather_risk_score: float
    carrier_avg_delay_minutes: float
    warehouse_congestion_score: float

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        features = np.array([[
            data.distance_km,
            data.route_congestion_score,
            data.weather_risk_score,
            data.carrier_avg_delay_minutes,
            data.warehouse_congestion_score
        ]])

        prediction = model.predict(features)[0]

        return {
            "status": "success",
            "predicted_delay_minutes": float(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Health Check
# -----------------------------
@app.get("/")
def health():
    return {"status": "API is running"}