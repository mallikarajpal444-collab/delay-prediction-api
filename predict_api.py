from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict

app = FastAPI(title="Delay Prediction API")

# -------------------------
# Lazy load model
# -------------------------

model = None

def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = joblib.load("delay_model.joblib")
    return model


# -------------------------
# Input schema
# -------------------------

class PredictionInput(BaseModel):
    data: Dict[str, float]


# -------------------------
# Prediction endpoint
# -------------------------

@app.post("/predict")
def predict(input_data: PredictionInput):

    model = get_model()

    data = input_data.data

    arr = [[
        data["distance"],
        data["carrier_rating"],
        data["traffic_index"],
        data["weather_risk"],
        data["port_congestion"]
    ]]

    prediction = model.predict(arr)[0]

    return {"predicted_delay_minutes": float(prediction)}


# -------------------------
# Health check
# -------------------------

@app.get("/")
def health():
    return {"status": "running"}