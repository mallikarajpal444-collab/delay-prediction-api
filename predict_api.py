from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict
import os
import uvicorn

app = FastAPI(title="Delay Prediction API")

# -------------------------
# Local model path
# -------------------------

MODEL_PATH = "delay_model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("delay_model.joblib not found in project directory")

print("Loading model...")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully")

# -------------------------
# Request schema
# -------------------------

class PredictionInput(BaseModel):
    data: Dict[str, float]

# -------------------------
# Prediction endpoint
# -------------------------

@app.post("/predict")
def predict(input_data: PredictionInput):

    data = input_data.data

    arr = np.array([[
        data["distance"],
        data["carrier_rating"],
        data["traffic_index"],
        data["weather_risk"],
        data["port_congestion"]
    ]])

    prediction = model.predict(arr)[0]

    return {"predicted_delay_minutes": float(prediction)}

# -------------------------
# Health check
# -------------------------

@app.get("/")
def health():
    return {"status": "running"}

# -------------------------
# Run server
# -------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)