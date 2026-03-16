from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import Dict
import requests
import os
import uvicorn

app = FastAPI(title="Delay Prediction API")

# -------------------------
# Model file locations
# -------------------------

MODEL_URL = "https://huggingface.co/malli18/delay/resolve/main/delay_model.pkl"
FEATURES_URL = "https://huggingface.co/malli18/delay/resolve/main/model_features.pkl"

MODEL_PATH = "delay_model.pkl"
FEATURES_PATH = "model_features.pkl"


def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        r = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"{path} downloaded. Size:", os.path.getsize(path))


# -------------------------
# Ensure files exist
# -------------------------

download_file(MODEL_URL, MODEL_PATH)
download_file(FEATURES_URL, FEATURES_PATH)


# -------------------------
# Load model + features
# -------------------------

model = pickle.load(open(MODEL_PATH, "rb"))
features = pickle.load(open(FEATURES_PATH, "rb"))


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

    df = pd.DataFrame([data])

    # add missing features
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # enforce column order
    df = df[features]

    prediction = model.predict(df)[0]

    return {
        "predicted_delay_minutes": float(prediction)
    }


# -------------------------
# Health check
# -------------------------

@app.get("/")
def health():
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)