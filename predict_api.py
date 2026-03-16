from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from typing import Dict
import requests
import os
import uvicorn

app = FastAPI(title="Delay Prediction API")

MODEL_URL = "https://huggingface.co/malli18/delay/resolve/main/delay_model.pkl"
MODEL_PATH = "delay_model.pkl"


def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        r = requests.get(url, stream=True)
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"{path} downloaded. Size:", os.path.getsize(path))


# Ensure model exists
download_file(MODEL_URL, MODEL_PATH)

# Load model
model = pickle.load(open(MODEL_PATH, "rb"))


class PredictionInput(BaseModel):
    data: Dict[str, float]


@app.post("/predict")
def predict(input_data: PredictionInput):

    data = input_data.data

    # convert dict values → numpy array
    arr = np.array([list(data.values())])

    prediction = model.predict(arr)[0]

    return {
        "predicted_delay_minutes": float(prediction)
    }


@app.get("/")
def health():
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)