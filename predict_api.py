from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from typing import Dict
import uvicorn

app = FastAPI(title="Delay Prediction API")

# Load model
model = pickle.load(open("delay_model.pkl", "rb"))

# Load feature order
features = pickle.load(open("model_features.pkl", "rb"))


# Request schema
class PredictionInput(BaseModel):
    data: Dict[str, float]


@app.post("/predict")
def predict(input_data: PredictionInput):

    data = input_data.data

    df = pd.DataFrame([data])

    # add missing features
    for col in features:
        if col not in df.columns:
            df[col] = 0

    # reorder columns
    df = df[features]

    prediction = model.predict(df)[0]

    return {
        "predicted_delay_minutes": float(prediction)
    }


@app.get("/")
def health():
    return {"status": "running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)