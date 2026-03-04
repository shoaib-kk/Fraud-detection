from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("models/lightgbm.pkl")

@app.get("/")
def root():
    return {"message": "Fraud detection API"}

@app.post("/predict")
def predict(transaction: dict):

    df = pd.DataFrame([transaction])

    proba = model.predict_proba(df)[0, 1]
    pred = int(proba > 0.1)

    return {
        "fraud_probability": float(proba),
        "prediction": pred
    }