

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List
from io import StringIO

app = FastAPI(
    title="Exoplanet AI Prediction Backend",
    description="API for exoplanet detection model prediction",
    version="1.0"
)

# Allow CORS for local frontend development and deployment
origins = [
    "http://localhost",
    "http://localhost:3000",  # React default
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    # Add your deployed frontend URLs here
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load saved ML components once on startup
try:
    model = joblib.load("models/best_exoplanet_model.pkl")
    scaler = joblib.load("models/advanced_scaler.pkl")
    selector = joblib.load("models/feature_selector.pkl")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    model, scaler, selector = None, None, None

class PredictionInput(BaseModel):
    features: List[float]  # numerical features matching training feature count

class BatchPredictionRequest(BaseModel):
    data: List[List[float]]  # list of feature vectors

@app.get("/")
def index():
    return {"message": "Exoplanet Prediction API is online"}

@app.post("/predict/single")
def predict_single(input_data: PredictionInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        X = np.array(input_data.features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        if selector:
            X_scaled = selector.transform(X_scaled)
        
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0, 1]
        
        return {
            "prediction": bool(prediction),
            "confidence": float(probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    content = await file.read()
    try:
        df = pd.read_csv(StringIO(content.decode('utf-8')))
        numeric_df = df.select_dtypes(include=[np.number])
        
        X = numeric_df.values
        X_scaled = scaler.transform(X)
        if selector:
            X_scaled = selector.transform(X_scaled)
        
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        results = []
        for i in range(len(predictions)):
            results.append({
                "row": i + 1,
                "prediction": bool(predictions[i]),
                "confidence": float(probabilities[i])
            })
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing uploaded file: {e}")

@app.get("/model/info")
def get_model_info():
    try:
        with open("models/advanced_model_info.json", "r") as f:
            import json
            info = json.load(f)
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading model info: {e}")
