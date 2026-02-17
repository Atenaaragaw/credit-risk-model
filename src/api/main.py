import pandas as pd
import joblib 
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from src.api.pydantic_models import CustomerFeatures, PredictionResponse

LOCAL_MODEL_PATH = "./model/credit_risk_model.joblib" 
model = None

# Ensure the variable is named 'app'
app = FastAPI(title="PrecisionCredit API")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = joblib.load(LOCAL_MODEL_PATH)
    except Exception:
        model = None
    yield

app.router.lifespan_context = lifespan

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(features: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Logic to convert and predict...
    data_dict = features.model_dump()
    X_new = pd.DataFrame([data_dict])
    
    probability = model.predict_proba(X_new.values)[:, 1][0]
    prediction = "HIGH RISK" if probability > 0.5 else "LOW RISK"
    
    return PredictionResponse(
        risk_prediction=prediction,
        risk_probability=float(probability)
    )