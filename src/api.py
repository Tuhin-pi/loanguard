from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LoanGuard API",
    description="Loan default prediction API with ML",
    version="1.0.0"
)

# Load model at startup
MODEL_PATH = "models/best_model.json"
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully")

class LoanApplication(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    AMT_GOODS_PRICE: float
    DAYS_BIRTH: int
    DAYS_EMPLOYED: int
    DAYS_REGISTRATION: float
    DAYS_ID_PUBLISH: int
    CNT_FAM_MEMBERS: float
    CNT_CHILDREN: int
    EXT_SOURCE_1: float = 0.5
    EXT_SOURCE_2: float = 0.5
    EXT_SOURCE_3: float = 0.5
    CODE_GENDER: int = 1
    NAME_CONTRACT_TYPE: int = 0
    FLAG_OWN_CAR: int = 0
    FLAG_OWN_REALTY: int = 1

class PredictionResponse(BaseModel):
    default_probability: float
    risk_level: str
    recommendation: str

def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    df["AGE_YEARS"] = abs(df["DAYS_BIRTH"]) / 365
    df["YEARS_EMPLOYED"] = abs(df["DAYS_EMPLOYED"]) / 365
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    df = df.fillna(df.median())
    
    return df

@app.get("/")
def root():
    return {"message": "LoanGuard API is running", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features = engineer_features(application.dict())
        proba = model.predict_proba(features)[0][1]
        
        if proba < 0.3:
            risk_level = "LOW"
            recommendation = "Approve — low default risk"
        elif proba < 0.6:
            risk_level = "MEDIUM"
            recommendation = "Review manually — moderate risk"
        else:
            risk_level = "HIGH"
            recommendation = "Reject — high default risk"
        
        logger.info(f"Prediction: {proba:.4f} | Risk: {risk_level}")
        
        return PredictionResponse(
            default_probability=round(float(proba), 4),
            risk_level=risk_level,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))