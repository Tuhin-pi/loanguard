from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import pandas as pd
import logging
import os
from datetime import datetime
import csv

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

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
    yield

app = FastAPI(
    title="LoanGuard API",
    description="Loan default prediction API with ML",
    version="1.0.0",
    lifespan=lifespan
)

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

def log_prediction(features: dict, prediction: float):
    os.makedirs("data/predictions", exist_ok=True)
    log_path = "data/predictions/prediction_log.csv"
    
    features["prediction"] = prediction
    features["timestamp"] = datetime.now().isoformat()
    
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=features.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(features)


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
        input_dict = application.dict()
        features = engineer_features(input_dict)
        proba = model.predict_proba(features)[0][1]
        
        # Log prediction for drift monitoring
        log_prediction(input_dict, float(proba))
        
        if proba < 0.3:
            risk_level = "LOW"
            recommendation = "Approve — low default risk"
        elif proba < 0.6:
            risk_level = "MEDIUM"
            recommendation = "Review manually — moderate risk"
        else:
            risk_level = "HIGH"
            recommendation = "Reject — high default risk"
        
        return PredictionResponse(
            default_probability=round(float(proba), 4),
            risk_level=risk_level,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))