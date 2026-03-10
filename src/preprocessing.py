import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

FEATURES = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH",
    "CNT_FAM_MEMBERS", "CNT_CHILDREN", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "CODE_GENDER", "NAME_CONTRACT_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"
]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Fix known anomaly in this dataset
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    
    # Convert age from days to years
    df["AGE_YEARS"] = abs(df["DAYS_BIRTH"]) / 365
    
    # Employment years
    df["YEARS_EMPLOYED"] = abs(df["DAYS_EMPLOYED"]) / 365
    
    # Credit to income ratio — important feature
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    
    # Annuity to income ratio
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    
    logger.info("Feature engineering complete")
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cat_cols = df.select_dtypes(include="object").columns
    
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN")
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

def prepare_features(df: pd.DataFrame):
    df = clean_data(df)
    df = encode_categoricals(df)
    
    feature_cols = FEATURES + ["AGE_YEARS", "YEARS_EMPLOYED", "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]
    
    # Only keep columns that exist
    feature_cols = [f for f in feature_cols if f in df.columns]
    
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["TARGET"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from ingestion import load_data, validate_data
    df = load_data()
    validate_data(df)
    X_train, X_test, y_train, y_test = prepare_features(df)
    print("Features ready:", X_train.columns.tolist())