import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = "data/raw/application_train.csv"

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    
    df = pd.read_csv(path)
    logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def validate_data(df: pd.DataFrame) -> bool:
    required_columns = ["TARGET", "AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH"]
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    null_target = df["TARGET"].isnull().sum()
    if null_target > 0:
        raise ValueError(f"TARGET column has {null_target} null values")
    
    logger.info("Data validation passed")
    return True

if __name__ == "__main__":
    df = load_data()
    validate_data(df)
    print(df.head())
    print(df["TARGET"].value_counts())