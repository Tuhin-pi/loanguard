import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd
import pickle
import os
import logging

logger = logging.getLogger(__name__)

def save_best_model():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("loanguard-credit-risk")
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No runs found. Train a model first.")
    
    best_run = runs[0]
    print(f"Best run: {best_run.info.run_id} | AUC: {best_run.data.metrics['roc_auc']:.4f}")
    
    # Load and save model locally
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)
    
    os.makedirs("models", exist_ok=True)
    model.save_model("models/best_model.json")
    print("✅ Model saved to models/best_model.json")
    
    return model

if __name__ == "__main__":
    save_best_model()