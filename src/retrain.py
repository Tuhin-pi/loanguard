import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pandas as pd
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

def get_current_best_auc() -> float:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("loanguard-credit-risk")
    
    if not experiment:
        return 0.0
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.roc_auc DESC"],
        max_results=1
    )
    
    if not runs:
        return 0.0
    
    return runs[0].data.metrics.get("roc_auc", 0.0)

def retrain_and_promote(X_train, y_train, X_test, y_test) -> dict:
    current_best_auc = get_current_best_auc()
    logger.info(f"Current best AUC: {current_best_auc:.4f}")
    
    mlflow.set_experiment("loanguard-credit-risk")
    
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 10,
        "random_state": datetime.now().microsecond,  # Different seed each retrain
        "eval_metric": "auc"
    }
    
    with mlflow.start_run(run_name=f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        new_auc = roc_auc_score(y_test, y_pred_proba)
        
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", new_auc)
        mlflow.log_metric("previous_best_auc", current_best_auc)
        mlflow.xgboost.log_model(model, "model")
        
        # Promote only if new model is better
        if new_auc > current_best_auc:
            model.save_model("models/best_model.json")
            promoted = True
            logger.info(f"✅ New model promoted! AUC: {new_auc:.4f} > {current_best_auc:.4f}")
            mlflow.set_tag("promoted", "true")
        else:
            promoted = False
            logger.info(f"⚠️ Model NOT promoted. AUC: {new_auc:.4f} <= {current_best_auc:.4f}")
            mlflow.set_tag("promoted", "false")
        
        return {
            "new_auc": new_auc,
            "previous_auc": current_best_auc,
            "promoted": promoted
        }

if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from ingestion import load_data, validate_data
    from preprocessing import prepare_features
    from drift_detector import load_reference_data, simulate_drift, detect_drift
    
    print("Loading data...")
    df = load_data()
    validate_data(df)
    X_train, X_test, y_train, y_test = prepare_features(df)
    
    print("\nChecking for drift...")
    drifted_sample = simulate_drift(X_test.sample(1000, random_state=42))
    drift_result = detect_drift(drifted_sample)
    
    if drift_result["should_retrain"]:
        print(f"\n🔁 Drift detected! Retraining model...")
        retrain_result = retrain_and_promote(X_train, y_train, X_test, y_test)
        print(f"New AUC: {retrain_result['new_auc']:.4f}")
        print(f"Promoted: {retrain_result['promoted']}")
    else:
        print("\n✅ No significant drift. Retraining not needed.")