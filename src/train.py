import mlflow
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def train_model(X_train, y_train, X_test, y_test):
    
    mlflow.set_experiment("loanguard-credit-risk")
    
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 10,  # handles class imbalance
        "random_state": 42,
        "eval_metric": "auc"
    }
    
    with mlflow.start_run():
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=100
        )
        
        # Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("precision", report["1"]["precision"])
        mlflow.log_metric("recall", report["1"]["recall"])
        mlflow.log_metric("f1", report["1"]["f1-score"])
        
        mlflow.xgboost.log_model(model, "model")
        
        logger.info(f"ROC-AUC: {auc:.4f}")
        print(f"\n✅ Model trained — ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, y_pred))
        
        return model, auc

if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from ingestion import load_data, validate_data
    from preprocessing import prepare_features
    
    df = load_data()
    validate_data(df)
    X_train, X_test, y_train, y_test = prepare_features(df)
    model, auc = train_model(X_train, y_train, X_test, y_test)