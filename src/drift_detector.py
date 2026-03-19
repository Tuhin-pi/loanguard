import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric
import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REFERENCE_DATA_PATH = "data/processed/reference_data.csv"
DRIFT_REPORTS_PATH = "data/drift_reports"
DRIFT_THRESHOLD = 0.5  # If more than 50% features drift, trigger retraining

def load_reference_data() -> pd.DataFrame:
    if not os.path.exists(REFERENCE_DATA_PATH):
        raise FileNotFoundError("Reference data not found. Run preprocessing.py first.")
    df = pd.read_csv(REFERENCE_DATA_PATH)
    # Drop target for drift comparison
    return df.drop(columns=["TARGET"], errors="ignore")

def detect_drift(current_data: pd.DataFrame) -> dict:
    os.makedirs(DRIFT_REPORTS_PATH, exist_ok=True)
    
    reference_data = load_reference_data()
    
    # Align columns
    common_cols = [col for col in reference_data.columns if col in current_data.columns]
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]
    
    # Run Evidently drift report
    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric()
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data
    )
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"{DRIFT_REPORTS_PATH}/drift_report_{timestamp}.html"
    report.save_html(report_path)
    logger.info(f"Drift report saved to {report_path}")
    
    # Extract drift results
    result = report.as_dict()
    
    dataset_drift = result["metrics"][1]["result"]["dataset_drift"]
    drift_share = result["metrics"][1]["result"]["share_of_drifted_columns"]
    
    drift_summary = {
        "timestamp": timestamp,
        "dataset_drift_detected": dataset_drift,
        "drift_share": drift_share,
        "should_retrain": drift_share >= DRIFT_THRESHOLD,
        "report_path": report_path
    }
    
    # Save summary as JSON
    summary_path = f"{DRIFT_REPORTS_PATH}/drift_summary_{timestamp}.json"
    with open(summary_path, "w") as f:
        json.dump(drift_summary, f, indent=2)
    
    logger.info(f"Drift share: {drift_share:.2%} | Should retrain: {drift_summary['should_retrain']}")
    
    return drift_summary

def simulate_drift(X_reference: pd.DataFrame, drift_factor: float = 2.0) -> pd.DataFrame:
    """Simulate drifted data for testing purposes"""
    drifted = X_reference.copy()
    numeric_cols = drifted.select_dtypes(include=np.number).columns
    
    for col in numeric_cols[:5]:  # Drift first 5 numeric features
        drifted[col] = drifted[col] * drift_factor + np.random.normal(0, drifted[col].std(), len(drifted))
    
    logger.info(f"Simulated drift on {len(numeric_cols[:5])} features")
    return drifted

if __name__ == "__main__":
    reference = load_reference_data()
    
    # Test with non-drifted data first
    print("\n--- Testing with normal data ---")
    normal_sample = reference.sample(1000, random_state=42)
    result = detect_drift(normal_sample)
    print(f"Drift detected: {result['dataset_drift_detected']}")
    print(f"Drift share: {result['drift_share']:.2%}")
    print(f"Should retrain: {result['should_retrain']}")
    
    # Test with simulated drifted data
    print("\n--- Testing with drifted data ---")
    drifted_sample = simulate_drift(reference.sample(1000, random_state=42))
    result = detect_drift(drifted_sample)
    print(f"Drift detected: {result['dataset_drift_detected']}")
    print(f"Drift share: {result['drift_share']:.2%}")
    print(f"Should retrain: {result['should_retrain']}")