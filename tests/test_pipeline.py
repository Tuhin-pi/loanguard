import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append("src")

from ingestion import validate_data
from preprocessing import clean_data, prepare_features
from drift_detector import simulate_drift, load_reference_data

# ── Ingestion tests ──────────────────────────────────────────

def test_validate_data_passes_with_valid_df():
    df = pd.DataFrame({
        "TARGET": [0, 1, 0],
        "AMT_INCOME_TOTAL": [100000, 200000, 150000],
        "AMT_CREDIT": [500000, 300000, 400000],
        "DAYS_BIRTH": [-12000, -15000, -10000]
    })
    assert validate_data(df) == True

def test_validate_data_fails_missing_column():
    df = pd.DataFrame({"TARGET": [0, 1]})
    with pytest.raises(ValueError):
        validate_data(df)

def test_validate_data_fails_null_target():
    df = pd.DataFrame({
        "TARGET": [0, None],
        "AMT_INCOME_TOTAL": [100000, 200000],
        "AMT_CREDIT": [500000, 300000],
        "DAYS_BIRTH": [-12000, -15000]
    })
    with pytest.raises(ValueError):
        validate_data(df)

# ── Preprocessing tests ──────────────────────────────────────

def test_clean_data_creates_new_features():
    df = pd.DataFrame({
        "DAYS_BIRTH": [-12005],
        "DAYS_EMPLOYED": [-2500],
        "AMT_INCOME_TOTAL": [135000],
        "AMT_CREDIT": [568800],
        "AMT_ANNUITY": [20560]
    })
    result = clean_data(df)
    assert "AGE_YEARS" in result.columns
    assert "YEARS_EMPLOYED" in result.columns
    assert "CREDIT_INCOME_RATIO" in result.columns
    assert "ANNUITY_INCOME_RATIO" in result.columns

def test_clean_data_fixes_employed_anomaly():
    df = pd.DataFrame({
        "DAYS_BIRTH": [-12005],
        "DAYS_EMPLOYED": [365243],  # Known anomaly value
        "AMT_INCOME_TOTAL": [135000],
        "AMT_CREDIT": [568800],
        "AMT_ANNUITY": [20560]
    })
    result = clean_data(df)
    assert pd.isna(result["DAYS_EMPLOYED"].iloc[0])

def test_age_calculation_is_correct():
    df = pd.DataFrame({
        "DAYS_BIRTH": [-365 * 30],  # 30 years
        "DAYS_EMPLOYED": [-2500],
        "AMT_INCOME_TOTAL": [135000],
        "AMT_CREDIT": [568800],
        "AMT_ANNUITY": [20560]
    })
    result = clean_data(df)
    assert abs(result["AGE_YEARS"].iloc[0] - 30.0) < 0.1

# ── Drift detection tests ────────────────────────────────────

def test_simulate_drift_changes_distribution():
    df = pd.DataFrame(np.random.randn(100, 5),
                      columns=["A", "B", "C", "D", "E"])
    drifted = simulate_drift(df, drift_factor=3.0)
    assert not df.equals(drifted)

def test_simulate_drift_preserves_shape():
    df = pd.DataFrame(np.random.randn(100, 5),
                      columns=["A", "B", "C", "D", "E"])
    drifted = simulate_drift(df)
    assert drifted.shape == df.shape

# ── API tests ────────────────────────────────────────────────

def test_api_health_endpoint():
    from fastapi.testclient import TestClient
    import sys
    sys.path.insert(0, "src")

    # Only test if model exists
    if not os.path.exists("models/best_model.json"):
        pytest.skip("Model not trained yet")

    from api import app
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_predict_endpoint():
    from fastapi.testclient import TestClient

    if not os.path.exists("models/best_model.json"):
        pytest.skip("Model not trained yet")

    from api import app
    client = TestClient(app)

    payload = {
        "AMT_INCOME_TOTAL": 135000,
        "AMT_CREDIT": 568800,
        "AMT_ANNUITY": 20560,
        "AMT_GOODS_PRICE": 450000,
        "DAYS_BIRTH": -12005,
        "DAYS_EMPLOYED": -2500,
        "DAYS_REGISTRATION": -3648.0,
        "DAYS_ID_PUBLISH": -2700,
        "CNT_FAM_MEMBERS": 2.0,
        "CNT_CHILDREN": 0,
        "EXT_SOURCE_1": 0.6,
        "EXT_SOURCE_2": 0.7,
        "EXT_SOURCE_3": 0.6,
        "CODE_GENDER": 1,
        "NAME_CONTRACT_TYPE": 0,
        "FLAG_OWN_CAR": 0,
        "FLAG_OWN_REALTY": 1
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert 0 <= data["default_probability"] <= 1