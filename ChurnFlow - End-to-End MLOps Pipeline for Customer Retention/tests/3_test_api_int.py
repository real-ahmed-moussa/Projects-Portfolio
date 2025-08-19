# Import Libraries
import pandas as pd
from fastapi.testclient import TestClient
import os
from app import app


# Local Development & Containerized Paths
BASE = "/app" if os.path.exists("/app") else "/home/ahmedmoussa/training-files/mlops_ppln/ppln"
BASE = os.getenv("APP_BASE", BASE)
PROCESSED_TEST = f'{BASE}/data/processed/test.csv'

# Constants
TARGET = "Churn"

def test_predict_endpoint_smoke():
    # 1. Build Payload from a Real Row to Ensure Category Codes are Valid
    df = pd.read_csv(PROCESSED_TEST)
    assert TARGET in df.columns, f"{TARGET} must exist in processed test CSV"
    payload = df.drop(columns=[TARGET]).iloc[0].to_dict()

    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200 and h.json().get("ok") is True

        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200, f"API returned {resp.status_code}, body: {resp.text}"
        data = resp.json()

        assert "Churn_Prediction" in data and "Churn_Probability" in data
        assert data["Churn_Prediction"] in (0, 1)
        assert 0.0 <= data["Churn_Probability"] <= 1.0
