# Import Libraries
import json
import math
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn
import os, requests

# Local Development & Containerized Paths
BASE = "/app" if os.path.exists("/app") else "/home/ahmedmoussa/training-files/mlops_ppln/ppln"
BASE = os.getenv("APP_BASE", BASE)
MODEL_PATH =  f"{BASE}/models/churn_pred_model.pkl"
PREPROCESSOR_PATH =  f"{BASE}/models/preprocessor.pkl"
FEATURE_NAMES_PATH = f"{BASE}/models/feature_names.json"
VERSION_LOG_PATH   = f"{BASE}/logs/version_log.csv"

MONITOR_URL = os.getenv("MONITOR_URL", "http://monitor:8001")  # service name in compose

# Constants
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
ALL_INPUT_COLS = NUMERIC_COLS + CATEGORICAL_COLS
TARGET = "Churn"


# [1] Pydantic request/response Models
class CustomerData(BaseModel):
    tenure: float = Field(..., description="Customer tenure in months")
    MonthlyCharges: float
    TotalCharges: float
    gender: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: int
    InternetService: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    Contract: int
    PaperlessBilling: int
    PaymentMethod: int

class PredictResponse(BaseModel):
    Churn_Prediction: int
    Churn_Probability: float

class BatchRequest(BaseModel):
    records: List[CustomerData]


# [2] App & Global Artifacts
app = FastAPI(title="Churn Prediction API", version="1.0.0")
_model = None
_preprocessor = None
_feature_names: Optional[List[str]] = None


# [3] Helper Functions
def _safe_load(path: str, kind: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail=f"{kind} not found at: {path}")
    try:
        return joblib.load(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load {kind}: {e}")

def _load_feature_names_if_any() -> Optional[List[str]]:
    if os.path.exists(FEATURE_NAMES_PATH):
        try:
            with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
                names = json.load(f)
            if isinstance(names, list) and len(names) > 0:
                return names
        except Exception:
            pass
    return None

def _load_artifacts():
    global _model, _preprocessor, _feature_names
    _model = _safe_load(MODEL_PATH, "Model")
    _preprocessor = _safe_load(PREPROCESSOR_PATH, "Preprocessor")
    _feature_names = _load_feature_names_if_any()

def _coerce_input_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # keep only expected columns; error if missing expected ones
    missing = [c for c in ALL_INPUT_COLS if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {missing}")
    extra = [c for c in df.columns if c not in ALL_INPUT_COLS]
    if extra:
        # Silently drop unknown fields to avoid leakage
        df = df.drop(columns=extra)

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL_COLS:
        # Enforce integers; treat NaN as -1 (will be handled by preprocessor imputer)
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(int)

    # Basic NA guard for numerics
    for c in NUMERIC_COLS:
        if df[c].isna().any():
            # replace NaN with median of provided batch (tiny batches OK)
            df[c] = df[c].fillna(df[c].median())

    return df[ALL_INPUT_COLS]

def _predict_core(frame: pd.DataFrame):
    Xt = _preprocessor.transform(frame)
    if _feature_names is not None and Xt.shape[1] != len(_feature_names):
        raise HTTPException(status_code=500, detail="Transformed feature shape mismatch with persisted feature_names.")
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(Xt)[:, 1]
    elif hasattr(_model, "decision_function"):
        # logistic-like fallback
        z = _model.decision_function(Xt)
        proba = 1.0 / (1.0 + np.exp(-z))
    else:
        proba = np.full(shape=(Xt.shape[0],), fill_value=np.nan)
    preds = _model.predict(Xt)
    return preds, proba


# [3] Startup: Load Artifacts
@app.on_event("startup")
def _startup():
    _load_artifacts()


# [4] Routes
@app.get("/health")
def health():
    return {
        "ok": _model is not None and _preprocessor is not None,
        "has_feature_names": _feature_names is not None,
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH,
    }

@app.get("/version")
def version():
    """
    If logs/version_log.csv exists, return the latest recorded version; otherwise 'latest'.
    """
    try:
        if os.path.exists(VERSION_LOG_PATH):
            df = pd.read_csv(VERSION_LOG_PATH)
            if not df.empty and "version" in df.columns:
                return {"version": str(df.iloc[-1]["version"])}
    except Exception:
        pass
    return {"version": "latest"}

@app.post("/reload")
def reload_artifacts():
    _load_artifacts()
    return {"reloaded": True}

def _async_tap_to_monitor(payload: dict):
    try:
        requests.post(f"{MONITOR_URL}/monitor", json={"records": [payload]}, timeout=1)
    except Exception:
        pass  # never block predictions

@app.post("/predict", response_model=PredictResponse)
def predict_churn(data: CustomerData, background_tasks: BackgroundTasks):
    """
    Single-record prediction.
    Expects integer-coded categoricals and numeric floats, aligned with training pipeline.
    """
    try:
        frame = pd.DataFrame([data.model_dump()])
        if _model is None or _preprocessor is None:
            _load_artifacts()
        frame = _coerce_input_types(frame)
        preds, proba = _predict_core(frame)
        p = float(proba[0]) if proba is not None and not math.isnan(proba[0]) else None
        result = {"Churn_Prediction": int(preds[0]), "Churn_Probability": p}
        background_tasks.add_task(_async_tap_to_monitor, data.model_dump())
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch")
def predict_batch(payload: BatchRequest):
    """
    Batch prediction.
    Body: {"records": [CustomerData, ...]}
    """
    try:
        if _model is None or _preprocessor is None:
            _load_artifacts()
        frame = pd.DataFrame([r.model_dump() for r in payload.records])
        frame = _coerce_input_types(frame)
        preds, proba = _predict_core(frame)
        out = []
        for i in range(len(frame)):
            p = float(proba[i]) if proba is not None and not math.isnan(proba[i]) else None
            out.append({"Churn_Prediction": int(preds[i]), "Churn_Probability": p})
        return {"results": out, "n": len(out)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
def reload_model():
    global model, preprocessor
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 5. Local Run
if __name__ == "__main__":
    # For production, prefer: uvicorn app:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)