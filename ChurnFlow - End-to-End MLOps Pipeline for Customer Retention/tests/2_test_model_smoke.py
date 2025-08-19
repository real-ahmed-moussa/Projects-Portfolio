# Import Libraries
import joblib
import numpy as np
import pandas as pd
import os


# Local Development & Containerized Paths
BASE = "/app" if os.path.exists("/app") else "/home/ahmedmoussa/training-files/mlops_ppln/ppln"
BASE = os.getenv("APP_BASE", BASE)
PREPROCESSOR_PATH = f'{BASE}/models/preprocessor.pkl'
MODEL_PATH = f'{BASE}/models/churn_pred_model.pkl'
PROCESSED_TEST = f'{BASE}/data/processed/test.csv'

# Constants
TARGET = "Churn"


def test_model_predicts_on_processed_rows():
    pre = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(PROCESSED_TEST)
    assert TARGET in df.columns, f"{TARGET} must exist in processed test CSV"

    X = df.drop(columns=[TARGET]).head(5)   # small batch
    Xt = pre.transform(X)

    # Predictions
    y_pred = model.predict(Xt)
    assert y_pred.shape[0] == Xt.shape[0], "Predictions should match batch size"
    assert set(np.unique(y_pred)).issubset({0, 1}), "Predictions must be binary {0,1}"

    # Probabilities (if available)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(Xt)[:, 1]
        assert y_proba.shape[0] == Xt.shape[0], "Probabilities should match batch size"
        assert np.all((y_proba >= 0) & (y_proba <= 1)), "Probabilities must be in [0,1]"
