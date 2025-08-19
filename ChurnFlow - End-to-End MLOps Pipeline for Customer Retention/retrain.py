# Import Libraries
import os
import re
import datetime as dt
from typing import Tuple

import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import feature_engineering as fe


# Local Development & Containerized Paths
BASE = "/app" if os.path.exists("/app") else "/home/ahmedmoussa/training-files/mlops_ppln/ppln"
BASE = os.getenv("APP_BASE", BASE)
NEW_DATA_PATH = f"{BASE}/data/processed/new_data.csv"

PREPROCESSOR_PATH = f"{BASE}/models/preprocessor.pkl"
FEATURE_NAMES_PATH = f"{BASE}/models/feature_names.json"
MODEL_DIR = f"{BASE}/models"
LATEST_MODEL_PATH = f"{MODEL_DIR}/churn_pred_model.pkl"

VERSION_LOG_PATH = f"{BASE}/logs/version_log.csv"
RETRAIN_LOG_PATH = f"{BASE}/logs/retrain_log.csv"

# Constants
TARGET = "Churn"
SEED = 42


# [1] Function to Ensure Directories Exist
def _ensure_dirs():
    os.makedirs(os.path.dirname(VERSION_LOG_PATH), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


# [2] Function to Get the Next Model Version
def _next_version(model_dir: str) -> str:
    """Scan existing files to find the next integer version (v1, v2, ...)."""
    os.makedirs(model_dir, exist_ok=True)
    versions = []
    for name in os.listdir(model_dir):
        m = re.match(r"churn_pred_model_v(\d+)\.pkl$", name)
        if m:
            versions.append(int(m.group(1)))
    nxt = (max(versions) + 1) if versions else 1
    return f"v{nxt}"


# [3] Function to Load New Data
def _load_new_data() -> pd.DataFrame:
    """Function to load new data."""
    if not os.path.exists(NEW_DATA_PATH):
        raise FileNotFoundError(f"New data file not found: {NEW_DATA_PATH}")
    df = pd.read_csv(NEW_DATA_PATH)
    if df.empty:
        raise ValueError("New data CSV is empty.")
    if TARGET not in df.columns:
        raise ValueError(f"New data must include target column '{TARGET}' for retraining.")
    return df


# [4] Retrain Model Function
def retrain_model():
    """
    Retrain on NEW_DATA_PATH, save versioned model, update 'latest', log results.
    Returns string version, e.g., 'v3'.
    """
    _ensure_dirs()

    # 1. Load New Raw Data for Retraining
    df = _load_new_data()

    # 2. Build a New Preprocessor on this Batch (keeps things consistent end-to-end)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET])

    pre = fe.create_feature_pipeline(train_df)
    x_train, x_test, y_train, y_test, feature_names = fe.transform_features(pre, train_df, test_df)
    
    # 3. Train Model
    model = LogisticRegression(
                                solver="liblinear",
                                penalty="l2",
                                C=1.0,
                                max_iter=1000,
                                random_state=SEED,
                            )
    model.fit(x_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, y_pred))

    # 5. Persist Preprocessor + Feature Names
    joblib.dump(pre, PREPROCESSOR_PATH)
    pd.Series(feature_names).to_json(FEATURE_NAMES_PATH, orient="values")

    # 6. Versioned Model File + Latest Pointer
    version = _next_version(MODEL_DIR)
    versioned_path = os.path.join(MODEL_DIR, f"churn_pred_model_{version}.pkl")

    joblib.dump(model, versioned_path)     # <-- write the versioned file
    if os.path.exists(LATEST_MODEL_PATH):
        os.remove(LATEST_MODEL_PATH)
    joblib.dump(model, LATEST_MODEL_PATH)  # <-- update "latest" pointer

    # 7. Logs
    _log_model_version(version, acc)
    _log_retrain_result(version, acc, len(df))

    return version


# [5] Function to Log Model Version
def _log_model_version(version: str, accuracy: float) -> None:
    row = {
            "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": version,
            "accuracy": accuracy,
            "model_path": f"{MODEL_DIR}/churn_pred_model_{version}.pkl",
            "latest_path": LATEST_MODEL_PATH,
        }
    try:
        df = pd.read_csv(VERSION_LOG_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=row.keys())
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(VERSION_LOG_PATH, index=False)


# [6] Function to Log Retrain Results
def _log_retrain_result(version: str, accuracy: float, n_rows: int) -> None:
    row = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": version,
        "accuracy": accuracy,
        "n_rows": n_rows,
        "data_path": NEW_DATA_PATH,
    }
    try:
        df = pd.read_csv(RETRAIN_LOG_PATH)
    except FileNotFoundError:
        df = pd.DataFrame(columns=row.keys())
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(RETRAIN_LOG_PATH, index=False)