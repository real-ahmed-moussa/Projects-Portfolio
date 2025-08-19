# Import Libraries
import json
from pathlib import Path
import os

import joblib
import pandas as pd


# Local Development & Containerized Paths
BASE = "/app" if os.path.exists("/app") else "/home/ahmedmoussa/training-files/mlops_ppln/ppln"
BASE = os.getenv("APP_BASE", BASE)
TRAIN_TRANSFORMED = f'{BASE}/data/processed/train_transformed.csv'
TEST_TRANSFORMED  = f'{BASE}/data/processed/test_transformed.csv'
PREPROCESSOR_PATH = f'{BASE}/models/preprocessor.pkl'
FEATURE_NAMES_PATH= f'{BASE}/models/feature_names.json'

# Constants
TARGET = "Churn"


def test_feature_names_match_preprocessor():
    pre = joblib.load(PREPROCESSOR_PATH)
    with open(FEATURE_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)

    assert isinstance(names, list) and len(names) > 0, "feature_names.json must contain a nonempty list."
    assert list(pre.get_feature_names_out()) == names, "Persisted feature names must match preprocessor.get_feature_names_out()"


def test_transformed_train_test_have_same_columns():
    train_df = pd.read_csv(TRAIN_TRANSFORMED)
    test_df  = pd.read_csv(TEST_TRANSFORMED)

    assert TARGET in train_df.columns and TARGET in test_df.columns, f"{TARGET} must be present in transformed CSVs."

    train_cols = [c for c in train_df.columns if c != TARGET]
    test_cols  = [c for c in test_df.columns if c != TARGET]

    assert train_cols == test_cols, "Train/Test transformed features must have identical ordered columns."
    assert len(train_cols) > 0, "Transformed features should not be empty!"
