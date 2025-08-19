# Import Libraries
from typing import List, Tuple
import json
import os

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


# Local Development & Containerized Paths
BASE = "/app" if os.path.exists("/app") else "/home/ahmedmoussa/training-files/mlops_ppln/ppln"
BASE = os.getenv("APP_BASE", BASE)
PROCESSED_TRAIN_PATH = f'{BASE}/data/processed/train.csv'
PROCESSED_TEST_PATH = f'{BASE}/data/processed/test.csv'

FEATURES_TRAIN_PATH = f'{BASE}/data/processed/train_transformed.csv'
FEATURES_TEST_PATH = f'{BASE}/data/processed/test_transformed.csv'

PREPROCESSOR_PATH    = f'{BASE}/models/preprocessor.pkl'
FEATURE_NAMES_PATH   = f'{BASE}/models/feature_names.json'

# Constants
TARGET_COL = "Churn"
NUMERIC_COLS: List[str] = ["tenure", "MonthlyCharges", "TotalCharges"]                      # already coerced by ingestion
CATEGORICAL_COLS: List[str] = [                                                             # ingestion label-encoded to integers; we still treat them as categorical
                                "gender", "Partner", "Dependents", "PhoneService", 
                                "MultipleLines", "InternetService", "OnlineSecurity", 
                                "OnlineBackup", "DeviceProtection", "TechSupport", 
                                "StreamingTV", "StreamingMovies", "Contract", 
                                "PaperlessBilling", "PaymentMethod"
                            ]


# [1] Load the Data
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load preprocessed data."""
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)

    if TARGET_COL not in train_df or TARGET_COL not in test_df:
        raise ValueError(f"Target column '{TARGET_COL}' must exist in processed CSVs.")

    if train_df.empty or test_df.empty:
        raise ValueError("Processed train/test CSVs are empty; run data_ingestion.py first.")

    return train_df, test_df


# [2] Functions to Check for Existing Columns
def _present(cols: List[str], df: pd.DataFrame) -> List[str]:
    """Return columns that exist (defensive against drift/schema changes)."""
    return [c for c in cols if c in df.columns and c != TARGET_COL]


# [3] Create Feature Pipeline
def create_feature_pipeline(train_df: pd.DataFrame) -> ColumnTransformer:
    """Create pipeline for feature engineering."""
    
    numeric_cols     = _present(NUMERIC_COLS, train_df)
    categorical_cols = _present(CATEGORICAL_COLS, train_df)
    
    # Numeric Features Pipeline
    numeric_pipeline = Pipeline(steps=[
                                        ('scaler', StandardScaler(with_mean=True, with_std=True))
                                    ])
    
    # Categorical Features Pipeline
    categorical_pipeline = Pipeline(steps=[
                                            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                                        ])
    
    # Combine Both Pipelines
    preprocessor = ColumnTransformer(
                                        transformers=[
                                                        ('num', numeric_pipeline, numeric_cols),
                                                        ('cat', categorical_pipeline, categorical_cols)
                                                    ],
                                                    remainder="drop",
                                                    verbose_feature_names_out=True
                                    )
    
    return preprocessor


# [4] Function to Get Feature Names
def _get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """After fitting, we can resolve transformed column names."""
    names = preprocessor.get_feature_names_out()
    return list(names)


# [5] Function to Transform the Features
def transform_features(preprocessor: ColumnTransformer, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, List[str]]:
    """Apply feature engineering transformations to train and test data."""
    x_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].astype(int)
    
    x_test  = test_df.drop(columns=[TARGET_COL])
    y_test  = test_df[TARGET_COL].astype(int)
    
    # Perform Transformations
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)

    # Resolve Feature Names
    feature_names = _get_feature_names(preprocessor)

    # Ensure 2D Numpy Arrays
    x_train_transformed = np.asarray(x_train_transformed)
    x_test_transformed = np.asarray(x_test_transformed)
    
    return x_train_transformed, x_test_transformed, y_train, y_test, feature_names


# [5] Save Transformed Data
def save_transformed_data(x_train: np.ndarray, x_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, feature_names: List[str]) -> None:
    """Function to save transformed data to CSV files."""
    # Combine Training Data
    train_data = pd.DataFrame(x_train, columns=feature_names)
    train_data[TARGET_COL] = y_train.values
    
    # Combine Testing Data
    test_data = pd.DataFrame(x_test, columns=feature_names)
    test_data[TARGET_COL] = y_test.values
    
    # Export the Data
    train_data.to_csv(FEATURES_TRAIN_PATH, index=False)
    test_data.to_csv(FEATURES_TEST_PATH, index=False)
    
    # Persist feature names for serving/monitoring alignment checks
    with open(FEATURE_NAMES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    print(f"[FE] Saved transformed train to {FEATURES_TRAIN_PATH}")
    print(f"[FE] Saved transformed test to {FEATURES_TEST_PATH}")
    print(f"[FE] Saved feature names to {FEATURE_NAMES_PATH}")
    

# [6] Main Function
if __name__ == "__main__":
    train_df, test_df = load_data()
    preprocessor = create_feature_pipeline(train_df)
    x_train, x_test, y_train, y_test, feature_names = transform_features(preprocessor, train_df, test_df)
    save_transformed_data(x_train, x_test, y_train, y_test, feature_names)

    # Save the Preprocessor for training/serving/monitoring
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"[FE] Saved preprocessor to {PREPROCESSOR_PATH}")