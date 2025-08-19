# Import Libraries
import random
import os
import pandas as pd
import numpy as np

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# Local Development & Containerized Paths
BASE = "/app" if os.path.exists("/app") else "/home/ahmedmoussa/training-files/mlops_ppln/ppln"
BASE = os.getenv("APP_BASE", BASE)

FEATURES_TRAIN_PATH = f'{BASE}/data/processed/train_transformed.csv'
FEATURES_TEST_PATH = f'{BASE}/data/processed/test_transformed.csv'
MODEL_PATH = f'{BASE}/models/churn_pred_model.pkl'

# Constants
TARGET_COL = "Churn"
SEED = 42


# [1] Function to Set Seed
def _set_seeds(seed: int = SEED) -> None:
    """Set seeds to allow for experiment reproduction."""
    random.seed(seed)
    np.random.seed(seed)


# [2] Function to Load and Split the Already Transformed Data (from feature engineering)
def _load_xy(train_path: str, test_path: str):
    """Function to load the data."""
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    if TARGET_COL not in train_df or TARGET_COL not in test_df:
        raise ValueError(f"Target column '{TARGET_COL}' must exist in transformed CSVs. Run feature_engineering.py first.")

    x_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL].astype(int)

    x_test  = test_df.drop(columns=[TARGET_COL])
    y_test  = test_df[TARGET_COL].astype(int)

    return x_train, y_train, x_test, y_test


# [3] Function to Train and Evaluate the Model
def _configure_mlflow():
    base = "/app" if os.path.exists("/app") else "/home/ahmedmoussa/training-files/mlops_ppln/ppln"
    base = os.getenv("APP_BASE", base)
    uri = os.getenv("MLFLOW_TRACKING_URI", f"file://{base}/mlruns")
    if uri.startswith("file://"):
        os.makedirs(uri.replace("file://", ""), exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "ChurnFlow"))

def train_and_evaluate():
    """Train, evaluate, and log model results."""
    _configure_mlflow()
    _set_seeds(SEED)

    # 1. Load Data
    x_train, y_train, x_test, y_test = _load_xy(FEATURES_TRAIN_PATH, FEATURES_TEST_PATH)

    # 2. ML Model - Can be replaced by better models in the future!
    model = LogisticRegression(
                                solver="liblinear",                 # good for small/medium OHE feature spaces
                                penalty="l2",
                                C=1.0,
                                max_iter=1000,
                                random_state=SEED,
                                class_weight="balanced"
                            )

    # 3. Fit
    model.fit(x_train, y_train)

    # 4. Predict
    y_pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
    else:
        # Fallback if probas not available
        y_proba = None

    # 5. Metrics
    metrics = {
                "accuracy":  float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
                "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
            }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    # 7. MLflow Logging
    with mlflow.start_run(run_name="logreg_baseline"):
        mlflow.log_params({
                            "model_type": "LogisticRegression",
                            "solver": "liblinear",
                            "penalty": "l2",
                            "C": 1.0,
                            "max_iter": 1000,
                            "random_state": SEED,
                            "class_weight": "balanced"
                        })
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"confusion_matrix_labels": [0, 1], "confusion_matrix": cm}, "confusion_matrix.json")

        # Signature & Example
        signature = infer_signature(x_train, model.predict(x_train))
        input_example = x_train.iloc[:1]

        mlflow.sklearn.log_model(
                                    sk_model=model,
                                    artifact_path="model",
                                    signature=signature,
                                    input_example=input_example
                                )

        # Export the Model
        joblib.dump(model, MODEL_PATH)

    # 8. Print Results
    print(
            "[TRAIN] done | "
            + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            + f" | saved to {MODEL_PATH}"
        )


# [4] Main Function
if __name__ == "__main__":
    train_and_evaluate()