# Import Libraries
import os
import datetime as dt
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import feature_engineering as fe


# [1] Containerized Paths
NEW_RAW_DATA_PATH = "/app/data/processed/new_data.csv"
NEW_PROCESSED_TRAIN_DATA_PATH = "/app/data/processed/new_data_train_transformed.csv"
NEW_PROCESSED_TEST_DATA_PATH = "/app/data/processed/new_data_test_transformed.csv"
MODEL_PATH = "/app/models/churn_pred_model.pkl"
PREPROCESSOR_PATH =  "/app/models/preprocessor.pkl"
RETRAIN_LOG_PATH = "/app/logs/retrain_log.csv"

MODEL_DIR = "/app/models"
VERSION_LOG_PATH = "/app/logs/version_log.csv"


# [2] Retrain Model Function
def retrain_model():
    """Retrain the model with new data when drift is detected."""
    
    # 1. Load new raw data for retraining
    data = pd.read_csv(NEW_RAW_DATA_PATH)

    # 2. Load the data pre-processor
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # 3. Split the data
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = fe.transform_features(preprocessor, train_df, test_df)
    
    # 4. Save the transformed data
    # Combine Training Data
    train_data = pd.DataFrame(x_train)
    train_data["Churn"] = y_train
    # Combine Testing Data
    test_data = pd.DataFrame(x_test)
    test_data["Churn"] = y_test
    
    # 5. Export the Data
    train_data.to_csv(NEW_PROCESSED_TRAIN_DATA_PATH, index=False)
    test_data.to_csv(NEW_PROCESSED_TEST_DATA_PATH, index=False)
    print("New data transformed and saved successfully!")

    # 6. Specify Model Parameters
    params = {
        "max_iter": 1000,
        "solver": "lbfgs",
        "penalty": "l2",
        "C": 1.1
    }

    # 7. Retrain the model
    model = LogisticRegression(**params)
    model.fit(x_train, y_train)

    # 8. Evaluate the retrained model
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"New model accuracy: {accuracy:.4f}")
    
    # 9. Get the next model version
    latest_version = get_latest_model_version()
    next_version = latest_version + 1
    model_file_name = f"churn_pred_model_v{next_version}.pkl"
    model_path = os.path.join(MODEL_DIR, model_file_name)
   

    # 10. Save the retrained model
    joblib.dump(model, MODEL_PATH)
    print(f"New model exported successfully to {model_path}")
    
    # 11. Log the retraining results
    log_model_version(next_version, accuracy)
    
    # 12. Update the latest model symlink
    latest_model_file_name = f"churn_pred_model.pkl"
    latest_model_path = os.path.join(MODEL_DIR, latest_model_file_name)
    if os.path.exists(latest_model_path):
        os.remove(latest_model_path)
    joblib.dump(model, MODEL_PATH) 


# [3] Log Retraining Results Function
def log_retraining_results(drift_results):
    """Log retraining events and new model info."""
    retrain_results = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "drift_detected": drift_results["drift_detected"],
        "number_of_drifted_features": drift_results["number_of_drifted_features"],
        "share_of_drifted_features": drift_results["share_of_drifted_features"],
        "retraining_triggered": True
    }

    # Load existing retrain logs
    try:
        retrain_log_df = pd.read_csv(RETRAIN_LOG_PATH)
    except FileNotFoundError:
        retrain_log_df = pd.DataFrame(columns=retrain_results.keys())

    # Append new retraining info
    retrain_log_df = pd.concat([retrain_log_df, pd.DataFrame([retrain_results])], ignore_index=True)
    retrain_log_df.to_csv(RETRAIN_LOG_PATH, index=False)


# [4] Function to Get the Latest Model Version
def get_latest_model_version():
    """Get the latest model version from the version log."""
    try:
        version_log = pd.read_csv(VERSION_LOG_PATH)
        latest_version = version_log["version"].max()
        return latest_version if not pd.isnull(latest_version) else 0
        
    except FileNotFoundError:
        return 0


# [5] Function to Log the Model Version
def log_model_version(version, accuracy):
    """Log the model version and accuracy."""
    
    version_info = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": version,
        "accuracy": accuracy
    }

    # Load existing version log
    try:
        version_log_df = pd.read_csv(VERSION_LOG_PATH)
    except FileNotFoundError:
        version_log_df = pd.DataFrame(columns=version_info.keys())

    # Append new version info
    version_log_df = pd.concat([version_log_df, pd.DataFrame([version_info])], ignore_index=True)
    version_log_df.to_csv(VERSION_LOG_PATH, index=False)
