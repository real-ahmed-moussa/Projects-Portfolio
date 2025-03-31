# Import Libraries
from datetime import datetime
import pandas as pd
import joblib
import subprocess

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DataDriftTable, DatasetDriftMetric, DatasetSummaryMetric

from flask import Flask, request, jsonify, render_template_string

import retrain as rt


# [1] Containerized Paths
MODEL_PATH =  "/app/models/churn_pred_model.pkl"
PREPROCESSOR_PATH =  "/app/models/preprocessor.pkl"
REFERENCE_DATA_PATH = "/app/data/processed/ref_data.csv"
DRIFT_LOG_PATH = "/app/logs/drift_results.csv"


# [2] Load Model and Preprocessor
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


# [3] Load Reference Data
reference_data = pd.read_csv(REFERENCE_DATA_PATH)


# [4] Convert categorical columns to strings
categorical_columns = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]
reference_data[categorical_columns] = reference_data[categorical_columns].astype(str)


# [5] Detect and Log Drift
def detect_drift(reference_data, current_data):
    """Detect drift and log results using Evidently AI."""
    
    # Create a Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(reference_data=reference_data, current_data=current_data)
    
    # Extract Drift Summary
    drift_summary = drift_report.as_dict()
    
    # Log Drift Results
    drift_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "drift_detected": drift_summary['metrics'][0]['result']['dataset_drift'],
        "number_of_drifted_features": drift_summary['metrics'][0]['result']['number_of_drifted_columns'],
        "share_of_drifted_features": drift_summary['metrics'][0]['result']['share_of_drifted_columns']
    }
    
    # Save Drift Results to a CSV file
    log_drift_results(drift_results)
    
    # Trigger retraining if drift exceeds threshold
    trigger_retraining(drift_results)

    return drift_results


# [6] Helper Function to Log Drift Results
def log_drift_results(drift_results):
    """Log drift results to a CSV file."""
    try:
        # Load existing drift logs
        drift_log_df = pd.read_csv(DRIFT_LOG_PATH)
    except FileNotFoundError:
        # Create an empty DataFrame if file doesn't exist
        drift_log_df = pd.DataFrame(columns=drift_results.keys())

    # Append new results to the DataFrame
    drift_log_df = pd.concat([drift_log_df, pd.DataFrame([drift_results])], ignore_index=True)

    # Save updated results back to CSV
    drift_log_df.to_csv(DRIFT_LOG_PATH, index=False)


# [7] Define drift threshold for retraining
DRIFT_THRESHOLD = 0.3                   # 30% of features drifted triggers retraining
def trigger_retraining(drift_results):
    """Trigger model retraining if drift exceeds threshold."""
    if drift_results["share_of_drifted_features"] >= DRIFT_THRESHOLD:
        print("Significant drift detected! Retraining the model...")

        # Call the retraining script
        rt.retrain_model()

        print("Model retraining completed. New model is deployed.")
        rt.log_retraining_results(drift_results)
        
    else:
        print("No significant drift detected. No retraining needed.")


# [8] Initialize the Flask App
app = Flask(__name__)


# [9] Root route for GET requests
@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Monitoring Dashboard! Use the /monitor endpoint with a POST request to analyze data."


# [10] Monitor route for POST requests
@app.route('/monitor', methods=['POST'])
def monitor():
    try:
        # Get input data
        data = request.json
        print("Received data:", data)  # Log input data

        # Convert input data to DataFrame
        df = pd.DataFrame([data])
        print("DataFrame created:", df)  # Log DataFrame

        # Ensure numerical columns are of the correct type
        numerical_columns = ["tenure", "MonthlyCharges", "TotalCharges"]
        df[numerical_columns] = df[numerical_columns].astype(float)
        
        # Ensure categorical columns in current data are strings
        df[categorical_columns] = df[categorical_columns].astype(int)

        # Log data types of reference and current data
        print("Reference Data Types:\n", reference_data.dtypes)
        print("Current Data Types:\n", df.dtypes)

        # Transform the Data
        feature_names = preprocessor.get_feature_names_out()  # Get feature names from the preprocessor
        transformed_df = preprocessor.transform(df)
        print("Data transformed successfully")  # Log transformation

        # Get model prediction
        prediction = model.predict(transformed_df)
        print("Model prediction:", prediction)  # Log prediction

        # Compare current data with reference data
        report = Report(metrics=[DataDriftTable(), DatasetDriftMetric(), DatasetSummaryMetric()])
        report.run(reference_data=reference_data, current_data=df)
        print("Report generated successfully")  # Log report generation
        
        drift_results = detect_drift(reference_data=reference_data, current_data=df)
        
        # Save the report to a file (optional)
        report.save_html("report.html")

        # Prepare response
        response = {
            "prediction": prediction.tolist(),  # Return predictions
            "report": report.as_dict(),         # Include Evidently report
            "drift_results": drift_results      # Include drift
        }

        return jsonify(response), 200

    except Exception as e:
        print("Error:", str(e))  # Log the error
        return jsonify({"error": str(e)}), 500


# [11] Route to view the report in the browser
@app.route('/view_report', methods=['GET'])
def view_report():
    try:
        # Render the saved report HTML file
        with open("report.html", "r") as f:
            report_html = f.read()
        return render_template_string(report_html)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# [12] Run the Flask App
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8001, debug=True)