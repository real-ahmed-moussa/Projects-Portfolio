import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# File Paths
TRAIN_FEATURES_PATH = '/home/ahmedmoussa/training-files/mlops_ppln/data/processed/train_transformed.csv'
TEST_FEATURES_PATH = '/home/ahmedmoussa/training-files/mlops_ppln/data/processed/test_transformed.csv'
MODEL_PATH = '/home/ahmedmoussa/training-files/mlops_ppln/models/churn_pred_model.pkl'


# Load Data
train_df = pd.read_csv(TRAIN_FEATURES_PATH)
test_df = pd.read_csv(TEST_FEATURES_PATH)


# Split Features and Target
x_train, y_train = train_df.drop(columns=["Churn"]), train_df["Churn"]
x_test, y_test = test_df.drop(columns=["Churn"]), test_df["Churn"]


# Start MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Point to the MLflow tracking server
mlflow.set_experiment("churn_prediction")

with mlflow.start_run():
    
    params = {
        "max_iter": 1000,
        "solver": "lbfgs",
        "penalty": "l2",
        "C": 1.0
    }
    
    # Train the Model
    model = LogisticRegression(**params)
    model.fit(x_train, y_train)
    
    # Predictions
    y_pred = model.predict(x_test)
    
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Log Parameters
    mlflow.log_params(params)
    
    # Log Metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    
    # Infer Signature and Create Input Example
    signature = infer_signature(x_train, model.predict(x_train))
    input_example = x_train.iloc[:1]  # Use the first row of the training data as an example
    
    # Log Model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="logistic_regression_model",
        signature=signature,
        input_example=input_example
    )
    
    print(f"Model trained. Accuracy: {accuracy:.4f}")


# Save the Model
joblib.dump(model, MODEL_PATH)