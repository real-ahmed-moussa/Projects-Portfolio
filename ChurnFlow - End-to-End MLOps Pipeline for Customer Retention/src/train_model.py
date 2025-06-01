# Import Libraries
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# [1] Containerized Paths
TRAIN_FEATURES_PATH = '/app/data/processed/train_transformed.csv'
TEST_FEATURES_PATH = '/app/data/processed/test_transformed.csv'
MODEL_PATH = '/app/models/churn_pred_model.pkl'


# [2] Load Data
train_df = pd.read_csv(TRAIN_FEATURES_PATH)
test_df = pd.read_csv(TEST_FEATURES_PATH)


# [3] Split Features and Target
x_train, y_train = train_df.drop(columns=["Churn"]), train_df["Churn"]
x_test, y_test = test_df.drop(columns=["Churn"]), test_df["Churn"]


# [4] Start MLflows
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Point to the MLflow tracking server
mlflow.set_experiment("churn_prediction")

mlflow.autolog()


# [5] Specify Model Parameters
params = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "penalty": "l2",
    "C": 1.1
}


# [6] Train the Model
model = LogisticRegression(**params)
model.fit(x_train, y_train)


# [7] Predictions
y_pred = model.predict(x_test)

# [8] Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# [9] Infer Signature and Create Input Example
signature = infer_signature(x_train, model.predict(x_train))
input_example = x_train.iloc[:1]  # Use the first row of the training data as an example

print(f"Model trained. Accuracy: {accuracy:.4f}")


# [10] Save the Model
joblib.dump(model, MODEL_PATH)