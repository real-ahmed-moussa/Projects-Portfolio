# Import Libraries
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# [1] Load the Model and the Preprocessor
# Model and Preprocessor Containerized Paths
MODEL_PATH =  "/app/models/churn_pred_model.pkl"
PREPROCESSOR_PATH =  "/app/models/preprocessor.pkl"

# Function to Load the Latest Model
def load_latest_model():
    "Load the latest version of the latest churn prediction model."
    try:
        model = joblib.load(MODEL_PATH)
        print("Latest model loaded successfully!")
        return model
    except FileNotFoundError:
        raise Exception("Model not found! Please retrain or check file paths.")

# Load the Model and Preprocessor
model = load_latest_model()
preprocessor = joblib.load(PREPROCESSOR_PATH)


# [2] Define App API
# Define FASTAPI App
app = FastAPI(title="Churn Prediction API")

# Define Request Schema
class CustomerData(BaseModel):
    tenure: float
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

# Define API Endpoint
@app.post("/predict")
def predict_churn(data: CustomerData):
    """Predict if a customer will churn."""
    try:
        input_df = pd.DataFrame([data.dict()])
        
        # Transform Features
        transformed_data = preprocessor.transform(input_df)
        
        # Make Prediction
        prediction = model.predict((transformed_data))[0]
        probability = model.predict_proba(transformed_data)[0, 1]
        
        return {
            "Churn Prediction": int(prediction),
            "Churn Probability": float(probability)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)