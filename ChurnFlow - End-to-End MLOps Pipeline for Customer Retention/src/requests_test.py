# Import Libraries
import requests

# URL of the Flask API
url = "http://localhost:8001/monitor"

# Request Payload
data = {
    "tenure": 12,
    "MonthlyCharges": 29.99,
    "TotalCharges": 349.88,
    "gender": 0,
    "Partner": 1,
    "Dependents": 0,
    "PhoneService": 1,
    "MultipleLines": 1,
    "InternetService": 0,
    "OnlineSecurity": 1,
    "OnlineBackup": 0,
    "DeviceProtection": 2,
    "TechSupport": 0,
    "StreamingTV": 1,
    "StreamingMovies": 0,
    "Contract": 1,
    "PaperlessBilling": 1,
    "PaymentMethod": 3
}

# Send POST request to the Flask API
response = requests.post(url, json=data)
print(response.status_code, response.json())
