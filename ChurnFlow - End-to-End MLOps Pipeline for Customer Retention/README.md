# 📌 ChurnFlow: End-to-End MLOps Pipeline for Customer Retention
> A production-ready MLOps solution designed to predict and mitigate customer churn using a fully automated pipeline.


## 📖 Overview
 - **ChurnFlow** implements a complete machine learning lifecycle for customer churn prediction in a telecom business context.
 - The project demonstrates expertise in **MLOps, automation, and real-time model serving**, integrating continuous monitoring and retraining for long-term reliability.
 - The pipeline was **containerized using Docker** and automated with **GitHub Actions**, ensuring seamless deployment and scalability.


## 🚀 Features
   - ✅ **Real-Time Model Serving:** FastAPI-based REST API for on-demand churn predictions.
   - ✅ **CI/CD Automation:** GitHub Actions for automatic builds, testing, and deployment.
   - ✅ **Drift Detection & Retraining:** EvidentlyAI triggers model retraining when data or prediction drift is detected.
   - ✅ **Experiment Tracking:** MLflow for model performance logging and version control.


## 🏢 Business Impact
   > *This pipeline significantly reduced the time-to-deployment and maintenance overhead for predictive models, enabling proactive customer retention strategies. Telecom teams could identify at-risk customers early, reducing churn rates and improving overall customer lifetime value.*


## ⚙️ Tech Stack
| Technology       | Purpose                                  |
| ---------------- | ---------------------------------------- |
| `Python`         | Core programming language                |
| `scikit-learn`   | ML model development                     |
| `EvidentlyAI`    | Drift detection & monitoring             |
| `MLflow`         | Model tracking & versioning              |
| `FastAPI`        | API for real-time inference              |
| `Docker`         | Containerization                         |
| `GitHub Actions` | CI/CD pipeline automation                |
| `joblib`         | Model serialization                      |
| `pandas`         | Data preprocessing & feature engineering |


## 📂 Project Structure
<pre>
📦 ChurnFlow - End-to-End MLOps Pipeline for Customer Retention
 ┣ 📂 src
 ┣ 📂 data
 ┣ 📂 logs
 ┣ 📂 mlartifacts
 ┣ 📂 mlruns
 ┣ 📂 models
 ┣ 📂 tests
 ┣ Dockerfile.api
 ┣ Dockerfile.monitor
 ┣ README.md
 ┣ requirements.txt
 ┗ report.html
</pre>


## 🛠️ Installation
1️⃣ **Clone the Repository**
<pre>
git clone https://github.com/ahmedmoussa/Projects-Portfolio.git
cd churnflow-mlops-pipeline
</pre>

2️⃣ **Build and Run Docker Containers**
<pre>
docker-compose up --build
</pre>

▶️ **Usage Example**
<pre>
curl -X POST -H "Content-Type: application/json" -d '{
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
}' http://127.0.0.1:8000/predict
</pre>


## 📊 Results
   - **Baseline Logistic Regression Model Accuracy:** 80%
   - **Baseline Logistic Regression Model AUC:** 84%
   - **EvidentlyAI reports:** Drift visualization available in `report.html`.


## 📝 License
This project is shared for portfolio purposes only and may not be used for commercial purposes without permission.