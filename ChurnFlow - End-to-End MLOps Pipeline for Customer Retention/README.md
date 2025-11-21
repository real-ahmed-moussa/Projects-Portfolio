<p align="center">
  <img src="imgs/ChurnFlow.png" alt="CF" width="500">
</p>

# 📌 ChurnFlow: End-to-End MLOps Pipeline for Customer Retention
> A production-ready MLOps solution designed to predict and mitigate customer churn using a fully automated pipeline.


## 📖 Overview
- **ChurnFlow** implements a complete machine learning lifecycle for customer churn prediction in a telecom business context.
- The project demonstrates expertise in **MLOps, automation, and real-time model serving**, integrating continuous monitoring and retraining for long-term reliability.
- The pipeline was **containerized using Docker** and automated with **GitHub Actions**, ensuring seamless deployment and scalability.


## 🏢 Business Impact
This pipeline significantly reduced the time-to-deployment and maintenance overhead for predictive models, enabling proactive customer retention strategies. Telecom teams could identify at-risk customers early, reducing churn rates and improving overall customer lifetime value.


## 🚀 Features
✅ **Real-Time Model Serving:** FastAPI-based REST API for on-demand churn predictions.  
✅ **CI/CD Automation:** GitHub Actions for automatic builds, testing, and deployment.  
✅ **Drift Detection & Retraining:** EvidentlyAI triggers model retraining when data or prediction drift is detected.  
✅ **Experiment Tracking:** MLflow for model performance logging and version control.  


## ⚙️ Tech Stack
| Technology       | Purpose                                  |
| ---------------- | ---------------------------------------- |
| `Python`         | Core programming language                |
| `scikit-learn`   | ML model development                     |
| `EvidentlyAI`    | Drift detection & monitoring             |
| `MLflow`         | Model tracking & versioning              |
| `FastAPI`        | REST API for real-time inference         |
| `Docker`         | Containerization                         |
| `GitHub Actions` | CI/CD pipeline automation                |
| `joblib`         | Model serialization                      |
| `pandas`         | Data preprocessing & feature engineering |


## 📂 Project Structure
<pre>
📦 ChurnFlow - End-to-End MLOps Pipeline for Customer Retention
┣ 📂 data                                      # Raw and processed datasets 
┣ 📂 logs                                      # Application & monitoring logs 
┣ 📂 mlruns                                    # MLflow experiment tracking 
┣ 📂 models                                    # Saved model versions (.pkl files) 
┣ 📂 tests                                     # Unit & integration tests 
┣ 📂 imgs
┣ 📜 app.py                                    # FastAPI prediction service 
┣ 📜 data_ingestion.py                         # Data ingestion script 
┣ 📜 feature_engineering.py                    # Feature engineering pipeline 
┣ 📜 train_model.py                            # Model training script 
┣ 📜 retrain.py                                # Retraining logic (triggered on drift) 
┣ 📜 monitor.py                                # EvidentlyAI drift monitoring service 
┣ 📜 requirements.txt                          # Python dependencies 
┣ 📜 docker-compose.yml                        # Multi-container orchestration 
┣ 📜 api.Dockerfile                            # Dockerfile for API service 
┣ 📜 monitor.Dockerfile                        # Dockerfile for monitoring service 
┣ 📜 mlflow.Dockerfile                         # Dockerfile for MLflow service
</pre>


## 🛠️ Installation
1️⃣ **Clone the Repository**
<pre>
git clone https://github.com/ahmedmoussa/Projects-Portfolio.git
cd 'ChurnFlow - End-to-End MLOps Pipeline for Customer Retention'
</pre>

2️⃣ **Create Virtual Environment**
<pre>
python -m venv venv
source venv/bin/activate
</pre>

3️⃣ **Install Dependencies**
<pre>
pip install -r requirements.txt
</pre>


▶️ **Usage**

1️⃣ **Train and Save Model**

Run the training script to train and version your model:
<pre>
python data_ingestion.py
python feature_engineering.py
python train_model.py
</pre>
This will:
 - Train a Logistic Regression model on the churn dataset.
 - Save the model to the `models/` folder.
 - Log parameters, metrics, and artifacts in MLflow (`mlruns/` directory).

2️⃣ **Start API Server**

Use Docker Compose to build and run the API container:
<pre>
docker compose up --build
</pre>
The API will be available at:
<pre>
http://localhost:8000/predict
</pre>

3️⃣ **Make Predictions**
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

4️⃣ **Monitor Drift**
 - EvidentlyAI Dashboard: http://localhost:8001/view_report
 - Detects data drift and performance degradation, triggering retraining when necessary.


🔄 **CI/CD Pipeline**
- **Tests:** Run via pytest on every push/PR to main
- **Docker Images:** Built & pushed to Docker Hub (api, monitor, mlflow) on successful main branch builds
- **Artifacts:** Stored in GitHub Actions + MLflow


## 📊 Results
   - **Baseline Logistic Regression Model Accuracy:** 80%
   - **Baseline Logistic Regression Model AUC:** 84%
   - **EvidentlyAI reports:** Drift visualization available in `report.html`.


## 📝 License
This project is shared for portfolio purposes only and may not be used for commercial purposes without permission.