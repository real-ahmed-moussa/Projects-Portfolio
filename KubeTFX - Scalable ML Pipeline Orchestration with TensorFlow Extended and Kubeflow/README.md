# 📌 KubeTFX: Scalable ML Pipeline Orchestration with TensorFlow Extended and Kubeflow
> End-to-end machine learning pipeline built with TFX and orchestrated via Kubeflow Pipelines on a local Minikube Kubernetes cluster, demonstrating cloud-native, scalable ML workflows.


## 📖 Overview
- This project implements a **TFX-based ML pipeline** for **medical insurance cost prediction**, orchestrated with **Kubeflow Pipelines** on **Minikube** to simulate production-ready, cloud-native ML workflows.
- Adapted from "Building Machine Learning Pipelines" (Hapke & Nelson, 2020), this version features a **custom dataset** and tailored pipeline components, containerized with Docker for reproducibility.
- Focused on **scalable orchestration, Kubernetes-native execution, and infrastructure-as-code practices** to bridge local development and cloud deployment readiness.


## 🚀 Features
- ✅ **End-to-End Orchestration:** Automates ingestion, schema validation, data transformation, model training, and evaluation under Kubeflow Pipelines.
- ✅ **Kubernetes-Native Execution:** Uses Minikube for local testing of scalable, cloud-compatible workflows.
- ✅ **Modular and Portable Design:** Dockerized components with YAML-based configuration for seamless reuse and portability.
- ✅ **TFX Ecosystem Integration:** Leverages TFDV, TFT, and TFMA for data validation, transformation, and model analysis.


## 🏢 Business Impact
   > *This project demonstrates how Kubernetes-based orchestration improves **scalability, reproducibility, and deployment readiness** in machine learning pipelines, accelerating the path from development to production while ensuring reliable, auditable workflows.*


## ⚙️ Tech Stack
| Technology                  | Purpose                                                            |
| --------------------------- | ------------------------------------------------------------------ |
| `Python`                    | Core programming language                                          |
| `TensorFlow Extended (TFX)` | Standardized ML pipeline framework                                 |
| `Kubeflow Pipelines`        | Orchestrates pipeline execution and task scheduling                |
| `Minikube`                  | Local Kubernetes cluster simulation                                |
| `Docker`                    | Encapsulates dependencies and ensures environment reproducibility  |
| `TF Data Validation (TFDV)` | Detects anomalies and infers schema from training data             |
| `TF Transform (TFT)`        | Applies consistent preprocessing for training and serving          |
| `TF Model Analysis (TFMA)`  | Evaluates model performance using configurable metrics and slicing |
| `SQLite`                    | Stores pipeline metadata for reproducibility and traceability      |



## 📂 Project Structure
<pre>
📦 KubeTFX - Scalable ML Pipeline Orchestration with TensorFlow Extended and Kubeflow
 ┣ 📂 pl_comps                # Custom pipeline components (TFX modules) 
 ┣ 📂 pl_yaml_output          # Compiled Kubeflow pipeline YAML outputs 
 ┣ 📜 base_pipeline.py        # Core pipeline definition (TFX DAG) 
 ┣ 📜 pipeline_run.py         # Script to compile & submit pipeline to Kubeflow
 ┣ 📜 pv.yaml                 # Persistent Volume definition for Minikube
 ┣ 📜 pvc.yaml                # Persistent Volume Claim for data storage 
 ┗ README.md
</pre>


## 🛠️ Installation
1️⃣ **Clone the Repository**
<pre>
git clone https://github.com/ahmedmoussa/Projects-Portfolio.git
cd 'KubeTFX - Scalable ML Pipeline Orchestration with TensorFlow Extended and Kubeflow'
</pre>

2️⃣ **Start Minikube Cluster**
<pre>
minikube start --cpus=4 --memory=8192
</pre>

3️⃣ **Apply Persistent Volumes (PV/PVC)**
<pre>
kubectl apply -f pv.yaml
kubectl apply -f pvc.yaml
</pre>

4️⃣ **Ensure Kubeflow Pipelines is running**
- Install via Kubeflow manifests or MiniKF.
- Verify:
<pre>
kubectl get pods -n kubeflow
</pre>

5️⃣ **Compile and submit Pipeline**
<pre>
python pipeline_run.py
</pre>
- The compiled YAML will be saved under `pl_yaml_output/` and submitted to the Kubeflow Pipelines UI.
- Access Kubeflow Pipelines Dashboard:
<pre>
minikube service ml-pipeline-ui -n kubeflow
</pre>


## 📊 Results
- **Prediction Task:** Regression model for medical insurance cost prediction.
- **Containerized, Portable Pipeline:** Validated on local Kubernetes cluster.
- **Cloud-ready Orchestration:** Demonstrated scalable, reproducible ML workflow management for real-world deployment scenarios.

## 📝 License
This project is shared for portfolio purposes only and may not be used for commercial purposes without permission.