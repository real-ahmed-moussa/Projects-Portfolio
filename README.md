# Projects-Portfolio

Portfolio of my projects to date. Only projects with no confidentiality requirements are posted.

---

## 📚 **List of Projects**

## [1] Insurance Policy Classification (TensorFlow)
[Insurance_Policy_Classification_TF.ipynb] This project implements a **TensorFlow Neural Network (TF NN)** to determine whether a client will purchase a new insurance policy based on historical data.

---

### ⚡️ **Technologies Used**
| Technology          | Purpose                                                  |
|-------------------- |----------------------------------------------------------|
| `TensorFlow`        | Neural network model development                         |
| `scikit-learn`      | Data preprocessing, feature engineering, model pipelines |
| `NumPy`             | Data manipulation                                        |
| `pandas`            | Data preprocessing & analysis                            |
| `matplotlib`        | Data visualization and customization                     |
| `seaborn`           | Statistical data visualization                           |

---

## [2] Insurance Policy Classification (Scikit-Learn)
[Insurance_Policy_Classification.ipynb] Implemented and compared **Logistic Regression, Random Forests, and SVMs** to predict the likelihood of a client purchasing a new insurance policy.

---

### ⚡️ **Technologies Used**
| Technology          | Purpose                                                                           |
|-------------------- |-----------------------------------------------------------------------------------|
| `scikit-learn`      | Data preprocessing, model development (Logistic Regression, Random Forests, SVMs) |
| `pandas`            | Data preprocessing & analysis                                                     |
| `NumPy`             | Data manipulation                                                                 |

---

## [3] Beans Multiclass Classification
[Beans_Multiclass_Classification.R] Applied **Bagging, Random Forests, Boosting, Mixture Discriminant Analysis, and Artificial Neural Networks (ANNs)** to classify bean types based on geometric characteristics.

---

### ⚡️ **Technologies Used**
| Technology          | Purpose                                                                                        |
|-------------------- |------------------------------------------------------------------------------------------------|
| `R Libraries`       | Model development (Bagging, Random Forests, Boosting, Mixture Discriminant Analysis, and ANNs) |
| `pandas`            | Data preprocessing & analysis                                                                  |
| `NumPy`             | Data manipulation                                                                              |

---

## [4] TFX Pipeline with Airflow Integration
[TFX Pipeline with Airflow Integration] This project demonstrates the execution of a **TensorFlow Extended (TFX) pipeline** on a local machine, with **Apache Airflow** used for triggering and monitoring the pipeline. The base pipeline code and training architecture were adapted from *Building Machine Learning Pipelines: Automating Model Lifecycles with TensorFlow (Hapke and Nelson, 2020).*

---

### ⚡️ **Technologies Used**
| Technology          | Purpose                              |
|-------------------- |--------------------------------------|
| `TensorFlow`        | Model training & transformation      |
| `Apache Airflow`    | Pipeline orchestration               |
| `Docker`            | Containerized execution              |
| `Python`            | Custom pipeline components           |

---

## [5] TFX Pipeline with Kubeflow Integration
[TFX Pipeline with Kubeflow Integration] This project demonstrates the execution of a **TensorFlow Extended (TFX) pipeline** using **Kubeflow Pipelines** on a Minikube cluster running on Docker. The base pipeline code and training architecture were adapted from *Building Machine Learning Pipelines: Automating Model Lifecycles with TensorFlow (Hapke and Nelson, 2020).*

---

### ⚡️ **Technologies Used**
| Technology          | Purpose                              |
|-------------------- |--------------------------------------|
| `TensorFlow`        | Model training & transformation      |
| `Kubeflow`          | Pipeline orchestration               |
| `Minikube`          | Local Kubernetes cluster management  |
| `Docker`            | Containerized execution              |
| `YAML`              | Pipeline configuration               |

---

## [6] Video Classification using ConvLSTM2D
[ConvLSTM2D Video Classification] This project implements a **video classification model** using **ConvLSTM2D** to analyze sequences of video frames for multi-class classification. It combines convolutional layers for spatial feature extraction with LSTM units to capture temporal dependencies.

---

### ⚡️ **Technologies Used**
| Technology          | Purpose                              |
|-------------------- |--------------------------------------|
| `TensorFlow`        | Model development                    |
| `OpenCV`            | Video frame extraction               |
| `NumPy`             | Data manipulation                    |

---

## [7] Tree-Based Models Hyperparameter Optimization
[Tree_Based_Models.ipynb] This project involves the development and optimization of tree-based machine learning models for classification tasks. The models include:

    Decision Trees

    Random Forests

    AdaBoost

    Gradient Boosting

    LightGBM

    XGBoost

    CatBoost

Hyperparameters for each model were automatically tuned using Particle Swarm Optimization (PSO) via the hyperparameter-optimizer library. This approach significantly improved model accuracy, robustness, and generalization compared to manual or grid search methods.

---

### ⚡️ **Technologies Used**
| Technology                  | Purpose                                                     |
|-----------------------------|-------------------------------------------------------------|
| `scikit-learn`              | Decision Trees, Random Forests, AdaBoost, Gradient Boosting |
| `LightGBM`                  | Gradient boosting with optimized performance                |
| `XGBoost`                   | Extreme gradient boosting                                   |
| `CatBoost`                  | Gradient boosting with categorical feature support          |
| `hyperparameter-optimizer`  | Automated hyperparameter tuning                             |
| `MLflow`                    | Experiment tracking and model management                    |
| `joblib`                    | Model serialization and versioning                          |
| `pandas`                    | Data preprocessing & transformation                         |

---
🔹 Notes

    Particle Swarm Optimization (PSO) is a nature-inspired global optimization algorithm that efficiently explores the hyperparameter space, outperforming traditional grid and random search strategies in both convergence speed and solution quality.
---

## [8] MLOps Churn Prediction Pipeline
[MLOps Churn Pipeline] This project demonstrates an end-to-end **MLOps pipeline** for customer churn prediction using a real-world telecom dataset. It integrates **CI/CD, drift detection, model retraining, versioning, and model performance tracking** with **MLflow**.

---

### ⚡️ **Technologies Used**
| Technology          | Purpose                              |
|-------------------- |--------------------------------------|
| `scikit-learn`      | Model development                    |
| `EvidentlyAI`       | Drift detection & monitoring         |
| `MLflow`            | Model tracking & performance logging |
| `FastAPI`           | API for inference                    |
| `Docker`            | Containerization                     |
| `GitHub Actions`    | CI/CD automation                     |
| `joblib`            | Model serialization and versioning   |
| `pandas`            | Data preprocessing & transformation  |

---

### 🏗️ **Pipeline Workflow**
1. **Model Training & Versioning:**
   - Train a **Logistic Regression Model** on telecom data.
   - Save models with versioned names (`churn_pred_model_v1.pkl`).
   - **Track model metrics and parameters with MLflow.**

2. **API Deployment:**
   - Serve predictions via `app.py` using **FastAPI**.
   - API listens on `http://localhost:8000/predict`.

3. **Drift Detection & Monitoring:**
   - **EvidentlyAI** compares incoming data with reference data.
   - Drift detection triggers retraining when performance drops.
   - Retrained models are versioned and tracked via MLflow.

4. **CI/CD with GitHub Actions:**
   - Automate Docker builds and model versioning.
   - Push images to Docker Hub.
   - Deploy updated services when drift is detected.

---