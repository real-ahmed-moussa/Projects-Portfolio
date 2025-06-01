# Projects-Portfolio

Portfolio of my projects to date. Only projects with no confidentiality requirements are posted.

---

## 📚 **List of Projects**

## Hyperparameter Optimizer (Python Package)
 - Internal optimization toolkit to streamline machine learning workflows by automating hyperparameter tuning using advanced metaheuristic algorithms.
 - This Python-based solution was designed for seamless integration into existing scikit-learn pipelines and supports a wide range of hyperparameter types—continuous, integer, and categorical. It also features customizable scoring metrics, cross-validation, and full traceability for audit and reproducibility.
 - **Business Impact**
 > **The toolkit significantly reduced the time and computational effort required to tune models, enabling data science teams to deliver higher-performing models faster. It has proven particularly valuable in accelerating model development cycles, improving model accuracy, and reducing manual trial-and-error in tuning processes.**
 
###  Install from PyPI: 
    pip install hyperparameter-optimizer

### Compatible with:
| `scikit-learn` | `LightGBM` | `XGBoost` | `CatBoost` |

### ⚡️ **Tech Stack**
| Technology          | Purpose                               |
|-------------------- |---------------------------------------|
| `scikit-learn`      | Model development                     |
| `PSO`               | Particle Swarm Optimization Algorithm |
| `PS`                | Pattern Search Algorithm              |

---

## DeepVision: Video Analysis
 - Deep learning solution for automated video content analysis by building a **ConvLSTM-based video classification model**.
 - The model architecture combined convolutional layers with LSTM units (ConvLSTM2D) to extract spatial features from individual frames and capture temporal dynamics across sequences. OpenCV was used to streamline frame extraction and preprocessing, while TensorFlow served as the core framework for model development, training, and evaluation.
 - **Business Impact**
 > **This solution addressed the challenge of accurately classifying complex video data in real time, enabling use cases such as automated quality control, surveillance anomaly detection, and content tagging. It improved processing efficiency by reducing the need for manual video review and enhanced the accuracy of insights derived from unstructured video data.**

### ⚡️ **Tech Stack**
| Technology          | Purpose                                 |
|-------------------- |-----------------------------------------|
| `TensorFlow`        | Model development                       |
| `OpenCV`            | Video frame extraction                  |
| `CNN`               | Spatial feature extraction from frames  |
| `LSTM`              | Temporal pattern learning across frames |
| `NumPy`             | Data manipulation                       |

---

## ChurnFlow: End-to-End MLOps Pipeline for Customer Retention
 - Production-grade MLOps pipeline to predict and mitigate customer churn for a telecom client.
 - This end-to-end solution integrated model development, CI/CD automation, real-time monitoring, and automated retraining workflows. Using a real-world telecom dataset, the pipeline enabled seamless deployment and continuous improvement of churn prediction models.
 - **Key Components**
 >  1. **Model Serving:** FastAPI-based REST endpoint for real-time inference.
    2. **CI/CD Automation:** GitHub Actions to manage code integration, model deployment, and retraining triggers.
    3. **Performance Monitoring:** MLflow for model versioning, experiment tracking, and lifecycle management.
    4. **Drift Detection:** Integrated EvidentlyAI to monitor feature and prediction drift in production, triggering retraining workflows as needed.
 - **Business Impact**
 > **This pipeline significantly reduced the time to deploy and maintain predictive models, improved model reliability in production, and enhanced the business’s ability to proactively retain at-risk customers.**

### ⚡️ **Tech Stack**
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

## AirflowTFX: Reproducible ML Pipeline Orchestration with TensorFlow Extended (TFX)
 - Modular and containerized ML pipeline based on *TensorFlow Extended (TFX)* and orchestrated with *Apache Airflow* to enable automated, repeatable workflows in a controlled local environment.
 - The pipeline architecture was based on industry-standard design patterns, extended from *"Building Machine Learning Pipelines" (Hapke & Nelson, 2020)*, and customized with Python to support domain-specific tasks and artifacts. *Docker* was used to ensure full environment reproducibility, aligning with production deployment standards.
 - **Key Components**
 >  1. **Pipeline Orchestration:** Configured Airflow DAGs to manage pipeline execution, logging, and task dependency tracking.
    2. **Custom TFX Components:** Designed Python-based components to address domain-specific preprocessing, transformation, and evaluation requirements.
    3. **Environment Reproducibility:** Employed Docker to encapsulate dependencies and configurations, ensuring consistent results across development and deployment environments.
 - **Business Impact**
 > This solution laid the groundwork for scalable, production-ready ML infrastructure, improving experiment reproducibility, modularity, and maintainability. It demonstrated how to integrate robust orchestration tools into ML pipelines, bridging the gap between model development and deployment operations.

### ⚡️ **Tech Stack**
| Technology                  | Purpose                                                              |
|---------------------------- |----------------------------------------------------------------------|
| `TensorFlow Extended (TFX)` | Core framework for building standardized, end-to-end ML pipelines.   |
| `Apache Airflow`            | Orchestrates pipeline execution through DAGs and task scheduling.    |
| `Docker`                    | Ensures reproducible environments across development and deployment. |
| `TF Data Validation (TFDV)` | Detects anomalies and infers schema from training data.              |
| `TF Transform (TFT)`        | Applies scalable, consistent preprocessing for training and serving. |
| `TF Model Analysis (TFMA)`  | Evaluates model performance using configurable metrics and slicing.  |
| `SQLite`                    | Stores pipeline metadata for reproducibility and traceability.       |
| `Airflow CLI`               | Manages and monitors DAG execution during development and testing.   |

---

## KubeTFX: Scalable ML Pipeline Orchestration with TensorFlow Extended and Kubeflow
 - *TensorFlow Extended (TFX)* pipeline using Kubeflow Pipelines on a local Minikube Kubernetes cluster, designed to simulate real-world production orchestration in a cloud-native environment.
 - The project adapted architectural principles from *"Building Machine Learning Pipelines" (Hapke & Nelson, 2020)* and emphasized modularity, reusability, and end-to-end ML lifecycle orchestration. *Docker* and Kubernetes were used to containerize components and manage execution environments, while YAML configurations ensured reproducibility and portability.
 - **Key Features**
 >  1. **End-to-End Pipeline Orchestration:** Managed preprocessing, model training, evaluation, and deployment workflows using Kubeflow Pipelines.
    2. **Kubernetes-Based Execution:** Utilized Minikube to simulate scalable infrastructure, validating cloud-readiness and resource management under a local environment.
    3. **Portable & Modular Design:** Encapsulated pipeline steps as reusable components with Docker and YAML to support easy adaptation across projects.
 - **Business Impact**
 > **This solution provided a robust framework for scalable machine learning operations, enabling seamless transition from development to production. It showcased how local Kubernetes environments can be leveraged for iterative pipeline testing, improving reproducibility, infrastructure-as-code practices, and deployment readiness.**

### ⚡️ **Tech Stack**
| Technology                  | Purpose                                                              |
|---------------------------- |----------------------------------------------------------------------|
| `TensorFlow Extended (TFX)` | Core framework for building standardized, end-to-end ML pipelines.   |
| `Kubeflow`                  | Orchestrates pipeline execution through DAGs and task scheduling.    |
| `Minikube`                  | Local Kubernetes cluster management.                                 |
| `Docker`                    | Ensures reproducible environments across development and deployment. |
| `TF Data Validation (TFDV)` | Detects anomalies and infers schema from training data.              |
| `TF Transform (TFT)`        | Applies scalable, consistent preprocessing for training and serving. |
| `TF Model Analysis (TFMA)`  | Evaluates model performance using configurable metrics and slicing.  |
| `SQLite`                    | Stores pipeline metadata for reproducibility and traceability.       |

---

## OptiTree: Benchmarking Tree-Based Models with Metaheuristic Optimization
 - Bench-marked suite of high-performance tree-based models, including Decision Trees, Random Forests, AdaBoost, Gradient Boosting, LightGBM, XGBoost, and CatBoost, on structured classification tasks.
 - Model optimization was fully automated using Particle Swarm Optimization (PSO) via the hyperparameter-optimizer library. This approach efficiently searched complex hyperparameter spaces, outperforming traditional tuning methods. MLflow was used to track experiments, and joblib enabled serialized model persistence.
 - **Business Impact**
 > **This solution provided a scalable framework for selecting and tuning models in high-stakes decision-making contexts, delivering improved predictive accuracy and faster iteration cycles.**

### ⚡️ **Tech Stack**
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

## PolicyPredict: Insurance Purchase Prediction with Deep Learning
 - End-to-end machine learning solution to predict customer likelihood of purchasing new insurance policies, enabling targeted marketing and improved customer engagement strategies.
 - The pipeline included data preprocessing, feature engineering, and the design of a fully connected neural network using TensorFlow. scikit-learn was used for pipeline orchestration and model evaluation, while pandas and NumPy facilitated efficient data manipulation. Model insights and performance metrics were visualized using matplotlib and seaborn to support stakeholder decision-making.
 - **Business Impact**
 > **This solution provided the business with actionable intelligence on customer purchasing behavior, increasing campaign efficiency and supporting data-driven policy recommendations.**

### ⚡️ **Tech Stack**
| Technology          | Purpose                                                  |
|-------------------- |----------------------------------------------------------|
| `TensorFlow`        | Neural network model development                         |
| `scikit-learn`      | Data preprocessing, feature engineering, model pipelines |
| `NumPy`             | Data manipulation                                        |
| `pandas`            | Data preprocessing & analysis                            |
| `matplotlib`        | Data visualization and customization                     |
| `seaborn`           | Statistical data visualization                           |

---

## PolicyPredict Lite: Insurance Purchase Modeling with Classical ML Algorithms
 - Multiple ML models to predict customer purchases of new insurance policies, supporting data-driven customer outreach and cross-selling initiatives.
 - The solution involved implementing Logistic Regression, Random Forests, and Support Vector Machines (SVMs) within a unified scikit-learn pipeline. Core stages included data preprocessing, feature selection, model training, and performance evaluation. pandas and NumPy were used for efficient data manipulation and exploratory analysis.
 - **Business Impact**
 > **This approach provided fast, interpretable insights into customer behavior, helping stakeholders prioritize leads and tailor communication strategies without the overhead of deep learning infrastructure.**

### ⚡️ **Tech Stack**
| Technology          | Purpose                                                                           |
|-------------------- |-----------------------------------------------------------------------------------|
| `scikit-learn`      | Data preprocessing, model development (Logistic Regression, Random Forests, SVMs) |
| `pandas`            | Data preprocessing & analysis                                                     |
| `NumPy`             | Data manipulation                                                                 |

---

## AgriGrain: Multiclass Classification of Grain Types Using Ensemble and Neural Models
 - ML-based solution to classify various grain types based on geometric features, enabling automation in agricultural quality control and grain sorting processes.
 - The project utilized a range of classification algorithms, including ensemble methods (Bagging, Random Forests, Boosting), Mixture Discriminant Analysis, and Artificial Neural Networks (ANNs). Models were implemented using R libraries, with additional data preprocessing and exploratory analysis conducted in Python using pandas and NumPy.
 - **Business Impact**
 > This solution enhanced the accuracy and efficiency of grain classification, reducing reliance on manual inspection and supporting scalable agri-tech applications.

***[Multiclass Classification]***

### ⚡️ **Tech Stack**
| Technology          | Purpose                                                                                        |
|-------------------- |------------------------------------------------------------------------------------------------|
| `R Libraries`       | Model development (Bagging, Random Forests, Boosting, Mixture Discriminant Analysis, and ANNs) |
| `pandas`            | Data preprocessing & analysis                                                                  |
| `NumPy`             | Data manipulation                                                                              |