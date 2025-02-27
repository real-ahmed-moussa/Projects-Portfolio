# Projects-Portfolio
Portfolio of my projects to date. Only projects with no confidentiality requirements are posted.

List of Projects
================
[1] Binary Classification
-------------------------
 1. Insurance_Policy_Classification TF ... Implement TF NN to determine whether a client will buy a new insurance policy or not.
 2. Insurance_Policy_Classification    ... Implement Logistic Regression, Random Forests, and SVMs to determine whether a client will buy a new inusrance policy or not.
 
 [2] Multi-class Classification
 ------------------------------
 1. Beans_Multiclass_Classification ... Implement Bagging, Random Forests, Boosting, Mixture Discriminant Analysis, and Artifical Neural Networks to determine the type of beans based on certain geometric characteristics.

 [3] TFX Pipeline Demonstration with Airflow Integration
 ------------------------------------------------------
This project demonstrates the execution of a TensorFlow Extended (TFX) pipeline on a local machine, with Apache Airflow used for triggering and monitoring the pipeline. The base pipeline code and training architecture were adapted from Building Machine Learning Pipelines: Automating Model Lifecycles with TensorFlow (Hapke and Nelson, 2020).

Key Enhancements & Features
    Hyperparameter Tuning: Introduced a new Tuner component to optimize the model's performance.
    Dataset & Application: A different public dataset was used for a regression task, shifting the focus to pipeline execution rather than model accuracy.
    Dependency Conflict Resolution: Addressed a conflict between TFX and Apache Airflow, which prevented both from being installed in the same environment.
        Implemented separate virtual environments, isolating TFX while allowing Airflow to manage and trigger the pipeline.
        Limitation: In the Airflow web UI, the TFX pipeline appears as a single task, allowing for scheduling and monitoring but lacking component-level visibility.

Future Improvements
    Enhancing Airflow's interface to improve visibility of pipeline component dependencies.
    Exploring alternative orchestration strategies for better integration between TFX and Apache Airflow.

This project serves as a foundational demonstration of MLOps principles, showcasing automated model lifecycle management and workflow orchestration with TFX and Airflow.

 [4] TFX Pipeline Demonstration with Kubeflow Integration
 ------------------------------------------------------
This project demonstrates the execution of a TensorFlow Extended (TFX) pipeline using Kubeflow Pipelines on a Minikube cluster running on Docker. The pipeline was deployed and managed with kubectl, ensuring a containerized and scalable machine learning workflow. The base pipeline code and training architecture were adapted from Building Machine Learning Pipelines: Automating Model Lifecycles with TensorFlow (Hapke and Nelson, 2020).

Key Enhancements & Features
    Hyperparameter Tuning: Integrated a Tuner component to optimize model performance.
    Dataset & Application: Utilized a different public dataset for a regression task, focusing on pipeline execution rather than model accuracy.
    Kubeflow on Minikube (Docker): The pipeline was deployed on a Minikube cluster running on Docker, enabling seamless local testing and orchestration.
    Persistent Storage Management:
        Implemented Persistent Volumes (PVs), Persistent Volume Claims (PVCs), and Persistent Volume Mounts to handle data storage and access efficiently.
        Ensured data persistence across pipeline runs, facilitating reproducibility.
    Pipeline Deployment with YAML Configuration:
        Defined and executed the Kubeflow pipeline using a YAML configuration file, streamlining deployment and automation.

Future Improvements
    Enhancing pipeline visibility and monitoring within the Kubeflow UI.
    Exploring cloud-based deployments for improved scalability and resource efficiency.

This project serves as a comprehensive demonstration of MLOps principles, showcasing automated model lifecycle management, pipeline orchestration, and persistent storage handling with TFX and Kubeflow Pipelines.