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
 1. This demonstration showcases the execution of a TensorFlow Extended (TFX) pipeline on a local machine, with Apache Airflow used for triggering and monitoring the pipeline. The base pipeline code and training architecture were adapted from the book Building Machine Learning Pipelines: Automating Model Lifecycles with TensorFlow (Hapke and Nelson, 2020). A new Tuner component has been added to the pipeline though. For this particular application, a different public dataset has been utilized for regression purposes. Note that the primary focus of this demonstration is not on the accuracy of the neural network model but on explaining the structure and execution of TFX pipelines.
 2. A key challenge encountered during the implementation was a dependency conflict between TFX and Apache Airflow, which prevented both packages from being installed in the same environment. To address this, I set up separate virtual environments for each package and leveraged Airflow to manage the TFX pipeline in its own isolated environment. However, this setup comes with a limitation: in the Airflow web UI, the TFX pipeline appears as a single task, which allows for scheduling and monitoring the entire pipeline but does not expose the dependencies between its individual components.
 3. Stay tuned for further insights on how to overcome this limitation and improve the visibility of component dependencies within the Airflow interface.
