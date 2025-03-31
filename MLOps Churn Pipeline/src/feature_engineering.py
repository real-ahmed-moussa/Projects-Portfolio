# Import Libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib


# [1] Containerized Paths
PROCESSED_TRAIN_PATH = '/app/data/processed/train.csv'
PROCESSED_TEST_PATH = '/app/data/processed/test.csv'

TRAIN_FEATURES_PATH = '/app/data/processed/train_transformed.csv'
TEST_FEATURES_PATH = '/app/data/processed/test_transformed.csv'


# [2] Load the Data
def load_data():
    """Load preprocessed data."""
    train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_PATH)
    return train_df, test_df


# [3] Create Feature Pipeline
def create_feature_pipeline():
    """Create pipeline for feature engineering."""
    
    numeric_features = [
        "tenure", "MonthlyCharges", "TotalCharges"
    ]
    categorical_features = [
        "gender", "Partner", "Dependents", "PhoneService", 
        "MultipleLines", "InternetService", "OnlineSecurity", 
        "OnlineBackup", "DeviceProtection", "TechSupport", 
        "StreamingTV", "StreamingMovies", "Contract", 
        "PaperlessBilling", "PaymentMethod"
    ]
    
    # Numeric Features Pipeline (impute missing values and perform scaling)
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical Features Pipeline - (impute missing values and perform one-hot-encoding)
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine Both Pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_features),
            ('cat', categorical_pipeline, categorical_features)
        ])
    
    return preprocessor


# [4] Function to Transform the Features
def transform_features(preprocessor, train_df, test_df):
    """Apply feature engineering transformations to train and test data."""
    
    x_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]
    
    x_test = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]
    
    # Perform Transformations
    x_train_transformed = preprocessor.fit_transform(x_train)
    x_test_transformed = preprocessor.transform(x_test)
    
    return x_train_transformed, x_test_transformed, y_train, y_test


# [5] Save Transformed Data
def save_transformed_data(x_train, x_test, y_train, y_test):
    """Function to save transformed data to CSV files."""
    
    # Combine Training Data
    train_data = pd.DataFrame(x_train)
    train_data["Churn"] = y_train
    
    # Combine Testing Data
    test_data = pd.DataFrame(x_test)
    test_data["Churn"] = y_test
    
    # Export the Data
    train_data.to_csv(TRAIN_FEATURES_PATH, index=False)
    test_data.to_csv(TEST_FEATURES_PATH, index=False)
    print("Data transformed and saved successfully!")
    

# [6] Main Function
if __name__ == "__main__":
    train_df, test_df = load_data()
    preprocessor = create_feature_pipeline()
    x_train, x_test, y_train, y_test = transform_features(preprocessor, train_df, test_df)
    save_transformed_data(x_train, x_test, y_train, y_test)
    
    # Save the Preprocessor Pipeline
    joblib.dump(preprocessor, "/home/ahmedmoussa/training-files/mlops_ppln/mlops-churn-pipeline/models/preprocessor.pkl")