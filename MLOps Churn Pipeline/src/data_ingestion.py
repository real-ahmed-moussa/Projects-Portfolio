# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# [1] Containerized Paths
RAW_DATA_PATH = '/app/data/raw/telco_customer_churn.csv'
PROCESSED_TRAIN_PATH = '/app/data/processed/train.csv'
PROCESSED_TEST_PATH = '/app/data/processed/test.csv'


# [2] Load Raw Data
def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    
    df = pd.read_csv(file_path)
    return df


# [3] Preprocess Data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the raw dataset - handle missing values and encode categorical variables."""
    
    # Drop customerID column
    df = df.drop(columns=['customerID'])
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))
    
    # Encode categorical variables
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    return df


# [4] Split and Save Data
def split_and_save(df):
    """Split into training and testing datasets and save them."""
    
    # Split the Data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save the Data
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)
    
    print("Data Saved Successfully!")


# [5] Main Function
if __name__ == '__main__':
    
    # Load Raw Data
    raw_df = load_data(RAW_DATA_PATH)
    
    # Preprocess Data
    processed_df = preprocess_data(raw_df)
    
    # Split and Save Data
    split_and_save(processed_df)