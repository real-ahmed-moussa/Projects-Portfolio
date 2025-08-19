# Import Libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Local Development & Containerized Paths
BASE = "/app" if os.path.exists("/app") else "/home/ahmedmoussa/training-files/mlops_ppln/ppln"
BASE = os.getenv("APP_BASE", BASE)
RAW_DATA_PATH = f'{BASE}/data/raw/telco_customer_churn.csv'
PROCESSED_TRAIN_PATH = f'{BASE}/data/processed/train.csv'
PROCESSED_TEST_PATH = f'{BASE}/data/processed/test.csv'
PROCESSED_REF_PATH = f'{BASE}/data/processed/ref_data.csv'

# Constants
TARGET_COL = "Churn"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
DROP_COLS = ["customerID"]
SEED = 42


# [1] Load Raw Data
def load_data(file_path: str) -> pd.DataFrame:
    """Load the raw dataset from a CSV file."""
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"Raw dataset is empty: {file_path}")
    return df


# [2] Preprocess Data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the raw dataset - handle missing values and encode categorical variables."""
    df = df.copy()

    # 1. Drop Safe-to-remove Columns
    for col in DROP_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # 2. Coerce Numeric Columns
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 3. Impute All Columns: numeric -> median; categorical -> mode
    for col in df.columns:
        if col == TARGET_COL:                                           # pass target column
            continue
        if pd.api.types.is_numeric_dtype(df[col]):                      # impute numeric columns
            df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().all():                                    # edge case: impute columns with no values
                df[col] = df[col].fillna("UNKNOWN")
            else:                                                       # impute categorical columns
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    
    # 4. Encode Target Column > Integer
    if TARGET_COL in df.columns and not pd.api.types.is_numeric_dtype(df[TARGET_COL]):
        mapping = {"No": 0, "Yes": 1}
        df[TARGET_COL] = df[TARGET_COL].map(mapping).astype("int64")
    
    # 5. Encode Categorical Features
    for col in df.columns:
        if col == TARGET_COL:                                           # pass target column
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):                  # encode categorical columns > integers
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str)).astype("int64")
        
    return df


# [3] Stratified Split and Save Data
def split_and_save(df: pd.DataFrame) -> None:
    """Split into training and testing datasets and save them."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataframe.")

    # Split the Data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df[TARGET_COL])
    
    # Save the Data
    train_df.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_PATH, index=False)
    train_df.sample(frac=0.1, random_state=42).to_csv(PROCESSED_REF_PATH, index=False)

    print(f"[INGEST] Saved train Successfully to {PROCESSED_TRAIN_PATH}")
    print(f"[INGEST] Saved test Successfully to {PROCESSED_TEST_PATH}")
    print(f"[INGEST] Saved ref Successfully to {PROCESSED_REF_PATH}")


# [4] Main Function
if __name__ == '__main__':
    # Load Raw Data
    raw_df = load_data(RAW_DATA_PATH)
    
    # Preprocess Data
    processed_df = preprocess_data(raw_df)
    
    # Split and Save Data
    split_and_save(processed_df)

    print("[INGEST] Done.")