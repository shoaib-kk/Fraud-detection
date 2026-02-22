import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
import kagglehub
from pathlib import Path 
from utilities import setup_logger
logger = setup_logger(__name__)

def read_data():
    data_path = Path("data/creditcard.csv")
    if data_path.exists():
        logger.info("Loading dataset from local file...")
        data = pd.read_csv(data_path)
        return data
    
    logger.info("Downloading dataset")
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    data = pd.read_csv(f"{path}/creditcard.csv")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)

    return data

def validate_data(data: pd.DataFrame):
    expected_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)] + ['Class']

    missing_columns = set(expected_columns) - set(data.columns)
    extra_columns = set(data.columns) - set(expected_columns)
    if missing_columns or extra_columns:
        raise ValueError(f"Data validation failed. Missing columns: {missing_columns}, Extra columns: {extra_columns}")
    class_types = data['Class'].unique()
    if not set(class_types).issubset({0, 1}):
        raise ValueError(f"Data validation failed. 'Class' column contains unexpected values: {class_types}")
    
    if (data['Amount'] < 0).any():
        raise ValueError("Data validation failed. 'Amount' column contains negative values.")
    if (data['Time'] < 0).any():
        raise ValueError("Data validation failed. 'Time' column contains negative values.")


def train_val_test_split(data: pd.DataFrame, test_size=0.2, val_size = 0.1, random_state=42, time_based = False):
    train_size = 1 - test_size - val_size
    assert train_size + val_size + test_size == 1
    assert 0 < train_size < 1
    assert 0 < val_size < 1
    assert 0 < test_size < 1 

    if time_based:
        # Sort by time and split based on time
        data = data.sort_values(by='Time')
        n_test = int(len(data) * test_size)
        test_df = data.iloc[-n_test:]
        train_and_val_df = data.iloc[:-n_test]
        n_train = int(len(train_and_val_df) * (train_size / (train_size + val_size)))
        train_df = train_and_val_df.iloc[:n_train]
        val_df = train_and_val_df.iloc[n_train:]
        return train_df, val_df, test_df
    
    train_and_val_df, test_df = train_test_split(data, test_size = test_size, random_state=random_state, stratify=data['Class'])
    
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_and_val_df, test_size=val_size_adjusted, random_state=random_state, stratify=train_and_val_df['Class'])
    return train_df, val_df, test_df

def seperate_features_and_target(data: pd.DataFrame):
    """Split features and target; keep all feature columns except Class."""
    X = data.drop(columns=['Class'])
    y = data['Class']
    return X, y

def clean_data(data: pd.DataFrame):
    # Check for missing values
    if data.isnull().sum().any():
        data = data.dropna()
    data["Class"] = data["Class"].astype(int)
    return data

def distribution(data: pd.DataFrame, set_name: str):
    # make sure to run on both test and train to ensure they are pretty similarly distributed 
    logger.info(f"Class distribution for {set_name}:")
    logger.info(f"Number of fraud cases {data['Class'].value_counts()}")
    logger.info(f"Percentage of non-fraud cases: {(1 - data['Class'].mean()) * 100:.4f}%")
    logger.info(f"Percentage of fraud cases: {data['Class'].mean() * 100:.4f}%")


def main():
    data = read_data()
    data = clean_data(data)
    train_data, val_data, test_data = train_val_test_split(data)
    print("==="*20)
    distribution(train_data, "train")
    distribution(val_data, "validation")
    distribution(test_data, "test")
    print("==="*20)
    X_val, y_val = seperate_features_and_target(val_data)
    X_train, y_train = seperate_features_and_target(train_data)
    X_test, y_test = seperate_features_and_target(test_data)

if __name__ == "__main__":
    main()