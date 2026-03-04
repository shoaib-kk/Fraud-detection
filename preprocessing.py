import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from utilities import setup_logger
logger = setup_logger(__name__)


def _ensure_sample_from_full(full_df: pd.DataFrame, sample_path: Path, n_rows: int = 5000) -> pd.DataFrame:
    """Create a small sample parquet if missing to keep the app offline-friendly."""
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df = full_df.head(n_rows)
    sample_df.to_parquet(sample_path, index=False)
    return sample_df


def read_data(use_full_dataset: bool = True):
    data_dir = Path("data")
    csv_path = data_dir / "creditcard.csv"
    full_parquet_path = data_dir / "creditcard_full.parquet"
    sample_path = data_dir / "sample.parquet"

    if not use_full_dataset:
        if sample_path.exists():
            logger.info("Loading sample dataset (cached parquet)...")
            return pd.read_parquet(sample_path)

        if full_parquet_path.exists():
            logger.info("Sample parquet missing; creating from cached full parquet.")
            full_df = pd.read_parquet(full_parquet_path)
            return _ensure_sample_from_full(full_df, sample_path)

        if csv_path.exists():
            logger.info("Sample parquet missing; creating from existing CSV.")
            full_df = pd.read_csv(csv_path)
            return _ensure_sample_from_full(full_df, sample_path)

        logger.info("Sample parquet missing and no cached data; downloading full dataset to create sample.")
        use_full_dataset = True

    if full_parquet_path.exists():
        logger.info("Loading full dataset from cached parquet...")
        return pd.read_parquet(full_parquet_path)

    if csv_path.exists():
        logger.info("Converting existing CSV to cached parquet...")
        full_df = pd.read_csv(csv_path)
        full_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(full_parquet_path, index=False)
        return full_df

    logger.info("Downloading full dataset (one-time Kaggle fetch)...")

    try:
        import streamlit as st  # Only used to access secrets when available
        os.environ.setdefault("KAGGLE_USERNAME", st.secrets.get("KAGGLE_USERNAME", ""))
        os.environ.setdefault("KAGGLE_KEY", st.secrets.get("KAGGLE_KEY", ""))
    except Exception:
        logger.info("Streamlit secrets not available; expecting KAGGLE_USERNAME/KAGGLE_KEY in env.")

    from kaggle.api.kaggle_api_extended import KaggleApi  # imported lazily to avoid auth prompts

    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        "mlg-ulb/creditcardfraud",
        path=str(data_dir),
        unzip=True
    )

    full_df = pd.read_csv(csv_path)
    full_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(csv_path, index=False)
    full_df.to_parquet(full_parquet_path, index=False)
    return full_df

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