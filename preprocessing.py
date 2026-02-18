import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import kagglehub

def read_data():
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    print("Path to dataset files:", path)
    data = pd.read_csv(f"{path}/creditcard.csv")
    return data

def train_val_test_split(data: pd.DataFrame, test_size=0.2, val_size = 0.1, random_state=42):
    train_size = 1 - test_size - val_size
    assert train_size + val_size + test_size == 1
    assert 0 < train_size < 1
    assert 0 < val_size < 1
    assert 0 < test_size < 1 

    train_and_val_df, test_df = train_test_split(data, test_size = test_size, random_state=random_state, stratify=data['Class'])
    
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_and_val_df, test_size=val_size_adjusted, random_state=random_state, stratify=train_and_val_df['Class'])
    return train_df, val_df, test_df

def seperate_features_and_target(data: pd.DataFrame):
    
    V1_V28 = [f'V{i}' for i in range(1, 29)]
    X = data[['Time', 'Amount'] + V1_V28]
    y = data['Class']
    return X, y

def clean_data(data: pd.DataFrame):
    # Check for missing values
    if data.isnull().sum().any():
        data = data.dropna()
    data["Class"] = data["Class"].astype(int)
    return data

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    feature_columns = ['Amount', 'Time']

    scaler.fit(X_train[feature_columns])
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[feature_columns] = scaler.transform(X_train[feature_columns])
    X_test_scaled[feature_columns] = scaler.transform(X_test[feature_columns])


    return X_train_scaled, X_test_scaled, scaler 

def distribution(data: pd.DataFrame, set_name: str):
    # make sure to run on both test and train to ensure they are pretty similarly distributed 
    print(f"Class distribution for {set_name}:")
    print(f"Number of fraud cases {data['Class'].value_counts()}")
    print(f"Percentage of non-fraud cases: {(1 - data['Class'].mean()) * 100:.4f}%")
    print(f"Percentage of fraud cases: {data['Class'].mean() * 100:.4f}%")


def main():
    data = read_data()
    data = clean_data(data)
    train_data, val_data, test_data = train_val_test_split(data)
    distribution(train_data, "train")
    distribution(val_data, "validation")
    distribution(test_data, "test")
    X_val, y_val = seperate_features_and_target(val_data)
    X_train, y_train = seperate_features_and_target(train_data)
    X_test, y_test = seperate_features_and_target(test_data)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

if __name__ == "__main__":
    main()