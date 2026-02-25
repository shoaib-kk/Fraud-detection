import sys
import numpy as np
import pandas as pd 
from sklearn.metrics import precision_recall_curve
import streamlit as st 
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from evaluation_metrics import evaluate_model


@st.cache_data
def load_data(data_path):
    path = Path(data_path)
    if path.exists():
        return pd.read_csv(path)
    else:
        st.warning(f"File not found: {path}")
        return pd.DataFrame()
    
def validate_data(df):
    expected_columns = {"y_true", "y_pred_proba"}
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        st.error(f"Data validation failed. Missing columns: {missing_columns}")
        return False
    return True

def confusion_counts(y_true, y_pred):
    true_p = np.sum((y_true == 1) & (y_pred == 1))
    true_n = np.sum((y_true == 0) & (y_pred == 0))
    false_p = np.sum((y_true == 0) & (y_pred == 1))
    false_n = np.sum((y_true == 1) & (y_pred == 0))

    return {"true_positive": true_p, "true_negative": true_n, "false_positive": false_p, "false_negative": false_n}
def point_metrics(counts: dict):
    true_positive, false_positive, false_negative = counts["true_positive"], counts["false_positive"], counts["false_negative"]
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

st.title("Fraud Threshold Tuning")
st.caption("Move slider to adjust threshold flagging rate")

with st.sidebar:
    st.header("Controls")
    threshold = st.slider("Threshold", min_value = 0.0, max_value = 1.0, value = 0.1, step = 0.01)
    st.divider()
    false_positive_cost = st.number_input("False positive cost", min_value=0.0, value=1.0, step=0.1)
    false_negative_cost = st.number_input("False negative cost", min_value=0.0, value=10.0, step=0.1)
    


try:
    df = load_data("reports/val_predictions.csv")
except FileNotFoundError:
    st.error("Validation predictions file not found. Please run generate_val_predictions.py to create it.")
    st.stop()
validate_data(df)
y_true = df["y_true"].values
y_pred_proba = df["y_pred_proba"].values
y_pred = (y_pred_proba >= threshold).astype(int)
counts = confusion_counts(y_true, y_pred)
metrics = point_metrics(counts)
flagged_rate = float((y_pred == 1).mean())













