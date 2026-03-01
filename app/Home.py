import sys
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from lightGBM_main_model import train_lightgbm
from logistic_regression_baseline import train_logistic_regression
from preprocessing import clean_data, read_data, seperate_features_and_target, train_val_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def confusion_counts(y_true, y_pred):
    true_p = np.sum((y_true == 1) & (y_pred == 1))
    true_n = np.sum((y_true == 0) & (y_pred == 0))
    false_p = np.sum((y_true == 0) & (y_pred == 1))
    false_n = np.sum((y_true == 1) & (y_pred == 0))
    return {"true_positive": true_p, "true_negative": true_n, "false_positive": false_p, "false_negative": false_n}
def point_metrics(counts: dict):
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}


@st.cache_data(show_spinner=False)
def load_and_split_data():
    data = read_data()
    data = clean_data(data)
    train_df, val_df, _ = train_val_test_split(data, test_size=0.2, val_size=0.1, random_state=42, time_based=False)
    return train_df, val_df


@st.cache_resource(show_spinner=True)
def train_models(train_df, val_df):
    X_train, y_train = seperate_features_and_target(train_df)
    X_val, y_val = seperate_features_and_target(val_df)

    logistic_regression = train_logistic_regression(X_train, y_train, {"C": 1.0, "solver": "lbfgs"})
    lgb_model, _ = train_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        params={
            "num_leaves": 128,
            "min_data_in_leaf": 16,
            "feature_fraction": 1.0,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
        },
        num_boost_round=1500,
        early_stopping_rounds=100,
        verbose_eval=False,
        pos_weight_multiplier=0.5,
    )

    preds = {
        "LightGBM": lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration or lgb_model.num_trees()),
        "Logistic Regression": logistic_regression.predict_proba(X_val)[:, 1],
    }
    return y_val.values, preds


st.title("Fraud Model Comparison")
st.caption("Comparison of LightGBM vs Logistic Regression on the validation set.")

with st.sidebar:
    st.header("Controls")
    threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    st.divider()
    false_positive_cost = st.number_input("False positive cost", min_value=0.0, value=1.0, step=0.1)
    false_negative_cost = st.number_input("False negative cost", min_value=0.0, value=10.0, step=0.1)


with st.spinner("Loading data and training models"):
    train_df, val_df = load_and_split_data()
    y_val, model_preds = train_models(train_df, val_df)


def summarize_model(name, proba):
    y_pred = (proba >= threshold).astype(int)
    counts = confusion_counts(y_val, y_pred)
    metrics = point_metrics(counts)
    flagged_rate = float((y_pred == 1).mean())
    est_cost = counts["false_positive"] * false_positive_cost + counts["false_negative"] * false_negative_cost
    aps = average_precision_score(y_val, proba)
    roc_auc = roc_auc_score(y_val, proba)
    return {
        "Model": name,
        "Flagged Rate": flagged_rate,
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "APS": aps,
        "ROC-AUC": roc_auc,
        "Estimated Cost": est_cost,
        "True positive": counts["true_positive"],
        "False positive": counts["false_positive"],
        "False negative": counts["false_negative"],
        "True negative": counts["true_negative"],
    }


summary_rows = [summarize_model(name, proba) for name, proba in model_preds.items()]
summary_df = pd.DataFrame(summary_rows).set_index("Model")



# Highlight models with the best APS and ROC-AUC
best_aps_model = summary_df["APS"].idxmax()
best_aps_value = summary_df.loc[best_aps_model, "APS"]
best_roc_model = summary_df["ROC-AUC"].idxmax()
best_roc_value = summary_df.loc[best_roc_model, "ROC-AUC"]


st.metric("Best APS", f"{best_aps_model}: {best_aps_value:.4f}")

st.metric("Best ROC-AUC", f"{best_roc_model}: {best_roc_value:.4f}")

# Display summary table 
st.dataframe(
    summary_df.style.format({
        "Flagged Rate": "{:.2%}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "F1": "{:.4f}",
        "APS": "{:.4f}",
        "ROC-AUC": "{:.4f}",
        "Estimated Cost": "${:,.2f}",
    }),
    width="stretch",
)

st.divider()
st.subheader("Precision-Recall Curves")

# Evaluate metrics across thresholds
thresholds = np.linspace(0, 1, 51)
curve_records = []
for name, proba in model_preds.items():
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        counts = confusion_counts(y_val, y_pred)
        metrics = point_metrics(counts)
        est_cost = counts["false_positive"] * false_positive_cost + counts["false_negative"] * false_negative_cost
        curve_records.append({
            "model": name,
            "threshold": float(t),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "estimated_cost": est_cost,
        })

curves_df = pd.DataFrame(curve_records)

fig_threshold, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

metric_styles = {
    "precision": {"color": "tab:blue", "linestyle": "-"},
    "recall": {"color": "tab:green", "linestyle": "--"},
    "f1": {"color": "tab:red", "linestyle": "-."},
}

for name in curves_df["model"].unique():
    df_m = curves_df[curves_df["model"] == name]
    for metric, style in metric_styles.items():
        ax_top.plot(df_m["threshold"], df_m[metric], label=f"{name} {metric.title()}", **style)

    # Mark best cost threshold per model
    best_index = df_m["estimated_cost"].idxmin()
    best_threshold = df_m.loc[best_index, "threshold"]
    ax_top.axvline(best_threshold, color="gray", linestyle=":", alpha=0.3)
    ax_bottom.axvline(best_threshold, color="gray", linestyle=":", alpha=0.3, label=f"{name} best t={best_threshold:.2f}")

ax_top.set_ylabel("Precision / Recall / F1")
ax_top.set_xlim(0, 1)
ax_top.set_ylim(0, 1)
ax_top.legend(ncol=2)
ax_top.grid(True, linestyle="--", alpha=0.4)

for name in curves_df["model"].unique():
    df_m = curves_df[curves_df["model"] == name]
    ax_bottom.plot(df_m["threshold"], df_m["estimated_cost"], label=f"{name} cost")

ax_bottom.set_xlabel("Decision Threshold")
ax_bottom.set_ylabel("Estimated Cost")
ax_bottom.grid(True, linestyle="--", alpha=0.4)
ax_bottom.legend()

st.divider()
st.subheader("Threshold Analysis")

st.pyplot(fig_threshold, width="stretch")

st.divider()
st.subheader("Explanation of Metrics")







