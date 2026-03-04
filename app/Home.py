import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

# Ensure imports work when Streamlit runs from app/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_engineering import engineer_features
from evaluation_metrics import confusion_counts, point_metrics

ARTIFACTS_DIR = Path("artifacts")
DEFAULT_THRESHOLD = 0.1


@st.cache_resource(show_spinner=True)
def load_artifacts():
    models = {
        "LightGBM": joblib.load(ARTIFACTS_DIR / "model_lgbm.pkl"),
        "Logistic Regression": joblib.load(ARTIFACTS_DIR / "model_logreg.pkl"),
    }

    with open(ARTIFACTS_DIR / "feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    with open(ARTIFACTS_DIR / "feature_stats.json", "r") as f:
        feature_stats = json.load(f)
    metrics = None
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    threshold = DEFAULT_THRESHOLD
    threshold_path = ARTIFACTS_DIR / "threshold.json"
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            threshold = json.load(f).get("threshold", DEFAULT_THRESHOLD)

    return models, feature_columns, feature_stats, metrics, threshold


@st.cache_data(show_spinner=False)
def load_demo_data():
    sample_path = Path("data/sample.parquet")
    if sample_path.exists():
        return pd.read_parquet(sample_path)

    # Synthetic fallback with a few positive cases to avoid empty-class warnings
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    rows = []
    for i in range(200):
        row = {
            "Time": float(i * 10),
            "Amount": float(50 + (i % 10)),
            "Class": 1 if i % 50 == 0 else 0,
        }
        for v in range(1, 29):
            row[f"V{v}"] = float(np.sin(i + v) * 0.1)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols)


def prepare_features(df: pd.DataFrame, feature_stats: dict, feature_columns: list[str]):
    df_fe, _ = engineer_features(df, feature_stats)
    X = df_fe.reindex(columns=feature_columns, fill_value=0.0)
    y = df_fe["Class"].values if "Class" in df_fe.columns else None
    return X, y


def summarize_model(name, proba, y_true, threshold, false_positive_cost, false_negative_cost):
    if y_true is None:
        return None
    y_pred = (proba >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    metrics = point_metrics(counts)
    flagged_rate = float((y_pred == 1).mean())
    est_cost = counts["false_positive"] * false_positive_cost + counts["false_negative"] * false_negative_cost
    aps = average_precision_score(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)
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


models, feature_columns, feature_stats, precomputed_metrics, default_threshold = load_artifacts()

demo_df = load_demo_data()
X_demo, y_demo = prepare_features(demo_df, feature_stats, feature_columns)

st.title("Fraud Model Comparison")
st.caption("Artifact-driven app: loads pre-trained models and metrics; demo data only.")

with st.sidebar:
    st.header("Controls")
    threshold = st.slider(
        "Decision threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(default_threshold),
        step=0.01,
    )
    false_positive_cost = st.number_input("False positive cost", min_value=0.0, value=1.0, step=0.1)
    false_negative_cost = st.number_input("False negative cost", min_value=0.0, value=10.0, step=0.1)
    st.caption("Models and metrics are loaded from artifacts/. No training runs in the app.")

with st.spinner("Running models on demo data"):
    preds_demo = {
        name: model.predict_proba(X_demo)[:, 1] for name, model in models.items()
    }

if y_demo is not None:
    summary_rows = [
        summarize_model(name, proba, y_demo, threshold, false_positive_cost, false_negative_cost)
        for name, proba in preds_demo.items()
    ]
    summary_rows = [row for row in summary_rows if row is not None]
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).set_index("Model")
        st.markdown("**Demo data metrics**")
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
else:
    st.info("Demo dataset has no labels; showing precomputed metrics only.")

if precomputed_metrics:
    st.divider()
    st.subheader("Precomputed metrics (from training run)")
    st.json(precomputed_metrics, expanded=False)

st.divider()
st.subheader("Score distribution (demo data)")
fig, ax = plt.subplots(figsize=(6, 3))
for name, proba in preds_demo.items():
    ax.hist(proba, bins=40, alpha=0.5, label=name)
ax.axvline(threshold, color="k", linestyle=":", alpha=0.7)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig, width="stretch")

st.caption("Artifacts loaded from artifacts/. Demo data from data/sample.parquet (with synthetic fallback). No training occurs in this app run.")
