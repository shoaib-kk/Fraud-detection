import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix

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


def compute_pr_curves(y_true: np.ndarray, preds: dict[str, np.ndarray]):
    curves = {}
    for name, proba in preds.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        curves[name] = (precision, recall)
    return curves


def compute_alert_vs_recall(y_true: np.ndarray, preds: dict[str, np.ndarray], thresholds: np.ndarray):
    records = {name: [] for name in preds}
    total = len(y_true)
    for t in thresholds:
        for name, proba in preds.items():
            y_pred = (proba >= t).astype(int)
            recall = (y_pred[y_true == 1].mean() if (y_true == 1).any() else 0.0)
            flagged = y_pred.sum()
            records[name].append((float(t), float(recall), int(flagged), flagged / total))
    return records


@st.cache_data(show_spinner=False)
def shap_bar_figure(model, X):
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)
    shap_pos = vals[1] if isinstance(vals, list) else vals
    shap.summary_plot(shap_pos, X, show=False, plot_type="bar")
    fig = plt.gcf()
    plt.close(fig)
    return fig


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
    focus_model = st.radio("Focus model", list(models.keys()), index=0)
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
        st.subheader("At-a-glance")
        c1, c2, c3, c4 = st.columns(4)
        row = summary_df.loc[focus_model]
        c1.metric("Model", focus_model)
        c2.metric("Precision", f"{row['Precision']:.3f}")
        c3.metric("Recall", f"{row['Recall']:.3f}")
        c4.metric("Alerts flagged", f"{row['Flagged Rate']:.2%}")
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

if y_demo is not None and len(np.unique(y_demo)) > 1:
    st.divider()
    st.subheader("Precision–Recall Curve (demo data)")
    pr_curves = compute_pr_curves(y_demo, preds_demo)
    fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
    for name, (prec, rec) in pr_curves.items():
        ax_pr.plot(rec, prec, label=name)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.grid(True, linestyle="--", alpha=0.4)
    ax_pr.legend()
    st.pyplot(fig_pr, width="stretch")

    st.subheader("Alert Volume vs Recall")
    thresholds = np.linspace(0, 1, 51)
    alert_curves = compute_alert_vs_recall(y_demo, preds_demo, thresholds)
    fig_alert, ax_alert = plt.subplots(figsize=(6, 4))
    for name, points in alert_curves.items():
        ts, recalls, flagged, flagged_rate = zip(*points)
        ax_alert.plot(flagged, recalls, label=name)
    ax_alert.set_xlabel("Alerts (count)")
    ax_alert.set_ylabel("Recall")
    ax_alert.grid(True, linestyle="--", alpha=0.4)
    ax_alert.legend()
    st.pyplot(fig_alert, width="stretch")
else:
    st.info("PR and alert-volume curves need labels with both classes in the demo data.")

st.divider()
st.subheader("Score distribution (demo data)")
fig, ax = plt.subplots(figsize=(6, 3))
for name, proba in preds_demo.items():
    alpha = 0.75 if name == focus_model else 0.35
    ax.hist(proba, bins=40, alpha=alpha, label=name, edgecolor="black" if name == focus_model else None)
ax.axvline(threshold, color="k", linestyle=":", alpha=0.7, label="Threshold")
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig, width="stretch")

if y_demo is not None:
    st.divider()
    st.subheader("Confusion Matrix (demo data)")
    proba = preds_demo.get(focus_model)
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_demo, y_pred, labels=[0, 1])
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
    ax_cm.set_title(focus_model)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm, width="stretch")
    plt.close(fig_cm)

st.divider()
st.subheader("SHAP Feature Importance (LightGBM)")
if focus_model == "LightGBM":
    try:
        sample_shap = X_demo.copy()
        if len(sample_shap) > 500:
            sample_shap = sample_shap.sample(500, random_state=42)
        lgb_model = models.get("LightGBM")
        fig_shap = shap_bar_figure(lgb_model, sample_shap)
        st.pyplot(fig_shap, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render SHAP summary: {e}")
else:
    st.info("SHAP available when LightGBM is the focus model.")

st.caption("Artifacts loaded from artifacts/. Demo data from data/sample.parquet (with synthetic fallback). No training occurs in this app run.")
