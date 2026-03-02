import sys
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from pipeline import split_for_training, build_feature_sets, load_or_train_models
from evaluation_metrics import confusion_counts, point_metrics
from utilities import load_json


@st.cache_data(show_spinner=False)
def load_feature_sets():
    splits = split_for_training(time_based=True)
    return build_feature_sets(*splits)


@st.cache_resource(show_spinner=True)
def load_models(retrain: bool):
    features = load_feature_sets()
    return load_or_train_models(features, retrain=retrain)


st.title("Fraud Model Comparison")
st.caption("Comparison of LightGBM vs Logistic Regression — validation metrics shown by default; toggle to view held-out test metrics.")

with st.sidebar:
    st.header("Controls")
    threshold = st.slider("Decision threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01,
        help="Fraud often uses a low threshold to maximize recall; adjust based on cost and flagged rate.")
    st.caption("At threshold {:.2f}, you'll see flagged rate/recall/cost below.".format(threshold))
    st.divider()
    false_positive_cost = st.number_input("False positive cost", min_value=0.0, value=1.0, step=0.1)
    false_negative_cost = st.number_input("False negative cost", min_value=0.0, value=10.0, step=0.1)
    st.divider()
    retrain_clicked = st.button(
        "Run training script (temp_main.py)",
        help="Runs the full pipeline including SHAP and calibration. May take a few minutes.",
    )


with st.spinner("Loading data, features, and models"):
    features = load_feature_sets()
    # If the sidebar retrain button was clicked, reload models with retrain=True
    models = load_models(retrain=retrain_clicked)
    y_val = features["y_calib"].values
    y_test = features["y_test"].values
    model_preds_val = {
        "LightGBM": models["lgb_model"].predict_proba(features["X_calib"])[:, 1],
        "Logistic Regression": models["log_model"].predict_proba(features["X_calib"])[:, 1],
    }
    model_preds_test = {
        "LightGBM": models["lgb_model"].predict_proba(features["X_test"])[:, 1],
        "Logistic Regression": models["log_model"].predict_proba(features["X_test"])[:, 1],
    }


def summarize_model(name, proba, y_true):
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


summary_rows_val = [summarize_model(name, proba, y_val) for name, proba in model_preds_val.items()]
summary_rows_test = [summarize_model(name, proba, y_test) for name, proba in model_preds_test.items()]
summary_df_val = pd.DataFrame(summary_rows_val).set_index("Model")
summary_df_test = pd.DataFrame(summary_rows_test).set_index("Model")



st.markdown("**Validation metrics (calibration split, not test)**")

# Highlight models with the best APS and ROC-AUC on validation
best_aps_model = summary_df_val["APS"].idxmax()
best_aps_value = summary_df_val.loc[best_aps_model, "APS"]
best_roc_model = summary_df_val["ROC-AUC"].idxmax()
best_roc_value = summary_df_val.loc[best_roc_model, "ROC-AUC"]

st.metric("Best APS (val)", f"{best_aps_model}: {best_aps_value:.4f}")
st.metric("Best ROC-AUC (val)", f"{best_roc_model}: {best_roc_value:.4f}")

st.dataframe(
    summary_df_val.style.format({
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

st.info(
    "Fraud framing: default threshold 0.10 leans toward high recall. Adjust costs/threshold to trade off flagged rate vs missed fraud; the table above shows the operational impact (flagged %, caught fraud, estimated cost)."
)

show_test = st.checkbox("Show test-set metrics", value=False, help="Test set is held-out and never used for training or calibration.")
if show_test:
    st.markdown("**Test metrics (held-out)**")
    st.dataframe(
        summary_df_test.style.format({
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
st.subheader("Threshold Analysis (validation/calibration set)")

# Evaluate metrics across thresholds on calibration/validation split
threshold_grid = np.linspace(0, 1, 51)
curve_records = []
for name, proba in model_preds_val.items():
    for t in threshold_grid:
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

st.pyplot(fig_threshold, width="stretch")

st.divider()
st.subheader("Explanation of Metrics")

st.divider()
st.subheader("Interpretability & Calibration (artifacts)")

interp_dir = Path("logs/interpretability")
artifacts = {
    "SHAP summary": interp_dir / "shap_summary.png",
    "Single transaction SHAP": interp_dir / "shap_single.png",
    "Reliability (calibration)": interp_dir / "reliability_lightgbm.png",
}

cols = st.columns(len(artifacts))
for col, (label, path) in zip(cols, artifacts.items()):
    if path.exists():
        col.image(str(path), caption=label, use_container_width=True)
    else:
        col.warning(f"Missing artifact: {path}")

calib_metrics_path = interp_dir / "calibration_metrics.json"
calib_metrics = load_json(calib_metrics_path)
if calib_metrics:
    st.markdown("**Calibration APS (LightGBM)**")
    st.json(calib_metrics, expanded=False)
else:
    st.info(f"Calibration metrics not found at {calib_metrics_path}.")

st.divider()
st.subheader("Model Status")

model_specs = {
    "LightGBM": {
        "model": Path("models/lightgbm.pkl"),
        "metrics": Path("models/lightgbm_metrics.json"),
    },
    "Logistic Regression": {
        "model": Path("models/logistic_regression.pkl"),
        "metrics": Path("models/logistic_regression_metrics.json"),
    },
}

status_rows = []
for name, spec in model_specs.items():
    model_exists = spec["model"].exists()
    metrics = load_json(spec["metrics"])
    trained_at = metrics.get("trained_at") if metrics else None
    status_rows.append({
        "Model": name,
        "Model file": "yes" if model_exists else "no",
        "Metrics file": "yes" if metrics else "no",
        "Trained at": trained_at or "n/a",
    })

status_df = pd.DataFrame(status_rows).set_index("Model")
st.dataframe(status_df)

st.caption("Retraining is optional. Use the sidebar button to run `temp_main.py` and refresh artifacts/status.")

if 'retrain_clicked' in locals() and retrain_clicked:
    st.divider()
    st.subheader("Retrain models (sidebar trigger)")
    with st.spinner("Retraining models... this may take a few minutes"):
        try:
            result = subprocess.run([sys.executable, "temp_main.py"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
            st.success("Training completed. Reload the page to see updated artifacts and status.")
            if result.stdout:
                st.expander("Training output").text(result.stdout[-4000:])
            if result.stderr:
                st.expander("Training errors").text(result.stderr[-4000:])
        except subprocess.CalledProcessError as exception:
            st.error("Training failed. See logs below.")
            st.expander("Training errors").text(exception.stderr or "(no stderr)")







