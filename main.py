import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score

from evaluation_metrics import evaluate_model, confusion_counts, point_metrics
from pipeline import (
    split_for_training,
    build_feature_sets,
    train_lightgbm_classifier,
    train_log_reg_with_grid,
    calibrate_prefit,
)
from preprocessing import distribution
from save_models import save_model, load_model
from utilities import setup_logger

logger = setup_logger(__name__)
DEFAULT_THRESHOLD = 0.1


def _save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _app_style_summary(y_true, proba, threshold: float):
    y_pred = (proba >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    metrics = point_metrics(counts)
    aps = average_precision_score(y_true, proba)
    roc_auc = average_precision_score(y_true, proba) if len(np.unique(y_true)) == 1 else average_precision_score(y_true, proba)
    # Use roc_auc_score only if both classes present to avoid errors
    try:
        from sklearn.metrics import roc_auc_score

        roc_auc = roc_auc_score(y_true, proba)
    except Exception:
        roc_auc = float("nan")
    return {
        "Model": "",
        "Flagged Rate": float((y_pred == 1).mean()),
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "APS": float(aps),
        "ROC-AUC": float(roc_auc),
        "threshold": threshold,
        "True positive": int(counts["true_positive"]),
        "False positive": int(counts["false_positive"]),
        "False negative": int(counts["false_negative"]),
        "True negative": int(counts["true_negative"]),
    }


def _print_table(title: str, rows: list):
    if not rows:
        return
    df = pd.DataFrame(rows).set_index("Model")
    print("\n" + title)
    print(df[["Flagged Rate", "Precision", "Recall", "F1", "APS", "ROC-AUC"]].to_string(float_format=lambda x: f"{x:.4f}"))


def run(build_new_models: bool = False, time_based: bool = True):
    train_df, early_df, calib_df, test_df = split_for_training(time_based=time_based)
    distribution(train_df, "train")
    distribution(early_df, "early_stop")
    distribution(calib_df, "calibration")
    distribution(test_df, "test")

    features = build_feature_sets(train_df, early_df, calib_df, test_df)
    X_train = features["X_train"]
    y_train = features["y_train"]
    X_early = features["X_early"]
    y_early = features["y_early"]
    X_calib = features["X_calib"]
    y_calib = features["y_calib"]
    X_test = features["X_test"]
    y_test = features["y_test"]

    # Logistic Regression
    if build_new_models:
        logger.info("Retraining logistic regression model.")
        log_model, log_params, log_score = train_log_reg_with_grid(X_train, y_train, X_early, y_early)
        save_model(log_model, "logistic_regression", {"average_precision_score": log_score}, log_params)
    else:
        try:
            log_model, loaded_metrics, log_params = load_model("logistic_regression")
            log_score = loaded_metrics.get("average_precision_score") or loaded_metrics.get("average_precision")
            logger.info("Loaded existing logistic regression model.")
        except FileNotFoundError:
            log_model, log_params, log_score = train_log_reg_with_grid(X_train, y_train, X_early, y_early)
            save_model(log_model, "logistic_regression", {"average_precision_score": log_score}, log_params)

    # LightGBM
    if build_new_models:
        lgb_model, lgb_params = train_lightgbm_classifier(X_train, y_train, X_early, y_early)
        save_model(lgb_model, "lightgbm", {"average_precision_score": None}, lgb_params)
    else:
        try:
            lgb_model, lgb_metrics, lgb_params = load_model("lightgbm")
            if not hasattr(lgb_model, "predict_proba"):
                logger.info("Existing LightGBM model lacks predict_proba; retraining.")
                lgb_model, lgb_params = train_lightgbm_classifier(X_train, y_train, X_early, y_early)
                save_model(lgb_model, "lightgbm", {"average_precision_score": None}, lgb_params)
        except FileNotFoundError:
            lgb_model, lgb_params = train_lightgbm_classifier(X_train, y_train, X_early, y_early)
            save_model(lgb_model, "lightgbm", {"average_precision_score": None}, lgb_params)
            lgb_metrics = {"average_precision_score": None}

    # Predictions
    lgb_val = lgb_model.predict_proba(X_calib, raw_score=False)[:, 1]
    lgr_val = log_model.predict_proba(X_calib)[:, 1]
    lgb_test = lgb_model.predict_proba(X_test, raw_score=False)[:, 1]
    lgr_test = log_model.predict_proba(X_test)[:, 1]

    # App-style summaries (matches Streamlit table columns)
    val_rows = []
    test_rows = []
    val_rows.append({"Model": "LightGBM", **_app_style_summary(y_calib.values, lgb_val, DEFAULT_THRESHOLD)})
    val_rows.append({"Model": "Logistic Regression", **_app_style_summary(y_calib.values, lgr_val, DEFAULT_THRESHOLD)})
    test_rows.append({"Model": "LightGBM", **_app_style_summary(y_test.values, lgb_test, DEFAULT_THRESHOLD)})
    test_rows.append({"Model": "Logistic Regression", **_app_style_summary(y_test.values, lgr_test, DEFAULT_THRESHOLD)})

    _print_table("Validation metrics (calibration split)", val_rows)
    _print_table("Test metrics (held-out)", test_rows)

    # Detailed metrics from evaluate_model for reference
    lgb_metrics = evaluate_model(y_test.values, lgb_test, threshold=DEFAULT_THRESHOLD)
    lgr_metrics = evaluate_model(y_test.values, lgr_test, threshold=DEFAULT_THRESHOLD)
    logger.info("LightGBM metrics summary:\n%s", lgb_metrics["summary"])
    logger.info("Logistic Regression metrics summary:\n%s", lgr_metrics["summary"])

    run_interpretability = build_new_models

    if run_interpretability:
        interp_dir = Path("logs/interpretability")
        interp_dir.mkdir(parents=True, exist_ok=True)

        sample_n = min(len(X_train), 1000)
        X_shap = X_train.sample(sample_n, random_state=42)
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_shap)
        shap_pos = shap_values[1] if isinstance(shap_values, list) else shap_values

        shap.summary_plot(shap_pos, X_shap, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(interp_dir / "shap_summary.png", dpi=150)
        plt.close()

        single_row = X_shap.iloc[[0]]
        single_values = explainer.shap_values(single_row)
        single_pos = single_values[1][0] if isinstance(single_values, list) else single_values[0]
        single_pos = np.ravel(single_pos)
        top_idx = np.argsort(np.abs(single_pos))[::-1][:10]
        plt.figure(figsize=(6, 4))
        plt.barh(single_row.columns[top_idx], single_pos[top_idx])
        plt.gca().invert_yaxis()
        plt.title("Single transaction SHAP (top 10)")
        plt.tight_layout()
        plt.savefig(interp_dir / "shap_single.png", dpi=150)
        plt.close()

        uncalibrated_aps = average_precision_score(y_test, lgb_test)
        calibrator_iso = calibrate_prefit(lgb_model, X_calib, y_calib, method="isotonic")
        calibrated_iso = calibrator_iso.predict_proba(X_test)[:, 1]
        calibrated_iso_aps = average_precision_score(y_test, calibrated_iso)

        calibrator_sig = calibrate_prefit(lgb_model, X_calib, y_calib, method="sigmoid")
        calibrated_sig = calibrator_sig.predict_proba(X_test)[:, 1]
        calibrated_sig_aps = average_precision_score(y_test, calibrated_sig)

        calib_metrics = {
            "aps_uncalibrated": float(uncalibrated_aps),
            "aps_isotonic": float(calibrated_iso_aps),
            "aps_sigmoid": float(calibrated_sig_aps),
        }
        _save_json(interp_dir / "calibration_metrics.json", calib_metrics)

        logger.info(
            "LightGBM APS — uncalibrated: %.4f | isotonic: %.4f | sigmoid: %.4f",
            uncalibrated_aps,
            calibrated_iso_aps,
            calibrated_sig_aps,
        )

        prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, lgb_test, n_bins=15)
        prob_true_iso, prob_pred_iso = calibration_curve(y_test, calibrated_iso, n_bins=15)
        prob_true_sig, prob_pred_sig = calibration_curve(y_test, calibrated_sig, n_bins=15)
        plt.figure(figsize=(6, 4))
        plt.plot(prob_pred_uncal, prob_true_uncal, label="Uncalibrated", marker="o")
        plt.plot(prob_pred_iso, prob_true_iso, label="Isotonic", marker="o")
        plt.plot(prob_pred_sig, prob_true_sig, label="Sigmoid", marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("LightGBM Reliability Diagram")
        plt.legend()
        plt.tight_layout()
        plt.savefig(interp_dir / "reliability_lightgbm.png", dpi=150)
        plt.close()

    # Save logistic regression metrics on test to disk for quick inspection
    save_model(log_model, "logistic_regression", lgr_metrics, log_params)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    axes[0].hist(lgb_test, bins=50)
    axes[0].set_title("LightGBM")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")

    axes[1].hist(lgr_test, bins=50)
    axes[1].set_title("Logistic Regression")
    axes[1].set_xlabel("Predicted probability")

    fig.suptitle("Prediction Probability Distribution", fontsize=12)
    fig.tight_layout()
    plt.show()


def main():
    run(build_new_models=True, time_based=True)


if __name__ == "__main__":
    main()
