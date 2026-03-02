import matplotlib.pyplot as plt
import numpy as np
import shap
from pathlib import Path
import json
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score
from utilities import setup_logger
from save_models import save_model, load_model
from evaluation_metrics import evaluate_model
from pipeline import (
    split_for_training,
    build_feature_sets,
    train_lightgbm_classifier,
    train_log_reg_with_grid,
    calibrate_prefit,
)
from preprocessing import distribution
logger = setup_logger(__name__)


def _save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def test(build_new_models: bool = False, time_based: bool = True):
    # Split into train / early-stop / calibration / test
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

    # Train or load logistic regression
    if build_new_models:
        logger.info("Retraining logistic regression model.")
        best_model, best_params, best_score = train_log_reg_with_grid(X_train, y_train, X_early, y_early)
        logger.info(f"Best parameters: {best_params}, Best score: {best_score:.4f}")
        save_model(best_model, "logistic_regression", {"average_precision_score": best_score}, best_params)
    else:
        try:
            best_model, loaded_metrics, best_params = load_model("logistic_regression")
            best_score = loaded_metrics.get("average_precision_score") or loaded_metrics.get("average_precision")
            logger.info("Loaded existing logistic regression model.")
        except FileNotFoundError:
            best_model, best_params, best_score = train_log_reg_with_grid(X_train, y_train, X_early, y_early)
            save_model(best_model, "logistic_regression", {"average_precision_score": best_score}, best_params)

    # Train or load LightGBM (sklearn API)
    if build_new_models:
        lgb_model, lightgbm_params = train_lightgbm_classifier(X_train, y_train, X_early, y_early)
        save_model(lgb_model, "lightgbm", {"average_precision_score": None}, lightgbm_params)
    else:
        try:
            lgb_model, lightgbm_metrics, lightgbm_params = load_model("lightgbm")
            if not hasattr(lgb_model, "predict_proba"):
                logger.info("Existing LightGBM model is a Booster without predict_proba; retraining with sklearn API.")
                lgb_model, lightgbm_params = train_lightgbm_classifier(X_train, y_train, X_early, y_early)
                save_model(lgb_model, "lightgbm", {"average_precision_score": None}, lightgbm_params)
        except FileNotFoundError:
            lgb_model, lightgbm_params = train_lightgbm_classifier(X_train, y_train, X_early, y_early)
            save_model(lgb_model, "lightgbm", {"average_precision_score": None}, lightgbm_params)
            lightgbm_metrics = {"average_precision_score": None}

    lgb_y_pred_proba = lgb_model.predict_proba(X_test, raw_score=False)[:, 1]
    lgr_y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    lgb_metrics = evaluate_model(y_test.values, lgb_y_pred_proba, threshold = 0.1)
    lgr_metrics = evaluate_model(y_test.values, lgr_y_pred_proba, threshold = 0.1)
    lgb_summary = lgb_metrics["summary"]
    lgr_summary = lgr_metrics["summary"]
    logger.info(f"LightGBM metrics: {lgb_summary}")
    logger.info(f"Logistic Regression metrics: {lgr_summary}")

    run_interpretability = build_new_models

    if run_interpretability:
        interp_dir = Path("logs/interpretability")
        interp_dir.mkdir(parents=True, exist_ok=True)

        # SHAP: global importance and single-transaction explanation
        sample_n = min(len(X_train), 1000)
        X_shap = X_train.sample(sample_n, random_state=42)
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_shap)
        shap_pos = shap_values[1] if isinstance(shap_values, list) else shap_values

        shap.summary_plot(shap_pos, X_shap, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(interp_dir / "shap_summary.png", dpi=150)
        plt.show()  

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
        plt.show()  
        # Calibration: use held-out calibration split and prefit estimator
        uncalibrated_aps = average_precision_score(y_test, lgb_y_pred_proba)

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

        prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, lgb_y_pred_proba, n_bins=15)
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

    # LightGBM is saved inside load_or_train_lightgbm when trained; save logistic regression here
    save_model(best_model, "logistic_regression", lgr_metrics, best_params)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

    axes[0].hist(lgb_y_pred_proba, bins=50)
    axes[0].set_title("LightGBM")
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")

    axes[1].hist(lgr_y_pred_proba, bins=50)
    axes[1].set_title("Logistic Regression")
    axes[1].set_xlabel("Predicted probability")

    fig.suptitle("Prediction Probability Distribution", fontsize=12)
    fig.tight_layout()
    plt.show()


def main():
    retrain = True  # Set to True to force retraining of models, False to load existing models if available
    test(build_new_models=retrain)


if __name__ == "__main__":
    main()

