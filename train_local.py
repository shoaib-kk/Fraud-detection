import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from pipeline import build_feature_sets, load_or_train_models
from preprocessing import clean_data, train_val_test_split
from evaluation_metrics import confusion_counts, point_metrics

ARTIFACTS_DIR = Path("artifacts")
DEFAULT_THRESHOLD = 0.1


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def compute_summary(y_true: np.ndarray, proba: np.ndarray, threshold: float):
    y_pred = (proba >= threshold).astype(int)
    counts = confusion_counts(y_true, y_pred)
    metrics = point_metrics(counts)
    aps = average_precision_score(y_true, proba)
    try:
        roc_auc = roc_auc_score(y_true, proba)
    except Exception:
        roc_auc = float("nan")
    return {
        "threshold": threshold,
        "flagged_rate": float((y_pred == 1).mean()),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "aps": float(aps),
        "roc_auc": float(roc_auc),
        "true_positive": int(counts["true_positive"]),
        "false_positive": int(counts["false_positive"]),
        "false_negative": int(counts["false_negative"]),
        "true_negative": int(counts["true_negative"]),
    }


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    data_path = Path("data/creditcard.csv")
    if not data_path.exists():
        raise FileNotFoundError("data/creditcard.csv not found. Place the full dataset locally before running train_local.py.")

    data = clean_data(pd.read_csv(data_path))

    train_df, val_df, test_df = train_val_test_split(
        data, test_size=0.2, val_size=0.1, random_state=42, time_based=True
    )

    calib_frac_of_val = 0.5
    n_calib = max(1, int(len(val_df) * calib_frac_of_val))
    calib_df = val_df.tail(n_calib)
    early_df = val_df.head(len(val_df) - n_calib)

    features = build_feature_sets(train_df, early_df, calib_df, test_df)
    models = load_or_train_models(features, retrain=True)

    feature_columns = list(features["X_train"].columns)
    feature_stats = features["fe_stats"]

    lgb_val = models["lgb_model"].predict_proba(features["X_calib"])[:, 1]
    lgr_val = models["log_model"].predict_proba(features["X_calib"])[:, 1]
    lgb_test = models["lgb_model"].predict_proba(features["X_test"])[:, 1]
    lgr_test = models["log_model"].predict_proba(features["X_test"])[:, 1]

    metrics = {
        "threshold": DEFAULT_THRESHOLD,
        "validation": {
            "LightGBM": compute_summary(features["y_calib"].values, lgb_val, DEFAULT_THRESHOLD),
            "Logistic Regression": compute_summary(features["y_calib"].values, lgr_val, DEFAULT_THRESHOLD),
        },
        "test": {
            "LightGBM": compute_summary(features["y_test"].values, lgb_test, DEFAULT_THRESHOLD),
            "Logistic Regression": compute_summary(features["y_test"].values, lgr_test, DEFAULT_THRESHOLD),
        },
    }

    def _print_table(title, data):
        df = pd.DataFrame(data).set_index("Model")
        print(f"\n{title}")
        print(
            df[["flagged_rate", "precision", "recall", "f1", "aps", "roc_auc"]]
            .rename(
                columns={
                    "flagged_rate": "Flagged Rate",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1",
                    "aps": "APS",
                    "roc_auc": "ROC-AUC",
                }
            )
            .to_string(float_format=lambda x: f"{x:.4f}")
        )

    val_rows = [
        {"Model": "LightGBM", **metrics["validation"]["LightGBM"]},
        {"Model": "Logistic Regression", **metrics["validation"]["Logistic Regression"]},
    ]
    test_rows = [
        {"Model": "LightGBM", **metrics["test"]["LightGBM"]},
        {"Model": "Logistic Regression", **metrics["test"]["Logistic Regression"]},
    ]

    _print_table("Validation metrics (calibration split)", val_rows)
    _print_table("Test metrics (held-out)", test_rows)

    joblib.dump(models["lgb_model"], ARTIFACTS_DIR / "model_lgbm.pkl")
    joblib.dump(models["log_model"], ARTIFACTS_DIR / "model_logreg.pkl")

    save_json(ARTIFACTS_DIR / "feature_columns.json", feature_columns)
    save_json(ARTIFACTS_DIR / "feature_stats.json", feature_stats)
    save_json(ARTIFACTS_DIR / "metrics.json", metrics)
    save_json(ARTIFACTS_DIR / "threshold.json", {"threshold": DEFAULT_THRESHOLD})

    print("Artifacts written to", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
