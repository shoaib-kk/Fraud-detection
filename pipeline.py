from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict

import lightgbm as lgb
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from feature_engineering import engineer_features
from logistic_regression_baseline import grid_search_logistic_regression
from preprocessing import (
    clean_data,
    read_data,
    seperate_features_and_target,
    train_val_test_split,
)
from save_models import load_model, save_model
from utilities import setup_logger

logger = setup_logger(__name__)


def split_for_training(
    time_based: bool = True,
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.1,
    calib_frac_of_val: float = 0.5,
):
    """Return train, early-stop, calibration, and test splits (raw data).
    If time_based=True, uses temporal ordering to avoid leakage.
    the validation portion is split into early-stop and calibration subsets.
    """
    data = clean_data(read_data())
    train_df, val_df, test_df = train_val_test_split(
        data,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        time_based=time_based,
    )

    if time_based:
        n_calib = max(1, int(len(val_df) * calib_frac_of_val))
        calib_df = val_df.tail(n_calib)
        early_df = val_df.head(len(val_df) - n_calib)
    else:
        early_df, calib_df = train_test_split(
            val_df,
            test_size=calib_frac_of_val,
            random_state=random_state,
            stratify=val_df["Class"],
        )

    return train_df, early_df, calib_df, test_df


def build_feature_sets(
    train_df,
    early_df,
    calib_df,
    test_df,
):
    """Apply feature engineering with train statistics to all splits."""
    train_fe, fe_stats = engineer_features(train_df)
    early_fe, _ = engineer_features(early_df, fe_stats)
    calib_fe, _ = engineer_features(calib_df, fe_stats)
    test_fe, _ = engineer_features(test_df, fe_stats)

    X_train, y_train = seperate_features_and_target(train_fe)
    X_early, y_early = seperate_features_and_target(early_fe)
    X_calib, y_calib = seperate_features_and_target(calib_fe)
    X_test, y_test = seperate_features_and_target(test_fe)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_early": X_early,
        "y_early": y_early,
        "X_calib": X_calib,
        "y_calib": y_calib,
        "X_test": X_test,
        "y_test": y_test,
        "fe_stats": fe_stats,
    }


def train_lightgbm_classifier(X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_data_in_leaf": 16,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "max_depth": -1,
        "n_estimators": 2000,
        "n_jobs": -1,
        "verbosity": -1,
        "boosting_type": "gbdt",
    }
    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )
    return clf, params


class _HoldoutCalibratedModel:
    """Wraps an already-fitted classifier with a light-weight holdout calibrator."""

    def __init__(self, base_model, calibrator, method: str):
        self.base_model = base_model
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X):
        base_probs = self.base_model.predict_proba(X)[:, 1]
        if self.method == "isotonic":
            calibrated = self.calibrator.predict(base_probs)
        else:  # sigmoid
            calibrated = self.calibrator.predict_proba(base_probs.reshape(-1, 1))[:, 1]
        calibrated = np.clip(calibrated, 0.0, 1.0)
        return np.vstack([1 - calibrated, calibrated]).T


def calibrate_prefit(model, X_calib, y_calib, method: str):
    """Calibrate a fitted model using a held-out calibration split.

    Newer sklearn versions removed cv="prefit". To avoid refitting the base model
    (slow for large datasets), we fit a one-dimensional calibrator on the base
    model's probabilities from the calibration split.
    """
    base_probs = model.predict_proba(X_calib)[:, 1]
    if method == "isotonic":
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(base_probs, y_calib)
    elif method == "sigmoid":
        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(base_probs.reshape(-1, 1), y_calib)
    else:
        raise ValueError("method must be 'isotonic' or 'sigmoid'")

    return _HoldoutCalibratedModel(model, calibrator, method)


def train_log_reg_with_grid(X_train, y_train, X_val, y_val):
    param_grid = {"C": [100, 150, 200, 250], "solver": ["liblinear"]}
    best_params, best_score, best_model = grid_search_logistic_regression(
        X_train, y_train, X_val, y_val, param_grid
    )
    return best_model, best_params, best_score


def load_or_train_models(features: Dict, retrain: bool = False):
    """Load persisted models if available, otherwise train and persist."""
    X_train = features["X_train"]
    y_train = features["y_train"]
    X_early = features["X_early"]
    y_early = features["y_early"]

    # Logistic Regression
    if retrain:
        log_model, log_params, log_score = train_log_reg_with_grid(X_train, y_train, X_early, y_early)
        save_model(log_model, "logistic_regression", {"average_precision_score": log_score}, log_params)
    else:
        try:
            log_model, log_metrics, log_params = load_model("logistic_regression")
            log_score = log_metrics.get("average_precision_score") or log_metrics.get("average_precision")
        except FileNotFoundError:
            log_model, log_params, log_score = train_log_reg_with_grid(X_train, y_train, X_early, y_early)
            save_model(log_model, "logistic_regression", {"average_precision_score": log_score}, log_params)

    # LightGBM
    if retrain:
        lgb_model, lgb_params = train_lightgbm_classifier(X_train, y_train, X_early, y_early)
        save_model(lgb_model, "lightgbm", {"average_precision_score": None}, lgb_params)
    else:
        try:
            lgb_model, lgb_metrics, lgb_params = load_model("lightgbm")
        except FileNotFoundError:
            lgb_model, lgb_params = train_lightgbm_classifier(X_train, y_train, X_early, y_early)
            save_model(lgb_model, "lightgbm", {"average_precision_score": None}, lgb_params)
            lgb_metrics = {"average_precision_score": None}

    return {
        "log_model": log_model,
        "log_params": log_params,
        "log_score": log_score,
        "lgb_model": lgb_model,
        "lgb_params": lgb_params,
    }
