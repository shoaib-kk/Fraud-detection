

from pathlib import Path
import pandas as pd
from lightGBM_main_model import train_lightgbm
from preprocessing import clean_data, read_data, seperate_features_and_target, train_val_test_split
from save_models import load_model, save_model
from utilities import setup_logger

logger = setup_logger(__name__)

def load_or_train_lightgbm(X_train, y_train, X_val, y_val, retrain: bool = True):
    lightgbm_params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 64,
        "min_data_in_leaf": 32,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": -1,
        "n_jobs": -1,
        "verbose": -1,
    }

    if not retrain:
        try:
            model, metrics, params = load_model("lightgbm")
            logger.info("Loaded existing LightGBM model.")
            return model, params, metrics
        except FileNotFoundError:
            logger.info("No saved LightGBM model found; training a new one.")

    model, aps = train_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        params=lightgbm_params,
        num_boost_round=2000,
        early_stopping_rounds=100,
        verbose_eval=False,
    )
    save_model(model, "lightgbm", {"average_precision_score": aps}, lightgbm_params)
    logger.info("Trained LightGBM. APS=%.4f", aps)
    return model, lightgbm_params, {"average_precision_score": aps}


def main() -> None:

    data = read_data()
    data = clean_data(data)
    train_df, val_df, test_df = train_val_test_split(data)

    X_train, y_train = seperate_features_and_target(train_df)
    X_val, y_val = seperate_features_and_target(val_df)
    X_test, y_test = seperate_features_and_target(test_df)

    model, params, metrics = load_or_train_lightgbm(X_train, y_train, X_val, y_val)
    logger.info("Using LightGBM params: %s", params)

    best_iter = model.best_iteration or None
    y_pred_proba = model.predict(X_test, num_iteration=best_iter)
    out_path = Path("val_predictions.csv")

    pd.DataFrame({"y_true": y_test, "y_pred_proba": y_pred_proba}).to_csv(out_path, index=False)
    logger.info("Wrote test-split predictions to %s", out_path.resolve())


if __name__ == "__main__":
    main()