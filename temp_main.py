import argparse

from logistic_regression_baseline import grid_search_logistic_regression
from preprocessing import read_data, train_val_test_split, seperate_features_and_target, clean_data, distribution
from feature_engineering import engineer_features
from lightGBM_main_model import train_lightgbm
from utilities import setup_logger
from save_models import save_model, load_model
from evaluation_metrics import evaluate_model
logger = setup_logger(__name__)

def test(build_new_models: bool = False):
    data = read_data()
    data = clean_data(data)
    train_data, val_data, test_data = train_val_test_split(data)
    distribution(train_data, "train")
    distribution(val_data, "validation")
    distribution(test_data, "test")

    train_data_fe, fe_stats = engineer_features(train_data)
    val_data_fe, _ = engineer_features(val_data, fe_stats)
    test_data_fe, _ = engineer_features(test_data, fe_stats)

    X_val, y_val = seperate_features_and_target(val_data_fe)
    X_train, y_train = seperate_features_and_target(train_data_fe)
    X_test, y_test = seperate_features_and_target(test_data_fe)

    # Load or train Logistic Regression
    param_grid = {
        'C': [100, 150, 200, 250], # so far c = 100 is best score at 0.6300 
        'solver': ['liblinear']
    }

    if build_new_models:
        logger.info("Retraining logistic regression model.")
        best_params, best_score, best_model = grid_search_logistic_regression(X_train, y_train, X_val, y_val, param_grid)
        logger.info(f"Best parameters: {best_params}, Best score: {best_score:.4f}")
    else:
        try:
            best_model, loaded_metrics, best_params = load_model("logistic_regression")
            best_score = loaded_metrics.get("average_precision_score") or loaded_metrics.get("average_precision")
            logger.info("Loaded existing logistic regression model.")
        except FileNotFoundError:
            best_params, best_score, best_model = grid_search_logistic_regression(X_train, y_train, X_val, y_val, param_grid)
            logger.info(f"Best parameters: {best_params}, Best score: {best_score:.4f}")

    # Load or train LightGBM
    if build_new_models:
        logger.info("Retraining LightGBM model.")
        lightgbm_params = {
            'objective': 'binary',
            'metric': 'average_precision',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 64,
            'min_data_in_leaf': 32,
            'feature_fraction': 0.85,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': -1,
            'n_jobs': -1,
            'verbose': -1,
        }
        lgb_model, lgb_aps = train_lightgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            lightgbm_params,
            num_boost_round=2000,
            early_stopping_rounds=100,
            verbose_eval=50,
        )
        logger.info(f"LightGBM Average Precision Score: {lgb_aps:.4f}")
    else:
        try:
            lgb_model, lgb_loaded_metrics, lightgbm_params = load_model("lightgbm")
            lgb_aps = lgb_loaded_metrics.get("average_precision_score") or lgb_loaded_metrics.get("average_precision")
            logger.info("Loaded existing LightGBM model.")
        except FileNotFoundError:
            logger.info("No existing LightGBM model found. Proceeding with training.")

            lightgbm_params = {
                'objective': 'binary',
                'metric': 'average_precision',
                'boosting_type': 'gbdt',
                'learning_rate': 0.03,
                'num_leaves': 64,
                'min_data_in_leaf': 32,
                'feature_fraction': 0.85,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'n_jobs': -1,
                'verbose': -1,
            }
            lgb_model, lgb_aps = train_lightgbm(
                X_train,
                y_train,
                X_val,
                y_val,
                lightgbm_params,
                num_boost_round=2000,
                early_stopping_rounds=100,
                verbose_eval=50,
            )

            logger.info(f"LightGBM Average Precision Score: {lgb_aps:.4f}")

    lgb_y_pred_proba = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    lgr_y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    lgb_metrics = evaluate_model(y_test.values, lgb_y_pred_proba, threshold = 0.1)
    lgr_metrics = evaluate_model(y_test.values, lgr_y_pred_proba, threshold = 0.1)
    lgb_summary = lgb_metrics["summary"]
    lgr_summary = lgr_metrics["summary"]
    logger.info(f"LightGBM metrics: {lgb_summary}")
    logger.info(f"Logistic Regression metrics: {lgr_summary}")

    save_model(lgb_model, "lightgbm", lgb_metrics, lightgbm_params)
    save_model(best_model, "logistic_regression", lgr_metrics, best_params)


def main():
    retrain = False  # Set to True to force retraining of models, False to load existing models if available
    test(build_new_models=retrain)
if __name__ == "__main__":
    main()

