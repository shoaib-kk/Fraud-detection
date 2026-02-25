import matplotlib.pyplot as plt
from logistic_regression_baseline import grid_search_logistic_regression
from preprocessing import read_data, train_val_test_split, seperate_features_and_target, clean_data, distribution
from feature_engineering import engineer_features
from utilities import setup_logger
from save_models import save_model, load_model
from evaluation_metrics import evaluate_model
from generate_val_predictions import load_or_train_lightgbm
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

    lgb_model, lightgbm_params, lgb_loaded_metrics = load_or_train_lightgbm(
        X_train,
        y_train,
        X_val,
        y_val,
        retrain=build_new_models,
    )
    lgb_aps = lgb_loaded_metrics.get("average_precision_score")

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

