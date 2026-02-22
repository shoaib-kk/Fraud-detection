import lightgbm as lgb
from sklearn.metrics import average_precision_score


def _compute_scale_pos_weight(y):
    positive = (y == 1).sum()
    negative = len(y) - positive
    if positive == 0:
        return 1.0
    return negative / positive


def train_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    params=None,
    num_boost_round: int = 2000,
    early_stopping_rounds: int = 100,
    verbose_eval: int | bool = False,
):

    base_params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "min_data_in_leaf": 32,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "max_depth": -1,
        "n_jobs": -1,
        "verbose": -1,
    }

    params = params or {}
    final_params = {**base_params, **params}
    if "scale_pos_weight" not in final_params:
        final_params["scale_pos_weight"] = _compute_scale_pos_weight(y_train)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = []
    if early_stopping_rounds:
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    if verbose_eval:
        callbacks.append(lgb.log_evaluation(period=verbose_eval if isinstance(verbose_eval, int) else 50))


    model = lgb.train(
        final_params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        valid_names=["valid"],
        callbacks=callbacks,
    )

    best_iteration = model.best_iteration or num_boost_round
    val_preds = model.predict(X_val, num_iteration=best_iteration)
    aps = average_precision_score(y_val, val_preds)
    return model, aps
