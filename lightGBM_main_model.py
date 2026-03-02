import lightgbm as lgb
from sklearn.metrics import average_precision_score


def _compute_scale_pos_weight(y, multiplier: float = 1.0):
    """Computes the scale_pos_weight parameter for LightGBM based on class imbalance in the target variable y."""
    positive = (y == 1).sum()
    negative = len(y) - positive
    if positive == 0:
        return 1.0
    return (negative / positive) * multiplier


def train_lightgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    params=None,
    num_boost_round: int = 1500,
    early_stopping_rounds: int = 100,
    verbose_eval: int | bool = False,
    pos_weight_multiplier: float | None = 0.5,
):

    base_params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 128,
        "min_data_in_leaf": 16,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "max_depth": -1,
        "n_jobs": -1,
        "verbose": -1,
    }

    params = params or {}
    final_params = {**base_params, **params}
    if "scale_pos_weight" not in final_params:
        multiplier = 1.0 if pos_weight_multiplier is None else pos_weight_multiplier
        final_params["scale_pos_weight"] = _compute_scale_pos_weight(y_train, multiplier)

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
