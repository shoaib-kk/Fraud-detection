import numpy as np
import pandas as pd


def engineer_features(data: pd.DataFrame, stats: dict | None = None):
    """Add log-amount, standardized amount, and cyclical time features.

    Pass the `stats` from the training split to val/test to avoid data leakage.
    """

    df = data.copy()
    amount_mean = stats.get("amount_mean") if stats else df["Amount"].mean()
    amount_std = stats.get("amount_std") if stats else df["Amount"].std(ddof=0)
    if amount_std == 0:
        amount_std = 1.0

    df["Log_Amount"] = np.log1p(df["Amount"])
    df["Amount_zscore"] = (df["Amount"] - amount_mean) / amount_std

    seconds_in_day = 24 * 60 * 60
    time_of_day = (df["Time"] % seconds_in_day) / seconds_in_day
    df["time_sin"] = np.sin(2 * np.pi * time_of_day)
    df["time_cos"] = np.cos(2 * np.pi * time_of_day)

    return df, {"amount_mean": amount_mean, "amount_std": amount_std}