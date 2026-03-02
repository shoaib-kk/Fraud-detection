import numpy as np 
import pandas as pd 
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,

)


def confusion_counts(y_true, y_pred):
    """Return basic confusion counts for binary labels."""
    true_p = np.sum((y_true == 1) & (y_pred == 1))
    true_n = np.sum((y_true == 0) & (y_pred == 0))
    false_p = np.sum((y_true == 0) & (y_pred == 1))
    false_n = np.sum((y_true == 1) & (y_pred == 0))
    return {
        "true_positive": true_p,
        "true_negative": true_n,
        "false_positive": false_p,
        "false_negative": false_n,
    }


def point_metrics(counts: dict):
    """Compute precision/recall/f1 from confusion counts."""
    true_positive = counts["true_positive"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {"precision": precision, "recall": recall, "f1": f1}

def evaluate_model(y_true, y_pred_proba, threshold=0.1):

    # Threshold shoule be relatively low for fraud detection to catch more positives, 
    # but can be tuned based on the costs of false positives vs false negatives in the specific use case.
    # so higher threshold -> should be more conservative, lower threshold -> more aggressive in flagging frauds.
    y_pred = (y_pred_proba >= threshold).astype(int)

    aps = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred)
    false_positives = conf_matrix[0, 1]
    false_negatives = conf_matrix[1, 0]
    summary = (
        f"APS: {aps:.4f} | ROC-AUC: {roc_auc:.4f} "
        f"Precision@{threshold:.2f}: {precision:.4f}  "
        f"F1: {f1:.4f}\n"
        f"Confusion matrix [[TN, FP], [FN, TP]]: {conf_matrix.tolist()}\n"
        f"{class_report}"
    )
    return {
        "threshold": threshold,
        "average_precision_score": aps,
        "roc_auc_score": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix.tolist(),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "classification_report": class_report,
        "summary": summary,
    }


def choose_best_threshold(
    y_true,
    y_pred_proba,
    false_negative_cost=10,
    false_positive_cost=1,
    thresholds=None
):
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    if thresholds is None:
        thresholds = np.unique(y_pred_proba)

    best_t = 0.5
    best_cost = float("inf")

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn * false_negative_cost + fp * false_positive_cost

        if cost < best_cost:
            best_cost = cost
            best_t = float(t)

    return best_t, best_cost