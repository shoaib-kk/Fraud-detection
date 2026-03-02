import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics  import average_precision_score
from sklearn.pipeline import Pipeline
from utilities import setup_logger
from sklearn.preprocessing import StandardScaler

logger = setup_logger(__name__)

def grid_search_logistic_regression(X_train, y_train, X_val, y_val, param_grid: dict):
    best_score = 0
    best_params = None
    best_pipeline = None

    # C represents inverse regularisation strength, small values - > strong regularisation 
    for C in param_grid['C']:
        for solver in param_grid['solver']:
            pipeline = train_logistic_regression(X_train, y_train, {'C': C, 'solver': solver})

            # use predict_proba to get probabilities for the positive test class
            # roc score instead of accuracy to better handle class imbalance ie,, the fact that there are very few fraud cases compared to non-fraud cases
            score = average_precision_score(y_val, pipeline.predict_proba(X_val)[:, 1])
            logger.debug(f"Evaluated Logistic Regression with C={C}, solver={solver}, Average Precision Score={score:.4f}")
            if score > best_score:
                best_score = score
                best_params = {'C': C, 'solver': solver}
                best_pipeline = pipeline
    return best_params, best_score, best_pipeline

def train_logistic_regression(X_train, y_train, params):

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42, **params))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

