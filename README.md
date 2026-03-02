## Credit Card Fraud Detection with Uncertainty-Aware Evaluation

End-to-end workflow for detecting credit card fraud on the highly imbalanced ULB dataset. The project covers ingestion, validation, feature engineering, model training (LightGBM and Logistic Regression), evaluation with threshold-cost analysis, and a Streamlit dashboard for interactive inspection.

### Why this matters
- Fraud cases are rare ($0.172\%$), so precision/recall trade-offs and cost-weighted thresholds are critical.
- Calibrated probability outputs and threshold sweeps give operators control over false positives vs. false negatives.
- Lightweight pipeline: one command to train, one command to visualize.

## Project Overview
- **Data**: `creditcard.csv` (time, amount, PCA components `V1`-`V28`, target `Class`), auto-downloaded via Kaggle if missing.
- **Feature engineering**: log amount, z-scored amount using train statistics, cyclical time-of-day sine/cosine features in [feature_engineering.py](feature_engineering.py).
- **Splitting**: Time-based by default into train / early-stop / calibration / test via [pipeline.py](pipeline.py) and [preprocessing.py](preprocessing.py) to reduce temporal leakage.
- **Models**:
	- **LightGBM** (sklearn `LGBMClassifier`) with early stopping and auto `scale_pos_weight` in [pipeline.py](pipeline.py).
	- **Logistic Regression** baseline grid-searched over C/solver in [pipeline.py](pipeline.py).
- **Evaluation**: APS, ROC-AUC, confusion matrix, costed threshold search, plus held-out calibration (isotonic/sigmoid) reliability diagrams.
- **Persistence**: Models and metadata saved to `models/` via [save_models.py](save_models.py) for reproducibility.
- **App**: Streamlit dashboard in [app/Home.py](app/Home.py) that uses the same pipeline/models as the CLI; shows validation (calibration split) by default with a test-set toggle.
- **Interpretability**: SHAP global importances and per-transaction explanations, plus reliability diagrams with calibrated vs uncalibrated probabilities.

## Pipeline
1) **Ingest & validate**: Load or download data; enforce expected schema via [preprocessing.py](preprocessing.py).
2) **Clean**: Drop missing rows, coerce target to int.
3) **Feature engineering**: Add `Log_Amount`, `Amount_zscore`, `time_sin`, `time_cos` using training stats to avoid leakage in [feature_engineering.py](feature_engineering.py).
4) **Split**: Time-based (default) into train / early-stop / calibration / test; avoids reusing validation for both early stopping and calibration.
5) **Train models** (shared in [pipeline.py](pipeline.py)):
	- Hyperparameter sweep for Logistic Regression (C, solver).
	- LightGBM `LGBMClassifier` with early stopping and class weighting.
6) **Calibrate**: Fit isotonic and sigmoid calibrators with `cv="prefit"` on the held-out calibration split; reliability diagram written to `logs/interpretability/reliability_lightgbm.png`.
7) **Persist**: Store `*.pkl`, metrics JSON, and params JSON in `models/` via [save_models.py](save_models.py).
8) **Visualize**: Streamlit dashboard plots precision/recall/F1 vs. threshold and estimated cost trade-offs in [app/Home.py](app/Home.py).

## Results (test split, threshold = 0.10)
| Model | APS | ROC-AUC | Precision@0.10 | Recall | F1 | False Positives | False Negatives |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | 0.7486 | 0.9146 | 0.6905 | 0.7733 | 0.7296 | 26 | 17 |
| Logistic Regression | 0.7775 | 0.9892 | 0.0081 | 0.9867 | 0.0160 | 9079 | 1 |

Notes:
- LightGBM offers far fewer false positives and higher precision; Logistic Regression maximizes recall at heavy false-positive cost.
- APS/ROC-AUC indicate strong ranking; operational threshold should reflect fraud costs and flagged-rate budgets.
- SHAP + calibration artifacts: see `logs/interpretability/*.png` and `logs/interpretability/calibration_metrics.json`.

### Validation split (app)
The dashboard shows validation (calibration split) by default and can toggle held-out test metrics. All metrics come from the shared pipeline/models—no separate app training path.



## Quickstart
1) **Install deps** (inside a virtual environment):
	```bash
	pip install lightgbm scikit-learn pandas numpy matplotlib streamlit kagglehub joblib shap
	```
2) **Train & evaluate (saves models/metrics)**:
	```bash
	python temp_main.py
	```
	- Uses time-based split into train/early-stop/calibration/test, feature engineering, grid-searches Logistic Regression, trains LightGBM, calibrates on the held-out calibration split, evaluates on test, and writes artifacts to `models/` + `logs/interpretability/`.
3) **Generate validation/test predictions only**:
	```bash
	python generate_val_predictions.py
	```
4) **Launch dashboard**:
	```bash
	streamlit run app/Home.py
	```
	- Compare models, sweep thresholds/costs, and view calibration/SHAP artifacts using the same persisted pipeline models.

Artifacts:
- SHAP summary and single-transaction plots: `logs/interpretability/shap_summary.png`, `logs/interpretability/shap_single.png`
- Reliability diagram (calibrated vs. uncalibrated LightGBM): `logs/interpretability/reliability_lightgbm.png`

## Repo Map
- Data prep: [preprocessing.py](preprocessing.py), [feature_engineering.py](feature_engineering.py)
- Modeling: [lightGBM_main_model.py](lightGBM_main_model.py), [logistic_regression_baseline.py](logistic_regression_baseline.py)
- Training flows: [temp_main.py](temp_main.py), [generate_val_predictions.py](generate_val_predictions.py)
- Evaluation: [evaluation_metrics.py](evaluation_metrics.py)
- Persistence & logging: [save_models.py](save_models.py), [utilities.py](utilities.py)
- App: [app/Home.py](app/Home.py)

## Key Takeaways
- Extreme imbalance demands metrics beyond accuracy; APS and ROC-AUC remain primary ranking metrics.
- Decision thresholds should be tuned to business costs (false positives vs. false negatives); see cost analysis utilities in [evaluation_metrics.py](evaluation_metrics.py).
- Feature scaling and simple baselines (Logistic Regression) provide strong recall; boosting improves precision and reduces false alarms.

## Next Steps
- Add calibration (Platt/Isotonic) to stabilize probabilities.
- Incorporate time-based validation to mimic production drift.
- Extend the Streamlit app with per-transaction explanations and cost sliders per segment.
- Package requirements and Docker recipe for reproducible deployment.
