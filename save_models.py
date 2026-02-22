from pathlib import Path
import joblib
import json
from datetime import datetime
from utilities import setup_logger

logger = setup_logger(__name__)


def _to_serializable(obj):
    """Recursively convert common non-JSON types (numpy, Path, etc.) to plain Python."""
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(obj, (np.generic,)):
        return obj.item()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_model(model, model_name: str, metrics: dict, model_params: dict, out_dir: str = "models"):
    model_path = Path(f"{out_dir}/{model_name}.pkl")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # joblib saves and loads ML models efficiently
    joblib.dump(model, model_path)

    metrics_path = Path(f"{out_dir}/{model_name}_metrics.json")
    serializable_metrics = _to_serializable({
        "trained_at": datetime.utcnow().isoformat() + "Z",
        **metrics,
    })
    with open(metrics_path, "w") as f:
        json.dump(serializable_metrics, f, indent=4)

    params_path = Path(f"{out_dir}/{model_name}_params.json")
    serializable_params = _to_serializable(model_params)
    with open(params_path, "w") as f:
        json.dump(serializable_params, f, indent=4)


def load_model(model_name: str, out_dir: str = "models"):
    model_path = Path(f"{out_dir}/{model_name}.pkl")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = joblib.load(model_path)

    metrics_path = Path(f"{out_dir}/{model_name}_metrics.json")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    else:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    params_path = Path(f"{out_dir}/{model_name}_params.json")

    if not params_path.exists():
        raise FileNotFoundError(f"Model parameters file not found: {params_path}")
    else:
        with open(params_path, "r") as f:
            model_params = json.load(f)

    return model, metrics, model_params