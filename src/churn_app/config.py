from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


@dataclass(frozen=True)
class Settings:
    project_root: Path
    dataset_url: str
    raw_data_path: Path
    model_dir: Path
    reports_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    mlflow_registered_model_name: str
    scoring_metric: str
    prediction_threshold: float
    app_host: str
    app_port: int
    ui_port: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[2]
    return Settings(
        project_root=project_root,
        dataset_url=os.getenv(
            "DATASET_URL",
            "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        ),
        raw_data_path=_resolve_path(
            project_root,
            os.getenv("RAW_DATA_PATH", "data/raw/telco_churn.csv"),
        ),
        model_dir=_resolve_path(
            project_root,
            os.getenv("MODEL_DIR", "artifacts/model"),
        ),
        reports_dir=_resolve_path(
            project_root,
            os.getenv("REPORTS_DIR", "reports"),
        ),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"),
        mlflow_experiment_name=os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            "churn-prediction",
        ),
        mlflow_registered_model_name=os.getenv(
            "MLFLOW_REGISTERED_MODEL_NAME",
            "churn-classifier",
        ),
        scoring_metric=os.getenv("SCORING_METRIC", "f1"),
        prediction_threshold=float(os.getenv("PREDICTION_THRESHOLD", "0.5")),
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=int(os.getenv("APP_PORT", "8000")),
        ui_port=int(os.getenv("UI_PORT", "7860")),
    )
