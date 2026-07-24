from __future__ import annotations

from pathlib import Path
from typing import Any


def maybe_log_training_run(
    *,
    pipeline: Any,
    metrics: dict[str, float],
    params: dict[str, Any],
    tracking_uri: str,
    experiment_name: str,
    registered_model_name: str,
    model_path: Path,
) -> str | None:
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return None

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=params.get("selected_model", "training-run")) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        mlflow.log_metrics(metrics)
        if model_path.exists():
            mlflow.log_artifact(str(model_path))
        try:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                registered_model_name=registered_model_name,
            )
        except Exception:
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
            )
        return run.info.run_id

    return None
