from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from churn_app.config import get_settings
from churn_app.data import ensure_dataset, load_training_frame, split_features_target
from churn_app.mlflow_utils import maybe_log_training_run
from churn_app.modeling import save_training_artifacts, train_and_select_model


def main() -> None:
    settings = get_settings()
    dataset_path = ensure_dataset(settings.raw_data_path, settings.dataset_url)
    frame = load_training_frame(dataset_path)
    X, y = split_features_target(frame)

    result = train_and_select_model(
        X,
        y,
        scoring_metric=settings.scoring_metric,
    )

    model_path, metadata_path = save_training_artifacts(
        result,
        settings.model_dir,
        prediction_threshold=settings.prediction_threshold,
        extra_metadata={"dataset_path": str(dataset_path)},
    )

    mlflow_run_id = maybe_log_training_run(
        pipeline=result.pipeline,
        metrics=result.metrics,
        params={
            "dataset_path": dataset_path,
            "selected_model": result.model_name,
            "scoring_metric": settings.scoring_metric,
            "train_rows": result.train_rows,
            "valid_rows": result.valid_rows,
        },
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
        registered_model_name=settings.mlflow_registered_model_name,
        model_path=model_path,
    )

    if mlflow_run_id:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["mlflow_run_id"] = mlflow_run_id
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Selected model: {result.model_name}")
    print(json.dumps(result.metrics, indent=2))
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")
    if mlflow_run_id:
        print(f"MLflow run id: {mlflow_run_id}")


if __name__ == "__main__":
    main()
