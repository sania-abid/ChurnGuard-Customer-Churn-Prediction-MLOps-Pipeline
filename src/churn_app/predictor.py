from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from churn_app.data import prepare_inference_frame


@dataclass
class ChurnPredictor:
    pipeline: Any
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "ChurnPredictor":
        model_path = model_dir / "churn_model.joblib"
        metadata_path = model_dir / "model_metadata.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run scripts/train.py first."
            )

        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        pipeline = joblib.load(model_path)
        return cls(pipeline=pipeline, metadata=metadata)

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("prediction_threshold", 0.5))

    def predict_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        frame = prepare_inference_frame(records)
        probabilities = self.pipeline.predict_proba(frame)[:, 1]
        labels = np.where(probabilities >= self.threshold, "Yes", "No")
        results = []
        for label, probability in zip(labels, probabilities, strict=True):
            results.append(
                {
                    "prediction": str(label),
                    "churn_probability": float(probability),
                    "model_name": self.metadata.get("model_name", "unknown"),
                    "threshold": self.threshold,
                    "mlflow_run_id": self.metadata.get("mlflow_run_id"),
                }
            )
        return results

    def predict_one(self, record: dict[str, Any]) -> dict[str, Any]:
        return self.predict_records([record])[0]
