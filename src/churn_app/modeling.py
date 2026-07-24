from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_app.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", _make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def candidate_models(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            min_samples_leaf=2,
            random_state=random_state,
        ),
    }


def calculate_metrics(y_true: pd.Series, probabilities: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


@dataclass
class TrainingResult:
    model_name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    leaderboard: dict[str, dict[str, float]]
    feature_columns: list[str]
    train_rows: int
    valid_rows: int


def train_and_select_model(
    X: pd.DataFrame,
    y: pd.Series,
    scoring_metric: str = "f1",
    random_state: int = 42,
) -> TrainingResult:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    leaderboard: dict[str, dict[str, float]] = {}
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_metrics: dict[str, float] = {}
    best_score = float("-inf")

    for name, estimator in candidate_models(random_state).items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_valid)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = calculate_metrics(y_valid, probabilities, predictions)
        leaderboard[name] = metrics

        score = metrics.get(scoring_metric, metrics["f1"])
        if score > best_score:
            best_name = name
            best_pipeline = pipeline
            best_metrics = metrics
            best_score = score

    if best_pipeline is None:
        raise RuntimeError("No model was trained successfully.")

    return TrainingResult(
        model_name=best_name,
        pipeline=best_pipeline,
        metrics=best_metrics,
        leaderboard=leaderboard,
        feature_columns=list(FEATURE_COLUMNS),
        train_rows=len(X_train),
        valid_rows=len(X_valid),
    )


def save_training_artifacts(
    result: TrainingResult,
    model_dir: Path,
    prediction_threshold: float,
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "churn_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    joblib.dump(result.pipeline, model_path)

    metadata = {
        "saved_at_utc": datetime.now(UTC).isoformat(),
        "model_name": result.model_name,
        "metrics": result.metrics,
        "leaderboard": result.leaderboard,
        "feature_columns": result.feature_columns,
        "train_rows": result.train_rows,
        "valid_rows": result.valid_rows,
        "prediction_threshold": prediction_threshold,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model_path, metadata_path
