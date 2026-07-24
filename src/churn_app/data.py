from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_COLUMNS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [column for column in FEATURE_COLUMNS if column not in NUMERIC_COLUMNS]

EXPECTED_COLUMNS = ["customerID", *FEATURE_COLUMNS, "Churn"]


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_dataset(url: str, destination: Path) -> Path:
    ensure_parent_directory(destination)
    urllib.request.urlretrieve(url, destination)
    return destination


def ensure_dataset(dataset_path: Path, dataset_url: str) -> Path:
    if dataset_path.exists():
        return dataset_path
    return download_dataset(dataset_url, dataset_path)


def read_raw_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run scripts/download_data.py first."
        )
    return pd.read_csv(path)


def clean_telco_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [column.strip() for column in df.columns]

    missing_columns = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing expected columns: {missing_columns}")

    df["customerID"] = df["customerID"].astype(str).str.strip()
    df["Churn"] = df["Churn"].astype(str).str.strip()
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].astype(str).str.strip()

    df = df.drop_duplicates(subset=["customerID"]).reset_index(drop=True)
    return df


def load_training_frame(path: Path) -> pd.DataFrame:
    return clean_telco_dataframe(read_raw_dataset(path))


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_telco_dataframe(frame)
    X = df[FEATURE_COLUMNS].copy()
    y = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    return X, y


def prepare_inference_frame(records: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    missing = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for inference: {missing}")

    prepared = df[FEATURE_COLUMNS].copy()
    for column in ("SeniorCitizen", "tenure"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0).astype(int)
    for column in ("MonthlyCharges", "TotalCharges"):
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared["TotalCharges"] = prepared["TotalCharges"].fillna(
        prepared["MonthlyCharges"] * prepared["tenure"]
    )
    for column in CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype(str).str.strip()
    return prepared
