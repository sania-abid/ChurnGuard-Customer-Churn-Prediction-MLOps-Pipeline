# Churn Prediction MLOps Project

This repository recreates the kind of end-to-end churn project shown in the YouTube video you shared: a customer churn model that is trained on Telco-style data, tracked with optional MLflow, served through FastAPI, and prepared for Docker, CI, and AWS ECS deployment.

## What is included

- A modular training pipeline with preprocessing, candidate-model comparison, artifact saving, and optional MLflow logging
- A FastAPI inference service with `/health` and `/predict`
- A Gradio UI for manual testing
- Optional Great Expectations-based data validation
- Docker and `docker-compose` setup
- GitHub Actions CI plus a manual ECS deployment workflow
- Tests for preprocessing, training, and the API layer

## Project layout

```text
.
├── .github/workflows/
├── infra/
├── scripts/
├── src/churn_app/
├── tests/
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

## Quick start

### 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

To enable MLflow, Gradio, and Great Expectations:

```powershell
pip install -e ".[dev,mlops]"
```

### 2. Download the dataset

```powershell
python scripts\download_data.py
```

The downloader pulls the public IBM Telco churn CSV from GitHub into `data/raw/telco_churn.csv`.

### 3. Validate the dataset

```powershell
python scripts\validate_data.py
```

If `great_expectations` is installed the script uses it. Otherwise it falls back to a lightweight built-in validator.

### 4. Train the model

```powershell
python scripts\train.py
```

Artifacts are written to `artifacts/model/`:

- `churn_model.joblib`
- `model_metadata.json`

### 5. Run the API

```powershell
python -m uvicorn churn_app.api:app --reload
```

Then open:

- `http://127.0.0.1:8000/docs`

### 6. Run the UI

```powershell
python -m churn_app.ui
```

The Gradio app runs on port `7860` by default.

## Environment variables

Copy `.env.example` values into your own shell or CI secrets as needed.

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATASET_URL` | public GitHub CSV | Source dataset URL |
| `RAW_DATA_PATH` | `data/raw/telco_churn.csv` | Local dataset path |
| `MODEL_DIR` | `artifacts/model` | Saved model artifacts |
| `MLFLOW_TRACKING_URI` | `file:./mlruns` | MLflow backend |
| `MLFLOW_EXPERIMENT_NAME` | `churn-prediction` | Experiment name |
| `MLFLOW_REGISTERED_MODEL_NAME` | `churn-classifier` | Registry model name |
| `SCORING_METRIC` | `f1` | Model selection metric |
| `PREDICTION_THRESHOLD` | `0.5` | Classification threshold |

## Docker

Build and run the services:

```powershell
docker compose up --build
```

Services:

- FastAPI: `http://localhost:8000`
- MLflow: `http://localhost:5000`
- Gradio: `http://localhost:7860`

## GitHub Actions

`/.github/workflows/ci.yml` runs tests on every push and pull request, builds the Docker image, and exposes a manual ECS deployment path once AWS secrets are configured.

Secrets expected by the deployment job:

- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `ECR_REPOSITORY`
- `ECS_CLUSTER`
- `ECS_SERVICE`
- `ECS_TASK_EXECUTION_ROLE_ARN`

## AWS deployment notes

The repository includes an ECS task definition template in `infra/ecs-task-definition.json`. Update the image URI, region, roles, and networking values before using the deployment workflow.

## Run tests

```powershell
pytest
```

## Notes

- Docker is required only for the container workflow.
- MLflow, Gradio, and Great Expectations are optional at install time so the core project can still train and serve in a lighter Python environment.
