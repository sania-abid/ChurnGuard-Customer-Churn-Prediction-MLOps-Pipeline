# ChurnGuard – Customer Churn Prediction MLOps Pipeline

An end-to-end MLOps pipeline for customer churn prediction built with Python, FastAPI, Scikit-learn, MLflow, Docker, and AWS deployment support. The project demonstrates the complete machine learning lifecycle, including data validation, preprocessing, model training, experiment tracking, API deployment, containerization, CI/CD automation, and cloud-ready infrastructure.

---

# Overview

ChurnGuard predicts whether a customer is likely to leave a subscription-based service using supervised machine learning. The project follows modern MLOps practices by automating data validation, model training, experiment tracking, API serving, containerization, testing, and deployment workflows.

---

# Features

### Machine Learning Pipeline

- Automated data preprocessing
- Feature engineering
- Multiple model comparison
- Model evaluation
- Best model selection
- Saved model artifacts

### MLOps

- MLflow experiment tracking
- Great Expectations data validation
- Docker containerization
- GitHub Actions CI/CD
- AWS ECS deployment ready

### API

- FastAPI REST API
- Health check endpoint
- Prediction endpoint
- Pydantic request validation

### User Interface

- Interactive Gradio web application
- Manual prediction testing

### Testing

- Unit testing with Pytest
- API testing
- Data validation testing
- Model testing

---

# Tech Stack

## Programming

- Python

## Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib

## Backend

- FastAPI
- Uvicorn
- Pydantic

## MLOps

- MLflow
- Great Expectations
- Docker
- Docker Compose
- GitHub Actions

## Cloud

- AWS ECS
- Amazon ECR

## Testing

- Pytest

---

# Project Structure

```text
ChurnGuard/
│
├── .github/
│   └── workflows/
│
├── infra/
│
├── scripts/
│   ├── download_data.py
│   ├── train.py
│   └── validate_data.py
│
├── src/
│   └── churn_app/
│
├── tests/
│
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Architecture Diagram

<img width="373" height="415" alt="image" src="https://github.com/user-attachments/assets/1953264d-72a8-4757-aaf1-87d711bfb6ca" />


<p align="center">
  <img src="docs/architecture.png" width="900">
</p>

---

# Installation

## Clone Repository

```bash
git clone https://github.com/sania-abid/ChurnGuard-Customer-Churn-Prediction-MLOps-Pipeline.git

cd ChurnGuard-Customer-Churn-Prediction-MLOps-Pipeline
```

---

## Create Virtual Environment

```bash
python -m venv .venv

# Windows

.venv\Scripts\activate

# Linux/macOS

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

For optional MLOps components:

```bash
pip install -e ".[dev,mlops]"
```

---

# Download Dataset

```bash
python scripts/download_data.py
```

---

# Validate Dataset

```bash
python scripts/validate_data.py
```

---

# Train Model

```bash
python scripts/train.py
```

Generated artifacts:

- Trained Model
- Model Metadata

---

# Run FastAPI Server

```bash
uvicorn churn_app.api:app --reload
```

API Documentation:

```
http://127.0.0.1:8000/docs
```

---

# Run Gradio Interface

```bash
python -m churn_app.ui
```

---

# Docker

Build and start the application:

```bash
docker compose up --build
```

Services

- FastAPI
- MLflow
- Gradio

---

# CI/CD

GitHub Actions automatically performs:

- Code Quality Checks
- Automated Testing
- Docker Image Build
- AWS ECS Deployment Workflow

---

# Environment Variables

Backend configuration includes:

- DATASET_URL
- RAW_DATA_PATH
- MODEL_DIR
- MLFLOW_TRACKING_URI
- MLFLOW_EXPERIMENT_NAME
- MLFLOW_REGISTERED_MODEL_NAME
- SCORING_METRIC
- PREDICTION_THRESHOLD

---

# Future Improvements

- Model Monitoring
- Data Drift Detection
- Automated Retraining
- Kubernetes Deployment
- Feature Store Integration
- Prometheus & Grafana Monitoring

---

# License

This project was developed for educational and portfolio purposes.

---

# Author

**Sania Abid**

LinkedIn:
https://linkedin.com/in/saniaa-abid
