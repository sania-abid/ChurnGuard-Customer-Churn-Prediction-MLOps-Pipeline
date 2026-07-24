from __future__ import annotations

from fastapi.testclient import TestClient

from churn_app.api import create_app


class FakePredictor:
    def predict_one(self, record: dict) -> dict:
        return {
            "prediction": "Yes",
            "churn_probability": 0.91,
            "model_name": "fake-model",
            "threshold": 0.5,
            "mlflow_run_id": None,
        }


def test_predict_endpoint_returns_prediction() -> None:
    client = TestClient(create_app(predictor=FakePredictor()))
    response = client.post(
        "/predict",
        json={
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.85,
            "TotalCharges": 958.2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "Yes"
    assert payload["model_name"] == "fake-model"
