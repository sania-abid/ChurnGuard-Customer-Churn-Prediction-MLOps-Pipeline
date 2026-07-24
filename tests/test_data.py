from __future__ import annotations

import pandas as pd

from churn_app.data import clean_telco_dataframe, prepare_inference_frame


def test_clean_telco_dataframe_fills_blank_total_charges() -> None:
    frame = pd.DataFrame(
        [
            {
                "customerID": "0001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 10,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 50.0,
                "TotalCharges": " ",
                "Churn": "No",
            }
        ]
    )

    cleaned = clean_telco_dataframe(frame)
    assert cleaned.loc[0, "TotalCharges"] == 500.0


def test_prepare_inference_frame_orders_and_types_columns() -> None:
    prepared = prepare_inference_frame(
        [
            {
                "gender": "Male",
                "SeniorCitizen": "1",
                "Partner": "No",
                "Dependents": "No",
                "tenure": "5",
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": "80.5",
                "TotalCharges": "402.5",
            }
        ]
    )

    assert list(prepared.columns) == [
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
    assert prepared.loc[0, "SeniorCitizen"] == 1
    assert prepared.loc[0, "tenure"] == 5
