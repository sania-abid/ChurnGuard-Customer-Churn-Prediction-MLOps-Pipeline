from __future__ import annotations

from churn_app.config import get_settings
from churn_app.predictor import ChurnPredictor


def build_demo():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is not installed. Run `pip install -e \".[mlops]\"` first."
        ) from exc

    settings = get_settings()
    predictor = ChurnPredictor.from_model_dir(settings.model_dir)

    def predict(
        gender: str,
        senior_citizen: int,
        partner: str,
        dependents: str,
        tenure: int,
        phone_service: str,
        multiple_lines: str,
        internet_service: str,
        online_security: str,
        online_backup: str,
        device_protection: str,
        tech_support: str,
        streaming_tv: str,
        streaming_movies: str,
        contract: str,
        paperless_billing: str,
        payment_method: str,
        monthly_charges: float,
        total_charges: float,
    ) -> tuple[str, str, str]:
        result = predictor.predict_one(
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        )
        return (
            result["prediction"],
            f"{result['churn_probability']:.2%}",
            result["model_name"],
        )

    with gr.Blocks(title="Customer Churn Predictor") as demo:
        gr.Markdown("# Customer Churn Predictor")
        gr.Markdown("Use the trained model to estimate churn probability for a single customer.")

        with gr.Row():
            gender = gr.Dropdown(["Female", "Male"], value="Female", label="Gender")
            senior_citizen = gr.Dropdown([0, 1], value=0, label="Senior Citizen")
            partner = gr.Dropdown(["Yes", "No"], value="Yes", label="Partner")
            dependents = gr.Dropdown(["Yes", "No"], value="No", label="Dependents")

        with gr.Row():
            tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure")
            monthly_charges = gr.Slider(0, 150, value=79.85, step=0.01, label="Monthly Charges")
            total_charges = gr.Slider(0, 9000, value=958.2, step=0.01, label="Total Charges")

        with gr.Row():
            phone_service = gr.Dropdown(["Yes", "No"], value="Yes", label="Phone Service")
            multiple_lines = gr.Dropdown(
                ["No phone service", "No", "Yes"],
                value="No",
                label="Multiple Lines",
            )
            internet_service = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                value="Fiber optic",
                label="Internet Service",
            )

        with gr.Row():
            online_security = gr.Dropdown(
                ["No internet service", "No", "Yes"],
                value="No",
                label="Online Security",
            )
            online_backup = gr.Dropdown(
                ["No internet service", "No", "Yes"],
                value="Yes",
                label="Online Backup",
            )
            device_protection = gr.Dropdown(
                ["No internet service", "No", "Yes"],
                value="No",
                label="Device Protection",
            )

        with gr.Row():
            tech_support = gr.Dropdown(
                ["No internet service", "No", "Yes"],
                value="No",
                label="Tech Support",
            )
            streaming_tv = gr.Dropdown(
                ["No internet service", "No", "Yes"],
                value="Yes",
                label="Streaming TV",
            )
            streaming_movies = gr.Dropdown(
                ["No internet service", "No", "Yes"],
                value="Yes",
                label="Streaming Movies",
            )

        with gr.Row():
            contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                value="Month-to-month",
                label="Contract",
            )
            paperless_billing = gr.Dropdown(
                ["Yes", "No"],
                value="Yes",
                label="Paperless Billing",
            )
            payment_method = gr.Dropdown(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                value="Electronic check",
                label="Payment Method",
            )

        run_button = gr.Button("Predict")
        prediction = gr.Textbox(label="Prediction")
        probability = gr.Textbox(label="Churn Probability")
        model_name = gr.Textbox(label="Selected Model")

        run_button.click(
            predict,
            inputs=[
                gender,
                senior_citizen,
                partner,
                dependents,
                tenure,
                phone_service,
                multiple_lines,
                internet_service,
                online_security,
                online_backup,
                device_protection,
                tech_support,
                streaming_tv,
                streaming_movies,
                contract,
                paperless_billing,
                payment_method,
                monthly_charges,
                total_charges,
            ],
            outputs=[prediction, probability, model_name],
        )

    return demo


def main() -> None:
    settings = get_settings()
    demo = build_demo()
    demo.launch(server_name=settings.app_host, server_port=settings.ui_port)


if __name__ == "__main__":
    main()
