from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, HTTPException

from churn_app.config import get_settings
from churn_app.predictor import ChurnPredictor
from churn_app.schemas import ChurnFeatures, PredictionResponse


class PredictorProtocol(Protocol):
    def predict_one(self, record: dict) -> dict:
        ...


def create_app(predictor: PredictorProtocol | None = None) -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.predictor is None:
            app.state.predictor = ChurnPredictor.from_model_dir(settings.model_dir)
        yield

    app = FastAPI(
        title="Customer Churn Prediction API",
        version="0.1.0",
        description="Predict whether a telco customer is likely to churn.",
        lifespan=lifespan,
    )
    app.state.predictor = predictor

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "service": "customer-churn-api",
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        predictor_name = "loaded" if app.state.predictor is not None else "missing"
        return {"status": "ok", "predictor": predictor_name}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(features: ChurnFeatures) -> PredictionResponse:
        if app.state.predictor is None:
            raise HTTPException(status_code=503, detail="Predictor is not available.")
        result = app.state.predictor.predict_one(features.model_dump())
        return PredictionResponse(**result)

    return app


app = create_app()
