import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import mlflow.sklearn
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(
    os.environ.get("MODEL_DIR", str(ROOT / "deployment/model"))
)


class CustomerData(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float | None = Field(ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    threshold: float
    predicted_churn: bool
    model_uri: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = json.loads(
        (MODEL_DIR / "serving_config.json").read_text(encoding="utf-8")
    )
    threshold = float(config["threshold"])
    if not 0 < threshold < 1:
        raise ValueError("Invalid saved threshold.")

    pipeline = mlflow.sklearn.load_model(str(MODEL_DIR))
    classes = pipeline.classes_.tolist()
    positive_index = classes.index(config["positive_class"])

    app.state.pipeline = pipeline
    app.state.threshold = threshold
    app.state.positive_index = positive_index
    app.state.model_uri = config["model_uri"]

    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {
        "status": "ready",
        "model_uri": app.state.model_uri,
        "threshold": app.state.threshold,
    }


@app.get("/")
def root():
    return {"service": "Telco Churn Prediction API", "docs": "/docs"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    frame = pd.DataFrame([customer.model_dump()])

    # Match the numeric types used during training.
    for column in [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
    ]:
        frame[column] = pd.to_numeric(
            frame[column], errors="raise"
        ).astype("float64")

    probability = float(
        app.state.pipeline.predict_proba(frame)[
            0, app.state.positive_index
        ]
    )
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise RuntimeError("Model returned an invalid probability.")

    return {
        "churn_probability": probability,
        "threshold": app.state.threshold,
        "predicted_churn": probability >= app.state.threshold,
        "model_uri": app.state.model_uri,
    }