import json
import math
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import mlflow.sklearn
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import os

BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
MODEL_DIR = ROOT / "deployment/model"

config = json.loads(
    (MODEL_DIR / "serving_config.json").read_text(encoding="utf-8")
)
model = mlflow.sklearn.load_model(str(MODEL_DIR))

sample = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
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
    "MonthlyCharges": 70.35,
    "TotalCharges": 350.75,
}


def post(payload):
    request = Request(
        f"{BASE_URL}/predict",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urlopen(request, timeout=10)


def check_prediction(payload):
    frame = pd.DataFrame([payload])
    for column in [
        "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
    ]:
        frame[column] = pd.to_numeric(frame[column]).astype("float64")

    positive_index = model.classes_.tolist().index(
        config["positive_class"]
    )
    expected = float(model.predict_proba(frame)[0, positive_index])

    with post(payload) as response:
        assert response.status == 200
        result = json.load(response)

    assert math.isclose(
        result["churn_probability"], expected,
        rel_tol=1e-6, abs_tol=1e-8,
    )
    assert result["threshold"] == config["threshold"]
    assert result["predicted_churn"] == (
        expected >= config["threshold"]
    )
    assert result["model_uri"] == config["model_uri"]


def expect_rejected(payload):
    try:
        with post(payload):
            pass
    except HTTPError as error:
        status = error.code
        error.close()
        assert status == 422, f"Expected 422, received {status}"
    else:
        raise AssertionError("Invalid request was accepted")


with urlopen(f"{BASE_URL}/health", timeout=10) as response:
    assert response.status == 200
    health = json.load(response)
    assert health["status"] == "ready"
    assert health["model_uri"] == config["model_uri"]

print("PASS: health and model identity")

check_prediction(sample)
print("PASS: API prediction matches exported pipeline")

# Exercise different categorical values.
alternative = {
    **sample,
    "InternetService": "DSL",
    "Contract": "Two year",
    "PaymentMethod": "Credit card (automatic)",
}
check_prediction(alternative)
print("PASS: alternative categories match exported pipeline")

check_prediction({**sample, "TotalCharges": None})
print("PASS: missing TotalCharges handled by fitted imputer")

expect_rejected({**sample, "tenure": -1})
expect_rejected({**sample, "Contract": "unknown"})
expect_rejected({**sample, "unexpected_field": 123})

missing_field = sample.copy()
missing_field.pop("MonthlyCharges")
expect_rejected(missing_field)

print("PASS: negative, unknown, extra and missing inputs rejected")
print("\nAll API checks passed.")