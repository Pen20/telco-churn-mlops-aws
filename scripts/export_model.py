import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np

from run_pipeline import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deployment/model"

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)

config = json.loads(
    (ROOT / "artifacts/selected_config.json").read_text(
        encoding="utf-8"
    )
)

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists. Keep or rename it before exporting."
    )

model = mlflow.sklearn.load_model(config["model_uri"])

mlflow.sklearn.save_model(
    sk_model=model,
    path=str(OUTPUT),
    serialization_format="cloudpickle",
    pyfunc_predict_fn="predict_proba",
)

(OUTPUT / "serving_config.json").write_text(
    json.dumps(config, indent=2),
    encoding="utf-8",
)

# Verify the exported pipeline produces the same predictions.
_, X, _ = load_dataset(
    ROOT / "data/raw/Telco-Customer-Churn.csv"
)
sample = X.head(10)

exported = mlflow.sklearn.load_model(str(OUTPUT))
np.testing.assert_allclose(
    model.predict_proba(sample),
    exported.predict_proba(sample),
)

print("Exported model:", OUTPUT)
print("Threshold:", config["threshold"])
print("PASS: exported predictions match the selected model.")