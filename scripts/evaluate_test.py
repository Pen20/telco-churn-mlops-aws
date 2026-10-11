import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from run_pipeline import load_dataset

ROOT = Path(__file__).resolve().parents[1]
mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("Telco Churn Adaptation")

config = json.loads(
    (ROOT / "artifacts/selected_config.json").read_text(
        encoding="utf-8"
    )
)

data_path = ROOT / "data/raw/Telco-Customer-Churn.csv"
if hashlib.sha256(data_path.read_bytes()).hexdigest() != config["dataset_sha256"]:
    raise ValueError("Dataset changed since model selection.")

manifest_path = mlflow.artifacts.download_artifacts(
    run_id=config["source_run_id"],
    artifact_path="split_manifest.json",
)
manifest = json.loads(
    Path(manifest_path).read_text(encoding="utf-8")
)

test_ids = manifest["reserved_test_ids"]
assert set(test_ids).isdisjoint(manifest["train_ids"])
assert set(test_ids).isdisjoint(manifest["validation_ids"])

df, X, y = load_dataset(data_path)
lookup = pd.Series(df.index.to_numpy(), index=df["customerID"])
test_idx = lookup.loc[test_ids].to_numpy()

model = mlflow.sklearn.load_model(config["model_uri"])
probabilities = model.predict_proba(X.loc[test_idx])[:, 1]
predictions = (probabilities >= config["threshold"]).astype(int)
y_test = y.loc[test_idx]

metrics = {
    "test_average_precision": float(
        average_precision_score(y_test, probabilities)
    ),
    "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
    "test_precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "test_recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "test_f1": float(f1_score(y_test, predictions, zero_division=0)),
}
matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

with mlflow.start_run(run_name="selected-xgboost-final-test"):
    mlflow.log_params({
        "source_model_uri": config["model_uri"],
        "source_run_id": config["source_run_id"],
        "threshold_selection_run_id":
            config["threshold_selection_run_id"],
        "threshold": config["threshold"],
        "n_test": len(test_ids),
        "dataset_sha256": config["dataset_sha256"],
    })
    mlflow.log_metrics(metrics)
    mlflow.log_dict({
        "labels": [0, 1],
        "rows": "actual",
        "columns": "predicted",
        "matrix": matrix.tolist(),
    }, "test_confusion_matrix.json")
    mlflow.log_dict(config, "selected_config.json")

print(f"\nTest observations: {len(test_ids)}")
print(f"Threshold: {config['threshold']:.2f}")
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

print("\nConfusion matrix: rows=actual, columns=predicted")
print(matrix)