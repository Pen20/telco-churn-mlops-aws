import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from run_pipeline import load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODEL_URI = "models:/m-17aa3f43f3bd468d9496ee8909b5c319"

mlflow.set_tracking_uri(
    f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
)
mlflow.set_experiment("Telco Churn Adaptation")

# Retrieve the training run associated with this exact model.
model_info = mlflow.models.get_model_info(MODEL_URI)
source_run_id = model_info.run_id
if not source_run_id:
    raise RuntimeError("The model has no associated training run.")

source_run = mlflow.get_run(source_run_id)

data_path = ROOT / "data/raw/Telco-Customer-Churn.csv"
actual_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
if actual_hash != source_run.data.params["dataset_sha256"]:
    raise ValueError("Dataset changed since training.")

manifest_path = mlflow.artifacts.download_artifacts(
    run_id=source_run_id,
    artifact_path="split_manifest.json",
)
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))

df, X, y = load_dataset(data_path)

# Use the recorded validation IDs, rather than creating another split.
validation_ids = manifest["validation_ids"]
lookup = pd.Series(df.index.to_numpy(), index=df["customerID"])
validation_idx = lookup.loc[validation_ids].to_numpy()

assert set(validation_ids).isdisjoint(manifest["train_ids"])
assert set(validation_ids).isdisjoint(manifest["reserved_test_ids"])

X_val = X.loc[validation_idx]
y_val = y.loc[validation_idx]

model = mlflow.sklearn.load_model(MODEL_URI)
probabilities = model.predict_proba(X_val)[:, 1]

rows = []
for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
    predictions = (probabilities >= threshold).astype(int)
    rows.append({
        "threshold": float(threshold),
        "precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
    })

results = pd.DataFrame(rows)

# Deterministic tie-break: higher precision, then higher threshold.
ranked = results.sort_values(
    ["f1", "precision", "threshold"],
    ascending=[False, False, False],
)
best = ranked.iloc[0]

config = {
    "model_uri": MODEL_URI,
    "source_run_id": source_run_id,
    "threshold": float(best["threshold"]),
    "positive_class": 1,
    "selection_objective": "validation_f1",
    "dataset_sha256": actual_hash,
}

with mlflow.start_run(run_name="xgboost-threshold-selection") as run:
    mlflow.log_params({
        "source_model_uri": MODEL_URI,
        "source_run_id": source_run_id,
        "selection_objective": "validation_f1",
        "threshold": config["threshold"],
        "n_validation": len(y_val),
        "threshold_grid": "0.05 to 0.95, step 0.01",
    })
    mlflow.log_metrics({
        "selected_val_precision": float(best["precision"]),
        "selected_val_recall": float(best["recall"]),
        "selected_val_f1": float(best["f1"]),
    })
    mlflow.log_table(results, artifact_file="threshold_comparison.json")
    config["threshold_selection_run_id"] = run.info.run_id
    mlflow.log_dict(config, "selected_config.json")

output = ROOT / "artifacts"
output.mkdir(exist_ok=True)
(output / "selected_config.json").write_text(
    json.dumps(config, indent=2), encoding="utf-8"
)

print("\nTop validation thresholds:")
print(ranked.head(10).to_string(index=False))
print(f"\nSelected threshold: {config['threshold']:.2f}")
print(f"Validation precision: {best['precision']:.4f}")
print(f"Validation recall:    {best['recall']:.4f}")
print(f"Validation F1:        {best['f1']:.4f}")
print("Saved: artifacts/selected_config.json")
print("Final test data remain untouched.")