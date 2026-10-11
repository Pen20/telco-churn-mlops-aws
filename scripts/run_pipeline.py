"""Compare churn models using training and validation data."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL


def load_dataset(path):
    df = pd.read_csv(path)
    required = ["customerID", "Churn"] + FEATURES
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df["customerID"].isna().any() or df["customerID"].duplicated().any():
        raise ValueError("Customer IDs must be present and unique.")

    labels = df["Churn"].astype(str).str.strip()
    if not labels.isin(["Yes", "No"]).all():
        raise ValueError("Churn must contain only Yes or No.")

    X = df[FEATURES].copy()
    for column in NUMERIC:
        original = X[column]
        converted = pd.to_numeric(original, errors="coerce")
        blank = original.isna() | original.astype(str).str.strip().eq("")
        if (converted.isna() & ~blank).any():
            raise ValueError(f"Invalid numeric values in {column}")
        X[column] = converted.astype("float64")

    if np.isinf(X[NUMERIC].to_numpy()).any():
        raise ValueError("Numeric features contain infinite values.")
    if (X[NUMERIC] < 0).any().any():
        raise ValueError("Numeric features must be nonnegative.")
    if not X["SeniorCitizen"].isin([0.0, 1.0]).all():
        raise ValueError("SeniorCitizen must be 0 or 1.")

    for column in CATEGORICAL:
        if X[column].isna().any():
            raise ValueError(f"Missing categorical values in {column}")
        X[column] = X[column].astype(str).str.strip()
        if X[column].eq("").any():
            raise ValueError(f"Blank categorical values in {column}")

    y = labels.map({"No": 0, "Yes": 1}).astype(int)
    return df, X, y


def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("categorical", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        ), CATEGORICAL),
    ])


def main(args):
    if not 0 < args.threshold < 1:
        raise ValueError("Threshold must be between 0 and 1.")

    path = Path(args.input).resolve()
    df, X, y = load_dataset(path)

    trainval_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.25,
        stratify=y.iloc[trainval_idx],
        random_state=42,
    )

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("Telco Churn Adaptation")

    candidates = {
        "logistic-regression": LogisticRegression(
            max_iter=2000, random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=2,
        ),
    }

    split_manifest = {
        "seed": 42,
        "train_ids": df.iloc[train_idx]["customerID"].tolist(),
        "validation_ids": df.iloc[val_idx]["customerID"].tolist(),
        "reserved_test_ids": df.iloc[test_idx]["customerID"].tolist(),
    }
    data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    results = []

    for name, estimator in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", make_preprocessor()),
            ("classifier", estimator),
        ])

        with mlflow.start_run(run_name=f"{name}-validation") as run:
            mlflow.log_params({
                "model": name,
                "threshold": args.threshold,
                "split_seed": 42,
                "n_train": len(train_idx),
                "n_validation": len(val_idx),
                "n_test_reserved": len(test_idx),
                "dataset_sha256": data_hash,
                "python_version": platform.python_version(),
                "sklearn_version": sklearn.__version__,
                "xgboost_version": xgboost.__version__,
            })
            mlflow.log_params({
                f"classifier_{key}": value
                for key, value in estimator.get_params().items()
                if value is not None
            })
            mlflow.log_dict(split_manifest, "split_manifest.json")

            start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_seconds = time.perf_counter() - start

            probabilities = pipeline.predict_proba(X_val)[:, 1]
            predictions = (probabilities >= args.threshold).astype(int)

            metrics = {
                "val_roc_auc": float(roc_auc_score(y_val, probabilities)),
                "val_average_precision": float(
                    average_precision_score(y_val, probabilities)
                ),
                "val_precision": float(
                    precision_score(y_val, predictions, zero_division=0)
                ),
                "val_recall": float(
                    recall_score(y_val, predictions, zero_division=0)
                ),
                "val_f1": float(
                    f1_score(y_val, predictions, zero_division=0)
                ),
                "train_seconds": train_seconds,
            }
            mlflow.log_metrics(metrics)
            mlflow.log_dict({
                "labels": [0, 1],
                "rows": "actual",
                "columns": "predicted",
                "matrix": confusion_matrix(
                    y_val, predictions, labels=[0, 1]
                ).tolist(),
            }, "validation_confusion_matrix.json")

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="pipeline",
                input_example=X_train.head(3),
                serialization_format="cloudpickle",
                pyfunc_predict_fn="predict_proba",
            )

            reloaded = mlflow.sklearn.load_model(model_info.model_uri)
            np.testing.assert_allclose(
                reloaded.predict_proba(X_val.head(10)),
                pipeline.predict_proba(X_val.head(10)),
            )
            mlflow.set_tag("reload_check", "passed")
            mlflow.log_dict({
                "model_uri": model_info.model_uri,
                "threshold": args.threshold,
                "positive_class": 1,
                "features": FEATURES,
            }, "serving_config.json")

            results.append({
                "model": name,
                **metrics,
                "model_uri": model_info.model_uri,
                "run_id": run.info.run_id,
            })

    ranking = pd.DataFrame(results).sort_values(
        "val_average_precision", ascending=False
    )
    print("\nValidation results:")
    print(ranking[[
        "model", "val_average_precision", "val_roc_auc",
        "val_precision", "val_recall", "val_f1",
    ]].to_string(index=False))

    best = ranking.iloc[0]
    print("\nBest candidate by validation average precision:")
    print(best["model"])
    print("Model URI:", best["model_uri"])
    print("Threshold:", args.threshold)
    print("Reserved test observations:", len(test_idx))
    print("Final test data have not been evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(ROOT / "data/raw/Telco-Customer-Churn.csv"),
    )
    parser.add_argument(
        "--mlflow_uri",
        default=f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    main(parser.parse_args())