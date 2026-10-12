# Telco Churn MLOps on AWS

Customer churn prediction using scikit-learn pipelines, XGBoost, MLflow, FastAPI, Docker, GitHub Actions, and Amazon EC2.

Repository: [Pen20/telco-churn-mlops-aws](https://github.com/Pen20/telco-churn-mlops-aws)

This project adapts an existing Telco churn implementation and documents the machine learning pipeline, API, automated checks, and AWS deployment verified in this repository.

## Project status

- Training and validation completed.
- Decision threshold selected using validation data.
- Selected model evaluated on a reserved test set.
- Preprocessing and model exported as one pipeline.
- FastAPI predictions and input validation tested.
- Docker container tested locally.
- Container build and API checks passed in GitHub Actions.
- Container deployed and tested on Amazon EC2.
- Remote access from Windows verified through an SSH tunnel.

The EC2 instance is stopped between demonstrations. There is currently no public API URL or automated AWS deployment.

## What this project demonstrates

- Reproducible train, validation, and test splits.
- Preprocessing fitted exclusively on training data.
- Comparison of logistic regression and XGBoost.
- Experiment tracking and model logging with MLflow.
- Decision threshold selection for an imbalanced classification task.
- Export and reload checks for the complete prediction pipeline.
- An API with validated request and response schemas.
- Docker packaging with a health check.
- Automated container and API checks in GitHub Actions.
- Manual deployment to AWS EC2 with access through SSH forwarding.

## Dataset

The project uses the IBM Telco Customer Churn sample dataset.

The dataset contains:

- 7,043 customer records.
- 21 columns.
- 19 input features after excluding `customerID` and `Churn`.
- 5,174 customers labelled `No`.
- 1,869 customers labelled `Yes`.

The target is `Churn`, encoded as:

- `0`: no churn.
- `1`: churn.

The raw dataset is stored locally at:

```text
data/raw/Telco-Customer-Churn.csv
```

Raw data are excluded from Git.

To download the dataset in PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path data/raw | Out-Null

Invoke-WebRequest `
    -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" `
    -OutFile "data/raw/Telco-Customer-Churn.csv"
```

In Linux:

```bash
mkdir -p data/raw

curl --fail --location \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv \
  --output data/raw/Telco-Customer-Churn.csv
```

## Data preparation

The training script checks required columns, customer identifiers, target labels, and input values.

Numeric features are:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
```

Categorical features are:

```text
gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

Blank numeric values are converted to missing values. Missing numeric values are handled by the fitted median imputer.

The preprocessing pipeline applies:

- Median imputation and standard scaling to numeric features.
- One-hot encoding to categorical features.

Preprocessing is fitted on the training set and reused for validation, testing, and inference.

## Data splits

The dataset is split with stratification and a random seed of `42`.

| Split | Observations | Purpose |
|---|---:|---|
| Training | 4,225 | Fit preprocessing and models |
| Validation | 1,409 | Compare models and select the threshold |
| Test | 1,409 | Evaluate the selected model and threshold |

Customer identifiers for each split are recorded in a split manifest.

The raw dataset SHA-256 hash is logged to help verify that subsequent steps use the same dataset.

## Model comparison

Two candidate pipelines were evaluated:

- Logistic regression.
- XGBoost.

Initial validation results used a decision threshold of `0.5`.

| Model | Average precision | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.644330 | 0.837920 | 0.657343 | 0.502674 | 0.569697 |
| Logistic regression | 0.642275 | 0.835988 | 0.660066 | 0.534759 | 0.590842 |

XGBoost was selected using validation average precision.

The average precision difference is small. These results describe this split and do not establish a statistically significant advantage for XGBoost.

## Decision threshold

The selected model produces a probability of churn.

The threshold determines whether that probability becomes a positive churn prediction:

```python
predicted_churn = churn_probability >= threshold
```

Thresholds were compared using validation data. The selected threshold maximized validation F1 within the evaluated grid.

| Selected threshold | Validation precision | Validation recall | Validation F1 |
|---|---:|---:|---:|
| 0.37 | 0.585812 | 0.684492 | 0.631319 |

Compared with `0.5`, the selected threshold increased recall and F1 while reducing precision.

F1 was used as a learning objective. A business deployment should choose a threshold using the costs and benefits of retention actions.

## Final test results

The selected XGBoost pipeline was evaluated on the reserved test set using the fixed threshold of `0.37`.

| Metric | Test result |
|---|---:|
| Average precision | 0.660882 |
| ROC AUC | 0.846847 |
| Precision | 0.582781 |
| Recall | 0.705882 |
| F1 | 0.638452 |

Confusion matrix:

| Actual / Predicted | No churn | Churn |
|---|---:|---:|
| No churn | 846 | 189 |
| Churn | 110 | 264 |

The model identified 264 of the 374 customers who churned in the test set.

The test set was evaluated after model and threshold selection. It should not be used to tune subsequent changes.

## MLflow tracking

The experiment is named:

```text
Telco Churn Adaptation
```

The pipeline records information such as:

- Model parameters.
- Validation metrics.
- Dataset hash.
- Split identifiers.
- Training time.
- Package versions.
- Saved model.
- Model reload verification.

Threshold selection and final test evaluation are also logged.

The selected model from the verified execution has the URI:

```text
models:/m-17aa3f43f3bd468d9496ee8909b5c319
```

This URI identifies a model in the local MLflow store. It is not a public download URL.

For deployment, the model is exported to `deployment/model`. The deployed API loads that directory directly and does not require the local MLflow database.

## Main files

| File | Purpose |
|---|---|
| `scripts/run_pipeline.py` | Validate data, create splits, train candidates, and log results |
| `scripts/select_threshold.py` | Select a decision threshold using validation data |
| `scripts/evaluate_test.py` | Evaluate the selected model on reserved test data |
| `scripts/export_model.py` | Export and verify the selected pipeline |
| `scripts/test_api_adapted.py` | Check API predictions and input validation |
| `src/app/main.py` | FastAPI application |
| `deployment/model/` | Exported model and serving configuration |
| `requirements-serving.txt` | Dependencies for the adapted application |
| `dockerfile` | Container image definition |
| `.github/workflows/ci.yml` | Automated container build and API checks |

The original `requirements.txt` may contain dependencies from the earlier implementation. Use `requirements-serving.txt` for the adapted workflow described here.

## Local setup

Clone the repository:

```bash
git clone https://github.com/Pen20/telco-churn-mlops-aws.git
cd telco-churn-mlops-aws
```

The verified local environment used Python `3.12.10`.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements-serving.txt
python -m pip check
```

### Linux

Use an installed Python 3.12 interpreter:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements-serving.txt
python -m pip check
```

Run commands from the repository root unless stated otherwise.

## Train the models

Download the dataset before training.

```bash
python scripts/run_pipeline.py
```

The script prints validation results and the selected candidate's model URI.

The default MLflow backend is the local SQLite database:

```text
mlflow.db
```

The database and generated tracking artifacts are excluded from Git.

### View experiments

Install the full MLflow package if the `mlflow ui` command is unavailable:

```bash
python -m pip install mlflow==3.17.0
```

Then run:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

Keep this terminal open while using the dashboard.

The dashboard is local to the machine running MLflow.

## Select, evaluate, and export

After a new training execution, update `MODEL_URI` in `scripts/select_threshold.py` to the model URI printed by the training script.

Select the threshold:

```bash
python scripts/select_threshold.py
```

Evaluate the selected model and threshold:

```bash
python scripts/evaluate_test.py
```

Export the selected pipeline:

```bash
python scripts/export_model.py
```

The threshold selection script writes:

```text
artifacts/selected_config.json
```

The export contains:

```text
deployment/model/
├── MLmodel
├── model.pkl
├── conda.yaml
├── python_env.yaml
├── requirements.txt
└── serving_config.json
```

The export script checks that predictions from the exported pipeline match the selected model.

It refuses to overwrite an existing export. Preserve an existing model before creating a replacement.

The bundled export can be used for inference without downloading the dataset or rerunning training.

## Run the API locally

```bash
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000
```

Endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Service information |
| `/health` | GET | Model readiness and identity |
| `/predict` | POST | Predict churn |
| `/docs` | GET | Interactive API documentation |

Open:

```text
http://127.0.0.1:8000/docs
```

### Health response

```json
{
  "status": "ready",
  "model_uri": "models:/m-17aa3f43f3bd468d9496ee8909b5c319",
  "threshold": 0.37
}
```

The model URI in this response identifies the source of the exported model.

### Prediction request

```json
{
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
  "TotalCharges": 350.75
}
```

### Verified prediction response

```json
{
  "churn_probability": 0.6109859943389893,
  "threshold": 0.37,
  "predicted_churn": true,
  "model_uri": "models:/m-17aa3f43f3bd468d9496ee8909b5c319"
}
```

`predicted_churn` is `true` because the probability exceeds the threshold of `0.37`.

`TotalCharges` is required in the request but may be `null`. The fitted numeric imputer handles this missing value.

Invalid categories, negative values, unexpected fields, and missing required fields are rejected.

## API checks

With the API running locally:

```bash
python scripts/test_api_adapted.py
```

The checks verify:

- Health status and model identity.
- Agreement between API predictions and the exported pipeline.
- Predictions for alternative valid categories.
- Handling of missing `TotalCharges`.
- Rejection of invalid requests.

Verified output:

```text
PASS: health and model identity
PASS: API prediction matches exported pipeline
PASS: alternative categories match exported pipeline
PASS: missing TotalCharges handled by fitted imputer
PASS: negative, unknown, extra and missing inputs rejected

All API checks passed.
```

## Docker

Docker must be running before building or starting the container.

Build the image:

```bash
docker build -f dockerfile -t telco-churn-api:local .
```

Run it on local port `8001`:

```bash
docker run -d --name telco-churn-api -p 127.0.0.1:8001:8000 telco-churn-api:local
```

Inspect logs and health:

```bash
docker logs telco-churn-api
docker inspect --format '{{.State.Health.Status}}' telco-churn-api
```

The verified container reported:

```text
healthy
```

### Test the container from PowerShell

```powershell
$env:API_BASE_URL = "http://127.0.0.1:8001"

python scripts/test_api_adapted.py

Remove-Item Env:API_BASE_URL
```

### Test the container from Linux

```bash
API_BASE_URL=http://127.0.0.1:8001 python scripts/test_api_adapted.py
```

Stop and remove the local container when finished:

```bash
docker stop telco-churn-api
docker rm telco-churn-api
```

The image runs the application as a non-root user and includes a health check.

The exported model is bundled into the image.

## GitHub Actions

The workflow in `.github/workflows/ci.yml`:

1. Checks out the repository.
2. Builds the Docker image.
3. Starts a test container.
4. Waits for a healthy container.
5. Runs the API checks inside the container.
6. Prints logs and removes the test container.

The verified workflow passed all API checks.

This workflow validates the bundled serving model. It does not retrain the model, publish an image to a registry, or deploy to AWS.

## AWS EC2 deployment

The container was deployed and tested on an Ubuntu EC2 instance.

The verified instance used an x86-64 architecture. Deployment was performed manually.

### Install Docker and Git

On the EC2 instance:

```bash
sudo apt update
sudo apt install -y docker.io git

sudo systemctl enable --now docker
sudo docker version
```

### Clone and build

```bash
git clone https://github.com/Pen20/telco-churn-mlops-aws.git
cd telco-churn-mlops-aws

sudo docker build -f dockerfile -t telco-churn-api:aws .
```

### Start the container

```bash
sudo docker run -d \
  --name telco-churn-api \
  --restart unless-stopped \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  -p 127.0.0.1:8000:8000 \
  telco-churn-api:aws
```

The API port is bound to the EC2 instance's loopback interface.

### Verify readiness

```bash
sudo docker logs telco-churn-api

curl --fail http://127.0.0.1:8000/health

sudo docker inspect --format '{{.State.Health.Status}}' telco-churn-api
```

### Run API checks inside the AWS container

```bash
sudo docker exec -u 0 telco-churn-api mkdir -p /app/scripts

sudo docker cp scripts/test_api_adapted.py \
  telco-churn-api:/app/scripts/test_api_adapted.py

sudo docker exec \
  -e API_BASE_URL=http://127.0.0.1:8000 \
  telco-churn-api \
  python /app/scripts/test_api_adapted.py
```

All API checks passed on the EC2 deployment.

## Access through an SSH tunnel

An SSH tunnel forwards a port on the local computer to the API running on EC2.

The verified forwarding configuration was:

```text
Local address: 127.0.0.1:8002
Remote destination: 127.0.0.1:8000
```

In Windows PowerShell, replace the key path and public IP with your own values:

```powershell
ssh `
    -i "C:\path\to\your-key.pem" `
    -o IdentitiesOnly=yes `
    -o ExitOnForwardFailure=yes `
    -N `
    -L 127.0.0.1:8002:127.0.0.1:8000 `
    ubuntu@YOUR_EC2_PUBLIC_IP
```

Keep this terminal open while using the tunnel.

Open the API documentation on your local computer:

```text
http://127.0.0.1:8002/docs
```

In another PowerShell terminal, activate the project environment and run:

```powershell
$env:API_BASE_URL = "http://127.0.0.1:8002"

python scripts/test_api_adapted.py

Remove-Item Env:API_BASE_URL
```

All API checks passed from Windows through the SSH tunnel.

Press `Ctrl+C` in the tunnel terminal to close the tunnel.

Private SSH keys must remain outside the repository.

## Stop and resume the demonstration

When finished:

1. Close the SSH tunnel.
2. Stop the EC2 instance in the AWS console.
3. Confirm that its state is `Stopped`.

Stopping the instance does not delete its storage. Associated resources can still incur charges.

To resume:

1. Start the EC2 instance.
2. Check its current public IP address.
3. Connect to the instance.
4. Verify the container and health endpoint.
5. Recreate the SSH tunnel using the current IP address.

On the instance:

```bash
sudo docker ps
curl --fail http://127.0.0.1:8000/health
```

The container uses the `unless-stopped` restart policy. If necessary, start it explicitly:

```bash
sudo docker start telco-churn-api
```

## Limitations and future improvements

- Results are based on one stratified train/validation/test split.
- The decision threshold optimizes validation F1 rather than a business cost function.
- Probability calibration has not been evaluated.
- API validation does not implement every possible cross-field business rule.
- AWS deployment is manual.
- The application has no public HTTPS endpoint or application authentication.
- Production monitoring, drift detection, and automated retraining are not implemented.
- Direct serving dependencies are pinned, but a complete dependency lockfile is not provided.
- Earlier project files remain in the repository and can be reviewed for cleanup.

These are possible next steps rather than completed features.

## Model loading

The bundled model uses Python serialization. Load model artifacts only from sources you trust.

The export and API checks verify prediction consistency for the tested inputs. They do not establish model reliability for every deployment population.

## Repository

Source code, workflow, and documentation:

[Pen20/telco-churn-mlops-aws](https://github.com/Pen20/telco-churn-mlops-aws)