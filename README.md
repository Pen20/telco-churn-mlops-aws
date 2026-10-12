## \# Telco Churn MLOps on AWS

## 

## Customer churn prediction using scikit-learn pipelines, XGBoost,

## MLflow experiment tracking, FastAPI, Docker, GitHub Actions, and

## manual deployment to Amazon EC2.

## 

## This project adapts an existing Telco churn implementation.

## It documents the changes and deployment verified in

## \[Pen20/telco-churn-mlops-aws](https://github.com/Pen20/telco-churn-mlops-aws).

## \## Current status

## 

## \- Training and validation completed.

## \- Decision threshold selected using validation data.

## \- Selected model evaluated on a reserved test set.

## \- Complete preprocessing and model pipeline exported.

## \- FastAPI predictions and input validation tested.

## \- Docker image tested locally and in GitHub Actions.

## \- Container deployed and tested on Amazon EC2.

## \- Access from Windows verified through an SSH tunnel.

## 

## The EC2 instance is stopped between demonstrations.

## There is currently no public API URL or automated AWS deployment.

## 

## \## Changes made in this adaptation

## 

## \- Replaced separate training and serving feature transformations with

## &#x20; one fitted preprocessing pipeline.

## \- Added stratified training, validation, and test splits.

## \- Compared logistic regression and XGBoost.

## \- Added validation-based threshold selection.

## \- Recorded dataset hashes and split identifiers in MLflow.

## \- Exported a model that can run without the MLflow tracking server.

## \- Replaced the API with validated request fields and probability outputs.

## \- Added prediction consistency and invalid-input checks.

## \- Updated Docker packaging to use the exported model.

## \- Replaced the original Docker Hub workflow with container build and tests.

## \- Deployed the application manually on EC2 and verified remote predictions.

## 

## \## Dataset

## 

## The project uses the IBM Telco Customer Churn sample dataset:

## 

## \- 7,043 customers.

## \- 21 columns, including customer ID and target.

## \- Target: `Churn`, mapped from `No` / `Yes` to `0` / `1`.

## \- 5,174 customers labelled `No`.

## \- 1,869 customers labelled `Yes`.

## 

## Source:

## \[IBM dataset repository](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv).

## 

## Place the CSV at:

## 

## ```text

## data/raw/Telco-Customer-Churn.csv

## ```

## 

## Raw data are excluded from Git. The dataset source is credited separately

## from the source code.

## 

## \## Modelling approach

## 

## The data are split using stratification and random seed 42:

## 

## | Partition | Observations | Purpose |

## |---|---:|---|

## | Training | 4,225 | Fit preprocessing and classifiers |

## | Validation | 1,409 | Compare models and select the threshold |

## | Test | 1,409 | Evaluate the frozen model and threshold |

## 

## Customer IDs are excluded from model inputs.

## 

## Numeric features use median imputation and standardisation.

## Categorical features use a fitted `OneHotEncoder`.

## 

## Preprocessing is fitted only on the training partition.

## The exported pipeline applies the same transformations during inference.

## 

## \## Validation results

## 

## The initial comparison used a decision threshold of 0.50.

## 

## | Model | Average precision | ROC AUC | Precision | Recall | F1 |

## |---|---:|---:|---:|---:|---:|

## | XGBoost | 0.644330 | 0.837920 | 0.657343 | 0.502674 | 0.569697 |

## | Logistic regression | 0.642275 | 0.835988 | 0.660066 | 0.534759 | 0.590842 |

## 

## XGBoost was selected using validation average precision.

## Its advantage was small and does not establish statistical superiority.

## 

## For XGBoost, thresholds from 0.05 to 0.95 were compared in steps of 0.01.

## The selected threshold maximised validation F1:

## 

## | Selected threshold | Precision | Recall | F1 |

## |---:|---:|---:|---:|

## | 0.37 | 0.585812 | 0.684492 | 0.631319 |

## 

## F1 was used as a learning-project objective.

## A business application should choose a threshold using intervention costs,

## missed-churn costs, and operational capacity.

## 

## \## Final test results

## 

## The selected XGBoost pipeline and threshold 0.37 were evaluated on

## 1,409 reserved test observations.

## 

## | Metric | Value |

## |---|---:|

## | Average precision | 0.660882 |

## | ROC AUC | 0.846847 |

## | Precision | 0.582781 |

## | Recall | 0.705882 |

## | F1 | 0.638452 |

## 

## Confusion matrix:

## 

## | Actual / Predicted | No churn | Churn |

## |---|---:|---:|

## | No churn | 846 | 189 |

## | Churn | 110 | 264 |

## 

## The model identified 264 of 374 churners.

## These results describe one held-out split of this sample dataset.

## They do not establish performance on another telecom population or

## the effectiveness of a retention intervention.

## 

## \## Main files

## 

## | File | Purpose |

## |---|---|

## | `scripts/run\_pipeline.py` | Data validation, splitting, model comparison, MLflow logging |

## | `scripts/select\_threshold.py` | Threshold selection on validation data |

## | `scripts/evaluate\_test.py` | Final evaluation on reserved test data |

## | `scripts/export\_model.py` | Export and verify the selected pipeline |

## | `scripts/test\_api\_adapted.py` | API consistency and input validation checks |

## | `src/app/main.py` | Adapted FastAPI application |

## | `deployment/model/` | Exported pipeline and serving configuration |

## | `requirements-serving.txt` | Pinned serving dependencies |

## | `dockerfile` | Linux container configuration |

## | `.github/workflows/ci.yml` | Container build and automated API checks |

## 

## Other files inherited from the original repository may remain.

## The commands below use the adapted implementation.

## 

## \## Local setup: Windows PowerShell

## 

## The local environment used Python 3.12.10.

## 

## ```powershell

## py -3.12 -m venv .venv

## .\\.venv\\Scripts\\Activate.ps1

## python -m pip install -r requirements-serving.txt

## python -m pip check

## ```

## 

## The serving requirements also contain the packages needed by the current

## adapted training scripts. The original `requirements.txt` belongs to

## the inherited implementation.

## 

## \## Training and MLflow

## 

## With the full dataset in `data/raw`:

## 

## ```powershell

## python scripts/run\_pipeline.py

## ```

## 

## The pipeline creates a local SQLite tracking database at `mlflow.db`

## and uses the experiment `Telco Churn Adaptation`.

## 

## Open the dashboard:

## 

## ```powershell

## mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

## ```

## 

## Visit http://127.0.0.1:5000.

## 

## Runs record model parameters, validation metrics, the dataset SHA-256,

## split identifiers, confusion matrices, and saved pipelines.

## 

## \### Repeating model selection

## 

## Each training execution creates new model identifiers.

## Before rerunning threshold selection, update `MODEL\_URI` in

## `scripts/select\_threshold.py` using the chosen model URI printed by training.

## 

## Then run:

## 

## ```powershell

## python scripts/select\_threshold.py

## python scripts/evaluate\_test.py

## python scripts/export\_model.py

## ```

## 

## Threshold selection writes `artifacts/selected\_config.json`.

## Evaluation and export use this configuration.

## 

## Freeze the model and threshold before evaluating the test set.

## Do not use test results to tune them further.

## 

## Export refuses to overwrite an existing `deployment/model` directory.

## Preserve or rename the previous export before producing another version.

## 

## The local MLflow database and tracking artifacts are excluded from Git.

## The bundled deployment model can run independently of them.

## 

## \## Run the local API

## 

## ```powershell

## python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000

## ```

## 

## Endpoints:

## 

## \- `GET /health`: model readiness, identity, and threshold.

## \- `POST /predict`: churn probability and threshold-based classification.

## \- `/docs`: interactive API documentation.

## 

## The model directory defaults to `deployment/model`.

## It can be overridden with the `MODEL\_DIR` environment variable.

## 

## Requests contain 19 customer features.

## `TotalCharges` is required but may be `null`; the fitted imputer handles it.

## Tested invalid categories, negative values, extra fields, and missing

## required fields return HTTP 422.

## 

## \## Example prediction

## 

## ```json

## {

## &#x20; "gender": "Male",

## &#x20; "SeniorCitizen": 0,

## &#x20; "Partner": "Yes",

## &#x20; "Dependents": "No",

## &#x20; "tenure": 5,

## &#x20; "PhoneService": "Yes",

## &#x20; "MultipleLines": "No",

## &#x20; "InternetService": "Fiber optic",

## &#x20; "OnlineSecurity": "No",

## &#x20; "OnlineBackup": "Yes",

## &#x20; "DeviceProtection": "No",

## &#x20; "TechSupport": "No",

## &#x20; "StreamingTV": "Yes",

## &#x20; "StreamingMovies": "Yes",

## &#x20; "Contract": "Month-to-month",

## &#x20; "PaperlessBilling": "Yes",

## &#x20; "PaymentMethod": "Electronic check",

## &#x20; "MonthlyCharges": 70.35,

## &#x20; "TotalCharges": 350.75

## }

## ```

## 

## For the bundled model:

## 

## ```json

## {

## &#x20; "churn\_probability": 0.6109859943389893,

## &#x20; "threshold": 0.37,

## &#x20; "predicted\_churn": true,

## &#x20; "model\_uri": "models:/m-17aa3f43f3bd468d9496ee8909b5c319"

## }

## ```

## 

## The probability is a model estimate, not a certainty.

## Probability calibration has not been independently evaluated.

## 

## \## API tests

## 

## With the local API running:

## 

## ```powershell

## python scripts/test\_api\_adapted.py

## ```

## 

## Checks cover:

## 

## \- Health and model identity.

## \- Predictions matching the exported pipeline.

## \- Alternative categorical inputs.

## \- Missing `TotalCharges`.

## \- Rejection of tested invalid requests.

## 

## \## Docker

## 

## Build:

## 

## ```powershell

## docker build -f dockerfile -t telco-churn-api:local .

## ```

## 

## Run:

## 

## ```powershell

## docker run -d --name telco-churn-api -p 127.0.0.1:8001:8000 telco-churn-api:local

## ```

## 

## Visit http://127.0.0.1:8001/docs.

## 

## Test the container:

## 

## ```powershell

## $env:API\_BASE\_URL = "http://127.0.0.1:8001"

## python scripts/test\_api\_adapted.py

## Remove-Item Env:API\_BASE\_URL

## docker inspect --format '{{.State.Health.Status}}' telco-churn-api

## ```

## 

## The image runs as a non-root user and includes a health check.

## Only the adapted API, serving dependencies, and exported model are copied

## into the image.

## 

## Dependencies directly listed in `requirements-serving.txt` are pinned.

## Transitive dependencies and the base-image tag are not fully locked.

## 

## \## Continuous integration

## 

## GitHub Actions builds the Docker image, starts a test container,

## waits for a healthy API, and executes the API checks inside the container.

## 

## The workflow requires no Docker Hub or AWS credentials.

## It tests the bundled model; it does not retrain models or deploy to AWS.

## 

## \## Manual AWS EC2 deployment

## 

## The application was verified on an x86\_64 `t3.small` instance

## in `us-east-1`, running Ubuntu 26.04 LTS.

## 

## Install Docker and Git:

## 

## ```bash

## sudo apt update

## sudo apt install -y docker.io git

## sudo systemctl enable --now docker

## ```

## 

## Clone and build:

## 

## ```bash

## git clone https://github.com/Pen20/telco-churn-mlops-aws.git

## cd telco-churn-mlops-aws

## sudo docker build -f dockerfile -t telco-churn-api:aws .

## ```

## 

## Run:

## 

## ```bash

## sudo docker run -d \\

## &#x20; --name telco-churn-api \\

## &#x20; --restart unless-stopped \\

## &#x20; --log-opt max-size=10m \\

## &#x20; --log-opt max-file=3 \\

## &#x20; -p 127.0.0.1:8000:8000 \\

## &#x20; telco-churn-api:aws

## ```

## 

## Verify:

## 

## ```bash

## curl --fail --max-time 10 http://127.0.0.1:8000/health

## sudo docker inspect --format '{{.State.Health.Status}}' telco-churn-api

## ```

## 

## Run API tests on EC2:

## 

## ```bash

## sudo docker exec -u 0 telco-churn-api mkdir -p /app/scripts

## sudo docker cp scripts/test\_api\_adapted.py \\

## &#x20; telco-churn-api:/app/scripts/test\_api\_adapted.py

## sudo docker exec \\

## &#x20; -e API\_BASE\_URL=http://127.0.0.1:8000 \\

## &#x20; telco-churn-api \\

## &#x20; python /app/scripts/test\_api\_adapted.py

## ```

## 

## The model loads from the container filesystem.

## An MLflow tracking server is not required on EC2.

## 

## \## Access through an SSH tunnel

## 

## The EC2 API is bound to the instance's loopback interface.

## From Windows, replace the key path and public IP below:

## 

## ```powershell

## ssh -i "C:\\path\\to\\key.pem" -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes -N -L 127.0.0.1:8002:127.0.0.1:8000 ubuntu@PUBLIC\_IP

## ```

## 

## Keep the terminal open and visit http://127.0.0.1:8002/docs.

## 

## From another local terminal:

## 

## ```powershell

## $env:API\_BASE\_URL = "http://127.0.0.1:8002"

## python scripts/test\_api\_adapted.py

## Remove-Item Env:API\_BASE\_URL

## ```

## 

## This connection was tested successfully from Windows to EC2.

## 

## The localhost address is usable only on the computer running the tunnel.

## It is not a public demonstration URL.

## SSH access requires the correct private key and permitted network access.

## 

## \## Stopping and resuming

## 

## Close the SSH tunnel with Ctrl+C.

## Closing the tunnel does not stop EC2.

## 

## Stop the instance from the AWS EC2 console between demonstrations.

## Storage and other retained resources may continue consuming credits.

## 

## After starting the instance again:

## 

## 1\. Check its current public IPv4 address.

## 2\. Verify the container and `/health`.

## 3\. Reopen the SSH tunnel using the current address.

## 

## The container is configured with `--restart unless-stopped`.

## 

## \## Limitations and next steps

## 

## \- Manual AWS deployment; continuous deployment is not implemented.

## \- No public HTTPS endpoint or application-level authentication.

## \- No model drift monitoring or automatic retraining.

## \- No independent probability calibration evaluation.

## \- One held-out dataset split; no external population validation.

## \- Request validation does not yet enforce all cross-field service rules.

## \- Training environment and dependencies need a complete lockfile.

## \- Legacy files and documentation need further cleanup.

## 

## \## Repository and dataset

## 

## Project repository:

## \[Pen20/telco-churn-mlops-aws](https://github.com/Pen20/telco-churn-mlops-aws).

## 

## Dataset:

## \[IBM — Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d).

## 

## The dataset is an IBM sample dataset.

## Applicable third-party licence notices are retained separately.

## 

