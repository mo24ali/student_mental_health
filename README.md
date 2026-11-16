# Student Mental Health Risk Prediction

A research dashboard and machine-learning pipeline that classify a student profile into **Low**, **Medium**, or **High** mental-health risk categories. The Streamlit app accepts 15 academic, lifestyle, and support indicators, then displays the saved model's predicted category and estimated class probabilities. Apache Airflow can run the training workflow on demand.

> **For education and research only.** This model is not a clinical diagnosis or a validated mental-health screening tool. Its estimated probabilities are not clinically validated.

## What is included

- A Streamlit dashboard with an assessment form, model-performance reports, and project information.
- Input validation, missing-value imputation, feature encoding, and scaling inside the fitted model pipeline.
- Comparison and tuning of Logistic Regression, Random Forest, and XGBoost using five-fold stratified cross-validation on training data; final evaluation on a held-out test set.
- A manually triggered Airflow DAG that saves each run's outputs separately from the model served by the dashboard.
- Docker Compose setup for Streamlit, Airflow 3.4.0, and PostgreSQL 16, as well as Python-only workflows.

The included dataset has 2,000 records. The target is `Mental_Health_Risk`; `Student_ID` is excluded from the features. The split is stratified (80% training, 20% testing, random state 42). Preprocessing is fitted within the training pipelines, rather than on the test set.

## Quick start: Docker Compose

You need Docker Engine with the Compose plugin, free local ports **8501** and **8080**, and enough disk space for the images, database, and training artifacts. Run these commands from the repository root:

```bash
cp .env.example .env
```

Replace every `CHANGE_ME` value in `.env` with a different secret. Use a password without URL-special characters for `POSTGRES_PASSWORD` (it is embedded in a database URL). Generate an Airflow Fernet key and JWT secret, for example:

```bash
python3 -c 'import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())'
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

Use the first output for `AIRFLOW_FERNET_KEY` and the second for `AIRFLOW_SECRET_KEY`. Keep `.env` private and retain its values across restarts.

```bash
docker compose up -d --build
docker compose ps
```

| Service | URL | What to do |
| --- | --- | --- |
| Streamlit | http://localhost:8501 | Open **Risk Prediction**, complete the form, and submit it. **Model Performance** shows the saved evaluation reports. |
| Airflow | http://localhost:8080 | Sign in as `admin` using `AIRFLOW_ADMIN_PASSWORD` from `.env`. Find `student_mental_health_ml` and select **Trigger** to start a training run. |

The one-time `airflow-init` container should exit successfully (`Exited (0)`). The DAG has **no schedule**: opening either UI does not start training. Its tasks run validation → data preparation → model comparison → tuning → evaluation → artifact verification. Tuning can take several minutes. The dashboard reads `models/` and `reports/` from the host; training results are stored in the separate `airflow-artifacts` volume and do not automatically change predictions.

To inspect or stop the stack:

```bash
docker compose logs -f airflow-scheduler airflow-dag-processor airflow-apiserver
docker compose down
```

`docker compose down` keeps the database, authentication, logs, and artifact volumes; `docker compose down -v` removes them.

## Run locally with Python

Use Python **3.12** and run commands from the repository root. The repository already contains the raw dataset, saved model, and evaluation reports, so you can open the dashboard without retraining:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Visit http://localhost:8501. To regenerate the checked-in split, model, and reports **in place**, run the following in order (the comparison and tuning steps can take several minutes):

```bash
python -m src.preprocessing
python -m src.train
python -m src.tune
python -m src.evaluate
```

These commands write `data/processed/`, `models/best_model.joblib`, and `reports/`. The saved model is a fitted preprocessing-plus-classifier pipeline; inference does not need a separate preprocessing command. Load only trusted joblib files.

### Optional: Airflow without Docker

For local manual DAG runs, install Airflow into a **separate** Python 3.12 environment. On Ubuntu, for example:

```bash
python3.12 -m venv .airflow-venv
source .airflow-venv/bin/activate
AIRFLOW_VERSION=3.4.0
PYTHON_VERSION=3.12
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
python -m pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
python -m pip install 'numpy>=1.26,<3' 'pandas>=2.2,<3' 'scikit-learn>=1.5,<2' xgboost 'joblib>=1.4,<2' 'matplotlib>=3.8,<4'
python -m pip check
bash run_airflow.sh
```

Use the Airflow constraints only for the first install: the ML packages require pandas <3. The script starts `airflow standalone` with a local SQLite database under `.airflow/`. Open http://localhost:8080 and sign in as `admin` / `admin`, then trigger `student_mental_health_ml`. Runs save their outputs under `.airflow/artifacts/<run-id-hash>/`, without overwriting the dashboard's model or reports. Stop the server with **Ctrl+C**. Do not start this server and the Docker Airflow server on port 8080 at the same time.

## Training results and artifacts

The current checked-in `reports/best_model_metadata.json` identifies **Logistic Regression** as the selected model. The saved held-out test metrics in `reports/metrics/test_metrics.json` are **75.75% accuracy** and **0.7578 macro F1** on **400 test records**. These are results for the included model and dataset, not a guarantee for new students.

| Path | Purpose |
| --- | --- |
| `data/raw/student_mental_health.csv` | Input dataset for training |
| `data/processed/{train,test}.csv` | Saved untransformed train/test splits from the Python workflow |
| `models/best_model.joblib` | Fitted pipeline used for dashboard predictions |
| `reports/best_model_metadata.json` | Selected model, parameters, and summary scores |
| `reports/metrics/` | Cross-validation comparison, tuning results, test metrics, classification report, and confusion matrix |
| `reports/figures/confusion_matrix.png` | Test-set confusion matrix figure |
| `src/` | Validation, preprocessing, training, tuning, evaluation, and inference code |
| `dags/mental_health_training_pipeline.py` | Manual Airflow training workflow |
| `notebooks/` | Data audit, preprocessing, experiments, and inference exploration |
| `tests/` | Workflow and dashboard tests |

To use a new **Docker Airflow** run for inference, first review its metrics and model. The `verify_artifacts` task log prints its staging directory (`/opt/airflow/artifacts/<run-id-hash>`). In the commands below, replace `<run-id-hash>` with the actual directory name (including removing the angle brackets). Copy a candidate to the host, check it, then replace the served model deliberately:

```bash
docker compose cp airflow-scheduler:/opt/airflow/artifacts/<run-id-hash>/models/best_model.joblib ./models/candidate.joblib
docker compose exec streamlit python -c 'from src.predict import get_model_info; print(get_model_info("/opt/project/models/candidate.joblib"))'
# Back up models/best_model.joblib before replacing it with the reviewed candidate.
mv models/candidate.joblib models/best_model.joblib
```

The next assessment uses the promoted model. If you also want the performance page to show that run, copy and review its files from the staged `reports/` directory before replacing the host reports. Local Airflow runs use the same review-and-copy process from `.airflow/artifacts/`.

## Tests and troubleshooting

With the main Python environment active:

```bash
python -m pytest -q
```

- **Missing model or reports:** Check that `models/best_model.joblib` and `reports/metrics/` exist; the dashboard does not train a replacement automatically.
- **Docker startup errors:** Run `docker compose ps` and `docker compose logs <service>`; check `.env` and `docker compose config` if variable interpolation fails. For database problems, inspect `docker compose logs postgres airflow-init`.
- **DAG not visible:** Check `docker compose exec airflow-dag-processor airflow dags list-import-errors --local` and confirm the raw CSV exists. For local Airflow, inspect the task logs in the UI.
- **Airflow password changed in `.env`:** An existing `airflow-auth` volume retains the initial admin password file; changing `.env` alone does not update it.
