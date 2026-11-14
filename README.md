# student_mental_health

## Docker platform (local development)

Docker Compose runs Streamlit and Airflow 3.4.0 (API server, scheduler, and DAG
processor) on a private bridge network. Airflow uses LocalExecutor and PostgreSQL
16 for metadata; no worker queue is needed. The dataset is mounted read-only into
Airflow. DAG runs write to a persistent `airflow-artifacts` volume, with a separate
directory per run. Streamlit reads `models/` and `reports/` from the host read-only;
training cannot replace its production model. The fitted preprocessing pipeline
is contained in `models/best_model.joblib`.

Prerequisites: Docker Engine with the Compose plugin, ports 8501 and 8080 free,
and the existing `data/raw/student_mental_health.csv` and
`models/best_model.joblib` files. Allow several GB of free disk space for images,
PostgreSQL, and training artifacts. On first setup, create `.env` from
`.env.example` and replace **all** `CHANGE_ME` values. Use unique values:
`POSTGRES_PASSWORD` (avoid URL-special characters, because it appears in a
SQLAlchemy connection URL), `AIRFLOW_ADMIN_PASSWORD`, and two secret keys.
Generate a valid Fernet key and a JWT key with:

```bash
python3 -c 'import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())'
python3 -c 'import secrets; print(secrets.token_hex(32))'
```

Keep `.env` private; it is Git-ignored. The database password and keys must stay
the same across restarts. To build and start everything (the one-shot `airflow-init`
service migrates the database and writes the initial admin password file):

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f
```

Open Streamlit at **http://localhost:8501** and Airflow at
**http://localhost:8080**; log in to Airflow as `admin` with the password from
`.env`. `airflow-init` should show `Exited (0)` after setup. In the Airflow UI,
find `student_mental_health_ml` and select **Trigger**. The DAG has no schedule;
it runs validation → preprocessing → model comparison → tuning → evaluation →
artifact verification only when triggered. Training can take several minutes.
Inspect task logs in the UI or run:

```bash
docker compose logs -f airflow-scheduler airflow-dag-processor airflow-apiserver
docker compose exec airflow-scheduler airflow dags list
docker compose exec airflow-dag-processor airflow dags list-import-errors --local
docker compose down
docker compose up -d
```

`down` keeps named volumes (PostgreSQL, artifacts, logs, auth); **do not** use
`down -v` unless you intend to delete them. After a run, find the staging path
`/opt/airflow/artifacts/<run-id-hash>/` in its `verify_artifacts` task log.
To promote a reviewed, successful run, copy its model to a candidate file on
the host, test it, then explicitly replace the production file. For example:

```bash
docker compose cp airflow-scheduler:/opt/airflow/artifacts/<run-id-hash>/models/best_model.joblib ./models/candidate.joblib
docker compose exec streamlit python -c 'from src.predict import get_model_info; print(get_model_info("/opt/project/models/candidate.joblib"))'
# Back up models/best_model.joblib before replacing it with the reviewed candidate.
mv models/candidate.joblib models/best_model.joblib
```

The next Streamlit prediction loads the promoted model. If you want the
performance page to reflect that run too, copy its reports from the same
artifact directory and review them before replacing the files in `reports/`.
Never load untrusted joblib artifacts. To verify a model with real inputs:
submit the Streamlit assessment form; the result includes its class and
estimated probabilities. If the model is missing, the assessment displays an
error rather than retraining.

Troubleshooting: `docker compose ps` and `docker compose logs <service>` show
startup errors; check `.env` keys and `docker compose config` if interpolation
fails. For database connectivity, inspect `docker compose logs postgres
airflow-init`. For missing DAGs, check the import-errors command above and
confirm `data/raw/student_mental_health.csv` exists. If Airflow's admin password
was changed in `.env` after first startup, the existing `airflow-auth` volume
still holds the original password; update that file explicitly or reset the
volume after backing up any required data. If ports are occupied, stop the local
`run_airflow.sh` instance before starting Compose.
