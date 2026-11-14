#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

export PATH="$PROJECT_ROOT/.airflow-venv/bin:$PATH"
export AIRFLOW_HOME="$PROJECT_ROOT/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$PROJECT_ROOT/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS=admin:admin
export AIRFLOW__CORE__PARALLELISM=2
export AIRFLOW__API__HOST=127.0.0.1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Airflow 3's SimpleAuthManager reads this local, ignored JSON file on startup.
umask 077
python -c 'import json, os; from pathlib import Path; p = Path(os.environ["AIRFLOW_HOME"]) / "simple_auth_manager_passwords.json.generated"; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps({"admin": "admin"}) + "\n"); p.chmod(0o600)'

exec airflow standalone
