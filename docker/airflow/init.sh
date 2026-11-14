#!/usr/bin/env bash
set -euo pipefail

airflow db migrate
umask 077
python - <<'PY'
import json
import os
from pathlib import Path

passwords = Path(os.environ["AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_PASSWORDS_FILE"])
passwords.parent.mkdir(parents=True, exist_ok=True)
if not passwords.exists():
    passwords.write_text(json.dumps({"admin": os.environ["AIRFLOW_ADMIN_PASSWORD"]}) + "\n")
    passwords.chmod(0o600)
PY
