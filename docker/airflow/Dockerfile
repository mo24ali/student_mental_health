FROM apache/airflow:3.4.0-python3.12

ENV PYTHONPATH=/opt/airflow/project PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

COPY --chown=airflow:0 docker/airflow/requirements.txt /tmp/ml-requirements.txt
RUN pip install --no-cache-dir "apache-airflow==3.4.0" -r /tmp/ml-requirements.txt \
    && pip check && rm /tmp/ml-requirements.txt
COPY --chown=airflow:0 src/ /opt/airflow/project/src/
COPY --chown=airflow:0 docker/airflow/init.sh /opt/airflow/init.sh
RUN mkdir -p /opt/airflow/artifacts /opt/airflow/auth /opt/airflow/logs \
    && chmod +x /opt/airflow/init.sh
USER airflow
