FROM python:3.12.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/project STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /opt/project
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --home-dir /home/app --create-home app
COPY docker/streamlit/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt
COPY --chown=app:app app/ app/
COPY --chown=app:app src/ src/
USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c 'import urllib.request; urllib.request.urlopen("http://localhost:8501/_stcore/health", timeout=3)' || exit 1
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.headless=true"]
