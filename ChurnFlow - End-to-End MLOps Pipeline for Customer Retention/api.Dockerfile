FROM python:3.10.8-slim

WORKDIR /app

# OS deps (if Evidently/psutil/etc. need them, add here)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py retrain.py feature_engineering.py monitor.py data_ingestion.py train_model.py /app/

# We mount data/, models/, logs/, mlruns/ as volumes at runtime
ENV APP_BASE=/app MONITOR_URL=http://monitor:8001 MLFLOW_TRACKING_URI=http://mlflow:5000

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
