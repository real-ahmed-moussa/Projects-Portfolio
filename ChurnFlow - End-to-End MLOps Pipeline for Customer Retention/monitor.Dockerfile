FROM python:3.10.8-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY monitor.py retrain.py feature_engineering.py /app/

ENV APP_BASE=/app

EXPOSE 8001

CMD ["python3", "monitor.py"]
