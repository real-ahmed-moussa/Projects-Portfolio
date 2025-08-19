FROM python:3.10.8-slim

WORKDIR /app

RUN pip install mlflow

VOLUME ["/mlruns"]

EXPOSE 5000

CMD ["mlflow", "ui", "--backend-store-uri", "file:/mlflow/mlruns", "--host", "0.0.0.0", "--port", "5000"]
