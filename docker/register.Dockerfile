# Minimal MLflow client image for registering the bundled model at Compose startup.
FROM ghcr.io/mlflow/mlflow:v3.5.0
WORKDIR /app
ENTRYPOINT ["python3"]
