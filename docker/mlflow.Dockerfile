# syntax=docker/dockerfile:1
# MLflow tracking server: official image plus the PostgreSQL driver.
FROM ghcr.io/mlflow/mlflow:v3.5.0
RUN --mount=type=secret,id=extra_ca \
    if [ -s /run/secrets/extra_ca ]; then \
        cat /etc/ssl/certs/ca-certificates.crt /run/secrets/extra_ca > /tmp/ca.pem; export PIP_CERT=/tmp/ca.pem; \
    fi; \
    pip install --no-cache-dir psycopg2-binary==2.9.13 && rm -f /tmp/ca.pem
