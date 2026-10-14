# syntax=docker/dockerfile:1
# Training, evaluation and inference. Code, data and outputs are mounted at run time, so only a change of
# requirements.txt needs a rebuild.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg PYTHONPATH=/app/src
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 git make \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
# EXTRA_CA_CERT (optional, see .env.example) is only needed behind a TLS-intercepting proxy.
RUN --mount=type=secret,id=extra_ca \
    if [ -s /run/secrets/extra_ca ]; then \
        cat /etc/ssl/certs/ca-certificates.crt /run/secrets/extra_ca > /tmp/ca.pem; export PIP_CERT=/tmp/ca.pem; \
    fi; \
    pip install -r /tmp/requirements.txt && pip freeze > /opt/requirements.lock && rm -f /tmp/ca.pem
WORKDIR /app
RUN git config --global --add safe.directory /app
