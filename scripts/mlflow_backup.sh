#!/usr/bin/env bash
# Back up or restore the MLflow tracking database and artifact store of this compose project.
#   scripts/mlflow_backup.sh backup    -> mlflow_backup/mlflow_db.dump + mlflow_backup/mlartifacts.tar.gz
#   scripts/mlflow_backup.sh restore   -> replaces the current tracking data with that backup
set -euo pipefail
cd "$(dirname "$0")/.."
BACKUP_DIR=mlflow_backup

backup() {
    mkdir -p "$BACKUP_DIR"
    docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "$BACKUP_DIR/mlflow_db.dump"
    docker compose exec -T mlflow tar -C /mlartifacts -czf - . > "$BACKUP_DIR/mlartifacts.tar.gz"
    echo "backup written to $BACKUP_DIR/"
}

restore() {
    docker compose up -d --wait postgres mlflow
    docker compose stop mlflow
    docker compose cp "$BACKUP_DIR/mlflow_db.dump" postgres:/tmp/mlflow_db.dump
    docker compose exec -T postgres sh -c 'dropdb -U "$POSTGRES_USER" --if-exists "$POSTGRES_DB" \
        && createdb -U "$POSTGRES_USER" "$POSTGRES_DB" \
        && pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner /tmp/mlflow_db.dump && rm /tmp/mlflow_db.dump'
    docker compose run --rm --no-deps -T --entrypoint sh -v "$(pwd)/$BACKUP_DIR:/backup:ro" mlflow \
        -c 'rm -rf /mlartifacts/* && tar -C /mlartifacts -xzf /backup/mlartifacts.tar.gz'
    docker compose up -d --wait mlflow
    echo "restored the MLflow history from $BACKUP_DIR/"
}

case "${1:-}" in
    backup) backup ;;
    restore) restore ;;
    *) echo "usage: $0 backup|restore" >&2; exit 2 ;;
esac
