# Every step runs inside Docker; only Docker (with Compose v2) and the four CSVs in ./data are needed.
COMPOSE := docker compose
RUN     := $(COMPOSE) run --rm trainer
LATEX   := $(COMPOSE) run --rm latex

BASELINES   := $(sort $(wildcard configs/experiments/baseline_*.yaml))
LADDER      := $(sort $(wildcard configs/experiments/0*.yaml))
ABLATIONS   := $(sort $(wildcard configs/ablations/*.yaml))

.PHONY: all up down data-check experiments tune test-lock train predict report test lint backup restore

all: up data-check experiments tune test-lock train predict report test lint

up:                 ## build images, start PostgreSQL + MLflow (UI on http://localhost:$${MLFLOW_PORT:-5050})
	$(COMPOSE) --profile tools build
	$(COMPOSE) up -d --wait postgres mlflow

down:
	$(COMPOSE) down

data-check:         ## schema + SHA-256 of the input files
	$(RUN) python scripts/check_data.py

experiments:        ## baselines, the six-step improvement ladder and the ablations (dev folds)
	$(RUN) python scripts/run_experiment.py $(foreach f,$(BASELINES) $(LADDER) $(ABLATIONS),--config $(f))

tune:               ## 60-trial Optuna search + forward-nested optimism check
	$(RUN) python scripts/tune.py --trials 60 --nested-trials 40

test-lock:          ## one-shot locked test (refuses to repeat; add ARGS=--force to override)
	$(RUN) python scripts/evaluate_locked_test.py $(ARGS)

train:              ## fit the champion on all labels, export model/, register @champion
	$(RUN) python scripts/train_final.py

predict:            ## write outputs/ and check them with the assessment's score.py
	$(RUN) python scripts/predict.py

report:             ## figures + tables from MLflow, then build report/freight_report.pdf
	$(RUN) python scripts/make_report_assets.py
	$(LATEX) latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp report/main.pdf report/freight_report.pdf

test:
	$(RUN) python -m pytest -q tests

lint:
	$(RUN) sh -c "ruff check . && black --check ."

backup:             ## save the MLflow database + artifacts to mlflow_backup/
	scripts/mlflow_backup.sh backup

restore:            ## load mlflow_backup/ into a fresh stack (the run IDs cited in the report then resolve)
	scripts/mlflow_backup.sh restore
