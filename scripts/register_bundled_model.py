"""Register the checked-in PyFunc bundle with MLflow when the Compose stack starts."""

from __future__ import annotations

from pathlib import Path

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

MODEL_PATH = Path("model/pyfunc")
ALIAS = "champion"


def main() -> None:
    if not (MODEL_PATH / "MLmodel").is_file():
        raise FileNotFoundError(f"Bundled MLflow model is missing: {MODEL_PATH}")

    config = yaml.safe_load(Path("configs/final.yaml").read_text())
    model_name = config["registered_model_name"]
    client = MlflowClient()

    try:
        champion = client.get_model_version_by_alias(model_name, ALIAS)
    except MlflowException as exc:
        missing_model = exc.error_code == "RESOURCE_DOES_NOT_EXIST"
        missing_alias = (
            exc.error_code == "INVALID_PARAMETER_VALUE"
            and f"alias {ALIAS} not found" in str(exc)
        )
        if not (missing_model or missing_alias):
            raise
    else:
        print(
            f"MLflow model {model_name}@{ALIAS} already points to version "
            f"{champion.version}; leaving it unchanged."
        )
        return

    with mlflow.start_run(run_name="register-bundled-model") as run:
        mlflow.log_artifacts(str(MODEL_PATH), artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri, model_name)

    client.set_registered_model_alias(model_name, ALIAS, registered.version)
    print(
        f"Registered {model_name} version {registered.version} from {MODEL_PATH} "
        f"and set the @{ALIAS} alias."
    )


if __name__ == "__main__":
    main()
