"""Reuse a compatible registry version or train one before API startup."""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient

from app.model_service import SERVING_CONTRACT_TAG
from models.train import REGISTERED_MODEL_NAME, train


def main() -> None:
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.getenv("MODEL_NAME", REGISTERED_MODEL_NAME)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    compatible = [
        version
        for version in client.search_model_versions(f"name='{model_name}'")
        if version.tags.get("serving_contract") == SERVING_CONTRACT_TAG
    ]
    if compatible:
        latest = max(compatible, key=lambda version: int(version.version))
        client.set_registered_model_alias(model_name, "champion", latest.version)
        print(f"Reusing compatible {model_name} version {latest.version}.")
        return

    if model_name != REGISTERED_MODEL_NAME:
        raise RuntimeError(
            f"No compatible version exists for {model_name!r}, and automated training "
            f"registers only {REGISTERED_MODEL_NAME!r}."
        )

    print(f"No compatible {model_name} version exists; starting one training run.")
    train()


if __name__ == "__main__":
    main()
