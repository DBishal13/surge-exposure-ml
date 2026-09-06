"""Deploys (or updates) a Databricks Model Serving endpoint from the model
registered by ../registry/register_model.py.

Serverless Model Serving quota/availability on Databricks Free Edition
can't be confirmed without a live workspace -- if this fails with a quota
or entitlement error, that's a workspace-tier limit, not a bug in the
registration step; batch inference (loading the registered model with
`mlflow.pyfunc.load_model("models:/workspace.surge_exposure.claim_risk_model/latest")`
in a notebook) works regardless of serving availability.

Usage:
    python deploy_endpoint.py --profile surge-exposure
"""
import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
)

MODEL_NAME = "workspace.surge_exposure.claim_risk_model"
ENDPOINT_NAME = "surge-exposure-claim-risk"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="surge-exposure")
    parser.add_argument("--model-version", default="1", help="Registered model version to serve.")
    parser.add_argument("--workload-size", default="Small")
    args = parser.parse_args()

    w = WorkspaceClient(profile=args.profile)

    served_entity = ServedEntityInput(
        entity_name=MODEL_NAME,
        entity_version=args.model_version,
        workload_size=args.workload_size,
        scale_to_zero_enabled=True,
    )

    existing = [e for e in w.serving_endpoints.list() if e.name == ENDPOINT_NAME]
    if existing:
        print(f"Updating existing endpoint {ENDPOINT_NAME} to model version {args.model_version} ...")
        w.serving_endpoints.update_config(
            name=ENDPOINT_NAME, served_entities=[served_entity]
        )
    else:
        print(f"Creating endpoint {ENDPOINT_NAME} serving {MODEL_NAME} v{args.model_version} ...")
        w.serving_endpoints.create(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=[served_entity]),
        )

    print(f"\nEndpoint '{ENDPOINT_NAME}' requested. It takes a few minutes to come up --")
    print(f"check status with: databricks serving-endpoints get {ENDPOINT_NAME} --profile {args.profile}")
    print("Then query it with query_endpoint.py once state.ready == 'READY'.")


if __name__ == "__main__":
    main()
