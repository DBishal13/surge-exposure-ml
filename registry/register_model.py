"""Registers the best Databricks-logged training run into the Unity Catalog
Model Registry, under workspace.surge_exposure.claim_risk_model -- same
`workspace.<schema>` convention as surge-exposure-agent's UC functions,
since this Databricks Free Edition workspace can't create a dedicated
catalog (confirmed in that project; see its README).

Prerequisite: run training/train_model.py with
`--tracking-uri databricks://surge-exposure` first, so there's a Databricks
experiment to pull the winning run from. This script only reads that
experiment and calls mlflow.register_model -- it does not train anything.

Usage:
    databricks auth login --profile surge-exposure   # once, interactively
    python train_model.py --tracking-uri databricks://surge-exposure
    python register_model.py --profile surge-exposure --target severity
"""
import argparse

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "workspace.surge_exposure.claim_risk_model"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="surge-exposure")
    parser.add_argument("--experiment", default="/Users/beesal13dh@gmail.com/surge_exposure_claim_risk")
    parser.add_argument("--target", choices=["frequency", "severity"], default="severity",
                         help="severity had the stronger original correlation (r=0.52 vs r=0.37)")
    args = parser.parse_args()

    tracking_uri = f"databricks://{args.profile}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri("databricks-uc")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(
            f"No experiment '{args.experiment}' found via profile '{args.profile}'. "
            f"Run training/train_model.py --tracking-uri {tracking_uri} first."
        )

    run_name_filter = f"tags.mlflow.runName = 'gbm-vs-heuristic-target_{args.target}'"
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=run_name_filter,
        order_by=["metrics.trained_model_cv_r DESC"],
        max_results=1,
    )
    if not runs:
        raise SystemExit(f"No runs found matching {run_name_filter!r} in experiment {args.experiment}.")

    best_run = runs[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    print(f"Registering {model_uri} (cv_r={best_run.data.metrics.get('trained_model_cv_r'):.3f}) "
          f"as {MODEL_NAME} ...")

    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"\nRegistered {MODEL_NAME} version {result.version}.")
    print(f"View it in the workspace under Catalog > workspace > surge_exposure > models > claim_risk_model.")


if __name__ == "__main__":
    main()
