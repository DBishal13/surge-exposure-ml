"""Trains a real model on data/training_data.csv and compares it, honestly,
against the existing hand-picked heuristic (exposure_score = 0.6*surge_ft_norm
+ 0.4*flood_active) -- both scored against the same real NFIP claim outcomes.

Uses a local file-based MLflow tracking URI by default so this runs and is
fully verifiable without any Databricks auth. Point --tracking-uri at
"databricks" (after `databricks auth login --profile surge-exposure`) to log
these same runs into a Databricks-hosted MLflow experiment instead -- see
../registry/register_model.py for promoting the winning run from there.

Evaluation is cell-grouped (GroupKFold on cell_id), not a random split:
every building in a cell shares that cell's label, so a random split would
leak the answer between train and test.
"""
import argparse

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

FEATURES_NUMERIC = ["surge_ft", "height_m"]
REGIONS = [
    "south-beach-miami", "clearwater-beach", "fort-myers-beach", "french-quarter-nola",
    "galveston-seawall", "charleston-battery", "outer-banks-nags-head", "ocean-city-md",
]
TARGETS = ["target_frequency", "target_severity"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Named columns (not a raw numpy array) so the MLflow-logged model
    signature has real feature names -- serving/query_endpoint.py and
    agent_tool/predict_claim_risk.sql send requests keyed by these same
    names, not positional array indices."""
    X = df[FEATURES_NUMERIC].fillna(0.0).copy()
    X["flood_active"] = df["flood_active"].astype(float)
    for region in REGIONS:
        X[f"region_slug_{region}"] = (df["region_slug"] == region).astype(float)
    return X


def cross_validated_r(model, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
    """Out-of-fold predictions via GroupKFold, then Pearson r + RMSE against
    the real target -- directly comparable to the original study's r values,
    but cross-validated instead of fit-and-score-on-the-same-data."""
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    oof = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X.iloc[test_idx])
    r, _ = pearsonr(oof, y)
    rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    return float(r), rmse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db",
                         help="Use 'databricks://surge-exposure' (after `databricks auth login "
                              "--profile surge-exposure`) to log into that Databricks workspace's "
                              "MLflow experiment instead of a local sqlite store.")
    parser.add_argument("--experiment", default="surge_exposure_claim_risk")
    parser.add_argument("--data", default="../data/training_data.csv")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    # Databricks-hosted experiments must live under a workspace path.
    on_databricks = args.tracking_uri.startswith("databricks")
    experiment_name = f"/Users/beesal13dh@gmail.com/{args.experiment}" if on_databricks else args.experiment
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(args.data)
    X = build_features(df)
    groups = df["cell_id"].to_numpy()
    n_cells = len(set(groups))
    print(f"Training on {len(df)} buildings across {n_cells} grid cells.")
    if n_cells < 30:
        print(f"CAUTION: only {n_cells} cells (~1 per region at this grid size) -- with this few "
              f"groups, a model can 'win' mainly by learning per-region average payouts rather than "
              f"a real surge/severity relationship. Read cv_r improvements with that in mind.")

    for target in TARGETS:
        y = df[target].to_numpy(dtype=float)
        baseline_r, _ = pearsonr(df["exposure_score"], y)

        model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        model_r, model_rmse = cross_validated_r(model, X, y, groups)
        # Refit on all data for the artifact that gets logged/registered.
        model.fit(X, y)

        with mlflow.start_run(run_name=f"gbm-vs-heuristic-{target}"):
            mlflow.log_param("target", target)
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 3)
            mlflow.log_param("learning_rate", 0.05)
            mlflow.log_param("cv_scheme", "GroupKFold(5) on cell_id")
            mlflow.log_metric("n_buildings", len(df))
            mlflow.log_metric("n_cells", df["cell_id"].nunique())
            mlflow.log_metric("baseline_heuristic_r", baseline_r)
            mlflow.log_metric("trained_model_cv_r", model_r)
            mlflow.log_metric("trained_model_cv_rmse", model_rmse)
            mlflow.sklearn.log_model(model, artifact_path="model", input_example=X.iloc[:5])

            print(f"\n[{target}]")
            print(f"  hand-picked heuristic:  r = {baseline_r:.3f}")
            print(f"  trained GBM (5-fold CV): r = {model_r:.3f}  (rmse={model_rmse:.3f})")
            if max(abs(model_r), abs(baseline_r)) < 0.1:
                print(f"  -> both near zero -- no reliable signal for this target at {n_cells} cells.")
            elif abs(model_r) > abs(baseline_r):
                print(f"  -> the trained model beats the heuristic on this data.")
            else:
                print(f"  -> the heuristic beats the trained model on this data.")

    print(f"\nMLflow runs logged to {args.tracking_uri} / experiment '{experiment_name}'.")
    print(f"Run `mlflow ui --backend-store-uri {args.tracking_uri}` (from this directory) to browse them locally."
          if not on_databricks else "View them in the Databricks workspace under Experiments.")


if __name__ == "__main__":
    main()
