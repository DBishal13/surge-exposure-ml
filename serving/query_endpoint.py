"""Example real REST call against the live Model Serving endpoint.

Usage:
    python query_endpoint.py --profile surge-exposure \\
        --surge-ft 8.5 --height-m 6.0 --flood-active 0 --region fort-myers-beach
"""
import argparse

from databricks.sdk import WorkspaceClient

ENDPOINT_NAME = "surge-exposure-claim-risk"

# Must match training/train_model.py's feature order: [surge_ft, height_m, flood_active, one-hot(region_slug)...]
REGIONS = [
    "south-beach-miami", "clearwater-beach", "fort-myers-beach", "french-quarter-nola",
    "galveston-seawall", "charleston-battery", "outer-banks-nags-head", "ocean-city-md",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="surge-exposure")
    parser.add_argument("--surge-ft", type=float, required=True)
    parser.add_argument("--height-m", type=float, required=True)
    parser.add_argument("--flood-active", type=int, choices=[0, 1], required=True)
    parser.add_argument("--region", choices=REGIONS, required=True)
    args = parser.parse_args()

    # Column names must match training/train_model.py's build_features() exactly --
    # the model was logged with a named-column signature (input_example=X.iloc[:5]),
    # not a positional one.
    row = {"surge_ft": args.surge_ft, "height_m": args.height_m, "flood_active": float(args.flood_active)}
    for r in REGIONS:
        row[f"region_slug_{r}"] = 1.0 if r == args.region else 0.0

    w = WorkspaceClient(profile=args.profile)
    resp = w.serving_endpoints.query(name=ENDPOINT_NAME, dataframe_records=[row])
    print(f"Predicted (whichever target register_model.py --target registered): {resp.predictions[0]}")


if __name__ == "__main__":
    main()
