"""Joins real buildings.csv (surge exposure heuristic features) with real
claims_raw.csv (FEMA NFIP claims) on a lat/lon grid, producing one row per
building with a claim-derived label.

Buildings and claims can't be joined by exact address (NFIP redacts claim
locations to a generalized lat/lon for privacy -- see the grid probe this
script runs), so both are bucketed onto the same grid and every building in
a cell inherits that cell's aggregate claim outcome. This is the same
weak-label limitation the original surge-exposure validation study had
(see surge-exposure-agent/knowledge_base/docs/validation_study_findings.md)
-- stated here rather than hidden, and it's why train_model.py evaluates
with a *cell-grouped* split instead of a random one.

Two targets, both continuous so they're directly comparable (same units of
comparison: Pearson r) to the original study's r=0.37 (frequency) / r=0.52
(severity) findings:
  - target_frequency: claims per building in the cell (claim_count / n_buildings)
  - target_severity:  mean amountPaidOnBuildingClaim of claims in the cell
"""
import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent


def grid_size(values: pd.Series) -> float:
    """FEMA generalizes claim lat/lon for privacy onto some fixed grid.
    Rather than assume a size, find it: the smallest non-zero gap between
    distinct observed values."""
    distinct = sorted(values.dropna().unique())
    gaps = [round(b - a, 6) for a, b in zip(distinct, distinct[1:]) if b - a > 1e-9]
    return min(gaps) if gaps else 0.01


def main() -> None:
    buildings = pd.read_csv(DATA_DIR / "buildings.csv")
    claims_path = DATA_DIR / "claims_raw.csv"
    if not claims_path.exists():
        print(f"Missing {claims_path} -- run fetch_nfip_claims.py first.", file=sys.stderr)
        sys.exit(1)
    claims = pd.read_csv(claims_path, low_memory=False)
    n_before = len(claims)
    claims = claims.dropna(subset=["latitude", "longitude"])
    print(f"Dropped {n_before - len(claims)} of {n_before} claims with no redacted "
          f"lat/lon (FEMA doesn't always retain a generalized location).", file=sys.stderr)

    cell_deg = max(grid_size(claims["latitude"]), grid_size(claims["longitude"]))
    print(f"Detected claim coordinate grid size: {cell_deg} degrees", file=sys.stderr)

    def cell_id(region: pd.Series, lat: pd.Series, lon: pd.Series) -> pd.Series:
        return (
            region.astype(str) + ":"
            + (lat / cell_deg).round().astype(int).astype(str) + ":"
            + (lon / cell_deg).round().astype(int).astype(str)
        )

    buildings["cell_id"] = cell_id(buildings["region_slug"], buildings["lat"], buildings["lon"])
    claims["cell_id"] = cell_id(claims["region_slug"], claims["latitude"], claims["longitude"])

    buildings_per_cell = buildings.groupby("cell_id").size().rename("n_buildings_in_cell")

    claims_per_cell = claims.groupby("cell_id").agg(
        n_claims_in_cell=("id", "count"),
        mean_building_payout=("amountPaidOnBuildingClaim", "mean"),
    )

    cell_stats = buildings_per_cell.to_frame().join(claims_per_cell, how="left")
    cell_stats["n_claims_in_cell"] = cell_stats["n_claims_in_cell"].fillna(0)
    cell_stats["mean_building_payout"] = cell_stats["mean_building_payout"].fillna(0.0)
    cell_stats["target_frequency"] = cell_stats["n_claims_in_cell"] / cell_stats["n_buildings_in_cell"]
    cell_stats["target_severity"] = cell_stats["mean_building_payout"]

    out = buildings.merge(cell_stats, on="cell_id", how="left")

    n_cells = out["cell_id"].nunique()
    n_cells_with_claims = (cell_stats["n_claims_in_cell"] > 0).sum()
    print(f"{len(buildings)} buildings bucketed into {n_cells} cells "
          f"({n_cells_with_claims} of them have >=1 real claim)", file=sys.stderr)
    print(f"Buildings per cell: min={cell_stats['n_buildings_in_cell'].min()}, "
          f"median={cell_stats['n_buildings_in_cell'].median():.0f}, "
          f"max={cell_stats['n_buildings_in_cell'].max()}", file=sys.stderr)

    baseline_freq_r = out["exposure_score"].corr(out["target_frequency"])
    baseline_sev_r = out["exposure_score"].corr(out["target_severity"])
    print(f"\nQuick baseline check (exposure_score vs. real claims, this dataset):", file=sys.stderr)
    print(f"  frequency r = {baseline_freq_r:.3f}  (original study: r=0.37)", file=sys.stderr)
    print(f"  severity  r = {baseline_sev_r:.3f}  (original study: r=0.52)", file=sys.stderr)

    out_path = DATA_DIR / "training_data.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
