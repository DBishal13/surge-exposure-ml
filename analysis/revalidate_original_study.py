"""Re-runs surge-exposure's own grid-cell validation methodology
(scripts/validate_exposure_bins.py, paper/paper.md) at this project's
larger scope -- all 8 covered regions instead of Lee County, FL alone --
using data/buildings.csv (already scored by that project's own pipeline)
and this project's independently-fetched data/claims_raw.csv (140,732 real
NFIP claims across 8 counties).

Methodology is replicated exactly, just at a larger geographic footprint:
snap buildings and claims to the same 0.1-degree grid (confirmed to match
surge-exposure/src/surge_exposure/data/nfip.py's snap_to_grid), aggregate
each side per cell, inner-join, and Pearson-correlate mean exposure_score
against claim_count and mean claim payout.

grid_size()/cell_id() below are copied from ../data/prepare_training_data.py
(a one-off analysis script, same as validate_exposure_bins.py was for the
original project) so both scripts partition cells identically -- keep them
in sync if either changes.

Usage:
    python analysis/revalidate_original_study.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_CSV = Path(__file__).resolve().parent / "8region_grid.csv"


def grid_size(values: pd.Series) -> float:
    distinct = sorted(values.dropna().unique())
    gaps = [round(b - a, 6) for a, b in zip(distinct, distinct[1:]) if b - a > 1e-9]
    return min(gaps) if gaps else 0.01


def cell_id(region: pd.Series, lat: pd.Series, lon: pd.Series, cell_deg: float) -> pd.Series:
    return (
        region.astype(str) + ":"
        + (lat / cell_deg).round().astype(int).astype(str) + ":"
        + (lon / cell_deg).round().astype(int).astype(str)
    )


def pearson(a: pd.Series, b: pd.Series) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    buildings = pd.read_csv(DATA_DIR / "buildings.csv")
    claims = pd.read_csv(DATA_DIR / "claims_raw.csv", low_memory=False)

    n_before = len(claims)
    claims = claims.dropna(subset=["latitude", "longitude"])
    print(f"Dropped {n_before - len(claims)} of {n_before} claims with no redacted lat/lon.")

    cell_deg = max(grid_size(claims["latitude"]), grid_size(claims["longitude"]))
    print(f"Grid size: {cell_deg} degrees (matches surge-exposure's own 0.1-degree snap_to_grid).")

    buildings = buildings.copy()
    claims = claims.copy()
    buildings["cell_id"] = cell_id(buildings["region_slug"], buildings["lat"], buildings["lon"], cell_deg)
    claims["cell_id"] = cell_id(claims["region_slug"], claims["latitude"], claims["longitude"], cell_deg)

    building_cells = buildings.groupby("cell_id").agg(
        region_slug=("region_slug", "first"),
        mean_exposure_score=("exposure_score", "mean"),
        building_count=("exposure_score", "size"),
    )

    claims["amount_paid_building_plus_contents"] = (
        claims["amountPaidOnBuildingClaim"].fillna(0) + claims["amountPaidOnContentsClaim"].fillna(0)
    )
    claim_cells = claims.groupby("cell_id").agg(
        claim_count=("id", "count"),
        mean_amount_paid_building_only=("amountPaidOnBuildingClaim", "mean"),
        mean_amount_paid_building_plus_contents=("amount_paid_building_plus_contents", "mean"),
    )

    merged = building_cells.join(claim_cells, how="inner")

    print(
        f"\n{len(merged)} of {buildings['cell_id'].nunique()} building cells have >=1 real claim "
        f"(inner join, matching the original study's methodology)."
    )
    print(merged.sort_values("mean_exposure_score", ascending=False).to_string())

    r_freq = pearson(merged["mean_exposure_score"], merged["claim_count"])
    r_sev_building_only = pearson(merged["mean_exposure_score"], merged["mean_amount_paid_building_only"])
    r_sev_building_plus_contents = pearson(
        merged["mean_exposure_score"], merged["mean_amount_paid_building_plus_contents"]
    )

    print(f"\n=== Cell-level correlation, {len(merged)} cells across 8 regions (original study's exact method) ===")
    print(f"mean_exposure_score vs claim_count                    : r = {r_freq:.3f}   "
          f"(Lee County only: r=0.37, Ian-window: r=0.25)")
    print(f"mean_exposure_score vs mean building-only payout      : r = {r_sev_building_only:.3f}   "
          f"(comparable to surge-exposure-ml's building-level r=0.471)")
    print(f"mean_exposure_score vs mean building+contents payout  : r = {r_sev_building_plus_contents:.3f}   "
          f"(Lee County only: r=0.52, Ian-window: r=0.516 -- original study's exact metric)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.reset_index().to_csv(OUT_CSV, index=False)
    print(f"\nWrote per-cell table to {OUT_CSV}")


if __name__ == "__main__":
    main()
