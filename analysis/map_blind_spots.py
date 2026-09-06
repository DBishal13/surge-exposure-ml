"""Systematically ranks all 13 grid cells by how much the heuristic
under- or over-rates real claim severity relative to the other cells, and
checks whether the trained model's honestly out-of-sample (GroupKFold,
never-seen-this-cell) predictions would have caught each blind spot in
advance -- as opposed to analysis/ANALYSIS.md's single NOLA anecdote,
which queried the live *fully-fit* serving endpoint and so could only
show what the model memorized, not what it would generalize to.

This is the input the agent (surge-exposure-agent) needs to state a
general rule instead of reciting one example: which regions is the
model's second opinion actually trustworthy for, cross-validated?

Usage:
    python analysis/map_blind_spots.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "training"))
from train_model import build_features  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_CSV = Path(__file__).resolve().parent / "blind_spot_map.csv"


def oof_predictions(X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Same GroupKFold scheme as training/train_model.py's cross_validated_r,
    but returns the actual out-of-fold predictions instead of a summary
    metric -- for a cell held out in a given fold, the model has literally
    never seen that cell's label or any of its buildings."""
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    oof = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = model.predict(X.iloc[test_idx])
    return oof


def normalize(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min())


def main() -> None:
    df = pd.read_csv(DATA_DIR / "training_data.csv")
    X = build_features(df)
    y = df["target_severity"].to_numpy(dtype=float)
    groups = df["cell_id"].to_numpy()

    print(f"Computing honest out-of-fold predictions for {len(df)} buildings, {len(set(groups))} cells "
          f"(GroupKFold(5) on cell_id -- a held-out cell's buildings are never seen during its own fold)...")
    oof = oof_predictions(X, y, groups)

    df = df.copy()
    df["model_oof_severity_pred"] = oof

    per_cell = df.groupby("cell_id").agg(
        region_slug=("region_slug", "first"),
        n_buildings=("exposure_score", "size"),
        mean_exposure_score=("exposure_score", "mean"),
        real_severity=("target_severity", "first"),
        model_oof_severity_pred=("model_oof_severity_pred", "mean"),
    )

    per_cell["norm_heuristic"] = normalize(per_cell["mean_exposure_score"])
    per_cell["norm_real"] = normalize(per_cell["real_severity"])
    per_cell["norm_model_oof"] = normalize(per_cell["model_oof_severity_pred"].clip(lower=0))
    per_cell["heuristic_blind_spot_gap"] = per_cell["norm_real"] - per_cell["norm_heuristic"]
    per_cell["model_still_blind_gap"] = per_cell["norm_real"] - per_cell["norm_model_oof"]

    ranked = per_cell.sort_values("heuristic_blind_spot_gap", ascending=False)
    print("\n=== Cells ranked by how badly the heuristic underrates real severity ===")
    print("(positive gap = heuristic underrates; the OOF columns show whether the")
    print(" trained model, having never seen this cell's label, would have caught it)\n")
    print(ranked[[
        "region_slug", "n_buildings", "mean_exposure_score", "real_severity",
        "model_oof_severity_pred", "heuristic_blind_spot_gap", "model_still_blind_gap",
    ]].to_string())

    def pearson(a: pd.Series, b: pd.Series) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    r_heuristic = pearson(per_cell["mean_exposure_score"], per_cell["real_severity"])
    r_model_oof = pearson(per_cell["model_oof_severity_pred"], per_cell["real_severity"])
    print(f"\nCell-level r, heuristic vs. real severity  : {r_heuristic:.3f}")
    print(f"Cell-level r, model OOF vs. real severity  : {r_model_oof:.3f}  "
          f"(honest -- excludes memorized region lookups)")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    ranked.reset_index().to_csv(OUT_CSV, index=False)
    print(f"\nWrote ranked per-cell table to {OUT_CSV}")


if __name__ == "__main__":
    main()
