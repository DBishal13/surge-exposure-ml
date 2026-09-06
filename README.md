# Surge Exposure Risk Model

An MLflow / Unity Catalog Model Registry / Model Serving project built on
top of [surge-exposure-agent](https://github.com/DBishal13/surge-exposure-agent)'s
real building exposure data. It asks one honest question: **does an actual
trained model beat that project's hand-picked heuristic score, when checked
against real FEMA flood insurance claims?**

The existing `exposure_score = 0.6*surge_ft_norm + 0.4*flood_active`
formula was deliberately simple and hand-picked, not learned (see
[`scoring_methodology.md`](https://github.com/DBishal13/surge-exposure-agent/blob/main/knowledge_base/docs/scoring_methodology.md)
in that repo). This project trains a real model on the same real claims
data that formula was originally validated against, tracks every run in
MLflow, registers the result in Unity Catalog, and reports the comparison
honestly either way.

## Architecture

```
FEMA OpenFEMA API (real NFIP claims, no key)
        │  data/fetch_nfip_claims.py
        ▼
data/claims_raw.csv  ──┐
                        │  data/prepare_training_data.py
data/buildings.csv  ────┤  (grid-cell join -- see "Methodology")
  (from surge-exposure-agent, real, already validated)
                        ▼
data/training_data.csv
        │  training/train_model.py  (MLflow-tracked, local or Databricks)
        ▼
MLflow run: heuristic baseline vs. trained GBM, both scored against real claims
        │  registry/register_model.py
        ▼
Unity Catalog: workspace.surge_exposure.claim_risk_model
        │  serving/deploy_endpoint.py
        ▼
Model Serving endpoint  ──►  serving/query_endpoint.py (REST demo client)
        │  (optional)
        ▼
agent_tool/predict_claim_risk.sql  -- a possible 7th tool for the
already-deployed surge_exposure_agent
```

## Methodology: joining on a grid, not an address

NFIP claims are privacy-redacted -- FEMA doesn't publish exact claim
addresses, only a generalized lat/lon. Rather than assume a grid size,
`prepare_training_data.py` measures it directly from the real data (the
smallest gap between distinct observed coordinates) and buckets both
buildings and claims onto that same grid, so every building in a cell
inherits that cell's aggregate claim outcome. This is the same
weak-label limitation the original `surge-exposure` validation study had
(see its own
[`validation_study_findings.md`](https://github.com/DBishal13/surge-exposure-agent/blob/main/knowledge_base/docs/validation_study_findings.md)) --
stated here rather than hidden, and it's why evaluation below uses a
**cell-grouped** cross-validation split (`GroupKFold` on `cell_id`), not a
random one: a random split would leak the answer between train and test
since many buildings share a label.

## Real results (verified locally, then reproduced live on Databricks)

- **140,732 real NFIP claims** fetched live from FEMA's public OpenFEMA API
  (capped at 20,000/county to keep the fetch fast -- some of these counties,
  e.g. Orleans Parish post-Katrina, have 100k+ historical claims) across the
  same 8 coastal counties `surge-exposure`'s 7,717 buildings cover.
- FEMA's own privacy-generalized coordinate grid turned out to be **0.1°**
  (~11km) -- coarse enough that each region collapses to roughly **one grid
  cell** (13 cells total across 8 regions). That's a real, important
  limitation of this data source, not a bug: see the caveat below.

| Target | Hand-picked heuristic (r) | Trained GBM, 5-fold group CV (r) | Verdict |
|---|---:|---:|---|
| **frequency** (claims per building) | 0.012 | -0.019 | Both ≈ 0 -- **no reliable signal either way** at 13 cells |
| **severity** (mean $ paid per claim) | 0.471 | **0.728** | Trained model **beats** the heuristic on this data |

The severity heuristic number (r=0.471) is a useful sanity check on its
own: the original study found r=0.52 for Lee County alone using only
Hurricane Ian claims; this run used 8 counties and all-time claims (capped)
and landed in the same ballpark -- the methodology reproduces consistently
on a different sample.

**The caveat that matters**: with only 13 cells (~1 per region), a model
can "win" mainly by learning **per-region average payout** rather than a
real within-region surge-depth/severity relationship -- home values and
storm history differ a lot by region, for reasons that have nothing to do
with surge exposure. `train_model.py` prints this caution automatically
whenever `n_cells < 30`. Confirming a true causal within-region effect
would need finer-grained claim geocoding than NFIP's public redaction
allows, or address-level data via a data-sharing agreement, which is out of
scope for this project.

**Verified end-to-end on a real Databricks workspace** on 2026-09-06: the
severity run was registered as `workspace.surge_exposure.claim_risk_model`
v1 in Unity Catalog, deployed to a live Model Serving endpoint, and queried
for real predictions (~$109k for a high-surge/low-elevation building vs.
~$11k for a low-surge/high-elevation one — the right direction). Full
command output in [`EVIDENCE.md`](EVIDENCE.md).

## Run it yourself

**Fastest path**: open `RUN_ME_train_and_register.py` as a Databricks
notebook (Repos > Add repo > this URL > Run All) -- see the widget for
which target (`frequency`/`severity`) to register.

**CLI-driven / local** (what was actually used to develop and verify this):

```bash
pip install -r requirements.txt

# 1. Real data, no Databricks account needed for these two steps:
python data/fetch_nfip_claims.py
python data/prepare_training_data.py

# 2. Train + track locally (sqlite-backed MLflow, fully self-contained):
cd training && python train_model.py
mlflow ui --backend-store-uri sqlite:///mlflow.db   # browse the runs

# 3. Databricks-only from here -- needs a live workspace session:
databricks auth login --profile surge-exposure
python train_model.py --tracking-uri databricks://surge-exposure
cd ../registry && python register_model.py --profile surge-exposure --target severity
cd ../serving && python deploy_endpoint.py --profile surge-exposure
# wait for it to report READY, then:
python query_endpoint.py --profile surge-exposure --surge-ft 8.5 --height-m 6.0 \
  --flood-active 0 --region fort-myers-beach
```

## Known limitations

- **Grid-cell label granularity** (see caveat above) is the big one --
  everything downstream inherits it.
- **v2 API deprecation**: `data/fetch_nfip_claims.py` uses FEMA's
  `FimaNfipClaims` v2 endpoint, deprecated 2026-10-15 in favor of a renamed
  v3 dataset. Fine for now; re-point `BASE_URL` before that date.
- **Free Edition serving availability**: confirmed working (see
  `EVIDENCE.md`) — a Small, scale-to-zero endpoint took about 10 minutes to
  provision. Batch inference
  (`mlflow.pyfunc.load_model("models:/workspace.surge_exposure.claim_risk_model/latest")`
  in a notebook) also works and is instant by comparison if you don't need
  a live REST endpoint.
- **`agent_tool/predict_claim_risk.sql`** is a real, signature-matched
  starting point (not a vague sketch) but is not applied to the deployed
  `surge_exposure_agent` automatically -- it's an opt-in extension.

## Things learned the hard way

- MLflow 3.x's file-based tracking store is in maintenance mode and throws
  on use — default to a `sqlite:///` URI instead of `file:./mlruns`.
- Databricks-hosted MLflow experiments must live under a real workspace
  directory (`/Users/<your-email>/...`) — a placeholder path like
  `/Users/shared/...` fails with `NOT_FOUND: Parent directory does not
  exist` on a Free Edition workspace where that folder was never created.
- The installed `databricks-sdk`'s `EndpointCoreConfigInput` requires
  `name` as an explicit argument, not just implied by the surrounding
  `create()` call — omitting it fails with a `TypeError`, not a clearer
  serving-specific error.
- MLflow prints a 🏃 emoji in its "View run" URL; Windows terminals default
  to `cp1252`, which can't encode it, crashing the script's own success
  message — the run had already logged fine server-side. Run with
  `PYTHONIOENCODING=utf-8` to see the rest of the output.
- Some NFIP claims have null lat/lon even after generalization — drop them
  before computing grid cells, don't assume every row is geocoded.

## Credits

Built as an ML/AI-engineer-track addition to the
[databricks-ai-capstone](https://github.com/DBishal13/databricks-ai-capstone)
portfolio (DataExpert.io's free Databricks AI Bootcamp). Reuses real data
from [surge-exposure](https://github.com/DBishal13/surge-exposure) /
[surge-exposure-agent](https://github.com/DBishal13/surge-exposure-agent),
and real claims from FEMA's public
[OpenFEMA API](https://www.fema.gov/about/openfema/api) (no key required).

## License

[MIT](LICENSE)
