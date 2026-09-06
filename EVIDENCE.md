# Deployment Evidence

Filled in after running the Databricks-only steps for real (needs
`databricks auth login --profile surge-exposure` -- see README).

## 1. Local training run (verified, not Databricks-dependent)

```
Training on 7717 buildings across 13 grid cells.
CAUTION: only 13 cells (~1 per region at this grid size) -- with this few
groups, a model can 'win' mainly by learning per-region average payouts
rather than a real surge/severity relationship. Read cv_r improvements
with that in mind.

[target_frequency]
  hand-picked heuristic:  r = 0.012
  trained GBM (5-fold CV): r = -0.019  (rmse=32.417)
  -> both near zero -- no reliable signal for this target at 13 cells.

[target_severity]
  hand-picked heuristic:  r = 0.471
  trained GBM (5-fold CV): r = 0.728  (rmse=24852.333)
  -> the trained model beats the heuristic on this data.
```

140,732 real NFIP claims fetched from `data/fetch_nfip_claims.py` against
the live OpenFEMA API on 2026-09-05.

## 2. Unity Catalog Model Registry entry

<!-- Fill in after `databricks auth login --profile surge-exposure` +
     registry/register_model.py, e.g. output of:
     databricks registered-models get workspace.surge_exposure.claim_risk_model --profile surge-exposure -->

## 3. Model Serving endpoint

<!-- Fill in after serving/deploy_endpoint.py reports READY, e.g. output of:
     databricks serving-endpoints get surge-exposure-claim-risk --profile surge-exposure
     plus a real query_endpoint.py call and its response. -->
