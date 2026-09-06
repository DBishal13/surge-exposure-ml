# Deployment Evidence

All three stages below were run for real against a live Databricks Free
Edition workspace (`beesal13dh@gmail.com`) on 2026-09-06, after
`databricks auth login --profile surge-exposure`.

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

The same two runs were reproduced against the real Databricks-hosted MLflow
experiment (`train_model.py --tracking-uri databricks://surge-exposure`),
logged under `/Users/beesal13dh@gmail.com/surge_exposure_claim_risk`
(experiment ID `3988899851016166`) -- identical metrics to the local run
above, confirming the local-vs-Databricks tracking backends agree.

## 2. Unity Catalog Model Registry entry

`registry/register_model.py --profile surge-exposure --target severity`
pulled run `ec8f8d3ee0db4733ba5db19bc09c482d` (cv_r = 0.728, the severity
run) and registered it as version 1:

```
$ databricks registered-models get workspace.surge_exposure.claim_risk_model --profile surge-exposure
{
  "catalog_name": "workspace",
  "created_at": 1788702920690,
  "created_by": "beesal13dh@gmail.com",
  "full_name": "workspace.surge_exposure.claim_risk_model",
  "metastore_id": "8f260364-ecaf-4804-85f5-b458ffaf2117",
  "name": "claim_risk_model",
  "owner": "beesal13dh@gmail.com",
  "schema_name": "surge_exposure",
  "updated_at": 1788702940761,
  "updated_by": "beesal13dh@gmail.com"
}
```

Visible in the workspace under Catalog > workspace > surge_exposure >
models > claim_risk_model, version 1.

## 3. Model Serving endpoint

`serving/deploy_endpoint.py --profile surge-exposure --model-version 1`
requested a serverless endpoint. It took about 10 minutes to provision
(scale-to-zero, Small workload) before reaching `READY`:

```
$ databricks serving-endpoints get surge-exposure-claim-risk --profile surge-exposure
{
  "config": {
    "served_entities": [
      {
        "entity_name": "workspace.surge_exposure.claim_risk_model",
        "entity_version": "1",
        "name": "claim_risk_model-1",
        "scale_to_zero_enabled": true,
        "state": { "deployment": "DEPLOYMENT_READY", "deployment_state_message": "" },
        "workload_size": "Small",
        "workload_type": "CPU"
      }
    ],
    "traffic_config": {
      "routes": [
        { "served_entity_name": "claim_risk_model-1", "traffic_percentage": 100 }
      ]
    }
  },
  "name": "surge-exposure-claim-risk",
  "state": { "config_update": "NOT_UPDATING", "ready": "READY" }
}
```

Two real queries against the live endpoint, registered on the severity
target (predicted dollar payout per building):

```
$ python query_endpoint.py --profile surge-exposure \
    --surge-ft 8.5 --height-m 6.0 --flood-active 0 --region fort-myers-beach
Predicted: 108953.54

$ python query_endpoint.py --profile surge-exposure \
    --surge-ft 2.0 --height-m 12.0 --flood-active 0 --region ocean-city-md
Predicted: 10819.22
```

Low surge / high elevation predicts roughly 10x less expected payout than
high surge / low elevation, in the expected direction -- a basic sanity
check that the served model, end to end, is behaving reasonably, not just
returning noise.
