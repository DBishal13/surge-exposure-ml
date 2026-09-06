-- Optional: a 7th tool for the already-deployed surge_exposure_agent
-- (see ../../surge-exposure-agent), calling this project's Model Serving
-- endpoint via ai_query() instead of just the hand-picked heuristic score.
--
-- Not applied automatically -- paste into a Databricks SQL editor/notebook
-- connected to the surge_exposure_agent's workspace.surge_exposure schema
-- *after* serving/deploy_endpoint.py's endpoint is READY.
--
-- The named_struct fields below must match training/train_model.py's
-- build_features() column names exactly (that's what the served model's
-- signature was inferred from) -- surge_ft, height_m, flood_active, and one
-- one-hot region_slug_<region> column per one of the 8 covered regions.

USE CATALOG workspace;
USE SCHEMA surge_exposure;

CREATE OR REPLACE FUNCTION predict_claim_risk(p_building_id STRING)
RETURNS STRING
COMMENT 'Predicts real-claims-based severity risk for a building using the
trained model (surge-exposure-ml), as a second opinion alongside the
hand-picked exposure_score. Use when asked to compare the heuristic score
against a learned model, or when asked "how much would this building
actually cost to claim." Only meaningful within the 8 covered regions.'
RETURN
  SELECT ai_query(
    'surge-exposure-claim-risk',
    named_struct(
      'surge_ft', b.surge_ft,
      'height_m', b.height_m,
      'flood_active', CAST(b.flood_active AS DOUBLE),
      'region_slug_south-beach-miami', CAST(b.region_slug = 'south-beach-miami' AS DOUBLE),
      'region_slug_clearwater-beach', CAST(b.region_slug = 'clearwater-beach' AS DOUBLE),
      'region_slug_fort-myers-beach', CAST(b.region_slug = 'fort-myers-beach' AS DOUBLE),
      'region_slug_french-quarter-nola', CAST(b.region_slug = 'french-quarter-nola' AS DOUBLE),
      'region_slug_galveston-seawall', CAST(b.region_slug = 'galveston-seawall' AS DOUBLE),
      'region_slug_charleston-battery', CAST(b.region_slug = 'charleston-battery' AS DOUBLE),
      'region_slug_outer-banks-nags-head', CAST(b.region_slug = 'outer-banks-nags-head' AS DOUBLE),
      'region_slug_ocean-city-md', CAST(b.region_slug = 'ocean-city-md' AS DOUBLE)
    )
  )
  FROM lakebase_catalog.public.buildings b
  WHERE b.building_id = p_building_id;
