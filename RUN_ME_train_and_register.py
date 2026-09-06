# Databricks notebook source
# RUN_ME: fetches real NFIP claims, builds the training table, trains +
# MLflow-tracks a model against the existing surge-exposure heuristic, and
# registers the winning run into Unity Catalog.
#
# How to use this:
#   1. Databricks workspace > Repos (or "Git folders") > Add repo > paste
#      this repo's GitHub URL.
#   2. Open this file inside that repo folder and Run All.
#
# There is also a CLI-driven equivalent that runs from your local machine
# via the Databricks CLI/SDK -- see the README's "Run it yourself" section.
# That's what was actually used to develop and locally verify this project
# (the data fetch, join, and local-MLflow training run all work with zero
# Databricks auth; only registration and serving need the workspace).
#
# What this notebook does NOT do: deploy the Model Serving endpoint. Free
# Edition serving quota/availability can't be assumed, so that's a separate,
# explicit step -- see serving/deploy_endpoint.py and the README.

# COMMAND ----------

# MAGIC %pip install -q -r requirements.txt
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("target", "severity", "Which target to register: frequency or severity")

# COMMAND ----------

# MAGIC %sh python data/fetch_nfip_claims.py

# COMMAND ----------

# MAGIC %sh cd data && python prepare_training_data.py

# COMMAND ----------

# Inside a Databricks notebook, MLflow already talks to this workspace --
# no --tracking-uri override needed (that flag is for running the same
# script from a local machine instead).
# MAGIC %sh cd training && python train_model.py --tracking-uri databricks

# COMMAND ----------

target = dbutils.widgets.get("target")
import subprocess
subprocess.run(["python", "register_model.py", "--profile", "DEFAULT", "--target", target],
                cwd="registry", check=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next steps (not run automatically)
# MAGIC
# MAGIC - **Serve it**: from your local machine, `databricks auth login --profile
# MAGIC   surge-exposure` then `python serving/deploy_endpoint.py --profile
# MAGIC   surge-exposure`. Check readiness with `databricks serving-endpoints get
# MAGIC   surge-exposure-claim-risk`, then try `serving/query_endpoint.py`.
# MAGIC - **Wire into the agent** (optional): once the endpoint is READY, paste
# MAGIC   `agent_tool/predict_claim_risk.sql` into a SQL editor on the
# MAGIC   surge-exposure-agent's workspace to give that already-deployed agent a
# MAGIC   7th tool backed by this trained model.
