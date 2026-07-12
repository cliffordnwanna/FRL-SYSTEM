# Microsoft Fabric Production Deployment Guide

## Architecture: Medallion Lakehouse

```
Bronze  (Raw)    → Silver (Tokenized) → Gold (Embeddings) → Serving
   ↓                    ↓                    ↓                  ↓
Eventstream        Fabric Notebook       Fabric ML          SQL Endpoint
Delta Tables       (PySpark)             (PyTorch)          Power BI / API
```

## Bronze Layer — Raw Ingestion

**Source:** Core banking system, app event logs, CRM exports
**Tool:** Fabric Eventstream or Data Factory pipeline
**Schedule:** Transactions hourly, app events real-time, products/complaints nightly

Tables created as Delta format in Lakehouse:
- `bronze_transactions`
- `bronze_app_events`
- `bronze_counterparties`
- `bronze_products`
- `bronze_complaints`

## Silver Layer — Tokenization

**Tool:** Fabric Notebook (PySpark + Python)
**Schedule:** Nightly at 01:00 AM
**Script:** Adaptation of `src/tokenizer.py` for distributed Spark execution

```python
# Fabric Notebook (Silver Layer)
from src.tokenizer import FinancialTokenizer
import pyspark.pandas as ps

txns = spark.table("bronze_transactions").pandas_api()
app  = spark.table("bronze_app_events").pandas_api()
cps  = spark.table("bronze_counterparties").pandas_api()

tokenizer = FinancialTokenizer()
sequences = tokenizer.transform(txns, app, cps)

# Write to Silver Delta table
spark.createDataFrame(sequences_df).write.mode("overwrite").saveAsTable("silver_customer_sequences")
```

## Gold Layer — Embeddings

**Tool:** Fabric ML (MLflow-tracked PyTorch experiment)
**Schedule:** Nightly after Silver layer completes (02:00 AM)
**Models registered in:** Fabric ML Model Registry

Two jobs:
1. `train_encoder.py` — Re-trains CoLES monthly; daily uses existing model for inference
2. `train_gnn.py` — Re-trains GraphSAGE monthly; daily uses existing model for inference

Output Delta table: `gold_customer_embeddings`
- `customer_id`
- `aepd_stage`
- `embedding_vector` (64-dim, stored as array)
- `anomaly_score`
- `updated_at`

## Serving Layer

**SQL Endpoint** → Power BI semantic model (existing BI engineer reads this)
**FastAPI endpoint** → Azure Container Instance for real-time single-customer lookup

```python
# Real-time inference (FastAPI)
@app.get("/customer/{customer_id}/embedding")
async def get_embedding(customer_id: str):
    # Fetch from Gold Delta table via SQL endpoint
    # Return 64-dim vector + AEPD stage + anomaly score
```

## Incremental Embedding Updates

Instead of re-encoding all customers nightly, use delta updates:

```python
# Only re-encode customers with events in last 24 hours
new_events = spark.sql("""
    SELECT DISTINCT customer_id 
    FROM bronze_transactions 
    WHERE timestamp > current_timestamp() - INTERVAL 1 DAY
""")

# Re-tokenize and re-encode only these customers
# Merge into gold_customer_embeddings (overwrite affected rows)
```

## Cost Estimates

| Resource | Tier | Estimated Monthly Cost |
|---|---|---|
| Fabric capacity | F4 | ~$730/month |
| Azure Container Instance | 1 vCPU, 2GB | ~$40/month |
| Azure Storage (models) | LRS | ~$5/month |
| **Total** | | **~$775/month** |

*F2 tier is sufficient for POC. Upgrade to F4 for production workloads.*
