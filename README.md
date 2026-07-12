# Financial Representation Learning System (FRL-System)

> *From raw transactions to intelligent customer understanding — a production-grade blueprint for financial AI.*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/frl-system/blob/main/notebooks/00_synthetic_data_generator.ipynb)

---

## What This Is

The **Financial Representation Learning System** is an open-source blueprint for building deep customer intelligence from raw banking and fintech transaction data — without relying on hand-crafted features, manual segmentation rules, or labelled datasets.

Instead of asking *"which bucket does this customer fall into?"*, this system asks *"what does this customer's behaviour actually mean — and how does it relate to every other customer in the ecosystem?"*

The answer comes from three components working together:

1. **Event Tokenizer** — converts raw financial events into a structured vocabulary (a "financial language")
2. **Sequence Encoder (CoLES)** — learns a dense customer embedding from their event history using self-supervised contrastive learning (no labels required)
3. **Graph Neural Network (GraphSAGE)** — enriches each customer embedding with network-level signals by modelling relationships between customers, merchants, and products

The output is a **64-dimensional customer vector** — a compact, information-rich representation of who each customer is, what they do, and how they relate to the broader ecosystem. Every downstream application (segmentation, churn prediction, next-product recommendation, fraud flagging, anomaly detection) reads from this single source of truth.

---

## Why This Matters

Traditional customer analytics in financial services operates on aggregated, hand-crafted features: average balance, transaction count, recency score. These are useful but shallow. They describe what happened. They cannot describe *why*, *what next*, or *how this customer relates to everyone else*.

This system is inspired by the approach pioneered by Alibaba's AIPL model, Sberbank's CoLES paper (SIGMOD 2022), and recent research integrating graph structure into event sequence models ("Beyond Isolated Clients", WWW 2026). It brings that approach to a practical, runnable implementation that any bank or fintech can adapt.

**What makes this different from standard RFM or clustering:**

| Approach | Feature Source | Labels Required | Captures Network Effects | Updateable |
|---|---|---|---|---|
| RFM / K-Means | Hand-crafted | No | No | No |
| Supervised ML | Hand-crafted | **Yes** | No | Partial |
| **FRL-System** | **Learned from data** | **No** | **Yes** | **Yes** |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW DATA SOURCES                            │
│  Transactions │ App Events │ Products │ Complaints │ Counterparties │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LAYER 1: EVENT TOKENIZER                       │
│                                                                     │
│  Raw rows → structured financial vocabulary                         │
│  Amount bucketing (32 bins, log-scale)                              │
│  Categorical encoding (channel, event type, product)                │
│  Temporal features (hour bucket, day bucket, time-delta bucket)     │
│  Output: customer_id → [token_1, token_2, ..., token_256]           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LAYER 2: SEQUENCE ENCODER (CoLES / GRU)                │
│                                                                     │
│  Self-supervised contrastive learning on event sequences            │
│  No labels required — learns from the data itself                   │
│  Two random subsequences of same customer → similar embeddings      │
│  Different customers → distant embeddings                           │
│  Output: customer_id → 64-dim embedding vector                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LAYER 3: GRAPH BUILDER (Bipartite Graph)               │
│                                                                     │
│  Nodes: customers + counterparties/merchants                        │
│  Edges: transactions with features [amount, channel, recency]       │
│  Reveals: shared merchants, salary clusters, merchant type          │
│  Library: PyTorch Geometric                                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LAYER 4: GNN ENRICHMENT (GraphSAGE)                    │
│                                                                     │
│  Inductive learning — works on new customers not seen in training   │
│  Each customer embedding absorbs neighbourhood signals              │
│  Output: enriched 64-dim embedding per customer                     │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   LAYER 5: DOWNSTREAM APPLICATIONS                  │
│                                                                     │
│  ① Customer Segmentation (K-Means on embeddings → AEPD stages)     │
│  ② Next-Product Recommendation (MLP head on embedding)             │
│  ③ Anomaly / Fraud Detection (distance from cluster centroid)       │
│  ④ Churn Prediction (embedding + temporal features → binary class)  │
│  ⑤ UMAP Visualisation (2D projection for stakeholder reporting)     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## The AEPD Customer Journey Framework

Inspired by Alibaba's AIPL model, adapted for financial services:

| Stage | Label | Description | Signal |
|---|---|---|---|
| **Aware** | A | Account exists, no meaningful activity | Never transacted or only 1 low-value event |
| **Engaged** | E | Active on platform, exploring products | Logins, balance checks, product views — but limited transactions |
| **Product-Active** | P | Regular transactor, using core products | Consistent transaction frequency, multi-channel |
| **Deep** | D | Multi-product, loyal, long tenure | High ADB, product breadth, low complaints, referral signals |

The system learns to place customers into AEPD stages from their embeddings — without you manually defining the rules.

---

## Repository Structure

```
frl-system/
│
├── README.md                          ← You are here
├── .gitignore
├── requirements.txt                   ← All Python dependencies
│
├── data/
│   ├── synthetic/                     ← Generated CSVs (gitignored)
│   └── schemas/                       ← Table schema definitions (JSON)
│       ├── customers_schema.json
│       ├── transactions_schema.json
│       ├── app_events_schema.json
│       ├── counterparties_schema.json
│       ├── products_schema.json
│       └── complaints_schema.json
│
├── notebooks/
│   ├── 00_synthetic_data_generator.ipynb   ← Generate realistic data
│   ├── 01_event_tokenizer.ipynb            ← Build financial vocabulary
│   ├── 02_sequence_encoder.ipynb           ← Train CoLES embeddings
│   ├── 03_graph_builder.ipynb              ← Build customer-merchant graph
│   ├── 04_gnn_enrichment.ipynb             ← GraphSAGE enrichment
│   └── 05_downstream_tasks.ipynb           ← Segmentation, viz, applications
│
├── src/
│   ├── __init__.py
│   ├── config.py                      ← All constants and hyperparameters
│   ├── data_generator.py              ← Synthetic data factory (pure Python)
│   ├── tokenizer.py                   ← Event tokenizer class
│   ├── encoder.py                     ← CoLES sequence encoder
│   ├── graph_builder.py               ← Graph construction utilities
│   ├── gnn.py                         ← GraphSAGE model
│   └── downstream.py                  ← Segmentation, anomaly, next-product
│
├── production/
│   ├── fabric/
│   │   ├── README.md                  ← Fabric deployment guide
│   │   ├── medallion_architecture.md  ← Bronze/Silver/Gold schema
│   │   └── pipeline_specs.md          ← Pipeline configuration
│   └── azure/
│       ├── README.md                  ← Azure ML fallback guide
│       └── ml_pipeline.md             ← Azure ML pipeline spec
│
├── docs/
│   ├── architecture.md                ← Deep dive on design decisions
│   ├── decisions.md                   ← Why we chose each approach
│   ├── tokenization_guide.md          ← How financial tokenization works
│   └── results/                       ← Screenshots, metrics, UMAP plots
│
└── tests/
    ├── test_tokenizer.py
    ├── test_encoder.py
    └── test_graph_builder.py
```

---

## Quick Start

### Option 1: Run on Google Colab (Recommended)

Click the Colab badge at the top of each notebook. Run them in order (00 → 05). No local setup required.

### Option 2: Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/frl-system.git
cd frl-system

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data
python src/data_generator.py

# 5. Run notebooks in Jupyter
jupyter notebook notebooks/
```

### Option 3: Run with Claude Code (VS Code)

```bash
git clone https://github.com/YOUR_USERNAME/frl-system.git
# Open in VS Code → Claude Code handles dependencies and execution
```

---

## Notebook Guide

| # | Notebook | Runtime (Colab Free) | GPU Required |
|---|---|---|---|
| 00 | Synthetic Data Generator | ~2 min | No |
| 01 | Event Tokenizer | ~3 min | No |
| 02 | Sequence Encoder (CoLES) | ~12 min | **Yes (T4)** |
| 03 | Graph Builder | ~4 min | No |
| 04 | GNN Enrichment (GraphSAGE) | ~8 min | **Yes (T4)** |
| 05 | Downstream Tasks | ~5 min | No |

**Total end-to-end runtime: ~34 minutes on Colab free tier with T4 GPU.**

Enable GPU in Colab: `Runtime → Change runtime type → T4 GPU`

---

## Synthetic Dataset

The synthetic dataset is modelled after real financial institution data structures. It contains:

- **5,000 customers** across 6 behavioural archetypes
- **~180,000 transactions** over 6 months
- **~120,000 app events** (logins, product views, session data)
- **2,500 counterparties** (merchants, utilities, salary sources, individuals)
- **~8,000 product records** (loans, cards, savings products)
- **~3,000 complaint records**

No real customer data is used anywhere in this project.

### Customer Archetypes (Ground Truth for Validation)

| Archetype | AEPD Stage | Behaviour Pattern |
|---|---|---|
| `digital_native` | D | High app engagement, multi-product, low complaints |
| `salary_spender` | P | Monthly salary credit, concentrated spend 26–30th |
| `loan_seeker` | E | Repeated loan product views, rarely converts |
| `branch_dependent` | E | Low digital engagement, mostly in-branch/USSD |
| `dormant_rich` | A→E | High balance, very low transaction frequency |
| `churn_risk` | P→A | Declining logins, recent complaints, reducing balance |

---

## Production Deployment

### Microsoft Fabric (Primary)

```
Bronze  → Raw ingestion from core banking / event streams
Silver  → Tokenized event sequences (Delta tables)
Gold    → Customer embeddings + graph enrichments
Serving → SQL endpoint → Power BI + FastAPI inference endpoint
```

See `production/fabric/` for full deployment guide.

### Azure ML (Fallback)

```
Azure Data Lake Gen2 → Azure ML Pipeline → Azure ML Endpoint
→ Azure AI Search (vector index) → Power BI
```

See `production/azure/` for full deployment guide.

---

## Research Foundation

This project implements and synthesises ideas from:

1. **CoLES: Contrastive Learning for Event Sequences with Self-Supervision**
   Babaev et al., SIGMOD 2022 — the core sequence encoding approach
   [arxiv.org/abs/2002.08232](https://arxiv.org/abs/2002.08232)

2. **Behavior Sequence Transformer for E-commerce Recommendation in Alibaba**
   Chen et al., 2019 — the BST architecture that inspired the sequence modelling
   [arxiv.org/abs/1905.06874](https://arxiv.org/abs/1905.06874)

3. **Beyond Isolated Clients: Integrating Graph-Based Embeddings into Event Sequence Models**
   WWW 2026 — combining CoLES with GNN over customer-merchant bipartite graphs
   [arxiv.org/abs/2604.09085](https://arxiv.org/abs/2604.09085)

4. **Alibaba AIPL Framework** — the customer journey model (Awareness → Interest → Purchase → Loyalty) adapted here as AEPD

---

## Key Design Decisions

**Why GRU and not Transformer for the encoder?**
GRU trains CoLES-style on 5,000 customers in under 12 minutes on a free Colab T4. A full Transformer takes 30–40 minutes and requires Colab Pro. The MVP uses GRU. The production architecture upgrades to a Transformer (BST-style) — the swap is one line of code.

**Why GraphSAGE and not GCN?**
GraphSAGE is *inductive* — it generates embeddings for new nodes (customers) not seen during training by sampling their neighbourhood. GCN is *transductive* — it breaks when you add new customers. For any institution onboarding hundreds of new accounts daily, GCN is not a production option.

**Why self-supervised (no labels)?**
Financial institutions rarely have clean, large-scale labels for customer intent. CoLES learns from the structure of the data itself — two random subsequences of the same customer's history should produce similar embeddings. This means the system works from day one without a labelling project.

**Why a bipartite graph (customers + counterparties)?**
Individual customer sequences don't reveal network patterns — like 200 customers all sending money to the same merchant every Friday (a shared employer paying via bank transfer), or a cluster of customers who share the same utility provider (a geographic cluster). The graph makes these invisible patterns visible.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

Areas for contribution:
- Additional downstream task notebooks (credit scoring, wealth segmentation)
- Alternative encoder architectures (Transformer, LSTM, Mamba)
- Production deployment scripts (Terraform, Fabric notebooks)
- Evaluation benchmarks on public financial datasets

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built as a research and production blueprint for financial AI teams.
Inspired by work at the intersection of representation learning, graph neural networks, and financial services.

---

*If this project helps your team, consider starring the repo and sharing your implementation experience.*
