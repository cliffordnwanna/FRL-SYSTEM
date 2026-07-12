"""
config.py
---------
Single source of truth for all hyperparameters, constants, and paths.
Change values here — everything else reads from here.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "synthetic"
SCHEMA_DIR = ROOT_DIR / "data" / "schemas"
DOCS_DIR = ROOT_DIR / "docs" / "results"

# ─────────────────────────────────────────────
# SYNTHETIC DATA PARAMETERS
# ─────────────────────────────────────────────
N_CUSTOMERS = 5_000
N_COUNTERPARTIES = 2_500
HISTORY_DAYS = 180          # 6 months of event history
RANDOM_SEED = 42

# Transaction volume per archetype (avg transactions per customer per month)
TXN_VOLUME = {
    "digital_native":   45,
    "salary_spender":   20,
    "loan_seeker":      8,
    "branch_dependent": 6,
    "dormant_rich":     3,
    "churn_risk":       12,   # declining over time
}

ARCHETYPE_DISTRIBUTION = {
    "digital_native":   0.20,
    "salary_spender":   0.30,
    "loan_seeker":      0.15,
    "branch_dependent": 0.15,
    "dormant_rich":     0.10,
    "churn_risk":       0.10,
}

ARCHETYPE_TO_AEPD = {
    "digital_native":   "D",
    "salary_spender":   "P",
    "loan_seeker":      "E",
    "branch_dependent": "E",
    "dormant_rich":     "A",
    "churn_risk":       "P",   # was P, trending toward A
}

# ─────────────────────────────────────────────
# TOKENIZER PARAMETERS
# ─────────────────────────────────────────────
MAX_SEQ_LENGTH = 256        # max events per customer (pad/truncate to this)
AMOUNT_N_BINS = 32          # number of log-scale amount buckets
TIME_DELTA_N_BINS = 10      # number of time-gap buckets (in hours)

# Categorical vocabularies
CHANNELS = ["APP", "USSD", "POS", "WEB", "BRANCH", "AGENT"]
EVENT_TYPES = [
    "TXN_DEBIT", "TXN_CREDIT",
    "LOGIN", "VIEW_BALANCE", "VIEW_LOAN_PRODUCT",
    "VIEW_STATEMENT", "OPEN_TRANSFER_SCREEN",
    "PRODUCT_VIEW", "COMPLAINT_RAISED", "COMPLAINT_RESOLVED",
]
PRODUCT_CODES = [
    "TRANSFER", "AIRTIME_TOPUP", "BILL_PAYMENT",
    "LOAN_REPAYMENT", "POS_PURCHASE", "CARD_PAYMENT",
    "SAVINGS_DEPOSIT", "FIXED_DEPOSIT", "LOAN_DISBURSEMENT",
]
COUNTERPARTY_TYPES = [
    "FOOD_VENDOR", "UTILITY", "SALARY_SOURCE",
    "INDIVIDUAL", "SCHOOL_FEE", "TELECOMS",
    "ECOMMERCE", "FUEL_STATION", "HEALTH",
    "GOVERNMENT", "SUPERMARKET", "UNKNOWN",
]
HOUR_BUCKETS = ["NIGHT", "MORNING", "AFTERNOON", "EVENING"]   # 4 buckets
DAY_BUCKETS = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

# ─────────────────────────────────────────────
# ENCODER (CoLES) PARAMETERS
# ─────────────────────────────────────────────
EMBED_DIM_PER_FEATURE = 16     # embedding size for each token column
GRU_HIDDEN_DIM = 128
ENCODER_OUTPUT_DIM = 64        # final customer embedding size
COLES_N_SUBSEQUENCES = 2       # number of subsequences per customer
COLES_MIN_SEQ_FRACTION = 0.4   # min fraction of sequence per subsequence
ENCODER_EPOCHS = 30
ENCODER_BATCH_SIZE = 64
ENCODER_LR = 1e-3
ENCODER_TEMPERATURE = 0.07     # NT-Xent loss temperature

# ─────────────────────────────────────────────
# GRAPH PARAMETERS
# ─────────────────────────────────────────────
GRAPH_EDGE_MIN_TXN = 2         # min transactions to create an edge
GRAPHSAGE_HIDDEN_DIM = 64
GRAPHSAGE_OUTPUT_DIM = 64
GRAPHSAGE_N_LAYERS = 2
GNN_EPOCHS = 30
GNN_BATCH_SIZE = 512
GNN_LR = 1e-3

# ─────────────────────────────────────────────
# DOWNSTREAM PARAMETERS
# ─────────────────────────────────────────────
N_SEGMENTS = 4                 # K for K-Means (maps to A/E/P/D)
UMAP_N_COMPONENTS = 2
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
ANOMALY_THRESHOLD_PERCENTILE = 95   # top 5% distances = anomalies

# AEPD label mapping (cluster → stage — assigned post-hoc from cluster analysis)
AEPD_STAGES = ["A", "E", "P", "D"]
AEPD_COLORS = {
    "A": "#E74C3C",   # red — dormant/unengaged
    "E": "#F39C12",   # orange — exploring
    "P": "#2ECC71",   # green — active
    "D": "#3498DB",   # blue — deeply loyal
}
