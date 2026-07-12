"""
graph_builder.py
----------------
Builds a bipartite customer–counterparty graph from transaction data.

Graph structure:
  Nodes:
    - Customer nodes  (indexed 0 … N_customers-1)
    - Counterparty nodes (indexed N_customers … N_customers + N_counterparties - 1)

  Node features:
    - Customer nodes:  64-dim CoLES embedding (from encoder.py)
    - Counterparty nodes: 8-dim type-encoded feature vector

  Edges:
    - One directed edge per unique (customer → counterparty) pair
    - Edge features: [txn_count_bucket, total_amount_bucket,
                      dominant_channel_id, recency_bucket]

Why bipartite?
  Individual customer sequences are blind to network structure.
  The bipartite graph reveals patterns invisible to per-customer analysis:
    - Merchants receiving from many customers → identify merchant type
    - Customers sharing same salary source → same employer cluster
    - Customers sending to same food vendor → geographic cluster
    - Isolated customers (few edges) → high churn risk signal

Reference:
  "Beyond Isolated Clients: Integrating Graph-Based Embeddings
   into Event Sequence Models"  arXiv:2604.09085 (WWW 2026)
"""

import sys
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    GRAPH_EDGE_MIN_TXN, ENCODER_OUTPUT_DIM,
    COUNTERPARTY_TYPES, CHANNELS, AMOUNT_N_BINS, RANDOM_SEED
)

torch.manual_seed(RANDOM_SEED)

# Counterparty type → integer for node features
CP_TYPE_TO_ID = {t: i + 1 for i, t in enumerate(COUNTERPARTY_TYPES)}
CHANNEL_TO_ID = {c: i + 1 for i, c in enumerate(CHANNELS)}
N_CP_FEATURES = 8    # counterparty node feature dimension


# ─────────────────────────────────────────────
# NODE FEATURE BUILDERS
# ─────────────────────────────────────────────

def build_counterparty_features(counterparties: pd.DataFrame) -> np.ndarray:
    """
    Encode counterparty nodes into fixed-size feature vectors.

    Features per counterparty node:
      [type_id, is_merchant, bank_code_hash_bucket (1-8),
       zero-padding to N_CP_FEATURES]
    """
    features = []
    for _, row in counterparties.iterrows():
        type_id = CP_TYPE_TO_ID.get(row.get("counterparty_type", "UNKNOWN"), 0)
        is_merchant = int(row.get("is_merchant", 0))
        bank_hash = hash(str(row.get("bank_code", ""))) % 8 + 1

        feat = np.zeros(N_CP_FEATURES, dtype=np.float32)
        feat[0] = type_id / len(COUNTERPARTY_TYPES)     # normalised type
        feat[1] = is_merchant
        feat[2] = bank_hash / 8.0                        # normalised bank bucket
        # feat[3:] = zeros (reserved for future enrichment)
        features.append(feat)

    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────
# EDGE FEATURE BUILDERS
# ─────────────────────────────────────────────

def _bucket(value: float, n_bins: int, max_val: float) -> float:
    """Normalised bin ID for a scalar value."""
    return min(int(value / max_val * n_bins) + 1, n_bins) / n_bins


def build_edge_features(group: pd.DataFrame) -> np.ndarray:
    """
    Aggregate transaction group (customer, counterparty) → edge feature vector.

    Features:
      [txn_count_norm, total_amount_norm, dominant_channel_id,
       recency_bucket, success_rate]
    """
    txn_count = len(group)
    total_amount = group["amount"].sum() if "amount" in group.columns else 0.0
    recency_days = 0.0

    if "timestamp" in group.columns:
        ts = pd.to_datetime(group["timestamp"])
        most_recent = ts.max()
        ref_date = pd.Timestamp("2024-12-31")
        recency_days = (ref_date - most_recent).days

    dominant_channel = "APP"
    if "channel" in group.columns:
        dominant_channel = group["channel"].mode().iloc[0] if len(group) > 0 else "APP"

    success_rate = 1.0
    if "status" in group.columns:
        success_rate = (group["status"] == "SUCCESS").mean()

    return np.array([
        _bucket(txn_count, 20, 200),
        _bucket(total_amount, 32, 10_000_000),
        CHANNEL_TO_ID.get(dominant_channel, 1) / len(CHANNELS),
        _bucket(recency_days, 10, 180),
        success_rate,
    ], dtype=np.float32)


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────

class BipartiteGraphBuilder:
    """
    Builds a PyTorch Geometric-compatible bipartite graph.

    Node index mapping:
      [0, N_cust)                    → customer nodes
      [N_cust, N_cust + N_cp)        → counterparty nodes
    """

    def __init__(self, min_txn: int = GRAPH_EDGE_MIN_TXN):
        self.min_txn = min_txn
        self.customer_index = {}      # customer_id → node_idx
        self.counterparty_index = {}  # counterparty_id → node_idx
        self.n_customers = 0
        self.n_counterparties = 0

    def _build_index(self, customers: pd.DataFrame,
                     counterparties: pd.DataFrame):
        self.customer_index = {
            cid: i for i, cid in enumerate(customers["customer_id"])
        }
        self.n_customers = len(customers)

        self.counterparty_index = {
            cpid: self.n_customers + i
            for i, cpid in enumerate(counterparties["counterparty_id"])
        }
        self.n_counterparties = len(counterparties)

    def build(self, transactions: pd.DataFrame,
              customers: pd.DataFrame,
              counterparties: pd.DataFrame,
              embeddings: dict) -> dict:
        """
        Build the bipartite graph.

        Args:
            transactions:  raw transactions DataFrame
            customers:     customers DataFrame
            counterparties: counterparties DataFrame
            embeddings:    dict from encode_all_customers()

        Returns:
            graph_data dict compatible with PyTorch Geometric Data()
        """
        self._build_index(customers, counterparties)

        N = self.n_customers + self.n_counterparties
        n_feat_c = ENCODER_OUTPUT_DIM    # customer node features = embedding dim
        n_feat_cp = N_CP_FEATURES        # counterparty node features

        # Pad features to same dimension
        feat_dim = max(n_feat_c, n_feat_cp)
        node_features = np.zeros((N, feat_dim), dtype=np.float32)

        # Fill customer node features from embeddings
        print("  Building customer node features from embeddings...")
        for cust_id, idx in tqdm(self.customer_index.items()):
            if cust_id in embeddings:
                emb = embeddings[cust_id]["embedding"]
                node_features[idx, :len(emb)] = emb

        # Fill counterparty node features
        print("  Building counterparty node features...")
        cp_feats = build_counterparty_features(counterparties)
        for i, (_, row) in enumerate(counterparties.iterrows()):
            node_idx = self.counterparty_index.get(row["counterparty_id"])
            if node_idx is not None:
                f = cp_feats[i]
                node_features[node_idx, :len(f)] = f

        # Build edges from transactions
        print("  Building edges from transactions...")
        valid_txns = transactions[
            transactions["customer_id"].isin(self.customer_index) &
            transactions["counterparty_id"].isin(self.counterparty_index)
        ].copy()

        grouped = valid_txns.groupby(["customer_id", "counterparty_id"])

        edge_src, edge_dst, edge_feats = [], [], []

        for (cust_id, cp_id), grp in tqdm(grouped, desc="  Processing edges"):
            if len(grp) < self.min_txn:
                continue

            src = self.customer_index[cust_id]
            dst = self.counterparty_index[cp_id]
            ef = build_edge_features(grp)

            # Bidirectional edges
            edge_src.extend([src, dst])
            edge_dst.extend([dst, src])
            edge_feats.extend([ef, ef])

        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_attr = np.array(edge_feats, dtype=np.float32)

        n_edges = len(edge_src)
        print(f"\n  ✅ Graph built:")
        print(f"     Nodes: {N:,} ({self.n_customers:,} customers + {self.n_counterparties:,} counterparties)")
        print(f"     Edges: {n_edges:,} (bidirectional, min_txn={self.min_txn})")
        print(f"     Node feature dim: {feat_dim}")
        print(f"     Edge feature dim: {edge_attr.shape[1] if len(edge_feats) > 0 else 0}")

        return {
            "x":              torch.tensor(node_features, dtype=torch.float),
            "edge_index":     torch.tensor(edge_index, dtype=torch.long),
            "edge_attr":      torch.tensor(edge_attr, dtype=torch.float),
            "n_customers":    self.n_customers,
            "n_counterparties": self.n_counterparties,
            "customer_index": self.customer_index,
            "counterparty_index": self.counterparty_index,
        }


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

def save_graph(graph_data: dict, path: Path):
    # Convert tensors to numpy for pickle portability
    saveable = {
        k: v.numpy() if isinstance(v, torch.Tensor) else v
        for k, v in graph_data.items()
    }
    with open(path, "wb") as f:
        pickle.dump(saveable, f)
    print(f"  ✓ Graph saved to {path}")


def load_graph(path: Path) -> dict:
    with open(path, "rb") as f:
        raw = pickle.load(f)
    # Restore tensors
    graph_data = {}
    tensor_keys = {"x", "edge_index", "edge_attr"}
    for k, v in raw.items():
        if k in tensor_keys and isinstance(v, np.ndarray):
            graph_data[k] = torch.tensor(v)
        else:
            graph_data[k] = v
    return graph_data
