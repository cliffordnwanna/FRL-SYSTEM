"""
gnn.py
------
GraphSAGE-based graph neural network for embedding enrichment.

Why GraphSAGE over GCN:
  GraphSAGE is INDUCTIVE — it generates embeddings for new nodes
  not seen during training by sampling and aggregating their neighbours.
  GCN is TRANSDUCTIVE — it breaks when you add new customers.
  For a financial institution onboarding hundreds of new accounts daily,
  GCN is not a production option.

Training: Unsupervised link prediction
  Positive pairs: connected (customer, counterparty) pairs
  Negative pairs: random unconnected pairs
  Loss: Binary Cross Entropy on dot product similarity

After training, extract enriched customer node embeddings.
These embeddings are now aware of the full network structure —
which merchants a customer shares with others, what clusters exist, etc.

Reference:
  Hamilton et al., "Inductive Representation Learning on Large Graphs"
  NeurIPS 2017 (GraphSAGE paper)
"""

import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    GRAPHSAGE_HIDDEN_DIM, GRAPHSAGE_OUTPUT_DIM, GRAPHSAGE_N_LAYERS,
    GNN_EPOCHS, GNN_LR, ENCODER_OUTPUT_DIM, RANDOM_SEED
)

torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class GraphSAGELayer(nn.Module):
    """Single GraphSAGE layer — mean aggregation, via PyTorch Geometric's SAGEConv."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = SAGEConv(in_dim, out_dim, aggr="mean")
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:           [N, in_dim]
            edge_index:  [2, E]
        Returns:
            out:         [N, out_dim]
        """
        out = self.norm(F.relu(self.conv(x, edge_index)))
        return F.normalize(out, p=2, dim=-1)


class GraphSAGEEncoder(nn.Module):
    """
    Multi-layer GraphSAGE encoder.
    Takes graph (x, edge_index) and outputs enriched node embeddings.
    """

    def __init__(self,
                 in_dim: int = None,
                 hidden_dim: int = GRAPHSAGE_HIDDEN_DIM,
                 out_dim: int = GRAPHSAGE_OUTPUT_DIM,
                 n_layers: int = GRAPHSAGE_N_LAYERS):
        super().__init__()

        if in_dim is None:
            in_dim = ENCODER_OUTPUT_DIM

        dims = [in_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList([
            GraphSAGELayer(dims[i], dims[i + 1])
            for i in range(n_layers)
        ])
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        return x


# ─────────────────────────────────────────────
# UNSUPERVISED LINK PREDICTION TRAINING
# ─────────────────────────────────────────────

def negative_sampling(edge_index: torch.Tensor, n_nodes: int,
                      n_neg: int) -> torch.Tensor:
    """Random negative edges (unconnected pairs)."""
    neg_src = torch.randint(0, n_nodes, (n_neg,))
    neg_dst = torch.randint(0, n_nodes, (n_neg,))
    return torch.stack([neg_src, neg_dst], dim=0)


def train_gnn(graph_data: dict,
              epochs: int = GNN_EPOCHS,
              lr: float = GNN_LR,
              device: torch.device = None) -> GraphSAGEEncoder:
    """
    Train GraphSAGE with unsupervised link prediction objective.

    Args:
        graph_data:  dict from BipartiteGraphBuilder.build()
        epochs:      training epochs
        lr:          learning rate
        device:      torch device

    Returns:
        Trained GraphSAGEEncoder
    """
    if device is None:
        device = DEVICE

    x = graph_data["x"].to(device)
    edge_index = graph_data["edge_index"].to(device)
    N = x.shape[0]
    in_dim = x.shape[1]
    n_edges = edge_index.shape[1]

    print(f"\n  Training GraphSAGE on {device}")
    print(f"  Nodes: {N:,} | Edges: {n_edges:,} | In dim: {in_dim}")
    print(f"  Epochs: {epochs} | LR: {lr}\n")

    model = GraphSAGEEncoder(in_dim=in_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        # Forward pass
        z = model(x, edge_index)   # [N, out_dim]

        # Positive pairs
        pos_src = edge_index[0]
        pos_dst = edge_index[1]
        pos_scores = (z[pos_src] * z[pos_dst]).sum(dim=-1)  # dot product

        # Negative pairs (random)
        neg_edges = negative_sampling(edge_index, N, n_edges)
        neg_src = neg_edges[0].to(device)
        neg_dst = neg_edges[1].to(device)
        neg_scores = (z[neg_src] * z[neg_dst]).sum(dim=-1)

        # BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )
        loss = (pos_loss + neg_loss) / 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        history.append(loss.item())

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f}")

    print(f"\n  ✅ GNN training complete. Final loss: {history[-1]:.4f}")
    model.eval()
    return model, history


# ─────────────────────────────────────────────
# INFERENCE — ENRICH CUSTOMER EMBEDDINGS
# ─────────────────────────────────────────────

@torch.no_grad()
def enrich_embeddings(model: GraphSAGEEncoder,
                      graph_data: dict,
                      original_embeddings: dict,
                      device: torch.device = None) -> dict:
    """
    Run graph-enriched forward pass and extract customer node embeddings.

    Args:
        model:               trained GraphSAGEEncoder
        graph_data:          dict from BipartiteGraphBuilder.build()
        original_embeddings: dict from encode_all_customers()
        device:              torch device

    Returns:
        enriched_embeddings: dict {customer_id: {"embedding": np.ndarray,
                                                  "aepd_stage": str,
                                                  "archetype": str}}
    """
    if device is None:
        device = DEVICE

    model.eval()
    model.to(device)

    x = graph_data["x"].to(device)
    edge_index = graph_data["edge_index"].to(device)

    # Full graph forward pass
    z = model(x, edge_index).cpu().numpy()  # [N, out_dim]

    n_customers = graph_data["n_customers"]
    customer_index = graph_data["customer_index"]

    enriched = {}
    for cust_id, node_idx in customer_index.items():
        enriched[cust_id] = {
            "embedding":  z[node_idx],
            "aepd_stage": original_embeddings.get(cust_id, {}).get("aepd_stage", "UNKNOWN"),
            "archetype":  original_embeddings.get(cust_id, {}).get("archetype", "UNKNOWN"),
            "original_embedding": original_embeddings.get(cust_id, {}).get("embedding"),
        }

    print(f"  ✅ Enriched {len(enriched):,} customer embeddings with graph signals")
    return enriched


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

def save_gnn(model: GraphSAGEEncoder, path: Path):
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "hidden_dim": GRAPHSAGE_HIDDEN_DIM,
            "out_dim": GRAPHSAGE_OUTPUT_DIM,
            "n_layers": GRAPHSAGE_N_LAYERS,
        }
    }, path)
    print(f"  ✓ GNN saved to {path}")


def load_gnn(path: Path, in_dim: int,
             device: torch.device = None) -> GraphSAGEEncoder:
    if device is None:
        device = DEVICE
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model = GraphSAGEEncoder(
        in_dim=in_dim,
        hidden_dim=cfg["hidden_dim"],
        out_dim=cfg["out_dim"],
        n_layers=cfg["n_layers"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model
