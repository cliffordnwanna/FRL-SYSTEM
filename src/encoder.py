"""
encoder.py
----------
CoLES-style self-supervised sequence encoder.

Learns customer embeddings from raw event sequences — no labels required.

Training objective (Contrastive Learning):
  - For each customer, sample 2 random subsequences of their event history
  - Train the model so that the 2 embeddings of the SAME customer are SIMILAR
  - Train the model so that embeddings of DIFFERENT customers are DISTANT
  - Loss: NT-Xent (Normalized Temperature-scaled Cross Entropy), same as SimCLR

Architecture:
  1. Embedding layer per feature column (vocab_size → embed_dim)
  2. Concatenate all feature embeddings → event vector
  3. GRU over sequence of event vectors → fixed-size customer embedding
  4. L2 normalise → unit sphere (required for cosine contrastive loss)

Reference:
  CoLES: Contrastive Learning for Event Sequences with Self-Supervision
  Babaev et al., SIGMOD 2022. arXiv:2002.08232
"""

import sys
import math
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    EMBED_DIM_PER_FEATURE, GRU_HIDDEN_DIM, ENCODER_OUTPUT_DIM,
    COLES_N_SUBSEQUENCES, COLES_MIN_SEQ_FRACTION,
    ENCODER_EPOCHS, ENCODER_BATCH_SIZE, ENCODER_LR,
    ENCODER_TEMPERATURE, RANDOM_SEED
)
from src.tokenizer import VOCAB_SIZES, N_FEATURES

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class CustomerEncoder(nn.Module):
    """
    GRU-based customer sequence encoder.

    Takes a padded token sequence of shape [batch, seq_len, n_features]
    and outputs a 64-dim L2-normalised customer embedding.
    """

    def __init__(self,
                 vocab_sizes: dict = VOCAB_SIZES,
                 embed_dim: int = EMBED_DIM_PER_FEATURE,
                 gru_hidden: int = GRU_HIDDEN_DIM,
                 output_dim: int = ENCODER_OUTPUT_DIM):
        super().__init__()

        self.embed_dim = embed_dim
        self.feature_names = list(vocab_sizes.keys())
        self.n_features = len(self.feature_names)

        # One embedding table per feature column
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(size, embed_dim, padding_idx=0)
            for name, size in vocab_sizes.items()
        })

        event_dim = embed_dim * self.n_features   # concatenated event vector size

        # GRU encoder
        self.gru = nn.GRU(
            input_size=event_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False,
        )

        # Projection to output dim
        self.proj = nn.Sequential(
            nn.Linear(gru_hidden, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, tokens: torch.Tensor,
                seq_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            tokens:       [batch, seq_len, n_features]  int tensor
            seq_lengths:  [batch] — actual length of each sequence (for masking)

        Returns:
            embedding:    [batch, output_dim]  float tensor, L2-normalised
        """
        batch, seq_len, n_feat = tokens.shape

        # Embed each feature column and concatenate
        embedded = []
        for i, name in enumerate(self.feature_names):
            col_tokens = tokens[:, :, i]          # [batch, seq_len]
            emb = self.embeddings[name](col_tokens)  # [batch, seq_len, embed_dim]
            embedded.append(emb)

        x = torch.cat(embedded, dim=-1)           # [batch, seq_len, event_dim]

        # Pack sequences if lengths provided (efficiency for padded sequences)
        if seq_lengths is not None:
            lengths_cpu = seq_lengths.cpu().clamp(min=1)
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )

        _, hidden = self.gru(x)                   # hidden: [n_layers, batch, hidden]
        last_hidden = hidden[-1]                  # [batch, hidden]

        embedding = self.proj(last_hidden)        # [batch, output_dim]
        embedding = F.normalize(embedding, p=2, dim=-1)  # L2 normalise

        return embedding


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class CoLESDataset(Dataset):
    """
    Dataset for CoLES contrastive training.

    For each customer, samples 2 random non-overlapping subsequences.
    The encoder should map both to similar embeddings.
    """

    def __init__(self, sequences: dict,
                 n_subsequences: int = COLES_N_SUBSEQUENCES,
                 min_frac: float = COLES_MIN_SEQ_FRACTION):
        self.customer_ids = list(sequences.keys())
        self.sequences = sequences
        self.n_subsequences = n_subsequences
        self.min_frac = min_frac

    def __len__(self):
        return len(self.customer_ids)

    def _sample_subsequence(self, tokens: np.ndarray, seq_len: int) -> tuple:
        """Sample a contiguous subsequence of at least min_frac of the sequence."""
        min_len = max(2, int(seq_len * self.min_frac))
        sub_len = random.randint(min_len, seq_len)
        start = random.randint(0, seq_len - sub_len)
        sub = tokens[start: start + sub_len]

        # Pad to MAX_SEQ_LENGTH
        from src.config import MAX_SEQ_LENGTH
        padded = np.zeros((MAX_SEQ_LENGTH, tokens.shape[1]), dtype=np.int32)
        padded[:sub_len] = sub
        return padded, sub_len

    def __getitem__(self, idx):
        cust_id = self.customer_ids[idx]
        seq_data = self.sequences[cust_id]
        tokens = seq_data["tokens"]
        seq_len = max(seq_data["seq_length"], 2)

        sub1, len1 = self._sample_subsequence(tokens, seq_len)
        sub2, len2 = self._sample_subsequence(tokens, seq_len)

        return {
            "sub1":   torch.tensor(sub1, dtype=torch.long),
            "len1":   torch.tensor(len1, dtype=torch.long),
            "sub2":   torch.tensor(sub2, dtype=torch.long),
            "len2":   torch.tensor(len2, dtype=torch.long),
            "cust_id": cust_id,
        }


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 temperature: float = ENCODER_TEMPERATURE) -> torch.Tensor:
    """
    NT-Xent Loss (Normalized Temperature-scaled Cross Entropy).
    Same as SimCLR. Both z1 and z2 are assumed to be L2-normalised.

    Args:
        z1, z2: [batch, dim] — two views of the same customers
    Returns:
        scalar loss
    """
    batch = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                     # [2B, dim]
    sim = torch.mm(z, z.T) / temperature               # [2B, 2B]

    # Mask out self-similarity
    mask = torch.eye(2 * batch, device=z.device).bool()
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch, 2 * batch, device=z.device),
        torch.arange(0, batch, device=z.device)
    ])

    loss = F.cross_entropy(sim, labels)
    return loss


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_encoder(sequences: dict,
                  epochs: int = ENCODER_EPOCHS,
                  batch_size: int = ENCODER_BATCH_SIZE,
                  lr: float = ENCODER_LR,
                  device: torch.device = None) -> CustomerEncoder:
    """
    Train the CoLES customer encoder on the provided sequences.

    Args:
        sequences: dict from FinancialTokenizer.transform()
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        device: torch device (auto-detected if None)

    Returns:
        Trained CustomerEncoder model
    """
    if device is None:
        device = DEVICE

    print(f"\n  Training CoLES Encoder on {device}")
    print(f"  Customers: {len(sequences):,} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"  Architecture: EmbedDim={EMBED_DIM_PER_FEATURE} × {N_FEATURES} features")
    print(f"                GRU hidden={GRU_HIDDEN_DIM} → Output={ENCODER_OUTPUT_DIM}D\n")

    dataset = CoLESDataset(sequences)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last=True)

    model = CustomerEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            sub1 = batch["sub1"].to(device)
            len1 = batch["len1"].to(device)
            sub2 = batch["sub2"].to(device)
            len2 = batch["len2"].to(device)

            z1 = model(sub1, len1)
            z2 = model(sub2, len2)

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        history.append(avg_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.6f}")

    print(f"\n  ✅ Training complete. Final loss: {history[-1]:.4f}")
    model.eval()
    return model, history


# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

@torch.no_grad()
def encode_all_customers(model: CustomerEncoder,
                         sequences: dict,
                         batch_size: int = 256,
                         device: torch.device = None) -> dict:
    """
    Encode all customers into embeddings using the trained model.

    Returns:
        dict: {customer_id: {"embedding": np.ndarray [output_dim],
                             "aepd_stage": str,
                             "archetype": str}}
    """
    if device is None:
        device = DEVICE

    model.eval()
    model.to(device)

    customer_ids = list(sequences.keys())
    embeddings = {}

    for i in tqdm(range(0, len(customer_ids), batch_size),
                  desc="  Encoding customers"):
        batch_ids = customer_ids[i: i + batch_size]
        tokens_list, lengths_list = [], []

        for cid in batch_ids:
            s = sequences[cid]
            tokens_list.append(s["tokens"])
            lengths_list.append(s["seq_length"])

        tokens_tensor = torch.tensor(
            np.stack(tokens_list), dtype=torch.long
        ).to(device)
        lengths_tensor = torch.tensor(lengths_list, dtype=torch.long).to(device)

        embs = model(tokens_tensor, lengths_tensor).cpu().numpy()

        for j, cid in enumerate(batch_ids):
            embeddings[cid] = {
                "embedding":  embs[j],
                "aepd_stage": sequences[cid].get("aepd_stage", "UNKNOWN"),
                "archetype":  sequences[cid].get("archetype", "UNKNOWN"),
            }

    print(f"\n  ✅ Encoded {len(embeddings):,} customers → {ENCODER_OUTPUT_DIM}D vectors")
    return embeddings


# ─────────────────────────────────────────────
# SAVE / LOAD
# ─────────────────────────────────────────────

def save_encoder(model: CustomerEncoder, path: Path):
    torch.save({
        "model_state": model.state_dict(),
        "vocab_sizes": VOCAB_SIZES,
        "config": {
            "embed_dim": EMBED_DIM_PER_FEATURE,
            "gru_hidden": GRU_HIDDEN_DIM,
            "output_dim": ENCODER_OUTPUT_DIM,
        }
    }, path)
    print(f"  ✓ Encoder saved to {path}")


def load_encoder(path: Path, device: torch.device = None) -> CustomerEncoder:
    if device is None:
        device = DEVICE
    ckpt = torch.load(path, map_location=device)
    model = CustomerEncoder(
        vocab_sizes=ckpt["vocab_sizes"],
        embed_dim=ckpt["config"]["embed_dim"],
        gru_hidden=ckpt["config"]["gru_hidden"],
        output_dim=ckpt["config"]["output_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model


def save_embeddings(embeddings: dict, path: Path):
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"  ✓ Embeddings saved to {path}")


def load_embeddings(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
