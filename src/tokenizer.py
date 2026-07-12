"""
tokenizer.py
------------
Converts raw financial event rows into padded integer token sequences
per customer — the "financial language" that feeds the sequence encoder.

Key design decisions:
  - One token = one event row (not a transaction batch, not an episode)
  - Each event is encoded as a tuple of integer IDs, one per feature column
  - Numeric amounts are log-bucketed into 32 bins
  - Time gaps between events are bucketed into 10 bins
  - Sequences are padded/truncated to MAX_SEQ_LENGTH (256)
  - The model learns what "episodes" are — we don't hard-code them

Output format per customer:
  {
    "customer_id": "CUST_000001",
    "token_sequences": np.ndarray of shape [MAX_SEQ_LENGTH, N_FEATURES],
    "seq_length": int  (actual length before padding)
  }
"""

import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    MAX_SEQ_LENGTH, AMOUNT_N_BINS, TIME_DELTA_N_BINS,
    CHANNELS, EVENT_TYPES, PRODUCT_CODES, COUNTERPARTY_TYPES,
    HOUR_BUCKETS, DAY_BUCKETS, DATA_DIR
)

# ─────────────────────────────────────────────
# VOCABULARY
# ─────────────────────────────────────────────

# PAD token = 0 for every vocabulary
PAD_ID = 0

def build_vocab() -> dict:
    """
    Build vocabulary mappings for each categorical feature.
    Index 0 is always reserved for PAD.
    """
    def _to_vocab(items: list) -> dict:
        return {item: i + 1 for i, item in enumerate(items)}

    return {
        "channel":          _to_vocab(CHANNELS),
        "event_type":       _to_vocab(EVENT_TYPES),
        "product_code":     _to_vocab(PRODUCT_CODES),
        "counterparty_type": _to_vocab(COUNTERPARTY_TYPES),
        "hour_bucket":      _to_vocab(HOUR_BUCKETS),
        "day_bucket":       _to_vocab(DAY_BUCKETS),
        # amount_bucket and time_delta_bucket are numeric → 1-indexed bin IDs
    }

VOCAB = build_vocab()

VOCAB_SIZES = {
    "channel":            len(CHANNELS) + 1,
    "event_type":         len(EVENT_TYPES) + 1,
    "product_code":       len(PRODUCT_CODES) + 1,
    "counterparty_type":  len(COUNTERPARTY_TYPES) + 1,
    "hour_bucket":        len(HOUR_BUCKETS) + 1,
    "day_bucket":         len(DAY_BUCKETS) + 1,
    "amount_bucket":      AMOUNT_N_BINS + 1,
    "time_delta_bucket":  TIME_DELTA_N_BINS + 1,
}

N_FEATURES = len(VOCAB_SIZES)   # 8 features per event token

# Amount bin edges (log-scale, ₦100 to ₦10M)
AMOUNT_BIN_EDGES = np.logspace(np.log10(100), np.log10(10_000_000), AMOUNT_N_BINS + 1)

# Time delta bin edges in hours (0 to 720h = 30 days)
TIME_DELTA_BIN_EDGES = np.concatenate([
    [0], np.logspace(np.log10(0.1), np.log10(720), TIME_DELTA_N_BINS)
])


# ─────────────────────────────────────────────
# FEATURE ENCODERS
# ─────────────────────────────────────────────

def encode_amount(amount: float) -> int:
    """Log-bucket amount into 1–32. Returns 0 (PAD) for missing/zero."""
    if pd.isna(amount) or amount <= 0:
        return PAD_ID
    bin_id = int(np.digitize(amount, AMOUNT_BIN_EDGES))
    return min(bin_id, AMOUNT_N_BINS)   # cap at max bin


def encode_time_delta(delta_hours: float) -> int:
    """Bucket time gap into 1–10. Returns 1 for very short gaps."""
    if pd.isna(delta_hours) or delta_hours < 0:
        return 1
    bin_id = int(np.digitize(delta_hours, TIME_DELTA_BIN_EDGES))
    return min(max(bin_id, 1), TIME_DELTA_N_BINS)


def encode_hour(ts: pd.Timestamp) -> int:
    hour = ts.hour
    if 5 <= hour < 12:
        bucket = "MORNING"
    elif 12 <= hour < 17:
        bucket = "AFTERNOON"
    elif 17 <= hour < 21:
        bucket = "EVENING"
    else:
        bucket = "NIGHT"
    return VOCAB["hour_bucket"].get(bucket, PAD_ID)


def encode_day(ts: pd.Timestamp) -> int:
    day_names = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
    return VOCAB["day_bucket"].get(day_names[ts.dayofweek], PAD_ID)


def encode_event_row(row: dict, prev_ts: pd.Timestamp, counterparty_map: dict) -> np.ndarray:
    """
    Encode a single event row into a 1D array of 8 integer token IDs.

    Feature order (N_FEATURES = 8):
      [channel, event_type, product_code, counterparty_type,
       hour_bucket, day_bucket, amount_bucket, time_delta_bucket]
    """
    ts = pd.Timestamp(row.get("timestamp", row.get("ts", pd.NaT)))

    # Time delta since previous event (hours)
    if prev_ts is not None and not pd.isna(ts):
        delta_hours = (ts - prev_ts).total_seconds() / 3600
    else:
        delta_hours = float("nan")

    # Counterparty type lookup
    cp_id = row.get("counterparty_id", None)
    cp_type = counterparty_map.get(cp_id, "UNKNOWN") if cp_id else "UNKNOWN"

    token = np.array([
        VOCAB["channel"].get(str(row.get("channel", "")), PAD_ID),
        VOCAB["event_type"].get(str(row.get("event_type", row.get("txn_type", ""))), PAD_ID),
        VOCAB["product_code"].get(str(row.get("product_code", "")), PAD_ID),
        VOCAB["counterparty_type"].get(cp_type, PAD_ID),
        encode_hour(ts) if not pd.isna(ts) else PAD_ID,
        encode_day(ts) if not pd.isna(ts) else PAD_ID,
        encode_amount(row.get("amount", 0.0)),
        encode_time_delta(delta_hours),
    ], dtype=np.int32)

    return token


# ─────────────────────────────────────────────
# MAIN TOKENIZER CLASS
# ─────────────────────────────────────────────

class FinancialTokenizer:
    """
    Converts merged event dataframes into padded token sequences per customer.

    Usage:
        tokenizer = FinancialTokenizer()
        sequences = tokenizer.fit_transform(transactions, app_events, counterparties)
        tokenizer.save(path)
    """

    def __init__(self, max_seq_length: int = MAX_SEQ_LENGTH):
        self.max_seq_length = max_seq_length
        self.vocab = VOCAB
        self.vocab_sizes = VOCAB_SIZES
        self.n_features = N_FEATURES

    def _build_counterparty_map(self, counterparties: pd.DataFrame) -> dict:
        return dict(zip(
            counterparties["counterparty_id"],
            counterparties["counterparty_type"]
        ))

    def _merge_events(self, transactions: pd.DataFrame,
                      app_events: pd.DataFrame) -> pd.DataFrame:
        """
        Merge transaction and app event tables into a unified event stream.
        This is the combined "financial language corpus".
        """
        txn_cols = {
            "txn_id": "event_id",
            "customer_id": "customer_id",
            "counterparty_id": "counterparty_id",
            "amount": "amount",
            "txn_type": "event_type",
            "channel": "channel",
            "product_code": "product_code",
            "timestamp": "timestamp",
        }
        app_cols = {
            "event_id": "event_id",
            "customer_id": "customer_id",
            "event_type": "event_type",
            "channel": "channel",
            "timestamp": "timestamp",
        }

        txn_df = transactions.rename(columns={k: v for k, v in txn_cols.items()
                                              if k in transactions.columns})
        txn_df = txn_df.reindex(columns=[
            "event_id", "customer_id", "counterparty_id",
            "amount", "event_type", "channel", "product_code", "timestamp"
        ])
        # txn_type values are "CREDIT"/"DEBIT"; EVENT_TYPES vocab uses "TXN_CREDIT"/"TXN_DEBIT"
        txn_df["event_type"] = "TXN_" + txn_df["event_type"].astype(str)

        app_df = app_events.reindex(columns=[
            "event_id", "customer_id", "event_type",
            "channel", "timestamp"
        ])

        merged = pd.concat([txn_df, app_df], ignore_index=True)
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])
        merged = merged.sort_values(["customer_id", "timestamp"])
        return merged

    def transform(self, transactions: pd.DataFrame,
                  app_events: pd.DataFrame,
                  counterparties: pd.DataFrame,
                  customers: pd.DataFrame = None) -> dict:
        """
        Build token sequences for all customers.

        Returns:
            dict: {
                customer_id: {
                    "tokens":      np.ndarray [max_seq_length, n_features],
                    "seq_length":  int,
                    "aepd_stage":  str (if customers df provided)
                }
            }
        """
        cp_map = self._build_counterparty_map(counterparties)
        events = self._merge_events(transactions, app_events)

        # Optional: ground truth AEPD stage per customer
        aepd_map = {}
        if customers is not None and "aepd_stage" in customers.columns:
            aepd_map = dict(zip(customers["customer_id"], customers["aepd_stage"]))
        archetype_map = {}
        if customers is not None and "archetype" in customers.columns:
            archetype_map = dict(zip(customers["customer_id"], customers["archetype"]))

        sequences = {}
        grouped = events.groupby("customer_id")

        for cust_id, group in tqdm(grouped, desc="  Tokenizing customers"):
            group = group.sort_values("timestamp")
            rows = group.to_dict("records")

            tokens = []
            prev_ts = None

            for row in rows:
                token = encode_event_row(row, prev_ts, cp_map)
                tokens.append(token)
                ts = pd.Timestamp(row.get("timestamp", pd.NaT))
                if not pd.isna(ts):
                    prev_ts = ts

            # Truncate to last MAX_SEQ_LENGTH events (most recent)
            if len(tokens) > self.max_seq_length:
                tokens = tokens[-self.max_seq_length:]

            seq_length = len(tokens)

            # Pad to max_seq_length with zero vectors
            padded = np.zeros((self.max_seq_length, self.n_features), dtype=np.int32)
            if seq_length > 0:
                padded[:seq_length] = np.array(tokens)

            sequences[cust_id] = {
                "tokens":     padded,
                "seq_length": seq_length,
                "aepd_stage": aepd_map.get(cust_id, "UNKNOWN"),
                "archetype":  archetype_map.get(cust_id, "UNKNOWN"),
            }

        return sequences

    def save(self, path: Path):
        """Save tokenizer state (vocab + config) for reproducibility."""
        state = {
            "vocab": self.vocab,
            "vocab_sizes": self.vocab_sizes,
            "n_features": self.n_features,
            "max_seq_length": self.max_seq_length,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        print(f"  ✓ Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "FinancialTokenizer":
        with open(path, "rb") as f:
            state = pickle.load(f)
        tok = cls(max_seq_length=state["max_seq_length"])
        tok.vocab = state["vocab"]
        tok.vocab_sizes = state["vocab_sizes"]
        tok.n_features = state["n_features"]
        return tok


# ─────────────────────────────────────────────
# CONVENIENCE FUNCTION
# ─────────────────────────────────────────────

def tokenize_from_csvs(data_dir: Path = None) -> tuple:
    """
    Load CSVs and tokenize in one call. Convenience for notebooks.

    Returns:
        (sequences: dict, tokenizer: FinancialTokenizer)
    """
    if data_dir is None:
        data_dir = DATA_DIR

    print("\n  Loading data...")
    transactions   = pd.read_csv(data_dir / "transactions.csv")
    app_events     = pd.read_csv(data_dir / "app_events.csv")
    counterparties = pd.read_csv(data_dir / "counterparties.csv")
    customers      = pd.read_csv(data_dir / "customers.csv")

    print(f"  → {len(transactions):,} transactions | {len(app_events):,} app events")

    tokenizer = FinancialTokenizer()
    sequences = tokenizer.transform(transactions, app_events, counterparties, customers)

    print(f"\n  ✅ Tokenized {len(sequences):,} customers")
    print(f"     Sequence shape per customer: ({MAX_SEQ_LENGTH}, {N_FEATURES})")
    print(f"     Vocabulary sizes: {VOCAB_SIZES}")

    return sequences, tokenizer
