"""
data_generator.py
-----------------
Synthetic financial data factory.

Generates realistic-shaped data modelled after core banking system tables
found in any retail bank or fintech. No real customer data is used.

Outputs (all saved to data/synthetic/):
  - customers.csv
  - transactions.csv
  - app_events.csv
  - counterparties.csv
  - products.csv
  - complaints.csv

Usage:
  python src/data_generator.py
  # or from a notebook:
  from src.data_generator import generate_all
  generate_all()
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm

# Allow running from repo root or from src/
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    N_CUSTOMERS, N_COUNTERPARTIES, HISTORY_DAYS, RANDOM_SEED,
    TXN_VOLUME, ARCHETYPE_DISTRIBUTION, ARCHETYPE_TO_AEPD,
    CHANNELS, EVENT_TYPES, PRODUCT_CODES, COUNTERPARTY_TYPES,
    DATA_DIR
)

# ─────────────────────────────────────────────
# SEED
# ─────────────────────────────────────────────
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

END_DATE = datetime(2024, 12, 31)
START_DATE = END_DATE - timedelta(days=HISTORY_DAYS)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _rand_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def _assign_archetypes(n: int) -> list:
    archetypes = list(ARCHETYPE_DISTRIBUTION.keys())
    probs = list(ARCHETYPE_DISTRIBUTION.values())
    return list(np.random.choice(archetypes, size=n, p=probs))


# ─────────────────────────────────────────────
# 1. COUNTERPARTIES
# ─────────────────────────────────────────────
def generate_counterparties(n: int = N_COUNTERPARTIES) -> pd.DataFrame:
    """
    Merchants, utility companies, salary sources, and individuals
    that customers send money to or receive money from.
    """
    ctype_weights = [0.15, 0.08, 0.05, 0.35, 0.05,
                     0.08, 0.08, 0.05, 0.04, 0.03, 0.03, 0.01]

    bank_codes = [f"BNK{str(i).zfill(3)}" for i in range(1, 30)]

    records = []
    for i in range(n):
        ctype = np.random.choice(COUNTERPARTY_TYPES, p=ctype_weights)
        records.append({
            "counterparty_id": f"CP_{str(i+1).zfill(5)}",
            "counterparty_type": ctype,
            "bank_code": random.choice(bank_codes),
            "is_merchant": int(ctype not in ["INDIVIDUAL", "SALARY_SOURCE"]),
        })

    df = pd.DataFrame(records)
    print(f"  ✓ Counterparties: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# 2. CUSTOMERS
# ─────────────────────────────────────────────
def generate_customers(n: int = N_CUSTOMERS) -> pd.DataFrame:
    archetypes = _assign_archetypes(n)
    account_types = ["SAVINGS", "CURRENT", "PREMIUM_SAVINGS"]
    onboarding_channels = ["APP", "BRANCH", "AGENT", "WEB"]
    regions = ["SOUTH_WEST", "SOUTH_EAST", "NORTH_WEST",
               "NORTH_CENTRAL", "SOUTH_SOUTH", "NORTH_EAST"]

    onboarding_weights = {
        "digital_native":   [0.70, 0.10, 0.10, 0.10],
        "salary_spender":   [0.40, 0.30, 0.20, 0.10],
        "loan_seeker":      [0.50, 0.20, 0.20, 0.10],
        "branch_dependent": [0.05, 0.70, 0.20, 0.05],
        "dormant_rich":     [0.20, 0.60, 0.15, 0.05],
        "churn_risk":       [0.45, 0.35, 0.15, 0.05],
    }

    records = []
    for i, arch in enumerate(archetypes):
        tenure = random.randint(1, 84)
        records.append({
            "customer_id": f"CUST_{str(i+1).zfill(6)}",
            "archetype": arch,
            "aepd_stage": ARCHETYPE_TO_AEPD[arch],
            "account_type": random.choice(account_types),
            "tenure_months": tenure,
            "onboarding_channel": np.random.choice(
                onboarding_channels, p=onboarding_weights[arch]
            ),
            "region": random.choice(regions),
            "avg_daily_balance_bucket": _balance_bucket(arch),
            "created_at": (END_DATE - timedelta(days=tenure * 30)).strftime("%Y-%m-%d"),
        })

    df = pd.DataFrame(records)
    print(f"  ✓ Customers: {len(df):,} rows | Archetypes: {df['archetype'].value_counts().to_dict()}")
    return df


def _balance_bucket(arch: str) -> str:
    buckets = ["MICRO", "LOW", "MID", "HIGH", "PREMIUM"]
    weights = {
        "digital_native":   [0.05, 0.20, 0.40, 0.25, 0.10],
        "salary_spender":   [0.10, 0.35, 0.40, 0.12, 0.03],
        "loan_seeker":      [0.20, 0.45, 0.30, 0.04, 0.01],
        "branch_dependent": [0.15, 0.40, 0.35, 0.08, 0.02],
        "dormant_rich":     [0.02, 0.08, 0.20, 0.35, 0.35],
        "churn_risk":       [0.25, 0.40, 0.30, 0.04, 0.01],
    }
    return np.random.choice(buckets, p=weights[arch])


# ─────────────────────────────────────────────
# 3. TRANSACTIONS
# ─────────────────────────────────────────────
def generate_transactions(customers: pd.DataFrame,
                          counterparties: pd.DataFrame) -> pd.DataFrame:
    """
    NIP/NIBSS-style transaction records. Each row = one financial event.
    Amount distribution is log-normal (realistic for retail banking).
    """
    cp_ids = counterparties["counterparty_id"].tolist()
    salary_cps = counterparties[
        counterparties["counterparty_type"] == "SALARY_SOURCE"
    ]["counterparty_id"].tolist()

    channel_weights = {
        "digital_native":   [0.60, 0.10, 0.15, 0.10, 0.05, 0.00],
        "salary_spender":   [0.35, 0.25, 0.25, 0.10, 0.05, 0.00],
        "loan_seeker":      [0.45, 0.25, 0.15, 0.10, 0.05, 0.00],
        "branch_dependent": [0.05, 0.40, 0.30, 0.05, 0.15, 0.05],
        "dormant_rich":     [0.20, 0.15, 0.20, 0.10, 0.30, 0.05],
        "churn_risk":       [0.30, 0.30, 0.20, 0.10, 0.10, 0.00],
    }

    amount_params = {
        "digital_native":   (9.5, 1.2),    # log-normal (mu, sigma) in NGN
        "salary_spender":   (10.0, 1.0),
        "loan_seeker":      (8.5, 1.3),
        "branch_dependent": (9.8, 1.1),
        "dormant_rich":     (12.0, 1.5),
        "churn_risk":       (8.8, 1.2),
    }

    records = []
    txn_counter = 1

    for _, cust in tqdm(customers.iterrows(), total=len(customers),
                        desc="  Generating transactions"):
        arch = cust["archetype"]
        n_txns = max(1, int(np.random.poisson(
            TXN_VOLUME[arch] * (HISTORY_DAYS / 30)
        )))

        # churn_risk customers have declining volume in last 60 days
        if arch == "churn_risk":
            n_txns = max(1, int(n_txns * 0.6))

        mu, sigma = amount_params[arch]

        for _ in range(n_txns):
            ts = _rand_date(START_DATE, END_DATE)
            is_credit = random.random() < 0.25   # 25% credits, 75% debits

            # salary credit: once a month, 25th–28th
            if arch == "salary_spender" and is_credit and salary_cps:
                cp_id = random.choice(salary_cps)
                amount = float(np.random.lognormal(11.5, 0.3))  # ₦100K-ish salary
                product = "TXN_CREDIT"
                # pin salary day
                ts = ts.replace(day=min(ts.day, 28))
                ts = ts.replace(day=max(ts.day, 25))
            else:
                cp_id = random.choice(cp_ids)
                amount = float(np.random.lognormal(mu, sigma))
                product = random.choice([
                    "TRANSFER", "AIRTIME_TOPUP", "BILL_PAYMENT",
                    "POS_PURCHASE", "CARD_PAYMENT"
                ])

            records.append({
                "txn_id": f"TXN_{str(txn_counter).zfill(8)}",
                "customer_id": cust["customer_id"],
                "counterparty_id": cp_id,
                "amount": round(min(amount, 5_000_000), 2),
                "txn_type": "CREDIT" if is_credit else "DEBIT",
                "channel": np.random.choice(CHANNELS, p=channel_weights[arch]),
                "product_code": product,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "SUCCESS" if random.random() > 0.03 else "FAILED",
            })
            txn_counter += 1

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    print(f"  ✓ Transactions: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# 4. APP EVENTS
# ─────────────────────────────────────────────
def generate_app_events(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Digital platform event log — logins, screen views, product exploration.
    High-signal for intent detection (e.g., loan_seeker repeatedly views loan screen).
    """
    # Monthly app session rates by archetype
    session_rates = {
        "digital_native":   30,
        "salary_spender":   12,
        "loan_seeker":      20,   # high app usage but specific screens
        "branch_dependent": 3,
        "dormant_rich":     5,
        "churn_risk":       8,    # declining
    }

    # Probability of each event type per session, per archetype
    event_probs = {
        "digital_native":   [0.15, 0.20, 0.05, 0.10, 0.20, 0.10, 0.10, 0.05, 0.05],
        "salary_spender":   [0.20, 0.25, 0.02, 0.15, 0.15, 0.10, 0.08, 0.03, 0.02],
        "loan_seeker":      [0.25, 0.10, 0.30, 0.05, 0.10, 0.08, 0.07, 0.03, 0.02],
        "branch_dependent": [0.30, 0.20, 0.02, 0.15, 0.10, 0.10, 0.08, 0.03, 0.02],
        "dormant_rich":     [0.35, 0.30, 0.05, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01],
        "churn_risk":       [0.20, 0.20, 0.05, 0.15, 0.10, 0.10, 0.10, 0.07, 0.03],
    }
    # EVENT_TYPES order:
    # TXN_DEBIT, TXN_CREDIT, VIEW_LOAN_PRODUCT, VIEW_STATEMENT, LOGIN,
    # VIEW_BALANCE, OPEN_TRANSFER_SCREEN, PRODUCT_VIEW, COMPLAINT_RAISED, COMPLAINT_RESOLVED
    # (trim to 9 for app events — no direct TXN_ events here)
    app_event_types = [
        "LOGIN", "VIEW_BALANCE", "VIEW_LOAN_PRODUCT", "VIEW_STATEMENT",
        "OPEN_TRANSFER_SCREEN", "PRODUCT_VIEW", "COMPLAINT_RAISED",
        "SESSION_IDLE", "LOGOUT"
    ]

    records = []
    event_counter = 1

    for _, cust in tqdm(customers.iterrows(), total=len(customers),
                        desc="  Generating app events"):
        arch = cust["archetype"]
        n_sessions = max(1, int(np.random.poisson(
            session_rates[arch] * (HISTORY_DAYS / 30)
        )))

        for _ in range(n_sessions):
            session_start = _rand_date(START_DATE, END_DATE)
            session_id = f"SES_{str(event_counter).zfill(9)}"
            n_events_in_session = max(1, int(np.random.poisson(4)))
            probs = event_probs[arch]

            for j in range(n_events_in_session):
                etype = np.random.choice(app_event_types, p=probs)
                ts = session_start + timedelta(seconds=j * random.randint(10, 120))
                records.append({
                    "event_id": f"EVT_{str(event_counter).zfill(9)}",
                    "customer_id": cust["customer_id"],
                    "session_id": session_id,
                    "event_type": etype,
                    "channel": "APP",
                    "session_duration_secs": random.randint(30, 900),
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                })
                event_counter += 1

    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    print(f"  ✓ App events: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# 5. PRODUCTS
# ─────────────────────────────────────────────
def generate_products(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Product holdings per customer — loan, card, fixed deposit, etc.
    Reflects real product penetration rates in retail banking.
    """
    product_types = ["PERSONAL_LOAN", "CREDIT_CARD", "FIXED_DEPOSIT",
                     "OVERDRAFT", "INVESTMENT_PLAN"]

    uptake_rates = {
        "digital_native":   [0.40, 0.55, 0.30, 0.20, 0.25],
        "salary_spender":   [0.50, 0.30, 0.15, 0.25, 0.10],
        "loan_seeker":      [0.15, 0.10, 0.05, 0.08, 0.03],   # wants but rarely gets
        "branch_dependent": [0.25, 0.10, 0.20, 0.10, 0.05],
        "dormant_rich":     [0.10, 0.20, 0.60, 0.05, 0.40],
        "churn_risk":       [0.20, 0.15, 0.05, 0.15, 0.05],
    }

    records = []
    prod_counter = 1

    for _, cust in customers.iterrows():
        arch = cust["archetype"]
        rates = uptake_rates[arch]
        for i, ptype in enumerate(product_types):
            if random.random() < rates[i]:
                records.append({
                    "product_id": f"PROD_{str(prod_counter).zfill(7)}",
                    "customer_id": cust["customer_id"],
                    "product_type": ptype,
                    "status": "ACTIVE" if random.random() > 0.15 else "CLOSED",
                    "opened_at": _rand_date(
                        START_DATE - timedelta(days=365), END_DATE
                    ).strftime("%Y-%m-%d"),
                })
                prod_counter += 1

    df = pd.DataFrame(records)
    print(f"  ✓ Products: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# 6. COMPLAINTS
# ─────────────────────────────────────────────
def generate_complaints(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Customer complaint records — a high-signal churn and sentiment indicator.
    """
    complaint_types = [
        "FAILED_TRANSACTION", "WRONG_DEBIT", "APP_ISSUE",
        "CARD_DECLINED", "LOAN_DISPUTE", "ACCOUNT_LOCKED", "OTHER"
    ]
    complaint_rates = {
        "digital_native":   0.05,
        "salary_spender":   0.10,
        "loan_seeker":      0.08,
        "branch_dependent": 0.12,
        "dormant_rich":     0.03,
        "churn_risk":       0.35,   # churn risk customers complain a lot
    }

    records = []
    comp_counter = 1

    for _, cust in customers.iterrows():
        arch = cust["archetype"]
        if random.random() < complaint_rates[arch]:
            n_comps = max(1, int(np.random.poisson(
                1.5 if arch == "churn_risk" else 1.0
            )))
            for _ in range(n_comps):
                raised = _rand_date(START_DATE, END_DATE)
                resolved = raised + timedelta(days=random.randint(1, 14)) \
                    if random.random() > 0.2 else None
                records.append({
                    "complaint_id": f"COMP_{str(comp_counter).zfill(6)}",
                    "customer_id": cust["customer_id"],
                    "complaint_type": random.choice(complaint_types),
                    "status": "RESOLVED" if resolved else "OPEN",
                    "raised_at": raised.strftime("%Y-%m-%d %H:%M:%S"),
                    "resolved_at": resolved.strftime("%Y-%m-%d") if resolved else None,
                    "severity": random.choice(["LOW", "MEDIUM", "HIGH"]),
                })
                comp_counter += 1

    df = pd.DataFrame(records)
    print(f"  ✓ Complaints: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# SCHEMA SAVER
# ─────────────────────────────────────────────
def save_schemas(dfs: dict):
    """Save column schemas as JSON for documentation."""
    import json

    SCHEMA_DIR = Path(__file__).parent.parent / "data" / "schemas"
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)

    type_map = {
        "object": "string", "int64": "integer",
        "float64": "float", "bool": "boolean"
    }
    for name, df in dfs.items():
        schema = {
            "table": name,
            "row_count": len(df),
            "columns": [
                {"name": col,
                 "dtype": type_map.get(str(df[col].dtype), str(df[col].dtype)),
                 "example": str(df[col].iloc[0]) if len(df) > 0 else ""}
                for col in df.columns
            ]
        }
        with open(SCHEMA_DIR / f"{name}_schema.json", "w") as f:
            json.dump(schema, f, indent=2)
    print(f"  ✓ Schemas saved to {SCHEMA_DIR}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────
def generate_all(output_dir: Path = None, verbose: bool = True) -> dict:
    """
    Generate all synthetic tables and save to CSV.

    Returns:
        dict of {table_name: DataFrame}
    """
    if output_dir is None:
        output_dir = DATA_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "="*60)
        print("  FINANCIAL REPRESENTATION LEARNING SYSTEM")
        print("  Synthetic Data Generator")
        print("="*60)
        print(f"\n  Config: {N_CUSTOMERS:,} customers | {HISTORY_DAYS} days history")
        print(f"  Output: {output_dir}\n")

    # Generate
    counterparties = generate_counterparties()
    customers      = generate_customers()
    transactions   = generate_transactions(customers, counterparties)
    app_events     = generate_app_events(customers)
    products       = generate_products(customers)
    complaints     = generate_complaints(customers)

    dfs = {
        "customers":      customers,
        "transactions":   transactions,
        "app_events":     app_events,
        "counterparties": counterparties,
        "products":       products,
        "complaints":     complaints,
    }

    # Save CSVs
    if verbose:
        print("\n  Saving CSVs...")
    for name, df in dfs.items():
        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        if verbose:
            size_mb = path.stat().st_size / 1e6
            print(f"    → {name}.csv ({len(df):,} rows, {size_mb:.1f} MB)")

    save_schemas(dfs)

    if verbose:
        total_rows = sum(len(df) for df in dfs.values())
        print(f"\n  ✅ Done. Total rows generated: {total_rows:,}")
        print("="*60 + "\n")

    return dfs


if __name__ == "__main__":
    generate_all()
