"""
downstream.py
-------------
Downstream applications built on top of customer embeddings.

All downstream tasks read from the same enriched embedding space.
This is the key architectural principle: one representation, many applications.

Tasks implemented:
  1. AEPD Segmentation      — K-Means clustering → stage assignment
  2. UMAP Visualisation     — 2D projection for stakeholder reporting
  3. Anomaly Detection      — distance from cluster centroid
  4. Next-Product Prediction — MLP classifier on embedding
  5. Segment Profile Report  — descriptive stats per segment
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    N_SEGMENTS, UMAP_N_COMPONENTS, UMAP_N_NEIGHBORS, UMAP_MIN_DIST,
    ANOMALY_THRESHOLD_PERCENTILE, AEPD_STAGES, AEPD_COLORS, RANDOM_SEED
)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def embeddings_to_matrix(embeddings: dict) -> tuple:
    """
    Convert embedding dict to numpy matrix for sklearn/UMAP.

    Returns:
        (X, customer_ids, metadata_df)
    """
    customer_ids = list(embeddings.keys())
    X = np.array([embeddings[cid]["embedding"] for cid in customer_ids])
    meta = pd.DataFrame({
        "customer_id": customer_ids,
        "aepd_true": [embeddings[cid].get("aepd_stage", "UNKNOWN") for cid in customer_ids],
        "archetype":  [embeddings[cid].get("archetype", "UNKNOWN") for cid in customer_ids],
    })
    return X, customer_ids, meta


# ─────────────────────────────────────────────
# 1. AEPD SEGMENTATION
# ─────────────────────────────────────────────

def segment_customers(embeddings: dict,
                      n_segments: int = N_SEGMENTS) -> pd.DataFrame:
    """
    Cluster customer embeddings into AEPD stages using K-Means.

    The cluster-to-AEPD mapping is done post-hoc by comparing
    cluster centroids to known archetype distributions.

    Returns:
        DataFrame with customer_id, cluster_id, aepd_pred, aepd_true
    """
    X, customer_ids, meta = embeddings_to_matrix(embeddings)

    print(f"\n  Running K-Means (k={n_segments}) on {len(X):,} embeddings...")
    kmeans = KMeans(n_clusters=n_segments, random_state=RANDOM_SEED,
                    n_init=10, max_iter=300)
    cluster_ids = kmeans.fit_predict(X)

    # Silhouette score (quality metric, higher = better separated clusters)
    sil = silhouette_score(X, cluster_ids, sample_size=min(2000, len(X)))
    print(f"  Silhouette Score: {sil:.4f}  (range: -1 to 1, higher is better)")

    # Post-hoc AEPD label assignment
    # Map each cluster to an AEPD stage based on which archetype dominates it
    archetype_to_aepd = {
        "digital_native": "D", "dormant_rich": "A",
        "loan_seeker": "E", "branch_dependent": "E",
        "salary_spender": "P", "churn_risk": "P",
    }

    meta["cluster_id"] = cluster_ids
    cluster_to_aepd = {}
    for c in range(n_segments):
        mask = meta["cluster_id"] == c
        if mask.sum() == 0:
            cluster_to_aepd[c] = "E"
            continue
        arch_counts = meta[mask]["archetype"].map(archetype_to_aepd).value_counts()
        cluster_to_aepd[c] = arch_counts.index[0] if len(arch_counts) > 0 else "E"

    meta["aepd_pred"] = meta["cluster_id"].map(cluster_to_aepd)

    # Evaluation vs ground truth
    valid = meta["aepd_true"] != "UNKNOWN"
    if valid.sum() > 0:
        le = LabelEncoder()
        true_enc = le.fit_transform(meta.loc[valid, "aepd_true"])
        pred_enc = le.transform(
            meta.loc[valid, "aepd_pred"].map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
        )
        ari = adjusted_rand_score(true_enc, pred_enc)
        print(f"  Adjusted Rand Index vs ground truth: {ari:.4f}")

    # Cluster sizes
    dist = meta["aepd_pred"].value_counts()
    print(f"\n  Predicted AEPD Distribution:")
    for stage in AEPD_STAGES:
        count = dist.get(stage, 0)
        pct = count / len(meta) * 100
        print(f"    {stage}: {count:,} customers ({pct:.1f}%)")

    meta["centroid_distance"] = np.linalg.norm(
        X - kmeans.cluster_centers_[cluster_ids], axis=1
    )

    return meta, kmeans


# ─────────────────────────────────────────────
# 2. UMAP VISUALISATION
# ─────────────────────────────────────────────

def compute_umap(embeddings: dict,
                 n_neighbors: int = UMAP_N_NEIGHBORS,
                 min_dist: float = UMAP_MIN_DIST) -> np.ndarray:
    """Reduce embeddings to 2D for visualisation."""
    try:
        from umap import UMAP
    except ImportError:
        print("  umap-learn not installed. Run: pip install umap-learn")
        return None

    X, _, _ = embeddings_to_matrix(embeddings)
    print(f"\n  Running UMAP on {len(X):,} embeddings...")
    reducer = UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=RANDOM_SEED,
        verbose=False,
    )
    coords = reducer.fit_transform(X)
    print("  ✅ UMAP complete")
    return coords


def plot_umap(coords: np.ndarray, meta: pd.DataFrame,
              color_by: str = "aepd_pred",
              title: str = "Customer Embedding Space (UMAP)",
              save_path: Path = None):
    """
    Plot 2D UMAP projection coloured by AEPD stage or archetype.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, col, col_title in [
        (axes[0], "aepd_pred", "Predicted AEPD Stage"),
        (axes[1], "archetype", "Customer Archetype"),
    ]:
        if col not in meta.columns:
            continue

        labels = meta[col].values
        unique_labels = sorted(set(labels))

        # Color palette
        if col == "aepd_pred":
            cmap = {s: AEPD_COLORS.get(s, "#999999") for s in unique_labels}
        else:
            palette = plt.cm.tab10.colors
            cmap = {l: palette[i % 10] for i, l in enumerate(unique_labels)}

        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=cmap[label], label=label,
                s=8, alpha=0.6, linewidths=0,
            )

        ax.set_title(col_title, fontsize=11)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(markerscale=3, fontsize=9, loc="best",
                  framealpha=0.7, edgecolor="gray")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  ✓ Plot saved to {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 3. ANOMALY DETECTION
# ─────────────────────────────────────────────

def detect_anomalies(meta: pd.DataFrame,
                     threshold_percentile: float = ANOMALY_THRESHOLD_PERCENTILE
                     ) -> pd.DataFrame:
    """
    Flag customers with unusually large distance from their cluster centroid.
    These are candidates for fraud review or churn intervention.
    """
    threshold = np.percentile(meta["centroid_distance"], threshold_percentile)
    meta["is_anomaly"] = meta["centroid_distance"] > threshold

    n_anomalies = meta["is_anomaly"].sum()
    print(f"\n  Anomaly Detection (top {100 - threshold_percentile:.0f}%):")
    print(f"  Threshold distance: {threshold:.4f}")
    print(f"  Flagged: {n_anomalies:,} customers ({n_anomalies/len(meta)*100:.1f}%)")

    # What archetypes are most anomalous?
    if "archetype" in meta.columns:
        arch_anomaly = (
            meta[meta["is_anomaly"]]["archetype"].value_counts() /
            meta["archetype"].value_counts()
        ).sort_values(ascending=False)
        print(f"\n  Anomaly rate by archetype:")
        for arch, rate in arch_anomaly.items():
            print(f"    {arch}: {rate*100:.1f}%")

    return meta


# ─────────────────────────────────────────────
# 4. NEXT-PRODUCT PREDICTION
# ─────────────────────────────────────────────

def train_next_product_classifier(embeddings: dict,
                                   products: pd.DataFrame,
                                   product_type: str = "PERSONAL_LOAN",
                                   test_size: float = 0.2) -> dict:
    """
    Train a simple MLP to predict whether a customer will take a product.
    Demonstrates the embedding's utility for downstream supervised tasks.

    Args:
        embeddings:   enriched embeddings dict
        products:     products DataFrame
        product_type: which product to predict
        test_size:    train/test split ratio

    Returns:
        dict with model, metrics, and feature importances
    """
    # Build labels: 1 if customer has this product, 0 otherwise
    has_product = set(
        products[products["product_type"] == product_type]["customer_id"]
    )

    X_rows, y_rows, cust_ids = [], [], []
    for cid, data in embeddings.items():
        X_rows.append(data["embedding"])
        y_rows.append(1 if cid in has_product else 0)
        cust_ids.append(cid)

    X = np.array(X_rows)
    y = np.array(y_rows)

    pos_rate = y.mean() * 100
    print(f"\n  Next-Product Prediction: {product_type}")
    print(f"  Customers with product: {y.sum():,} ({pos_rate:.1f}%)")
    print(f"  Training MLP on {len(X):,} customer embeddings...\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=200,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred,
                                target_names=["No Product", product_type]))

    return {
        "model": clf,
        "report": report,
        "product_type": product_type,
        "X_test": X_test,
        "y_test": y_test,
    }


# ─────────────────────────────────────────────
# 5. SEGMENT PROFILE REPORT
# ─────────────────────────────────────────────

def generate_segment_profiles(meta: pd.DataFrame,
                               transactions: pd.DataFrame,
                               products: pd.DataFrame = None) -> pd.DataFrame:
    """
    Generate descriptive statistics for each AEPD segment.
    This is the business-facing output — what does each segment look like?
    """
    profiles = []

    for stage in AEPD_STAGES:
        seg_custs = meta[meta["aepd_pred"] == stage]["customer_id"].tolist()
        if not seg_custs:
            continue

        seg_txns = transactions[transactions["customer_id"].isin(seg_custs)]

        profile = {
            "aepd_stage": stage,
            "n_customers": len(seg_custs),
            "pct_of_total": f"{len(seg_custs)/len(meta)*100:.1f}%",
            "avg_txn_count": seg_txns.groupby("customer_id").size().mean()
                             if len(seg_txns) > 0 else 0,
            "avg_txn_amount": seg_txns["amount"].mean()
                              if "amount" in seg_txns.columns else 0,
            "top_channel": seg_txns["channel"].mode().iloc[0]
                           if len(seg_txns) > 0 and "channel" in seg_txns.columns
                           else "N/A",
        }

        if products is not None:
            seg_prods = products[products["customer_id"].isin(seg_custs)]
            profile["avg_products"] = (
                seg_prods.groupby("customer_id").size().reindex(seg_custs).fillna(0).mean()
            )

        profiles.append(profile)

    df = pd.DataFrame(profiles)
    print("\n  AEPD Segment Profiles:")
    print(df.to_string(index=False))
    return df
