# Design Decisions

## Decision 1: GRU encoder, not Transformer, for MVP
**Rationale:** GRU trains CoLES-style on 5,000 customers in under 12 minutes on a free Colab T4. A full Transformer takes 30–40 minutes and requires Colab Pro. The GRU is the right tool for the prototype. The production path upgrades to a Transformer (BST-style, arXiv:1905.06874) — the swap is one class in `encoder.py`.

## Decision 2: GraphSAGE, not GCN
**Rationale:** GraphSAGE is inductive. It generates embeddings for new nodes (new customers) not seen during training by sampling and aggregating their neighbourhood. GCN is transductive — it requires the full graph at inference time. For a financial institution adding hundreds of new accounts daily, GCN is not a production option.

## Decision 3: Token = one event row (not an episode)
**Rationale:** The model learns what episodes are via self-attention and recurrence. Hard-coding episodes (e.g., "5 consecutive loan screen views = loan intent") is feature engineering — exactly what we're trying to replace. Feed raw events. Let the model find structure.

## Decision 4: Self-supervised training (no labels)
**Rationale:** Financial institutions rarely have clean, large-scale labels for customer intent. The FRL system works from day one without a labelling project. CoLES and GraphSAGE are both self-supervised. Labels are only used in downstream tasks (optional, for evaluation or supervised heads).

## Decision 5: Bipartite graph (customers + counterparties), not customer-only
**Rationale:** Individual customer sequences are blind to network effects. The bipartite graph reveals patterns invisible to per-customer analysis: shared merchants, employer clusters, geographic proximity, social transfer networks. See "Beyond Isolated Clients" (arXiv:2604.09085) for empirical evidence.

## Decision 6: 64-dim embedding output
**Rationale:** Sufficient to encode complex behaviour patterns while remaining computationally tractable for downstream ML tasks (KMeans, MLP, cosine search). Production may benefit from 128-dim after scaling.

## Decision 7: Log-scale amount bucketing (32 bins)
**Rationale:** Transaction amounts span 4+ orders of magnitude (₦100 to ₦10M+). Linear bins would put 95% of transactions in the first bin. Log-scale bins give equal resolution across the full range.

## Decision 8: Microsoft Fabric as primary production target
**Rationale:** Fabric unifies data engineering (Lakehouse), ML (MLflow), and BI (Power BI) in one governed platform. Reduces the number of vendor relationships and data movement steps vs. a fragmented Azure stack. Azure ML is the fallback for teams already invested in that ecosystem.
