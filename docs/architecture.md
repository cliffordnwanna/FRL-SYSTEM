# System Architecture Deep Dive

## The Core Principle

Traditional customer analytics asks: **"Which bucket does this customer fall into?"**

The Financial Representation Learning System asks: **"What does this customer's behaviour actually mean — in the context of every other customer, merchant, and product in the ecosystem?"**

The answer is a **learned vector representation** — a 64-dimensional embedding that encodes everything the system knows about a customer, updated as their behaviour changes.

## Why This Architecture

### Problem 1: Hand-crafted features are a ceiling
RFM scores (Recency, Frequency, Monetary) are useful but shallow. They describe what happened. They cannot describe intent, trajectory, or network position. The FRL system learns features from data rather than having them imposed by a human analyst.

### Problem 2: Individual sequences miss network signals
A customer sending ₦45,000 to a merchant every Tuesday is meaningless in isolation. The same behaviour pattern across 200 customers — all sending to the same merchant — reveals a shared food subscription, a community of practice, or an employer paying via bank transfer. The bipartite graph makes invisible patterns visible.

### Problem 3: Labelled data is scarce
Financial institutions rarely have clean, large-scale intent labels. The FRL system is entirely self-supervised: CoLES trains on sequence structure alone, GraphSAGE trains on link prediction. No labelling project required.

## Component Details

### Event Tokenizer
Converts raw event rows into a structured vocabulary — a "financial language." Each event becomes a tuple of 8 integer IDs. Numeric amounts are log-bucketed (not normalised) so that the distance between ₦500 and ₦1,000 (consumer spend) is treated differently from ₦500,000 and ₦1,000,000 (business transaction). Time gaps between events are encoded as a separate feature — a critical intent signal.

### CoLES Encoder
Implements the contrastive learning approach from Babaev et al. (SIGMOD 2022). The training signal: two random subsequences of the same customer's history should produce similar embeddings. Different customers should be distant. No labels. No manual annotation. The loss function (NT-Xent, same as SimCLR) learns to organise the embedding space so that behaviorally similar customers cluster together.

### Bipartite Graph
Nodes: customers + counterparties. Edges: transactions. Edge features encode volume, channel, and recency. The graph structure captures signals that sequence models miss: shared merchants, salary source clusters, geographic proximity (customers near the same utility provider), and social graphs (customers who regularly send to each other).

### GraphSAGE
Inductive graph learning (Hamilton et al., NeurIPS 2017). Each customer node aggregates information from its 2-hop neighbourhood — who else transacts with the same merchants, what those merchants are, how connected this customer is in the network. The result is an enriched embedding that knows both the customer's personal history (from CoLES) and their network position (from GraphSAGE).

## Design Decisions Log

See `docs/decisions.md` for the full rationale behind each architectural choice.
