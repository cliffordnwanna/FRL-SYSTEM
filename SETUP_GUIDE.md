# Step-by-Step Setup Guide
## Get from this ZIP to a live GitHub repo in 20 minutes

---

## Phase 1 — GitHub Repo Setup (5 minutes)

### 1.1 Create the repo on GitHub
1. Go to github.com → New repository
2. Name: `frl-system`
3. Description: `Financial Representation Learning System — customer embeddings via CoLES + GraphSAGE`
4. Visibility: **Public** (important — Colab badge links need public access)
5. Do NOT initialise with README (you already have one)
6. Click **Create repository**

### 1.2 Connect this project to GitHub
Open your terminal (or VS Code terminal / Claude Code terminal):

```bash
# Navigate to the unzipped folder
cd path/to/frl-system

# Initialise git
git init

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/frl-system.git

# Stage everything
git add .

# First commit
git commit -m "feat: initial project structure — FRL System MVP"

# Push
git branch -M main
git push -u origin main
```

### 1.3 Update Colab badge URLs
In every notebook and README, replace `YOUR_USERNAME` with your actual GitHub username:

```bash
# Find and replace in all files (Mac/Linux)
grep -r "YOUR_USERNAME" . --include="*.md" --include="*.ipynb" -l
# Then open each file and replace
```

---

## Phase 2 — Run on Google Colab (25 minutes)

### 2.1 Open Notebook 00
Go to: `github.com/YOUR_USERNAME/frl-system`
Click `notebooks/00_synthetic_data_generator.ipynb`
Click **Open in Colab** badge

### 2.2 Enable GPU
`Runtime → Change runtime type → T4 GPU → Save`

### 2.3 Update repo URL in notebook
In the first code cell, change:
```python
REPO_URL = "https://github.com/YOUR_USERNAME/frl-system.git"
```
to your actual username.

### 2.4 Run all notebooks in order
Run each notebook top-to-bottom:
- 00 → generates CSVs
- 01 → tokenizes events
- 02 → trains CoLES encoder (~12 min)
- 03 → builds graph
- 04 → trains GraphSAGE (~8 min)
- 05 → downstream tasks + dashboard

### 2.5 Save outputs to Google Drive
Add this to each notebook to persist files between sessions:
```python
from google.colab import drive
drive.mount('/content/drive')

import shutil
shutil.copy("data/synthetic/customer_embeddings.pkl",
            "/content/drive/MyDrive/frl-system/customer_embeddings.pkl")
```

---

## Phase 3 — Document and Publish (15 minutes)

### 3.1 Screenshot the UMAP plot
The UMAP output from Notebook 05 is your key visual. Save it as `docs/results/05_umap.png`.

### 3.2 Commit results
```bash
git add docs/results/
git commit -m "docs: add MVP results — UMAP, dashboard, training curves"
git push
```

### 3.3 Create a GitHub Release
- Go to repo → Releases → Create new release
- Tag: `v0.1.0-mvp`
- Title: `FRL System MVP — End-to-End Working Prototype`
- Describe what was achieved

---

## Phase 4 — Medium / LinkedIn Post

Key points to include in your post:
1. The problem: banks have millions of events but treat customers as RFM scores
2. The insight: customers have a "financial language" — events as tokens
3. The architecture: CoLES (sequence) + GraphSAGE (network) + AEPD (journey)
4. The result: one embedding, four downstream applications
5. The code: link to GitHub repo
6. The production path: Microsoft Fabric medallion architecture

Suggested title:
> "I built a Financial Representation Learning System in a weekend — here's the full architecture and code"

---

## Troubleshooting

**"Module not found" error in Colab:**
```python
import sys
sys.path.insert(0, "/content/frl-system")
```

**Out of memory on free Colab:**
Reduce `N_CUSTOMERS` in `src/config.py` from 5000 to 2000 before running Notebook 00.

**PyTorch Geometric install fails:**
```python
!pip install torch-geometric -q
!pip install torch-scatter torch-sparse -q -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

**UMAP install fails:**
```python
!pip install umap-learn -q
```
