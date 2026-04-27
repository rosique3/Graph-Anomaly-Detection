# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Master's thesis applying **Graph Autoencoders (GAE)** and **Variational Graph Autoencoders (VGAE)** for unsupervised structural anomaly detection on the Nashville Meetup social network (Kaggle dataset, 2015–2017). The network is modelled as three complementary graphs:

| Graph | Nodes | Edges | Semantics |
|---|---|---|---|
| **M** — Members | 11,372 | 1,176,368 | Shared groups between two members |
| **G** — Groups | 456 | 6,692 | Shared members between two groups |
| **MG** — Bipartite | 25,233 | 45,583 | Events attended by a member in a group |

Each graph gets its own independent GAE and VGAE model, producing six experiments total. Anomaly scores are per-node mean reconstruction error on edges.

## Environment Setup

```bash
# Conda (recommended)
conda activate TFM_grafos_V2

# Manual pip (PyTorch must be installed before PyG)
pip install torch>=2.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install -r requirements.txt
```

Requires Python 3.12. CUDA 11.8+ recommended (graph M is large; CPU is slow).

## Common Commands

```bash
# Regenerate GraphML files from processed CSVs
python graphml.py

# Launch Jupyter for notebooks
jupyter notebook
```

## Notebook Pipeline (run in order)

| Notebook | Status | Output |
|---|---|---|
| `01_data_loading_and_preprocessing.ipynb` | Done | `data/processed/*.csv` |
| `02_eda_metadata.ipynb` | Done | `markdowns/eda_metadata.md` |
| `03_eda_graph.ipynb` | Done | `markdowns/eda_graphs.md`, GraphML files |
| `04_feature_engineering.ipynb` | Done | `data/graph_data/data_{M,G,MG}.pt` |
| `05_gae_members.ipynb` | **Pending** | `results/scores/scores_M.csv`, `results/embeddings/embeddings_M.npy` |
| `06_gae_groups.ipynb` | **Pending** | `results/scores/scores_G.csv`, `results/embeddings/embeddings_G.npy` |
| `07_gae_bipartite.ipynb` | **Pending** | `results/scores/scores_MG.csv`, `results/embeddings/embeddings_MG.npy` |
| `08_results_comparison.ipynb` | **Pending** | `results/scores/scores_combined.csv` |

## Architecture

```
src/
├── data/
│   ├── loader.py          # CSV/edgelist/feature loading helpers
│   └── preprocessing.py   # Dedup, missing value handling, normalization, label encoding
├── graph/
│   ├── builder.py         # NetworkX graph construction + graph_to_pyg() conversion
│   └── features.py        # Structural feature extraction (degree, clustering, betweenness, pagerank)
├── models/
│   ├── gnn.py             # GCNEncoder, VariationalGCNEncoder, build_gae/vgae, train_epoch, get_anomaly_scores
│   └── baselines.py       # Isolation Forest, LOF, One-Class SVM (with StandardScaler)
└── utils/
    ├── metrics.py          # evaluate() → ROC-AUC/PR-AUC/F1; threshold_by_percentile()
    └── visualization.py    # plot_graph, plot_score_distribution, plot_roc_curve → reports/figures/
```

`graphml.py` (root) is a standalone script that rebuilds all GraphML exports from `data/processed/` CSVs. Bipartite nodes are prefixed (`member_*`, `group_*`) to avoid ID collisions.

## Key Design Decisions

- **No structural graph metrics in node features `X`** (degree, clustering, betweenness are excluded). The GAE learns structure from adjacency `A`; adding those features would create redundancy.
- **Feature matrices `X` per graph:** Member nodes use geographic features (location_level, lat/lon, 3 dims); Group nodes use log-scaled size/activity + category one-hot (35 dims). In the bipartite MG, features are zero-padded to a uniform dimension.
- **Anomaly score** = mean reconstruction error over a node's edges. Top 5th percentile by default via `threshold_by_percentile()`.
- **No ground truth labels** — evaluation is qualitative, cross-validating detected anomalies against EDA-identified candidates (leaf nodes, super-connectors, ghost groups, inactive members, etc.).
- **config.txt** stores three hex color codes used for graph visualizations.
