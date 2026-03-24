# 🔍 Graph Anomaly Detection — TFM

Detección de anomalías en grafos de redes sociales usando la plataforma **Meetup (Nashville)**.  
El objetivo es identificar miembros, grupos o eventos que se comporten de forma anómala dentro de la red de interacciones.

---

## 📁 Estructura del Proyecto

```
TFM/
├── data/
│   ├── raw/                        # Datos originales de Meetup (NO modificar)
│   │   ├── meta-members.csv        # Info de miembros (id, nombre, ciudad, lat/lon)
│   │   ├── meta-groups.csv         # Info de grupos (id, nombre, categoría, nº miembros)
│   │   ├── meta-events.csv         # Info de eventos (id, grupo, nombre, fecha)
│   │   ├── member-edges.csv        # Aristas miembro↔miembro (co-pertenencia a grupos)
│   │   ├── group-edges.csv         # Aristas grupo↔grupo (miembros compartidos)
│   │   ├── member-to-group-edges.csv  # Aristas miembro→grupo (membresía, ponderada)
│   │   └── rsvps.csv               # RSVPs: qué miembro fue a qué evento
│   ├── processed/                  # Datos limpios listos para modelar
│   └── external/                   # Datasets externos o benchmarks
├── notebooks/
│   ├── 01_data_loading.ipynb       # Carga y revisión de los CSVs
│   ├── 02_eda.ipynb                # Análisis exploratorio de la red
│   ├── 03_graph_construction.ipynb # Construcción del grafo con NetworkX
│   ├── 04_anomaly_detection.ipynb  # Modelos de detección (baselines + GAE)
│   └── 05_evaluation.ipynb         # Evaluación y comparativa de modelos
├── src/
│   ├── data/
│   │   ├── loader.py               # Funciones de carga de CSVs
│   │   └── preprocessing.py        # Limpieza y normalización
│   ├── graph/
│   │   ├── builder.py              # Construcción de grafos NetworkX / PyG
│   │   └── features.py             # Features estructurales (grado, betweenness...)
│   ├── models/
│   │   ├── baselines.py            # Isolation Forest, LOF, One-Class SVM
│   │   └── gnn.py                  # Graph Autoencoder (GAE) y VGAE
│   └── utils/
│       ├── metrics.py              # ROC-AUC, PR-AUC, F1, curva ROC
│       └── visualization.py        # Plots del grafo y distribución de scores
├── experiments/
│   ├── configs/gae_baseline.yaml   # Config de experimento GAE
│   └── results/                    # Resultados guardados
├── reports/figures/                # Gráficas exportadas para la memoria
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalación

```bash
# 1. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows

# 2. Instalar dependencias base
pip install -r requirements.txt

# 3. Instalar PyTorch (CPU — ajustar si tienes GPU CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Instalar PyTorch Geometric
pip install torch_geometric
```

---

## 📊 Dataset: Red Social de Meetup (Tennessee)

Este conjunto de datos ofrece una visión detallada de las interacciones entre usuarios y grupos en **meetup.com**, una plataforma diseñada para organizar y asistir a eventos presenciales. Es un recurso ideal para aplicar **análisis de grafos** y teoría de redes.

### 📝 Contexto

Las relaciones entre quién asiste a qué evento forman una red social compleja. Este dataset fue creado originalmente para la charla *"Principles of Network Analysis with NetworkX"*, presentada en conferencias como **PyNash** y **PyTennessee**.

A través de estos datos, se exploran los fundamentos de la teoría de grafos utilizando **NetworkX** (una biblioteca de Python) para extraer información sobre el tejido social de los grupos de Meetup en Tennessee.

### 🕸️ Datos de Grafos (Aristas)

Estos archivos contienen las conexiones y los "pesos" que definen la fuerza de cada relación.

| Archivo | Descripción | Peso (Weight) |
| --- | --- | --- |
| `member-to-group-edges.csv` | Red bipartita entre miembros y grupos. | Número de eventos asistidos. |
| `group-edges.csv` | Conexiones entre grupos. | Miembros compartidos entre grupos. |
| `member-edges.csv` | Conexiones entre miembros. | Grupos compartidos entre personas. |
| `rsvps.csv` | Datos crudos de asistencia. | Base para generar la red de miembros y grupos. |

### ℹ️ Metadatos

Información descriptiva para enriquecer el análisis de los nodos.

- **`meta-groups.csv`**: Detalles de cada grupo (nombre, categoría). Usa `group_id` como índice.
- **`meta-members.csv`**: Detalles de los usuarios (nombre, ubicación). Usa `member_id` como índice.
- **`meta-events.csv`**: Detalles de los eventos (nombre, fecha/hora). Usa `event_id` como índice.

---

## 🗺️ Flujo de Trabajo Completo

### Fase 1 — Carga de datos (`01_data_loading.ipynb`)

```python
import sys; sys.path.insert(0, ".")
from src.data.loader import load_csv, load_edgelist

members   = load_csv("meta-members.csv")
groups    = load_csv("meta-groups.csv")
m_edges   = load_edgelist("member-edges.csv",         source_col="member1",   target_col="member2")
g_edges   = load_edgelist("group-edges.csv",           source_col="group1",    target_col="group2")
mg_edges  = load_edgelist("member-to-group-edges.csv", source_col="member_id", target_col="group_id")
rsvps     = load_csv("rsvps.csv")
```

---

### Fase 2 — Análisis Exploratorio (EDA) (`02_eda.ipynb`)

Entender la estructura de la red antes de modelar.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Distribución de grados
print(members.describe())
print(f"Nº miembros: {len(members)}, Nº grupos: {len(groups)}")

# ¿Cuántas aristas hay?
print(f"Aristas miembro↔miembro: {len(m_edges)}")
print(f"Aristas grupo↔grupo: {len(g_edges)}")

# Distribución del peso (nº grupos en común)
m_edges["weight"].hist(bins=50)
plt.title("Distribución de pesos en aristas miembro↔miembro")
plt.show()

# Top grupos por nº de miembros
groups.nlargest(10, "num_members")[["group_name", "num_members"]]
```

**Qué buscar en el EDA:**
- Distribución de grados (¿sigue una ley de potencias? Typical en redes sociales)
- Nodos con grado extremadamente alto → posibles hubs o anomalías
- Pesos atípicos en las aristas → conexiones sospechosamente fuertes
- Comunidades vs nodos aislados

---

### Fase 3 — Construcción del Grafo (`03_graph_construction.ipynb`)

```python
from src.graph.builder import build_graph_from_edgelist, add_node_features
from src.graph.features import compute_node_features

# Grafo de miembros
G_members = build_graph_from_edgelist(
    m_edges, source_col="member1", target_col="member2", weight_col="weight"
)

# Añadir info de miembros como atributos de nodo
members_idx = members.set_index("member_id")
G_members = add_node_features(G_members, members_idx[["lat", "lon"]])

# Grafo de grupos (más pequeño, bueno para prototipar)
G_groups = build_graph_from_edgelist(
    g_edges, source_col="group1", target_col="group2", weight_col="weight"
)

# Grafo bipartito miembro ↔ grupo
G_bipartite = build_graph_from_edgelist(
    mg_edges, source_col="member_id", target_col="group_id", weight_col="weight"
)
```

> **Consejo**: Empieza con `G_groups` (grafo de grupos). Es mucho más pequeño que el de miembros y te permite iterar rápido.

---

### Fase 4 — Detección de Anomalías (`04_anomaly_detection.ipynb`)

#### 4a. Features estructurales + Modelos clásicos

```python
from src.graph.features import compute_node_features
from src.models.baselines import isolation_forest, local_outlier_factor

# Calcular features estructurales del grafo de grupos
features_df = compute_node_features(G_groups)
# → columnas: degree, clustering, betweenness, pagerank, triangles

# Isolation Forest
scores_if = isolation_forest(features_df, contamination=0.05)

# Local Outlier Factor
scores_lof = local_outlier_factor(features_df, contamination=0.05)
```

#### 4b. Graph Autoencoder (GAE/VGAE) — modelo principal del TFM

El GAE aprende a reconstruir las aristas del grafo. Los nodos cuyas conexiones son **difíciles de reconstruir** son candidatos a anomalías.

```python
import torch
from src.graph.builder import graph_to_pyg
from src.models.gnn import build_gae, train_epoch, get_anomaly_scores

# Convertir grafo a formato PyTorch Geometric
# (los features son: degree, clustering, betweenness, pagerank)
feature_keys = ["degree", "clustering", "betweenness", "pagerank"]
data = graph_to_pyg(G_groups, node_feature_keys=feature_keys)

# Construir y entrenar el GAE
in_channels = data.x.shape[1]
model = build_gae(in_channels=in_channels, out_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1, 201):
    loss = train_epoch(model, optimizer, data)
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

# Scores de anomalía (mayor = más anómalo)
scores_gae = get_anomaly_scores(model, data).numpy()
```

---

### Fase 5 — Evaluación y Visualización (`05_evaluation.ipynb`)

```python
import numpy as np
from src.utils.metrics import evaluate, threshold_by_percentile
from src.utils.visualization import plot_graph, plot_score_distribution, plot_roc_curve

# Ver distribución de scores
plot_score_distribution(scores_gae, title="GAE Anomaly Scores", save_as="gae_scores.png")

# Visualizar el grafo coloreado por score de anomalía
plot_graph(G_groups, anomaly_scores=dict(zip(G_groups.nodes(), scores_gae)),
           title="Grafo de Grupos — Anomaly Scores (GAE)", save_as="gae_graph.png")

# Si tienes etiquetas reales (ground truth):
# y_true = ...  # 0=normal, 1=anomalía
# results = evaluate(y_true, scores_gae)
# plot_roc_curve(y_true, scores_gae, save_as="roc_gae.png")

# Top 10 grupos más anómalos
nodes = list(G_groups.nodes())
top_anomalies = sorted(zip(nodes, scores_gae), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 grupos más anómalos:")
for group_id, score in top_anomalies:
    name = groups[groups["group_id"] == group_id]["group_name"].values
    print(f"  Grupo {group_id} ({name[0] if len(name) else 'N/A'}): score={score:.4f}")
```

---

## 📊 Modelos implementados

| Modelo | Tipo | Archivo | Cuándo usarlo |
|--------|------|---------|---------------|
| Isolation Forest | Baseline clásico | `src/models/baselines.py` | Primer experimento rápido |
| Local Outlier Factor | Baseline clásico | `src/models/baselines.py` | Anomalías locales de densidad |
| One-Class SVM | Baseline clásico | `src/models/baselines.py` | Frontera de decisión no lineal |
| **GAE** | Graph Neural Network | `src/models/gnn.py` | **Modelo principal del TFM** |
| **VGAE** | Graph Neural Network | `src/models/gnn.py` | Variante probabilística del GAE |

---

## 🎯 Estrategia recomendada para el TFM

1. **Prototipa en el grafo de grupos** (`G_groups`): es pequeño (~miles de nodos), rápido de entrenar.
2. **Corre primero los baselines** (Isolation Forest, LOF) para tener una referencia.
3. **Entrena el GAE** y compara sus scores con los baselines.
4. **Analiza cualitativamente** los grupos/miembros flaggeados: ¿tienen sentido como anomalías? (grupos con muy pocos miembros pero muchas conexiones, eventos sin asistentes, etc.)
5. **Escala al grafo de miembros** si los resultados son buenos (grafo mucho más grande).
6. **Documenta los resultados** en `reports/figures/` y actualiza la tabla de abajo.

---

## 📈 Resultados

| Modelo | Grafo | ROC-AUC | PR-AUC | F1 | Notas |
|--------|-------|---------|--------|----|-------|
| Isolation Forest | Grupos | — | — | — | |
| LOF | Grupos | — | — | — | |
| GAE | Grupos | — | — | — | |
| VGAE | Grupos | — | — | — | |
| GAE | Miembros | — | — | — | |

---

## 📚 Referencias

- Kipf, T. N., & Welling, M. (2016). *Variational Graph Auto-Encoders*. [arXiv:1611.07308](https://arxiv.org/abs/1611.07308)
- Liu, F. T., et al. (2008). *Isolation Forest*. ICDM.
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io
- NetworkX: https://networkx.org
