# Detección de Anomalías en Grafos mediante Graph Autoencoders

> **Trabajo de Fin de Máster** — Ciencia de Datos / Inteligencia Artificial  
> Dataset: [Nashville Meetup Network](https://www.kaggle.com/datasets/stkbailey/nashville-meetup) (Meetup Tennessee)  
> Framework principal: PyTorch Geometric · PyTorch · NetworkX

---

## Índice

1. [Descripción del proyecto](#1-descripción-del-proyecto)
2. [Objetivo del TFM](#2-objetivo-del-tfm)
3. [Cómo se detectan anomalías con GAEs](#3-cómo-se-detectan-anomalías-con-gaes)
4. [Dataset](#4-dataset)
5. [Estructura del repositorio](#5-estructura-del-repositorio)
6. [Notebooks — guía completa](#6-notebooks--guía-completa)
7. [Instalación y entorno](#7-instalación-y-entorno)
8. [Pipeline de ejecución](#8-pipeline-de-ejecución)
9. [Convenciones del proyecto](#9-convenciones-del-proyecto)
10. [Estado actual](#10-estado-actual)

---

## 1. Descripción del proyecto

Este TFM aplica **Graph Autoencoders (GAE)** y **Variational Graph Autoencoders (VGAE)** para detectar anomalías estructurales y de comportamiento en la red social de Meetup de Tennessee. El dataset representa las interacciones entre miembros, grupos y eventos de la plataforma meetup.com durante el período 2015–2017.

La red se modela como **tres grafos complementarios**, cada uno capturando una dimensión distinta de la misma realidad social:

| Grafo | Nodos | Aristas | Semántica del peso |
|---|---|---|---|
| **M** — Miembros | 11.372 | 1.176.368 | Grupos compartidos entre dos miembros |
| **G** — Grupos | 456 | 6.692 | Miembros compartidos entre dos grupos |
| **MG** — Bipartito | 25.233 | 45.583 | Eventos asistidos por un miembro en un grupo |

Sobre cada uno de estos grafos se entrena un modelo GAE/VGAE independiente, y los nodos con mayor error de reconstrucción son identificados como candidatos a anomalía.

---

## 2. Objetivo del TFM

### 2.1 Objetivo principal

Desarrollar y evaluar un sistema de detección de anomalías **no supervisado** basado en Graph Autoencoders que sea capaz de identificar nodos estructuralmente atípicos en redes sociales complejas, sin requerir etiquetas de anomalía previas.

La hipótesis central es que un GAE entrenado para reconstruir la estructura de un grafo asignará **errores de reconstrucción más altos** a los nodos cuyo entorno de vecindad difiere del patrón general de la red, capturando así diferentes tipos de anomalía: aislamiento estructural, hiperconectividad, comportamiento de puente atípico o participación artificial.

### 2.2 Objetivos específicos

**Análisis exploratorio profundo.**
Caracterizar la topología de los tres grafos: distribuciones de grado, coeficientes de clustering, centralidades, estructura de comunidades (Louvain) y patrones de comportamiento de miembros. Este análisis sirve como línea base para interpretar las anomalías que detecte el modelo.

**Feature engineering orientado al modelo.**
Construir matrices de features `X` para cada grafo a partir exclusivamente de metadatos originales (atributos de nodo), descartando deliberadamente métricas estructurales del grafo para evitar redundancia con lo que el GAE aprende de la matriz de adyacencia `A`.

**Implementación de GAE y VGAE.**
Implementar ambas variantes del modelo con PyTorch Geometric. El GAE maximiza la verosimilitud de reconstrucción de la matriz de adyacencia; el VGAE añade regularización variacional (KL-divergence) que actúa como regularizador implícito y produce embeddings más suaves. Entrenar un modelo por grafo, resultando en seis experimentos (2 modelos × 3 grafos).

**Definición de un anomaly score interpretable.**
El anomaly score de cada nodo se define como el **error de reconstrucción medio de sus aristas**: cuánto le cuesta al modelo reconstruir la conectividad real del nodo a partir de su embedding. Este score es directamente comparable entre nodos del mismo grafo y permite rankear los candidatos a anomalía.

**Validación cruzada con el EDA.**
Contrastar los nodos que el modelo puntúa como más anómalos con los candidatos identificados durante el EDA (nodos hoja, super-conectores, miembros inactivos, grupos fantasma, entidades no-persona...). Esta validación cualitativa es la principal forma de evaluación en un contexto no supervisado sin ground truth.

**Comparativa entre grafos.**
Analizar si los mismos nodos aparecen como anómalos en más de un grafo (lo que aumenta la confianza en la anomalía) y qué tipo de anomalía captura mejor cada representación.

### 2.3 Tipos de anomalía que se esperan detectar

| Tipo | Descripción | Grafo más relevante |
|---|---|---|
| **Aislamiento estructural** | Nodos con muy pocas conexiones respecto a la media (nodos hoja, grupos periféricos) | M, G |
| **Hiperconectividad** | Super-conectores con grado anómalamente alto; posible comportamiento artificial | M, G |
| **Puente inter-comunidad atípico** | Nodos con alta betweenness pero degree moderada (rol estructural desproporcionado) | M |
| **Inactividad real** | Miembros inscritos en grupos pero sin asistencia a eventos reales | MG |
| **Comportamiento explorador** | Miembros en muchos grupos con muy baja participación por grupo | MG |
| **Grupos fantasma** | Grupos con muchos miembros registrados pero casi ningún activo | MG |
| **Anomalía de naturaleza** | Entidades no-persona (empresas, organizaciones) registradas como miembros | M |
| **Componentes aisladas** | Subgrafos completamente desconectados de la componente principal | MG |

---

## 3. Cómo se detectan anomalías con GAEs

### 3.1 Intuición general

Un **Graph Autoencoder** aprende a comprimir cada nodo en un vector de baja dimensión (embedding) y luego intenta reconstruir las conexiones originales del grafo a partir de esos embeddings. La idea central es:

> Si el modelo se entrena sobre el comportamiento **mayoritario** de la red, los nodos **normales** serán fáciles de reconstruir (error bajo), mientras que los nodos **anómalos** — cuyo patrón de conexión difiere del resto — serán difíciles de reconstruir (error alto).

El error de reconstrucción de cada nodo actúa directamente como **anomaly score**: a mayor error, mayor probabilidad de ser una anomalía.

### 3.2 Arquitectura del GAE

El GAE tiene dos componentes:

**Encoder — Graph Convolutional Network (GCN):**
```
Z = GCN(X, A)
```
Recibe la matriz de features `X` y la matriz de adyacencia `A`, y produce una matriz de embeddings `Z` donde cada fila es la representación vectorial de un nodo. Las capas GCN agregan información del entorno de vecindad de cada nodo — es decir, el embedding de un nodo no solo depende de sus propias features, sino también de las de sus vecinos directos (y de los vecinos de sus vecinos en capas más profundas).

**Decoder — Producto interno:**
```
Â = σ(Z · Zᵀ)
```
Reconstruye la matriz de adyacencia calculando la similaridad entre cada par de embeddings. Si dos nodos tienen embeddings similares, el modelo predice que están conectados; si son disímiles, predice que no lo están.

**Función de pérdida — Binary Cross-Entropy:**
```
L = -[A · log(Â) + (1 - A) · log(1 - Â)]
```
Penaliza al modelo por predecir conexiones donde no las hay, y por no predecir conexiones donde sí las hay.

### 3.3 VGAE — extensión variacional

El **Variational Graph Autoencoder** extiende el GAE añadiendo una capa de regularización probabilística. En lugar de producir embeddings deterministas `Z`, el encoder produce **distribuciones gaussianas** `q(Z|X,A) = N(μ, σ²)` y samplea de ellas.

La función de pérdida añade un término de **KL-divergence** que fuerza a las distribuciones a estar cerca de una gaussiana estándar:
```
L_VGAE = L_reconstrucción + β · KL[q(Z|X,A) || p(Z)]
```

Esto actúa como regularizador que produce embeddings más suaves y generalizables, especialmente útil en grafos con nodos raros o comunidades pequeñas.

### 3.4 Anomaly score por nodo

Una vez entrenado el modelo, el anomaly score de cada nodo `v` se calcula como:

```
score(v) = (1/|N(v)|) · Σ_{u ∈ N(v)} BCE(A[v,u], Â[v,u])
```

Es decir: el **error de reconstrucción medio sobre todas las aristas del nodo**. Este score:
- Es alto cuando el modelo no consigue predecir correctamente con quién está conectado el nodo
- Captura tanto falsos positivos (predice conexiones que no existen) como falsos negativos (no predice conexiones que sí existen)
- Es directamente comparable entre todos los nodos del mismo grafo

Los nodos se ordenan por score descendente y se inspeccionan los percentiles superiores (p95, p99) como candidatos a anomalía.

### 3.5 Diagrama del pipeline

```
Metadatos                  Grafo
(meta_members,    ──────►  (nodos + aristas)
 meta_groups,              │
 meta_events)              │
      │                    ▼
      ▼             Matriz de adyacencia A
Feature Engineering        │
      │                    │
      ▼                    │
Matriz de features X ──────┤
                           ▼
                    ┌─────────────────┐
                    │   GCN Encoder   │
                    │  X, A  ──►  Z   │
                    └────────┬────────┘
                             │ embeddings Z
                    ┌────────▼────────┐
                    │  Inner-product  │
                    │  Decoder Z·Zᵀ  │
                    └────────┬────────┘
                             │ Â (adj. reconstruida)
                    ┌────────▼────────┐
                    │  Loss: BCE(A,Â) │
                    │  + KL (VGAE)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Anomaly Score  │
                    │  por nodo       │
                    └─────────────────┘
```

---

## 4. Dataset

**Fuente:** [Kaggle — Nashville Meetup Network](https://www.kaggle.com/datasets/stkbailey/nashville-meetup) (`stkbailey/nashville-meetup`)

El dataset captura la actividad de meetup.com en Tennessee durante noviembre 2015 – octubre 2017.

### Archivos de aristas

| Archivo | Descripción |
|---|---|
| `member-edges.csv` | Pares de miembros con co-membresía. Columnas: `member1`, `member2`, `weight` |
| `group-edges.csv` | Pares de grupos que comparten miembros. Columnas: `group1`, `group2`, `weight` |
| `member-to-group-edges.csv` | Membresías miembro-grupo con actividad. Columnas: `member_id`, `group_id`, `weight` |

### Archivos de metadatos (nodos)

| Archivo | Descripción |
|---|---|
| `meta-members.csv` | 24.591 miembros con nombre, ciudad, estado, coordenadas |
| `meta-groups.csv` | 602 grupos con nombre, categoría, número de miembros, organizador |
| `meta-events.csv` | 19.307 eventos con grupo, nombre y timestamp |

### Estadísticas clave

| Métrica | Valor |
|---|---|
| Miembros únicos | 24.591 |
| Grupos únicos | 602 |
| Categorías temáticas | 31 |
| Eventos | 19.307 |
| Ventana temporal | Nov 2015 – Oct 2017 |
| Ciudad principal | Nashville, TN (60.1% de miembros) |
| Grupo más grande | Nashville Hiking Meetup (15.838 miembros) |

---

## 5. Estructura del repositorio

```
tfm-anomaly-detection-graphs/
│
├── README.md                        ← Este archivo
│
├── data/
│   ├── raw/                         ← CSVs originales de Kaggle (no modificar)
│   ├── processed/                   ← CSVs limpios (output del notebook 01)
│   └── graph_data/                  ← Objetos PyG .pt (output del notebook 04)
│       ├── data_M.pt
│       ├── data_G.pt
│       └── data_MG.pt
│
├── notebooks/
│   ├── 01_data_loading_and_preprocessing.ipynb
│   ├── 02_eda_metadata.ipynb
│   ├── 03_eda_graphs.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_gae_members.ipynb         ← GAE sobre grafo M
│   ├── 06_gae_groups.ipynb          ← GAE sobre grafo G
│   ├── 07_gae_bipartite.ipynb       ← GAE sobre grafo MG
│   └── 08_results_comparison.ipynb  ← Comparativa entre grafos
│
├── src/
│   ├── models/
│   │   ├── gae.py                   ← Implementación GAE
│   │   └── vgae.py                  ← Implementación VGAE
│   ├── utils/
│   │   ├── anomaly_score.py         ← Cálculo del anomaly score
│   │   ├── evaluation.py            ← Métricas y visualizaciones
│   │   └── graph_utils.py           ← Utilidades de grafo
│   └── config.py                    ← Hiperparámetros y configuración global
│
├── results/
│   ├── embeddings/                  ← Embeddings Z guardados en .npy
│   ├── scores/                      ← Anomaly scores por grafo en .csv
│   └── figures/                     ← Gráficos generados
│
├── docs/
│   ├── eda_grafos.md                ← EDA completo de grafos
│   └── eda_metadata.md              ← EDA completo de metadatos
│
├── environment.yml                  ← Entorno Conda
└── requirements.txt                 ← Dependencias pip
```

---

## 6. Notebooks — guía completa

### Notebooks existentes

#### `01_data_loading_and_preprocessing.ipynb`
**Propósito:** Ingesta desde Kaggle y limpieza inicial.

- Descarga el dataset usando `kagglehub` y lo copia a `data/raw/`
- Inspecciona tipos, duplicados y nulos en los 7 archivos
- Elimina la columna `hometown` (79.96% de nulos, imputación inviable)
- Exporta CSVs limpios a `data/processed/`

**Output:** `data/processed/*.csv`

---

#### `02_eda_metadata.ipynb`
**Propósito:** Análisis exploratorio de los metadatos tabulares.

- **meta_members:** distribución geográfica, calidad del campo `state`, miembros internacionales, problema de centroides en coordenadas
- **meta_groups:** distribución por categoría, análisis de `num_members`, organizadores multi-grupo, integridad referencial de `organizer_id`
- **meta_events:** distribución temporal (serie mensual, día de semana, hora), 32 grupos truncados por límite de API, problema de horas por defecto (00:00 y 23:00)

**Output:** Figuras en `results/figures/`

---

#### `03_eda_graphs.ipynb`
**Propósito:** Análisis exploratorio de los tres grafos con NetworkX.

Para cada grafo (M, G, MG):
- Estadísticas globales: nodos, aristas, densidad, clustering, componentes
- Distribución de grado y segmentación por rangos
- Análisis de centralidades: degree, betweenness
- Coeficiente de clustering local
- Análisis de pesos de aristas
- Detección de comunidades: Louvain y Greedy Modularity
- Análisis de nodos puente inter-comunidad

**Hallazgos clave documentados:**
- Grafo M: modularidad Louvain 0.6755, 22 comunidades, 72.4% de nodos con clustering = 1.0
- Grafo G: correlación moderada entre tamaño y grado, hub tecnológico NashJS
- Grafo MG: ratio de actividad medio del 17%, 44% de miembros con un único evento

**Output:** Figuras en `results/figures/`

---

#### `04_feature_engineering.ipynb`
**Propósito:** Construcción de las matrices de features `X` para los tres grafos.

**Features de nodos miembro (grafo M y mitad miembro del MG):**
- `location_level`: variable ordinal geográfica (0=Internacional, 1=USA, 2=TN, 3=Nashville)
- `lat`, `lon`: coordenadas geográficas normalizadas

**Features de nodos grupo (grafo G y mitad grupo del MG):**
- `log_num_members`: tamaño en escala logarítmica
- `log_num_events`: actividad en escala logarítmica
- `is_truncated`: flag de grupos con ≥200 eventos (límite API)
- `has_valid_organizer`: integridad referencial del organizador
- one-hot de `category_name`: 31 columnas binarias

**Decisión clave:** No se incluyen métricas estructurales del grafo (degree, clustering, betweenness) para evitar redundancia con lo que el GAE aprende de `A`.

**Normalización:** StandardScaler sobre features continuas; binarias y one-hot sin escalar.

**Output:** `data/graph_data/data_M.pt`, `data_G.pt`, `data_MG.pt`

---

### Notebooks pendientes (detección de anomalías)

#### `05_gae_members.ipynb`
**Propósito:** Entrenamiento y evaluación de GAE/VGAE sobre el **grafo de miembros (M)**.

**Contenido:**
1. Carga de `data_M.pt` y configuración del experimento
2. Definición de la arquitectura GCN: capa de entrada → capa oculta → embedding
3. Entrenamiento del GAE (reconstrucción de adyacencia con BCE)
4. Entrenamiento del VGAE (BCE + KL-divergence)
5. Cálculo del anomaly score por nodo (error de reconstrucción medio sobre sus aristas)
6. Visualización de la distribución de scores (histograma, boxplot)
7. Ranking de los 50 nodos más anómalos con sus metadatos
8. Análisis cualitativo: ¿coinciden con los candidatos del EDA?
   - ¿Aparecen los 13 nodos hoja? ¿Los 97 super-conectores? ¿Pablo (alta betweenness, degree moderada)?
   - ¿Se detecta "GEEK by AKEIN Engineering" (entidad empresarial)?
9. Visualización de embeddings con t-SNE o UMAP, coloreados por comunidad Louvain
10. Comparativa GAE vs. VGAE en calidad de embeddings y distribución de scores

**Hiperparámetros a explorar:** tamaño del embedding (16, 32, 64), número de capas GCN, learning rate, épocas.

**Output:** `results/embeddings/embeddings_M.npy`, `results/scores/scores_M.csv`

---

#### `06_gae_groups.ipynb`
**Propósito:** Entrenamiento y evaluación de GAE/VGAE sobre el **grafo de grupos (G)**.

**Contenido:**
1. Carga de `data_G.pt` y configuración del experimento
2. Arquitectura adaptada: el grafo G es más pequeño (456 nodos) pero tiene features más ricas (35 dimensiones vs. 3 en M), por lo que el diseño del encoder puede diferir
3. Entrenamiento GAE y VGAE
4. Cálculo de anomaly scores
5. Análisis cualitativo de los grupos más anómalos:
   - ¿Aparecen los 28 grupos hoja? ¿Los 146 grupos ausentes del grafo?
   - ¿Se detectan los grupos con desajuste tamaño/grado? (Nashville Hiking Meetup vs. Stepping Out Social Dance)
   - ¿Aparecen las microcomunidades C0 y C3?
6. Visualización de embeddings coloreados por `category_name`
7. Análisis de si el modelo captura la estructura temática en el espacio de embedding

**Output:** `results/embeddings/embeddings_G.npy`, `results/scores/scores_G.csv`

---

#### `07_gae_bipartite.ipynb`
**Propósito:** Entrenamiento y evaluación de GAE/VGAE sobre el **grafo bipartito (MG)**.

**Contenido:**
1. Carga de `data_MG.pt` y configuración del experimento
2. Consideraciones especiales del bipartito:
   - 25.233 nodos (vs. 11.372 en M y 456 en G): mayor coste computacional
   - Heterogeneidad de features: padding con ceros para unificar la dimensión de X
   - No conexidad: tratamiento de las 24 componentes aisladas
3. Entrenamiento GAE y VGAE
4. Cálculo de scores separados para nodos miembro y nodos grupo
5. Análisis de anomalías en miembros:
   - ¿Se identifican los 10.834 miembros inactivos (un solo evento)?
   - ¿Aparecen los 151 exploradores (muchos grupos, poca asistencia)?
   - ¿Se detectan los 61 miembros hiperactivos?
6. Análisis de anomalías en grupos:
   - ¿Aparecen los grupos fantasma (Nashville Social Crew: 4.017 registrados, 3 activos)?
   - ¿Se detectan los 37 grupos con un único miembro activo?
7. Visualización de embeddings con distinción por tipo de nodo (miembro vs. grupo)

**Output:** `results/embeddings/embeddings_MG.npy`, `results/scores/scores_MG.csv`

---

#### `08_results_comparison.ipynb`
**Propósito:** Comparativa transversal de resultados entre los tres grafos.

**Contenido:**
1. Carga de los tres archivos de scores
2. Análisis de solape: ¿qué nodos aparecen como anómalos en más de un grafo?
   - Un nodo miembro anómalo en M y también en MG tiene doble evidencia
   - Un grupo anómalo en G y en MG idem
3. Construcción de un **ranking combinado** para miembros y grupos
4. Comparativa GAE vs. VGAE en los tres grafos: ¿cuál detecta anomalías más interpretables?
5. Validación final contra todos los candidatos identificados en el EDA:
   - Tabla resumen: candidato EDA → ¿detectado por GAE M? ¿G? ¿MG?
6. Análisis de limitaciones: qué tipos de anomalía el modelo no consigue capturar y por qué
7. Visualización conjunta: heatmap de scores normalizados para los top-50 nodos más anómalos

**Output:** `results/scores/scores_combined.csv`, figuras de comparativa

---

## 7. Instalación y entorno

### Requisitos del sistema

- Python 3.12
- CUDA 11.8+ (recomendado para GPU; funciona en CPU pero más lento para el grafo M)
- Conda (recomendado) o pip

### Instalación con Conda

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/tfm-anomaly-detection-graphs.git
cd tfm-anomaly-detection-graphs

# Crear el entorno
conda env create -f environment.yml
conda activate TFM_grafos_V2
```

### Instalación manual con pip

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install networkx pandas numpy scikit-learn matplotlib seaborn kagglehub
pip install umap-learn  # para visualización de embeddings
```

### `environment.yml`

```yaml
name: TFM_grafos_V2
channels:
  - pytorch
  - pyg
  - conda-forge
  - defaults
dependencies:
  - python=3.12
  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8
  - pyg
  - networkx
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyter
  - pip:
    - kagglehub
    - umap-learn
```

### Credenciales de Kaggle

El notebook 01 descarga el dataset automáticamente via `kagglehub`. Es necesario tener el archivo `~/.kaggle/kaggle.json` con las credenciales de la API de Kaggle:

```bash
# Descargar kaggle.json desde https://www.kaggle.com/settings → API → Create New Token
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

---

## 8. Pipeline de ejecución

Los notebooks deben ejecutarse en orden secuencial, ya que cada uno produce outputs que consume el siguiente:

```
01_data_loading   →   data/processed/*.csv
       ↓
02_eda_metadata   →   (análisis exploratorio, sin outputs para el pipeline)
       ↓
03_eda_graphs     →   (análisis exploratorio, sin outputs para el pipeline)
       ↓
04_feature_eng    →   data/graph_data/data_M.pt
                      data/graph_data/data_G.pt
                      data/graph_data/data_MG.pt
       ↓
05_gae_members    →   results/embeddings/embeddings_M.npy
                      results/scores/scores_M.csv
       ↓
06_gae_groups     →   results/embeddings/embeddings_G.npy
                      results/scores/scores_G.csv
       ↓
07_gae_bipartite  →   results/embeddings/embeddings_MG.npy
                      results/scores/scores_MG.csv
       ↓
08_results        →   results/scores/scores_combined.csv
                      (comparativa final y visualizaciones)
```

**Nota sobre tiempos de ejecución estimados (CPU):**

| Notebook | Tiempo estimado |
|---|---|
| 01 — Carga | ~2 min |
| 02-03 — EDA | ~5-10 min |
| 04 — Feature engineering | ~3 min |
| 05 — GAE Miembros | ~15-30 min (grafo denso, 1.17M aristas) |
| 06 — GAE Grupos | ~2-5 min (grafo pequeño) |
| 07 — GAE Bipartito | ~10-20 min |
| 08 — Comparativa | ~5 min |

---

## 9. Convenciones del proyecto

### Nomenclatura de DataFrames

| Variable | Contenido |
|---|---|
| `meta_members` | DataFrame con metadatos de los 24.591 miembros |
| `meta_groups` | DataFrame con metadatos de los 602 grupos |
| `meta_events` | DataFrame con los 19.307 eventos |
| `member_edges` | DataFrame con las 1.176.368 aristas del grafo M |
| `group_edges` | DataFrame con las 6.692 aristas del grafo G |
| `member_group_edges` | DataFrame con las 45.583 aristas del grafo MG |

### Nomenclatura de grafos

| Variable | Descripción |
|---|---|
| `M` | Grafo de miembros (NetworkX) |
| `G` | Grafo de grupos (NetworkX) |
| `MG` | Grafo bipartito miembro-grupo (NetworkX) |
| `data_M` | Objeto `torch_geometric.data.Data` del grafo M |
| `data_G` | Objeto `torch_geometric.data.Data` del grafo G |
| `data_MG` | Objeto `torch_geometric.data.Data` del grafo MG |

### Prefijos de nodos en el bipartito

Los nodos del grafo bipartito llevan prefijo para evitar colisiones entre los espacios de IDs de miembros y grupos:
- Nodos miembro: `member_<member_id>` (ej: `member_2069`)
- Nodos grupo: `group_<group_id>` (ej: `group_339011`)

### Estructura del objeto `Data` de PyG

```python
data_M.x            # Tensor [num_nodes, num_features] — features de nodo
data_M.edge_index   # Tensor [2, num_edges*2] — aristas en ambas direcciones
data_M.edge_weight  # Tensor [num_edges*2] — pesos de arista
data_M.member_ids   # Lista de member_ids en el orden del tensor x

data_G.group_ids    # Lista de group_ids en el orden del tensor x

data_MG.node_ids    # Lista de IDs (member_* y group_*) en orden del tensor x
data_MG.n_members   # Número de nodos miembro (primeros n_members nodos)
data_MG.n_groups    # Número de nodos grupo (últimos n_groups nodos)
```

---

## 10. Estado actual

| Fase | Estado | Notebook |
|---|---|---|
| Carga y preprocesamiento | ✅ Completado | `01` |
| EDA tabular | ✅ Completado | `02` |
| EDA de grafos | ✅ Completado | `03` |
| Feature engineering | ✅ Completado | `04` |
| GAE — Grafo de miembros | 🔲 Pendiente | `05` |
| GAE — Grafo de grupos | 🔲 Pendiente | `06` |
| GAE — Grafo bipartito | 🔲 Pendiente | `07` |
| Comparativa de resultados | 🔲 Pendiente | `08` |

---

## Referencias principales

- Kipf, T. N., & Welling, M. (2016). *Variational Graph Auto-Encoders*. NeurIPS Workshop.
- Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR.
- Ma, X., et al. (2021). *A Comprehensive Study on Large-Scale Graph Training*. NeurIPS.
- Ding, K., et al. (2019). *Deep Anomaly Detection on Attributed Networks*. SDM.
- PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/

---

*Repositorio del TFM — Detección de Anomalías en Grafos con GAE · Meetup Tennessee*