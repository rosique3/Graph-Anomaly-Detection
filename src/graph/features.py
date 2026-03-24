"""
features.py
Extracción de features estructurales de grafos para detección de anomalías.
"""

import networkx as nx
import pandas as pd
import numpy as np


def compute_node_features(G: nx.Graph) -> pd.DataFrame:
    """
    Calcula features estructurales por nodo:
    - Grado, grado de entrada/salida (si dirigido)
    - Coeficiente de clustering
    - Centralidad de Betweenness y PageRank
    - Número de triángulos (si no dirigido)
    """
    features = {}

    # Grado
    degree = dict(G.degree())
    features["degree"] = degree

    if G.is_directed():
        features["in_degree"] = dict(G.in_degree())
        features["out_degree"] = dict(G.out_degree())

    # Clustering
    try:
        features["clustering"] = nx.clustering(G)
    except Exception:
        features["clustering"] = {n: 0 for n in G.nodes()}

    # Betweenness (puede ser lento en grafos grandes)
    features["betweenness"] = nx.betweenness_centrality(G, normalized=True)

    # PageRank
    features["pagerank"] = nx.pagerank(G, alpha=0.85)

    # Triángulos (solo grafos no dirigidos)
    if not G.is_directed():
        features["triangles"] = nx.triangles(G)

    df = pd.DataFrame(features)
    df.index.name = "node_id"
    print(f"[features] Features extraídos: {df.shape[1]} features, {df.shape[0]} nodos")
    return df


def compute_edge_features(G: nx.Graph) -> pd.DataFrame:
    """
    Calcula features por arista:
    - Peso (si existe)
    - Jaccard coefficient de los nodos extremos
    """
    rows = []
    for u, v, data in G.edges(data=True):
        row = {"source": u, "target": v}
        row["weight"] = data.get("weight", 1.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"[features] Edge features: {df.shape}")
    return df
