"""
builder.py
Construcción de grafos a partir de edge lists y node features.
Compatible con NetworkX y PyTorch Geometric.
"""

import networkx as nx
import pandas as pd
import numpy as np


def build_graph_from_edgelist(
    edges_df: pd.DataFrame,
    source_col: str = "source",
    target_col: str = "target",
    weight_col: str = None,
    directed: bool = False,
) -> nx.Graph:
    """
    Construye un grafo NetworkX desde un DataFrame de aristas.

    Args:
        edges_df: DataFrame con columnas source y target (y opcionalmente peso).
        source_col: Nombre de la columna origen.
        target_col: Nombre de la columna destino.
        weight_col: Columna de pesos (opcional).
        directed: Si True, crea un DiGraph.

    Returns:
        Grafo NetworkX.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    for _, row in edges_df.iterrows():
        kwargs = {"weight": row[weight_col]} if weight_col and weight_col in row else {}
        G.add_edge(row[source_col], row[target_col], **kwargs)
    print(f"[builder] Grafo creado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    return G


def add_node_features(G: nx.Graph, features_df: pd.DataFrame) -> nx.Graph:
    """
    Agrega atributos de nodos al grafo.

    Args:
        G: Grafo NetworkX.
        features_df: DataFrame indexado por node_id con columnas de features.

    Returns:
        Grafo con atributos añadidos.
    """
    for node_id, row in features_df.iterrows():
        if G.has_node(node_id):
            G.nodes[node_id].update(row.to_dict())
    print(f"[builder] Features añadidos a {len(features_df)} nodos")
    return G


def graph_to_pyg(G: nx.Graph, node_feature_keys: list = None):
    """
    Convierte un grafo NetworkX a formato PyTorch Geometric Data.
    Requiere: torch, torch_geometric.
    """
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.utils import from_networkx
    except ImportError:
        raise ImportError("Instala torch y torch_geometric para usar esta función.")

    data = from_networkx(G, group_node_attrs=node_feature_keys)
    print(f"[builder] PyG Data: {data}")
    return data
