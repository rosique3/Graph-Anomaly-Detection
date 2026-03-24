"""
visualization.py
Funciones de visualización para grafos y resultados de anomalías.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_graph(G: nx.Graph, anomaly_scores=None, node_labels=None,
               title: str = "Grafo", save_as: str = None, pos=None):
    """
    Visualiza el grafo coloreando nodos por score de anomalía.

    Args:
        G: Grafo NetworkX.
        anomaly_scores: Dict o array de scores por nodo (opcional).
        node_labels: Dict {node_id: label} para etiquetar nodos.
        title: Título del plot.
        save_as: Nombre de archivo para guardar en reports/figures/.
        pos: Layout de posiciones (si None, usa spring_layout).
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = pos or nx.spring_layout(G, seed=42)

    if anomaly_scores is not None:
        nodes = list(G.nodes())
        if isinstance(anomaly_scores, dict):
            colors = [anomaly_scores.get(n, 0) for n in nodes]
        else:
            colors = list(anomaly_scores)
        nx.draw_networkx(G, pos=pos, ax=ax, node_color=colors,
                         cmap=cm.RdYlGn_r, node_size=80,
                         with_labels=node_labels is None,
                         labels=node_labels, edge_color="gray",
                         alpha=0.8, arrows=False)
        sm = plt.cm.ScalarMappable(cmap=cm.RdYlGn_r,
                                   norm=plt.Normalize(min(colors), max(colors)))
        plt.colorbar(sm, ax=ax, label="Anomaly Score")
    else:
        nx.draw_networkx(G, pos=pos, ax=ax, node_size=80,
                         edge_color="gray", alpha=0.8)

    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    if save_as:
        out = FIGURES_DIR / save_as
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[viz] Figura guardada: {out}")
    plt.show()


def plot_score_distribution(scores: np.ndarray, threshold: float = None,
                            title: str = "Distribución de Anomaly Scores",
                            save_as: str = None):
    """Histograma de los scores de anomalía."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=50, color="steelblue", alpha=0.7, edgecolor="white")
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Umbral: {threshold:.3f}")
        ax.legend()
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Frecuencia")
    ax.set_title(title)
    plt.tight_layout()
    if save_as:
        out = FIGURES_DIR / save_as
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[viz] Figura guardada: {out}")
    plt.show()


def plot_roc_curve(y_true, scores, save_as: str = None):
    """Curva ROC."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend()
    plt.tight_layout()
    if save_as:
        out = FIGURES_DIR / save_as
        plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
