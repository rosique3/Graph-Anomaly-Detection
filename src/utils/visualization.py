"""
visualization.py
Project-wide visual style and figure utilities.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

# ── Output ───────────────────────────────────────────────────────────────────
FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
# Okabe-Ito (2008): colorblind-safe and grayscale-separable.
PALETTE = {
    "M":       "#0072B2",   # blue      – Members graph
    "G":       "#E69F00",   # amber     – Groups graph
    "MG":      "#009E73",   # teal      – Bipartite graph
    "anomaly": "#D55E00",   # vermilion – anomaly highlight
    "neutral": "#999999",   # grey      – baselines / reference lines
}

# Ordered sequence for categories, communities, ordinal series.
CAT_COLORS = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#56B4E9", "#F0E442", "#CC79A7", "#000000",
]

# Colormaps
CMAP_SEQ  = "viridis"   # perceptually uniform, grayscale-safe
CMAP_DIV  = "RdBu_r"    # diverging
CMAP_HEAT = "YlOrRd"    # heatmaps

# ── Figure sizes ─────────────────────────────────────────────────────────────
FS = {
    "single":  (6.0,  4.0),
    "wide":    (8.0,  4.0),
    "r1c2":    (12.0, 4.0),
    "r1c3":    (14.0, 4.5),
    "r2c2":    (12.0, 8.0),
    "r3c1":    (6.0,  11.0),
    "map":     (10.0, 6.0),
    "heatmap": (8.0,  6.0),
}

DPI_SCREEN = 120
DPI_PRINT  = 300

# ── Typography ────────────────────────────────────────────────────────────────
FONT = {
    "title":      11,
    "axis_label": 10,
    "tick":        9,
    "legend":      9,
    "annotation":  8,
}


# ── Theme ─────────────────────────────────────────────────────────────────────
def apply_theme() -> None:
    """Set project-wide rcParams. Call once at the top of every notebook."""
    sns.set_theme(
        style="ticks",   # removes top/right spines; outward ticks → Nature style
        rc={
            "figure.dpi":            DPI_SCREEN,
            "savefig.dpi":           DPI_PRINT,
            "figure.figsize":        list(FS["single"]),
            "savefig.bbox":          "tight",
            "savefig.pad_inches":    0.05,

            "font.family":           "sans-serif",
            "font.sans-serif":       ["DejaVu Sans", "Arial", "Helvetica Neue"],
            "font.size":             FONT["tick"],
            "axes.titlesize":        FONT["title"],
            "axes.titleweight":      "normal",
            "axes.labelsize":        FONT["axis_label"],
            "xtick.labelsize":       FONT["tick"],
            "ytick.labelsize":       FONT["tick"],
            "legend.fontsize":       FONT["legend"],
            "legend.title_fontsize": FONT["legend"],

            "axes.linewidth":        0.8,
            "axes.axisbelow":        True,
            "axes.grid":             False,   # opt-in per chart via polish()

            "xtick.direction":       "out",
            "ytick.direction":       "out",
            "xtick.major.size":      4.0,
            "ytick.major.size":      4.0,
            "xtick.major.width":     0.8,
            "ytick.major.width":     0.8,

            "legend.frameon":        False,
            "legend.loc":            "best",

            "lines.linewidth":       1.5,
            "patch.linewidth":       0.4,
        },
    )


def polish(
    ax: plt.Axes,
    *,
    grid: bool = True,
    grid_axis: str = "y",
) -> None:
    """
    Per-chart finishing pass: explicit spine removal + optional grid.

    Call after all plotting calls, before tight_layout / savefig.
    Maps and network graphs should skip this (they call ax.axis('off')).

    Args:
        ax:        Target Axes.
        grid:      Whether to draw a grid (default True).
        grid_axis: 'x', 'y', or 'both'.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(grid, axis=grid_axis, color="#E5E5E5", linewidth=0.6, zorder=0)


def save_figure(fig: plt.Figure, name: str, *, dpi: int = DPI_PRINT) -> Path:
    """
    Save a figure to results/figures/<name>.

    Naming convention: {notebook_id}_{description}.png
    Examples: 02_members_states.png, 05_anomaly_scores_M.png

    Args:
        fig:  Figure to save.
        name: Filename with extension.
        dpi:  Resolution in DPI (default 300).

    Returns:
        Absolute path of the saved file.
    """
    out = FIGURES_DIR / name
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    print(f"[viz] saved -> {out}")
    return out


# ── Reusable chart functions ──────────────────────────────────────────────────

def plot_graph(
    G: nx.Graph,
    anomaly_scores=None,
    node_labels=None,
    title: str = "Graph",
    save_as: str | None = None,
    pos=None,
) -> None:
    fig, ax = plt.subplots(figsize=FS["wide"])
    pos = pos or nx.spring_layout(G, seed=42)

    if anomaly_scores is not None:
        nodes = list(G.nodes())
        colors = (
            [anomaly_scores.get(n, 0) for n in nodes]
            if isinstance(anomaly_scores, dict)
            else list(anomaly_scores)
        )
        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_color=colors, cmap=CMAP_SEQ,
            node_size=60, with_labels=node_labels is None,
            labels=node_labels, edge_color="#CCCCCC",
            width=0.4, alpha=0.9, arrows=False,
        )
        sm = plt.cm.ScalarMappable(
            cmap=CMAP_SEQ,
            norm=mpl.colors.Normalize(min(colors), max(colors)),
        )
        plt.colorbar(sm, ax=ax, label="Anomaly score", shrink=0.8, pad=0.02)
    else:
        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_color=PALETTE["neutral"], node_size=60,
            edge_color="#CCCCCC", width=0.4, alpha=0.9,
        )

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    if save_as:
        save_figure(fig, save_as)
    plt.show()


def plot_score_distribution(
    scores: np.ndarray,
    threshold: float | None = None,
    title: str = "Anomaly score distribution",
    graph: str = "M",
    save_as: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=FS["single"])
    color = PALETTE.get(graph, PALETTE["neutral"])
    ax.hist(scores, bins=50, color=color, alpha=0.85,
            edgecolor="white", linewidth=0.3)
    if threshold is not None:
        ax.axvline(
            threshold, color=PALETTE["anomaly"],
            linewidth=1.2, linestyle="--",
            label=f"Threshold: {threshold:.3f}",
        )
        ax.legend()
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Count")
    ax.set_title(title)
    polish(ax)
    plt.tight_layout()
    if save_as:
        save_figure(fig, save_as)
    plt.show()


def plot_roc_curve(
    y_true,
    scores,
    graph: str = "M",
    save_as: str | None = None,
) -> None:
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    color = PALETTE.get(graph, PALETTE["neutral"])

    fig, ax = plt.subplots(figsize=FS["single"])
    ax.plot(fpr, tpr, color=color, linewidth=1.8, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["neutral"],
            linewidth=0.8, linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend()
    polish(ax, grid=False)
    plt.tight_layout()
    if save_as:
        save_figure(fig, save_as)
    plt.show()
