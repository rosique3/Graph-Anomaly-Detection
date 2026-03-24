"""
metrics.py
Métricas de evaluación para detección de anomalías en grafos.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float = None) -> dict:
    """
    Evaluación completa de un detector de anomalías.

    Args:
        y_true: Etiquetas reales (0=normal, 1=anomalía).
        scores: Scores de anomalía (mayor = más anómalo).
        threshold: Umbral para binarizar scores. Si None, usa la mediana.

    Returns:
        Diccionario con métricas.
    """
    if threshold is None:
        threshold = np.median(scores)

    y_pred = (scores >= threshold).astype(int)

    metrics = {
        "ROC-AUC": roc_auc_score(y_true, scores),
        "PR-AUC": average_precision_score(y_true, scores),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
    }

    print("\n=== Resultados de Evaluación ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Anomalía"], zero_division=0))

    return metrics


def threshold_by_percentile(scores: np.ndarray, percentile: float = 95) -> float:
    """Calcula un umbral basado en percentil (p.ej., top 5% como anomalías)."""
    return np.percentile(scores, percentile)
