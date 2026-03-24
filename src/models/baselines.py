"""
baselines.py
Modelos baseline para detección de anomalías en grafos.
Métodos clásicos aplicados sobre features estructurales.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


def isolation_forest(features_df: pd.DataFrame, contamination: float = 0.05, random_state: int = 42) -> np.ndarray:
    """
    Detecta anomalías con Isolation Forest.

    Returns:
        Array de scores de anomalía (más negativo = más anómalo).
    """
    X = StandardScaler().fit_transform(features_df)
    model = IsolationForest(contamination=contamination, random_state=random_state, n_jobs=-1)
    scores = model.fit_predict(X)
    anomaly_scores = model.score_samples(X)
    print(f"[baselines] IsolationForest: {(scores == -1).sum()} anomalías detectadas")
    return anomaly_scores


def local_outlier_factor(features_df: pd.DataFrame, n_neighbors: int = 20, contamination: float = 0.05) -> np.ndarray:
    """
    Detecta anomalías con Local Outlier Factor.
    """
    X = StandardScaler().fit_transform(features_df)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=False)
    preds = lof.fit_predict(X)
    scores = lof.negative_outlier_factor_
    print(f"[baselines] LOF: {(preds == -1).sum()} anomalías detectadas")
    return scores


def one_class_svm(features_df: pd.DataFrame, nu: float = 0.05) -> np.ndarray:
    """
    Detecta anomalías con One-Class SVM.
    """
    X = StandardScaler().fit_transform(features_df)
    model = OneClassSVM(nu=nu, kernel="rbf")
    preds = model.fit_predict(X)
    scores = model.score_samples(X)
    print(f"[baselines] OneClassSVM: {(preds == -1).sum()} anomalías detectadas")
    return scores
