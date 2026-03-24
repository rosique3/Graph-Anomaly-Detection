"""
preprocessing.py
Limpieza y transformación de datos antes de construir el grafo.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Elimina filas duplicadas."""
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    print(f"[preprocessing] Duplicados eliminados: {before - len(df)}")
    return df


def handle_missing(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """Gestiona valores nulos. Estrategias: 'drop', 'mean', 'median', 'zero'."""
    if strategy == "drop":
        df = df.dropna()
    elif strategy == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    elif strategy == "median":
        df = df.fillna(df.median(numeric_only=True))
    elif strategy == "zero":
        df = df.fillna(0)
    print(f"[preprocessing] Nulos gestionados con estrategia: '{strategy}'")
    return df


def normalize_features(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """Normaliza columnas numéricas. Métodos: 'standard', 'minmax'."""
    num_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print(f"[preprocessing] Normalización '{method}' aplicada a {len(num_cols)} columnas")
    return df


def encode_labels(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Codifica etiquetas de anomalía a 0 (normal) / 1 (anomalía)."""
    df[label_col] = df[label_col].astype(int)
    print(f"[preprocessing] Distribución de etiquetas:\n{df[label_col].value_counts()}")
    return df
