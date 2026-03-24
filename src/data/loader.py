"""
loader.py
Funciones para cargar datasets de detección de anomalías en grafos.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def load_csv(filename: str, folder: str = "raw", **kwargs) -> pd.DataFrame:
    """Carga un archivo CSV desde data/raw o data/processed."""
    path = DATA_DIR / folder / filename
    print(f"[loader] Cargando: {path}")
    return pd.read_csv(path, **kwargs)


def load_edgelist(filename: str, folder: str = "raw", **kwargs) -> pd.DataFrame:
    """Carga una lista de aristas (source, target, [peso, ...])."""
    path = DATA_DIR / folder / filename
    print(f"[loader] Cargando edge list: {path}")
    return pd.read_csv(path, **kwargs)


def load_node_features(filename: str, folder: str = "raw", **kwargs) -> pd.DataFrame:
    """Carga atributos/features de los nodos."""
    path = DATA_DIR / folder / filename
    print(f"[loader] Cargando node features: {path}")
    return pd.read_csv(path, index_col=0, **kwargs)


def save_processed(df: pd.DataFrame, filename: str) -> None:
    """Guarda un DataFrame en data/processed."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    print(f"[loader] Guardado en: {out}")
