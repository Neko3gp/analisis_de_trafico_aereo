"""Punto de entrada de la ingesta: python -m src.ingest.run_ingest

Cada corrida se SUMA al historial guardado (sin duplicar), porque las APIs reales solo dan
la foto del momento y las correlaciones clima/retrasos necesitan varias horas de datos.
Conviene correrla periódicamente (p. ej. cada hora).
"""
import asyncio

import pandas as pd

from src.config import load_config
from src.ingest.clients import fetch_all
from src.storage import Storage

# tabla -> (columnas que identifican una fila, columnas de fecha)
TABLE_KEYS = {
    "flights": (["flight_iata", "dep_iata", "dep_time"], ["dep_time"]),
    "weather": (["airport", "ts"], ["ts"]),
    "air_quality": (["airport", "ts"], ["ts"]),
    "sensors": (["ts"], ["ts"]),
}


def merge_history(old: pd.DataFrame, new: pd.DataFrame, keys: list, time_cols: list) -> pd.DataFrame:
    """Une lo guardado con lo nuevo; ante una fila repetida gana la más reciente."""
    if old is None or old.empty:
        return new
    if new is None or new.empty:
        return old
    parts = []
    for df in (old, new):
        d = df.copy()
        for c in time_cols:
            d[c] = pd.to_datetime(d[c], errors="coerce")
        parts.append(d)
    merged = pd.concat(parts, ignore_index=True)
    return merged.drop_duplicates(keys, keep="last").reset_index(drop=True)


def run_ingest(env_path: str = ".env") -> dict:
    cfg = load_config(env_path)
    storage = Storage(cfg.db_path)
    fresh = asyncio.run(fetch_all(cfg, storage))
    tables = {}
    for name, df in fresh.items():
        keys, time_cols = TABLE_KEYS[name]
        tables[name] = merge_history(storage.load_table(name), df, keys, time_cols)
        storage.save_table(name, tables[name])
    modo = "DEMO (datos simulados)" if cfg.demo_mode else "APIs reales"
    print(f"Ingesta lista [{modo}]: " + ", ".join(f"{k}={len(v)}" for k, v in tables.items()))
    return tables


if __name__ == "__main__":
    run_ingest()
