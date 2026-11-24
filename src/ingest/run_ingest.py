"""Punto de entrada de la ingesta de datos.

Se puede ejecutar directamente desde la terminal:
    python -m src.ingest.run_ingest

Estrategia de persistencia histórica:
Cada ejecución se SUMA al historial persistido en SQLite sin generar filas duplicadas.
Esto es fundamental porque las APIs de nivel gratuito solo ofrecen el estado instantáneo,
mientras que los análisis estadísticos de correlación entre retrasos y clima requieren
acumular varias horas o días de observaciones históricas.
"""

import asyncio

import pandas as pd

from src.config import load_config
from src.ingest.clients import fetch_all
from src.storage import Storage

# Mapeo de tabla a: (claves que identifican de forma unívoca un registro, columnas temporales)
TABLE_KEYS = {
    "flights": (["flight_iata", "dep_iata", "dep_time"], ["dep_time"]),
    "weather": (["airport", "ts"], ["ts"]),
}


def merge_history(
    old: pd.DataFrame,
    new: pd.DataFrame,
    keys: list,
    time_cols: list,
) -> pd.DataFrame:
    """Fusiona datos históricos existentes con nuevas observaciones eliminando duplicados.

    Si una clave compuesta ya existe en la base de datos, la versión más reciente ('last')
    prevalece, permitiendo actualizar estados de vuelos que pasaron de programados a aterrizados.

    Args:
        old (pd.DataFrame): Datos previamente almacenados en SQLite.
        new (pd.DataFrame): Nuevos datos descargados en la ejecución actual.
        keys (list): Lista de columnas que componen la clave primaria lógica.
        time_cols (list): Lista de columnas de fecha/hora que deben ser parseadas a datetime.

    Returns:
        pd.DataFrame: DataFrame unificado y deduplicado.
    """
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

    # Concatenar y conservar la última observación observada para cada clave
    merged = pd.concat(parts, ignore_index=True)
    return merged.drop_duplicates(keys, keep="last").reset_index(drop=True)


def run_ingest(env_path: str = ".env") -> dict:
    """Orquesta el ciclo completo de ingesta de datos y actualización en SQLite.

    Pasos ejecutados:
    1. Carga la configuración desde el archivo .env.
    2. Conecta con el motor de almacenamiento SQLite.
    3. Descarga asíncronamente las lecturas actuales de vuelos y clima (o genera demo).
    4. Fusiona las nuevas observaciones con el historial existente para evitar duplicados.
    5. Guarda las tablas actualizadas ('flights' y 'weather') en la base de datos.

    Args:
        env_path (str): Ruta al archivo de configuración de entorno (.env).

    Returns:
        dict: Diccionario con los DataFrames consolidados {'flights': df, 'weather': df}.
    """
    cfg = load_config(env_path)
    storage = Storage(cfg.db_path)

    # Ejecutar la descarga asíncrona concurrente
    fresh = asyncio.run(fetch_all(cfg))

    tables = {}
    for name, df in fresh.items():
        keys, time_cols = TABLE_KEYS[name]
        # Fusionar con el historial existente y persistir en SQLite
        tables[name] = merge_history(storage.load_table(name), df, keys, time_cols)
        storage.save_table(name, tables[name])

    modo = "DEMO (datos simulados)" if cfg.demo_mode else "APIs reales"
    print(f"Ingesta lista [{modo}]: " + ", ".join(f"{k}={len(v)}" for k, v in tables.items()))
    return tables


if __name__ == "__main__":
    run_ingest()
