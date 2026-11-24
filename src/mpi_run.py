"""Módulo de procesamiento distribuido por aeropuerto con MPI y Multiprocessing.

Modos de ejecución:
1. Con MPI (computación en cluster / múltiples procesos nativos):
       mpirun -n 4 python -m src.mpi_run [ruta_al_.env]
   - Rank 0 (coordinador): Lee los datos de SQLite, construye la tabla horaria
     y reparte equitativamente los aeropuertos a los nodos (mediante scatter).
   - Ranks 1..N (trabajadores): Procesan sus aeropuertos asignados utilizando el
     motor numérico configurado (Numba / NumPy).
   - Rank 0: Recolecta los resultados (mediante gather) y es el ÚNICO proceso
     autorizado para escribir en la base de datos SQLite (evitando bloqueos).

2. Sin mpi4py:
   - Se activa automáticamente un fallback transparente usando `multiprocessing.Pool`
     para aprovechar todos los núcleos de la máquina local.
"""

import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd

from src.clean import HOURLY_COLS, build_hourly, clean_flights
from src.compute.backend import Backend, get_backend
from src.config import load_config
from src.storage import Storage

# Esquema de salida de la tabla 'results'
RESULT_COLS = [
    "airport",
    "n_rows",
    "mean_delay_rate",
    "std",
    "outliers",
    "corr_rain_delay",
    "corr_wind_delay",
    "corr_vis_delay",
]


def partition(items: list, n: int) -> list:
    """Reparte una lista de elementos en n particiones mediante round-robin.

    Si el número de procesos n supera la cantidad de elementos, los rangos sobrantes
    reciben listas vacías, garantizando que siempre se devuelvan exactamente n listas.

    Args:
        items (list): Lista de elementos a repartir.
        n (int): Número de particiones deseadas.

    Returns:
        list[list]: Lista de n sublistas con los elementos distribuidos.

    Raises:
        ValueError: Si n < 1.
    """
    if n < 1:
        raise ValueError("n debe ser >= 1")
    return [list(items[i::n]) for i in range(n)]


def process_airport(df_airport: pd.DataFrame, backend: Backend, airport: str = None) -> dict:
    """Calcula estadísticas y correlaciones climáticas con retrasos para un aeropuerto específico.

    Métricas computadas:
    - n_rows: Cantidad de intervalos horarios analizados.
    - mean_delay_rate: Promedio de la tasa de retraso.
    - std: Desviación estándar de la tasa de retraso.
    - outliers: Número de horas con tasas de retraso atípicas (|z| > 3).
    - corr_rain_delay: Correlación de Pearson entre precipitación (lluvia) y tasa de retraso.
    - corr_wind_delay: Correlación entre velocidad del viento y tasa de retraso.
    - corr_vis_delay: Correlación entre visibilidad y tasa de retraso.

    Args:
        df_airport (pd.DataFrame): Datos horarios correspondientes al aeropuerto.
        backend (Backend): Instancia del motor de cálculo (NumbaBackend o Backend).
        airport (str, optional): Código del aeropuerto. Si es None, se extrae del DataFrame.

    Returns:
        dict: Diccionario estructurado según las columnas de `RESULT_COLS`.
    """
    if airport is None and not df_airport.empty:
        airport = str(df_airport["airport"].iloc[0])

    n = len(df_airport)
    if n == 0:
        nan = float("nan")
        return dict(zip(RESULT_COLS, [airport, 0, nan, nan, 0, nan, nan, nan]))

    rate = df_airport["delay_rate"].to_numpy(dtype=float)
    col = lambda name: df_airport[name].to_numpy(dtype=float)  # noqa: E731
    mean, std = backend.mean_std(rate)

    return {
        "airport": airport,
        "n_rows": n,
        "mean_delay_rate": mean,
        "std": std,
        "outliers": int(backend.outlier_mask(rate).sum()),
        "corr_rain_delay": backend.pearson(col("rain_mm"), rate),
        "corr_wind_delay": backend.pearson(col("wind_speed"), rate),
        "corr_vis_delay": backend.pearson(col("visibility"), rate),
    }


def _get_comm():
    """Obtiene el comunicador global de MPI (MPI.COMM_WORLD) si mpi4py está disponible.

    Returns:
        MPI.Comm | None: Instancia del comunicador MPI o None si no está presente.
    """
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except (ImportError, RuntimeError):  # Sin mpi4py o sin biblioteca libmpi compartida
        return None


def _worker(args):
    """Función de trabajo auxiliar para ejecución con multiprocessing.Pool."""
    airport, df, backend_name = args
    return process_airport(df, get_backend(backend_name), airport=airport)


def _process_pool(items: list, backend_name: str) -> list:
    """Ejecuta el procesamiento paralelo mediante multiprocessing cuando no se dispone de MPI.

    Args:
        items (list): Lista de tuplas (codigo_aeropuerto, DataFrame).
        backend_name (str): Nombre del motor de cálculo ('numba' o 'numpy').

    Returns:
        list[dict]: Lista de resultados calculados para cada aeropuerto.
    """
    jobs = [(ap, df, backend_name) for ap, df in items]
    if len(jobs) <= 1:
        return [_worker(j) for j in jobs]
    try:
        # Usar tantos núcleos como existan en el sistema (hasta la cantidad de tareas)
        with mp.Pool(min(os.cpu_count() or 1, len(jobs))) as pool:
            return pool.map(_worker, jobs)
    except Exception as exc:  # noqa: BLE001 - entornos restringidos sin multiprocessing
        print(f"[mpi_run] multiprocessing falló ({exc}); se procesa en serie", file=sys.stderr)
        return [_worker(j) for j in jobs]


def _load_hourly(storage: Storage) -> pd.DataFrame:
    """Carga los vuelos y mediciones meteorológicas desde SQLite y construye la tabla horaria."""
    tables = {name: storage.load_table(name) for name in ("flights", "weather")}
    if not tables["flights"].empty:
        tables["flights"] = clean_flights(tables["flights"])
    hourly = build_hourly(tables)
    hourly["hour"] = pd.to_datetime(hourly["hour"])
    return hourly


def _to_frame(results: list) -> pd.DataFrame:
    """Convierte una lista de diccionarios de métricas en un DataFrame ordenado según RESULT_COLS."""
    return pd.DataFrame(results, columns=RESULT_COLS)


def run(config, storage: Storage, comm=None):
    """Punto de orquestación distribuida para procesar los aeropuertos.

    Lógica de coordinación:
    - Si comm es None (sin MPI): Utiliza multiprocessing.
    - Si comm está activo (MPI):
      * Rank 0 divide los aeropuertos y los reparte vía scatter.
      * Cada rank ejecuta el cálculo numérico para su subconjunto.
      * Rank 0 recopila los resultados vía gather y los guarda en SQLite.
      * Los ranks secundarios retornan None.

    Args:
        config: Configuración del sistema (contiene backend y rutas).
        storage (Storage): Instancia de persistencia SQLite.
        comm (optional): Comunicador de MPI; si es None, se intenta resolver automáticamente.

    Returns:
        pd.DataFrame | None: DataFrame con resultados consolidados en Rank 0 (o None en ranks auxiliares).
    """
    comm = comm if comm is not None else _get_comm()

    # Fallback sin MPI
    if comm is None:
        print("[mpi_run] mpi4py no disponible: se usa multiprocessing", file=sys.stderr)
        hourly = _load_hourly(storage)
        items = [(ap, g) for ap, g in hourly.groupby("airport")]
        results = _to_frame(_process_pool(items, config.backend))
        _save(storage, hourly, results)
        return results

    # Flujo distribuido con MPI
    rank, size = comm.Get_rank(), comm.Get_size()
    chunks, hourly = None, None

    # Solo el proceso raíz (Rank 0) accede a la base de datos para preparar los datos
    if rank == 0:
        hourly = _load_hourly(storage)
        items = [(ap, g) for ap, g in hourly.groupby("airport")]
        chunks = partition(items, size)

    # 1. Distribuir particiones de aeropuertos a todos los procesos
    mine = comm.scatter(chunks, root=0)
    print(f"[proceso {rank} de {size}] aeropuertos: {[ap for ap, _ in mine]}", flush=True)

    # 2. Cada proceso computa localmente su carga asignada
    backend = get_backend(config.backend)
    local = [process_airport(df, backend, airport=ap) for ap, df in mine]

    # 3. Reunir los resultados calculados en el proceso raíz (Rank 0)
    gathered = comm.gather(local, root=0)

    # Ranks secundarios finalizan su tarea aquí
    if rank != 0:
        return None

    # 4. Rank 0 consolida los resultados y los persiste en SQLite
    results = _to_frame([r for part in gathered for r in part])
    _save(storage, hourly, results)
    return results


def _save(storage: Storage, hourly: pd.DataFrame, results: pd.DataFrame) -> None:
    """Guarda las tablas 'hourly' y 'results' en SQLite asegurando no sobrescribir con vacíos."""
    if hourly.empty:
        return  # Evitar sobreescribir resultados históricos si no se generó información nueva
    storage.save_table("hourly", hourly)
    storage.save_table("results", results)


def main():
    """Función principal de ejecución desde la línea de comandos."""
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else ".env")
    results = run(cfg, Storage(cfg.db_path))
    if results is None:
        return  # Ranks secundarios en MPI
    if results.empty:
        print("Sin datos: corre primero la ingesta (python -m src.ingest.run_ingest).")
    else:
        print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
