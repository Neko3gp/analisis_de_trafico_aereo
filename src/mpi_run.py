"""Procesamiento distribuido por aeropuerto.

Con MPI:  mpirun -n 4 python -m src.mpi_run
          rank 0 lee la base, reparte los aeropuertos (scatter), todos procesan con el
          backend elegido, rank 0 junta los resultados (gather) y es el único que escribe.
Sin mpi4py: se reparte con multiprocessing (con aviso).
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

RESULT_COLS = ["airport", "n_rows", "mean_delay_rate", "std", "outliers",
               "corr_rain_delay", "corr_wind_delay", "corr_vis_delay"]


def partition(items: list, n: int) -> list:
    """Reparte items en n listas (round-robin); los ranks sobrantes reciben una lista vacía."""
    if n < 1:
        raise ValueError("n debe ser >= 1")
    return [list(items[i::n]) for i in range(n)]


def process_airport(df_airport: pd.DataFrame, backend: Backend, airport: str = None) -> dict:
    """Estadísticas y correlaciones del clima con los retrasos para un aeropuerto."""
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
    """Comunicador de MPI, o None si mpi4py no está instalado."""
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except ImportError:
        return None


def _worker(args):
    airport, df, backend_name = args
    return process_airport(df, get_backend(backend_name), airport=airport)


def _process_pool(items: list, backend_name: str) -> list:
    jobs = [(ap, df, backend_name) for ap, df in items]
    if len(jobs) <= 1:
        return [_worker(j) for j in jobs]
    try:
        with mp.Pool(min(os.cpu_count() or 1, len(jobs))) as pool:
            return pool.map(_worker, jobs)
    except Exception as exc:  # noqa: BLE001 - p. ej. entornos sin multiprocessing
        print(f"[mpi_run] multiprocessing falló ({exc}); se procesa en serie", file=sys.stderr)
        return [_worker(j) for j in jobs]


def _load_hourly(storage: Storage) -> pd.DataFrame:
    tables = {name: storage.load_table(name) for name in ("flights", "weather")}
    if not tables["flights"].empty:
        tables["flights"] = clean_flights(tables["flights"])
    hourly = build_hourly(tables)
    hourly["hour"] = pd.to_datetime(hourly["hour"])
    return hourly


def _to_frame(results: list) -> pd.DataFrame:
    return pd.DataFrame(results, columns=RESULT_COLS)


def run(config, storage: Storage, comm=None):
    """Devuelve el DataFrame de resultados en rank 0 (y None en los demás ranks)."""
    comm = comm if comm is not None else _get_comm()

    if comm is None:
        print("[mpi_run] mpi4py no disponible: se usa multiprocessing", file=sys.stderr)
        hourly = _load_hourly(storage)
        items = [(ap, g) for ap, g in hourly.groupby("airport")]
        results = _to_frame(_process_pool(items, config.backend))
        _save(storage, hourly, results)
        return results

    rank, size = comm.Get_rank(), comm.Get_size()
    chunks, hourly = None, None
    if rank == 0:
        hourly = _load_hourly(storage)
        items = [(ap, g) for ap, g in hourly.groupby("airport")]
        chunks = partition(items, size)
    mine = comm.scatter(chunks, root=0)
    backend = get_backend(config.backend)
    local = [process_airport(df, backend, airport=ap) for ap, df in mine]
    gathered = comm.gather(local, root=0)
    if rank != 0:
        return None
    results = _to_frame([r for part in gathered for r in part])
    _save(storage, hourly, results)
    return results


def _save(storage: Storage, hourly: pd.DataFrame, results: pd.DataFrame) -> None:
    if hourly.empty:
        return  # nada que guardar: no se pisan resultados anteriores con vacíos
    storage.save_table("hourly", hourly)
    storage.save_table("results", results)


def main():
    cfg = load_config()
    results = run(cfg, Storage(cfg.db_path))
    if results is None:
        return
    if results.empty:
        print("Sin datos: corre primero la ingesta (python -m src.ingest.run_ingest).")
    else:
        print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
