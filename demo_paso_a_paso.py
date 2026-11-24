"""Recorre el sistema completo en ocho pasos e imprime lo que produce cada uno.

Uso:
    python demo_paso_a_paso.py            # datos simulados, base temporal (no toca data/)
    python demo_paso_a_paso.py --pausa    # espera Enter entre pasos (para capturas)
    python demo_paso_a_paso.py --real     # usa las claves del .env y la base real
    python demo_paso_a_paso.py --n 2000000   # rutas simuladas en el paso 7
"""
import argparse
import asyncio
import os
import tempfile
import time
import warnings

import pandas as pd

from src import analysis, mpi_run
from src.clean import build_hourly, clean_flights
from src.compute.backend import available_backends, benchmark, get_backend
from src.config import Config, load_config
from src.ingest.clients import fetch_all
from src.storage import Storage

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def titulo(texto: str, args) -> None:
    if args.pausa:
        input("\n[Enter para continuar] ")
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def muestra(nombre: str, df: pd.DataFrame, n: int = 3) -> None:
    print(f"--- tabla '{nombre}' ---")
    print(df.head(n).to_string(index=False))
    print(f"... ({len(df)} filas en total)")


async def llamada_simulada(latencia: float) -> None:
    await asyncio.sleep(latencia)


async def en_paralelo(n_llamadas: int, latencia: float) -> None:
    # gather lanza todas a la vez: dura ~ lo que una sola
    await asyncio.gather(*(llamada_simulada(latencia) for _ in range(n_llamadas)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pausa", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--n", type=int, default=100_000)
    args = ap.parse_args()

    # ---------------------------------------------------------------- Paso 1
    titulo("PASO 1: Configuración (.env)", args)
    if args.real:
        cfg = load_config(".env")
    else:
        ruta = os.path.join(tempfile.mkdtemp(), "demo.db")
        cfg = Config(demo_mode=True, backend="numba", db_path=ruta)
    print("Modo           :", "DEMO (datos simulados)" if cfg.demo_mode else "APIs reales")
    print("Aeropuertos    :", ", ".join(cfg.airports))
    print("Backend pedido :", cfg.backend)
    print("Base de datos  :", cfg.db_path)

    # ---------------------------------------------------------------- Paso 2
    titulo("PASO 2: Backends numéricos disponibles en este equipo", args)
    print("Disponibles :", ", ".join(available_backends()))
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        usado = get_backend(cfg.backend)   # si el pedido falta, cae al siguiente
    print("Se usará    :", usado.name)
    for a in avisos:
        print("Aviso       :", a.message)

    # ---------------------------------------------------------------- Paso 3
    titulo("PASO 3: Ingesta concurrente de las APIs (asyncio)", args)
    storage = Storage(cfg.db_path)
    t0 = time.perf_counter()
    tablas = asyncio.run(fetch_all(cfg, storage))   # asyncio.gather por dentro
    print(f"Ingesta terminada en {time.perf_counter() - t0:.2f} s")
    for nombre in ("flights", "weather"):
        muestra(nombre, tablas[nombre])
    print("\nSimulación con 20 consultas de 100 ms cada una (sin red):")
    t0 = time.perf_counter()
    for _ in range(20):
        time.sleep(0.1)
    serie = time.perf_counter() - t0
    t0 = time.perf_counter()
    asyncio.run(en_paralelo(20, 0.1))
    paralelo = time.perf_counter() - t0
    print(f"  una tras otra : {serie:.2f} s")
    print(f"  con asyncio   : {paralelo:.2f} s  (≈ {serie / paralelo:.0f}x más rápido)")

    # ---------------------------------------------------------------- Paso 4
    titulo("PASO 4: Almacenamiento en SQLite y caché de consultas", args)
    for nombre, df in tablas.items():
        storage.save_table(nombre, df)          # tabla lista para el procesamiento
    cache = ("demo/endpoint", {"airport": "CUU"})
    storage.put_cache(*cache, {"respuesta": "guardada"})
    print("Tablas guardadas:", ", ".join(f"{k}={len(v)}" for k, v in tablas.items()))
    print("Caché vigente   :", storage.get_cached(*cache, max_age_s=60))    # la devuelve
    print("Caché vencida   :", storage.get_cached(*cache, max_age_s=0))     # None
    print("Modo del diario :", storage.journal_mode(), "(WAL: lecturas mientras se escribe)")

    # ---------------------------------------------------------------- Paso 5
    titulo("PASO 5: Limpieza con Pandas y tabla por aeropuerto y hora", args)
    crudo = storage.load_table("flights")
    limpio = clean_flights(crudo)               # tipos, duplicados, rangos válidos
    horaria = build_hourly({"flights": limpio,
                            "weather": storage.load_table("weather")})
    print(f"Vuelos antes de limpiar: {len(crudo)} -> después: {len(limpio)}")
    print(f"Velocidades sin dato (NaN): antes {int(pd.to_numeric(crudo['speed'], errors='coerce').isna().sum())}"
          f" -> después {int(limpio['speed'].isna().sum())}")
    print("\nTabla por hora (retraso = 15 min o más):")
    print(horaria[["airport", "hour", "flights", "delayed", "delay_rate", "rain_mm", "visibility"]]
          .head(5).round({"delay_rate": 3, "rain_mm": 1, "visibility": 0}).to_string(index=False))
    print(f"... ({len(horaria)} filas en total)")

    # ---------------------------------------------------------------- Paso 6
    titulo("PASO 6: Reparto con MPI y procesamiento por aeropuerto", args)
    aeropuertos = sorted(horaria["airport"].dropna().unique())
    for rango, lote in enumerate(mpi_run.partition(aeropuertos, 4)):
        print(f"  proceso {rango} recibe: {lote}")
    resultados = mpi_run.run(cfg, storage)      # con mpirun: scatter, procesar y gather
    print()
    print(resultados.sort_values("airport").round(3).rename(columns={
        "mean_delay_rate": "retraso", "outliers": "atipicos", "corr_rain_delay": "r_lluvia",
        "corr_wind_delay": "r_viento", "corr_vis_delay": "r_visibilidad"}).to_string(index=False))

    # ---------------------------------------------------------------- Paso 7
    titulo(f"PASO 7: Comparación de backends ({args.n:,} rutas simuladas)", args)
    tiempos = benchmark(args.n)                 # simula n rutas con cada backend
    base = tiempos["numpy"]
    for nombre, seg in tiempos.items():
        print(f"  {nombre:<6} {seg:8.3f} s {base / seg:5.1f}x vs numpy")

    # ---------------------------------------------------------------- Paso 8
    titulo("PASO 8: Análisis: clima contra retrasos y tráfico", args)
    print("Retrasos según el clima:")
    print(analysis.weather_delay_table(horaria).round(3).to_string(index=False))
    print("\nRetrasos por aeropuerto:")
    print(analysis.delays_by_airport(limpio).round(2).to_string(index=False))
    print("\nCorrelación con tráfico y con tasa de retraso:")
    print(analysis.traffic_vs_environment(horaria).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
