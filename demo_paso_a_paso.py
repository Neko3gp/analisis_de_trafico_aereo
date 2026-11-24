"""Recorre el sistema completo en 6 pasos e imprime lo que produce cada uno.

Uso:
    python demo_paso_a_paso.py            # datos simulados, base temporal (no toca data/)
    python demo_paso_a_paso.py --pausa    # espera Enter entre pasos (para tomar capturas)
    python demo_paso_a_paso.py --real     # usa las claves del .env y la base real (data/trafico.db)
    python demo_paso_a_paso.py --n 2000000   # rutas simuladas en el paso 5
"""
import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from src import analysis, mpi_run
from src.clean import build_hourly, clean_flights
from src.compute.backend import benchmark, get_backend
from src.config import load_config
from src.ingest.clients import fetch_all
from src.ingest.run_ingest import TABLE_KEYS, merge_history
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
        env_path = ".env"
    else:  # configuración temporal en modo demo: no toca tu .env ni data/
        carpeta = tempfile.mkdtemp()
        env_path = os.path.join(carpeta, ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"DEMO_MODE=1\nDB_PATH={os.path.join(carpeta, 'demo.db')}\n")
    cfg = load_config(env_path)
    print("Modo           :", "DEMO (datos simulados)" if cfg.demo_mode else "APIs reales (AirLabs + OpenWeather)")
    print("Aeropuertos    :", ", ".join(cfg.airports))
    print("Cálculo        :", get_backend(cfg.backend).name)
    print("Base de datos  :", cfg.db_path)

    # ---------------------------------------------------------------- Paso 2
    titulo("PASO 2: Lectura concurrente de las APIs (asyncio) y guardado en SQLite", args)
    storage = Storage(cfg.db_path)
    t0 = time.perf_counter()
    tablas = asyncio.run(fetch_all(cfg))        # asyncio.gather por dentro
    print(f"Ingesta terminada en {time.perf_counter() - t0:.2f} s")
    for nombre in ("flights", "weather"):
        muestra(nombre, tablas[nombre])
        claves, fechas = TABLE_KEYS[nombre]     # se SUMA al historial sin duplicar
        storage.save_table(nombre, merge_history(storage.load_table(nombre), tablas[nombre], claves, fechas))
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

    # ---------------------------------------------------------------- Paso 3
    titulo("PASO 3: Limpieza con Pandas y tabla por aeropuerto y hora", args)
    crudo = storage.load_table("flights")
    limpio = clean_flights(crudo)               # tipos, duplicados, rangos válidos
    horaria = build_hourly({"flights": limpio, "weather": storage.load_table("weather")})
    print(f"Vuelos antes de limpiar: {len(crudo)} -> después: {len(limpio)}")
    print("\nTabla por hora (retraso = 15 min o más):")
    print(horaria[["airport", "hour", "flights", "delayed", "delay_rate", "rain_mm", "visibility"]]
          .head(5).round({"delay_rate": 3, "rain_mm": 1, "visibility": 0}).to_string(index=False))
    print(f"... ({len(horaria)} filas en total)")

    # ---------------------------------------------------------------- Paso 4
    titulo("PASO 4: Reparto con MPI y procesamiento por aeropuerto", args)
    aeropuertos = sorted(horaria["airport"].dropna().unique())
    for rango, lote in enumerate(mpi_run.partition(aeropuertos, 4)):
        print(f"  proceso {rango} recibe: {lote}")
    print()
    if shutil.which("mpirun"):                  # 4 procesos reales
        subprocess.run(["mpirun", "-n", "4", sys.executable, "-m", "src.mpi_run", env_path], check=False)
    else:
        print("(mpirun no está instalado: se procesa con multiprocessing)")
        mpi_run.run(cfg, storage)
    resultados = storage.load_table("results")
    print("\nResultados guardados (correlación de cada variable del clima con la tasa de retraso):")
    print(resultados.sort_values("airport").round(3).to_string(index=False))

    # ---------------------------------------------------------------- Paso 5
    titulo(f"PASO 5: Numba contra numpy ({args.n:,} rutas simuladas)", args)
    tiempos = benchmark(args.n)                 # simula n rutas con cada backend
    for nombre, seg in tiempos.items():
        print(f"  {nombre:<6} {seg:8.3f} s {tiempos['numpy'] / seg:5.1f}x vs numpy")

    # ---------------------------------------------------------------- Paso 6
    titulo("PASO 6: Respuestas a las preguntas de análisis", args)
    clima = analysis.weather_delay_table(horaria).set_index("condition")["mean_delay_rate"]
    entorno = analysis.traffic_vs_environment(horaria).set_index("variable")
    por_aeropuerto = analysis.delays_by_airport(limpio)
    por_hora = analysis.activity_by_hour(limpio).set_index("hour_of_day")["flights"]

    print("1. ¿Hay relación entre clima y retrasos? (correlación con la tasa de retraso)")
    print(entorno["corr_with_delay_rate"].round(2).to_string())
    print("\n2. ¿Aumentan los retrasos con lluvia o baja visibilidad? (tasa media de retraso)")
    print((clima * 100).round(1).astype(str).add(" %").to_string())
    print("\n3. ¿Qué aeropuertos tienen más retrasos? (retraso medio en minutos)")
    print(por_aeropuerto[["airport", "flights", "mean_delay_min", "delayed_pct"]].round(1).to_string(index=False))
    print(f"\n4. ¿Cómo cambia la actividad durante el día? Hora pico: {por_hora.idxmax()}:00 "
          f"({por_hora.max()} vuelos); hora más baja: {por_hora.idxmin()}:00 ({por_hora.min()} vuelos)")
    print("\n5. ¿Qué condiciones coinciden con más tráfico? (correlación con vuelos por hora)")
    print(entorno["corr_with_flights"].round(2).sort_values(key=abs, ascending=False).to_string())


if __name__ == "__main__":
    main()
