"""Demostración paso a paso del sistema completo de análisis de tráfico aéreo y clima.

Ejecuta el pipeline completo de forma pedagógica en 6 pasos sucesivos,
imprimiendo y explicando en la terminal la salida producida en cada etapa:
  Paso 1: Lectura e inspección de configuración (.env).
  Paso 2: Ingesta concurrente con asyncio (y comparativa de velocidad vs secuencial).
  Paso 3: Limpieza con Pandas y agregación horaria por aeropuerto.
  Paso 4: Reparto distribuido con MPI y procesamiento de estadísticas/correlaciones.
  Paso 5: Medición de rendimiento (Benchmark) de NumPy vs Numba.
  Paso 6: Respuestas cuantitativas a las preguntas de análisis del proyecto.

Uso:
    python demo_paso_a_paso.py            # modo demo seguro en base temporal (no toca data/)
    python demo_paso_a_paso.py --pausa    # pausa con Enter entre pasos (ideal para explicaciones/capturas)
    python demo_paso_a_paso.py --real     # usa credenciales del .env y base de datos real (data/trafico.db)
    python demo_paso_a_paso.py --n 2000000   # cantidad de rutas simuladas en el benchmark
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import time

# Permite ejecutar el script desde cualquier directorio (raíz o subcarpetas)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from src import analysis, mpi_run
from src.clean import build_hourly, clean_flights
from src.compute.backend import benchmark, get_backend
from src.config import load_config
from src.ingest.clients import fetch_all
from src.ingest.run_ingest import TABLE_KEYS, merge_history
from src.storage import Storage

# Opciones de visualización legible para pandas en la terminal
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def titulo(texto: str, args) -> None:
    """Imprime un separador visual de sección con opción de pausa interactiva."""
    if args.pausa:
        input("\n[Presiona Enter para continuar...] ")
    print(f"\n{'=' * 70}\n{texto}\n{'=' * 70}")


def muestra(nombre: str, df: pd.DataFrame, n: int = 3) -> None:
    """Muestra una vista previa de las primeras n filas de una tabla y su total de registros."""
    print(f"--- tabla '{nombre}' ---")
    print(df.head(n).to_string(index=False))
    print(f"... ({len(df)} filas en total)")


async def llamada_simulada(latencia: float) -> None:
    """Simula una llamada de red asíncrona no bloqueante."""
    await asyncio.sleep(latencia)


async def en_paralelo(n_llamadas: int, latencia: float) -> None:
    """Lanza n_llamadas concurrentemente con asyncio.gather."""
    # gather lanza todas las tareas a la vez, por lo que la duración total es ~ latencia
    await asyncio.gather(*(llamada_simulada(latencia) for _ in range(n_llamadas)))


def main() -> None:
    """Función principal del recorrido paso a paso."""
    ap = argparse.ArgumentParser(description="Recorrido didáctico paso a paso del proyecto.")
    ap.add_argument("--pausa", action="store_true", help="Pausa la ejecución entre cada paso")
    ap.add_argument("--real", action="store_true", help="Usa configuración y base de datos real (.env)")
    ap.add_argument("--n", type=int, default=100_000, help="Rutas simuladas para el benchmark")
    args = ap.parse_args()

    # ==========================================================================
    # Paso 1: Configuración del sistema
    # ==========================================================================
    titulo("PASO 1: Configuración (.env)", args)
    if args.real:
        env_path = ".env"
    else:
        # Configuración temporal en modo demo: no altera el archivo .env ni data/
        carpeta = tempfile.mkdtemp()
        env_path = os.path.join(carpeta, ".env")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"DEMO_MODE=1\nDB_PATH={os.path.join(carpeta, 'demo.db')}\n")

    cfg = load_config(env_path)
    print("Modo           :", "DEMO (datos simulados)" if cfg.demo_mode else "APIs reales (AirLabs + OpenWeather)")
    print("Aeropuertos    :", ", ".join(cfg.airports))
    print("Cálculo        :", get_backend(cfg.backend).name)
    print("Base de datos  :", cfg.db_path)

    # ==========================================================================
    # Paso 2: Ingesta concurrente y demostración de velocidad de asyncio
    # ==========================================================================
    titulo("PASO 2: Lectura concurrente de las APIs (asyncio) y guardado en SQLite", args)
    storage = Storage(cfg.db_path)
    t0 = time.perf_counter()
    tablas = asyncio.run(fetch_all(cfg))  # Descarga concurrente con asyncio.gather
    print(f"Ingesta terminada en {time.perf_counter() - t0:.2f} s")

    for nombre in ("flights", "weather"):
        muestra(nombre, tablas[nombre])
        claves, fechas = TABLE_KEYS[nombre]
        # Fusión incremental con el historial sin generar duplicados
        storage.save_table(
            nombre,
            merge_history(storage.load_table(nombre), tablas[nombre], claves, fechas),
        )

    # Demostración del beneficio de concurrencia: 20 peticiones de 100 ms
    print("\nSimulación comparativa con 20 consultas de 100 ms cada una:")
    t0 = time.perf_counter()
    for _ in range(20):
        time.sleep(0.1)
    serie = time.perf_counter() - t0

    t0 = time.perf_counter()
    asyncio.run(en_paralelo(20, 0.1))
    paralelo = time.perf_counter() - t0

    print(f"  Secuencial (una tras otra) : {serie:.2f} s")
    print(f"  Concurrente (con asyncio)   : {paralelo:.2f} s  (≈ {serie / paralelo:.0f}x más rápido)")

    # ==========================================================================
    # Paso 3: Limpieza de datos y consolidación horaria
    # ==========================================================================
    titulo("PASO 3: Limpieza con Pandas y tabla por aeropuerto y hora", args)
    crudo = storage.load_table("flights")
    limpio = clean_flights(crudo)  # Validación de tipos, eliminación de anomalías y duplicados
    horaria = build_hourly({"flights": limpio, "weather": storage.load_table("weather")})

    print(f"Vuelos antes de limpiar: {len(crudo)} -> después de limpiar: {len(limpio)}")
    print("\nTabla horaria consolidada (vuelo retrasado = 15 minutos o más):")
    print(
        horaria[["airport", "hour", "flights", "delayed", "delay_rate", "rain_mm", "visibility"]]
        .head(5)
        .round({"delay_rate": 3, "rain_mm": 1, "visibility": 0})
        .to_string(index=False)
    )
    print(f"... ({len(horaria)} filas en total)")

    # ==========================================================================
    # Paso 4: Paralelismo y procesamiento distribuido por aeropuerto
    # ==========================================================================
    titulo("PASO 4: Reparto con MPI y procesamiento por aeropuerto", args)
    aeropuertos = sorted(horaria["airport"].dropna().unique())
    for rango, lote in enumerate(mpi_run.partition(aeropuertos, 4)):
        print(f"  Proceso {rango} recibe aeropuertos: {lote}")
    print()

    # Ejecutar 4 procesos si mpirun está disponible; si no o si falla, usar multiprocessing
    ejecuto_mpi = False
    if shutil.which("mpirun"):
        proc = subprocess.run(
            ["mpirun", "-n", "4", sys.executable, "-m", "src.mpi_run", env_path],
            check=False,
        )
        ejecuto_mpi = proc.returncode == 0

    if not ejecuto_mpi:
        if shutil.which("mpirun"):
            print("(mpirun no pudo ejecutarse en este entorno: fallback automático a multiprocessing)")
        else:
            print("(mpirun no está instalado: se procesa localmente con multiprocessing)")
        mpi_run.run(cfg, storage)

    resultados = storage.load_table("results")
    print("\nResultados calculados y guardados en SQLite (correlaciones climáticas vs retrasos):")
    print(resultados.sort_values("airport").round(3).to_string(index=False))

    # ==========================================================================
    # Paso 5: Benchmark de aceleración de hardware (NumPy vs Numba)
    # ==========================================================================
    titulo(f"PASO 5: Numba contra NumPy ({args.n:,} rutas simuladas)", args)
    tiempos = benchmark(args.n)
    for nombre, seg in tiempos.items():
        aceleracion = tiempos["numpy"] / seg
        print(f"  {nombre:<6} {seg:8.3f} s  ({aceleracion:5.1f}x vs NumPy)")

    # ==========================================================================
    # Paso 6: Análisis y respuestas a preguntas de investigación
    # ==========================================================================
    titulo("PASO 6: Respuestas a las preguntas de análisis", args)
    clima = analysis.weather_delay_table(horaria).set_index("condition")["mean_delay_rate"]
    entorno = analysis.traffic_vs_environment(horaria).set_index("variable")
    por_aeropuerto = analysis.delays_by_airport(limpio)
    por_hora = analysis.activity_by_hour(limpio).set_index("hour_of_day")["flights"]

    print("1. ¿Existe relación entre clima y retrasos? (correlación con la tasa de retraso)")
    print(entorno["corr_with_delay_rate"].round(2).to_string())

    print("\n2. ¿Aumentan los retrasos ante lluvia o baja visibilidad? (tasa media de retraso)")
    print((clima * 100).round(1).astype(str).add(" %").to_string())

    print("\n3. ¿Qué aeropuertos presentan mayores retrasos? (retraso medio en minutos)")
    print(por_aeropuerto[["airport", "flights", "mean_delay_min", "delayed_pct"]].round(1).to_string(index=False))

    print(
        f"\n4. ¿Cómo varía la actividad durante el día? Hora pico: {por_hora.idxmax()}:00 "
        f"({por_hora.max()} vuelos); hora con menor actividad: {por_hora.idxmin()}:00 ({por_hora.min()} vuelos)"
    )

    print("\n5. ¿Qué condiciones meteorológicas coinciden con mayor tráfico? (correlación con vuelos por hora)")
    print(entorno["corr_with_flights"].round(2).sort_values(key=abs, ascending=False).to_string())


if __name__ == "__main__":
    main()
