#!/usr/bin/env bash
# ==============================================================================
# Script de automatización del pipeline completo:
# 1. Ingesta asíncrona de datos (AirLabs + OpenWeather o Demo).
# 2. Procesamiento paralelo por aeropuerto con MPI (o Multiprocessing).
# 3. Lanzamiento interactivo del dashboard web en Streamlit.
#
# Uso:
#   ./run_pipeline.sh [num_procesos]   (por defecto se usan 4 procesos)
# ==============================================================================

# Detener la ejecución si ocurre cualquier error no controlado
set -e

# Asegurar que el directorio de trabajo sea la raíz del proyecto
cd "$(dirname "$0")"

# Activar el entorno virtual si existe
[ -d .venv ] && source .venv/bin/activate

# Número de procesos para MPI (argumento 1, o 4 por omisión)
N=${1:-4}

# Asegurar la creación de la carpeta de datos
mkdir -p data

# ------------------------------------------------------------------------------
# Paso 1: Ingesta concurrente con asyncio
# ------------------------------------------------------------------------------
echo "== Ingesta (asyncio) =="
python -m src.ingest.run_ingest

# ------------------------------------------------------------------------------
# Paso 2: Procesamiento distribuido por aeropuerto
# ------------------------------------------------------------------------------
echo "== Procesamiento con $N procesos =="
if command -v mpirun >/dev/null 2>&1; then
  # Ejecución nativa con OpenMPI / MPICH
  mpirun -n "$N" python -m src.mpi_run
else
  # Fallback automático con el módulo multiprocessing de Python
  echo "mpirun no encontrado: se usa multiprocessing"
  python -m src.mpi_run
fi

# ------------------------------------------------------------------------------
# Paso 3: Dashboard interactivo con Streamlit
# ------------------------------------------------------------------------------
echo "== Dashboard =="
streamlit run src/app.py
