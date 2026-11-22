#!/usr/bin/env bash
# Ingesta -> procesamiento con MPI -> dashboard Streamlit.
# Uso: ./run_pipeline.sh [num_procesos]   (por defecto 4)
set -e
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

N=${1:-4}
mkdir -p data

echo "== Ingesta (asyncio) =="
python -m src.ingest.run_ingest

echo "== Procesamiento con $N procesos =="
if command -v mpirun >/dev/null 2>&1; then
  mpirun -n "$N" python -m src.mpi_run
else
  echo "mpirun no encontrado: se usa multiprocessing"
  python -m src.mpi_run
fi

echo "== Dashboard =="
streamlit run src/app.py
