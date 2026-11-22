#!/usr/bin/env bash
# Prepara el entorno local en Mac (Apple Silicon).
set -e
cd "$(dirname "$0")"

if ! command -v mpirun >/dev/null 2>&1; then
  echo "AVISO: no se encontró OpenMPI. Instálalo con: brew install open-mpi"
  echo "       (sin MPI el procesamiento usa multiprocessing)"
fi

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "Se creó .env: pega ahí tus claves de AirLabs, OpenWeather y OpenAQ."
fi

mkdir -p data
echo "Listo. Activa el entorno con: source .venv/bin/activate"
