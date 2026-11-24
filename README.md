# Tráfico aéreo y condiciones ambientales

Proyecto de Cómputo paralelo y distribuido. Pregunta de fondo: **¿el clima se relaciona con los retrasos y la actividad aérea?**

```
APIs (asyncio) → SQLite (caché + historial) → Pandas → MPI (por aeropuerto) → backend numérico → Streamlit
AirLabs (vuelos) · OpenWeather (clima)          numba | mlx | cupy | numpy
```

## Instalación en Mac

```bash
brew install open-mpi          # necesario para mpi4py
./setup.sh                     # crea .venv, instala dependencias y copia .env.example a .env
```

Opcional, GPU del chip Apple: `source .venv/bin/activate && pip install mlx` y `BACKEND=mlx` en `.env`.

## Claves de las APIs

Pega tus claves en el archivo `.env` (el que crea `setup.sh`):

| Variable | Dónde se consigue |
|---|---|
| `AIRLABS_API_KEY` | https://airlabs.co/ |
| `OPENWEATHER_API_KEY` | https://openweathermap.org/api |

**Sin claves el proyecto corre en modo demo** con datos simulados (también con `DEMO_MODE=1`). El dashboard indica en la barra lateral en qué modo está.

## Cómo correrlo

```bash
./run_pipeline.sh 4            # ingesta → procesamiento con 4 procesos MPI → dashboard
```

O por pasos (con el entorno activado, `source .venv/bin/activate`):

```bash
python -m src.ingest.run_ingest              # descarga en paralelo y guarda en data/trafico.db
mpirun -n 4 python -m src.mpi_run            # reparte aeropuertos entre procesos
streamlit run src/app.py                     # dashboard (puerto 8501)
```

El dashboard también tiene los botones «Actualizar datos» y «Procesar con MPI». Sin `mpirun` el procesamiento usa `multiprocessing` y avisa.

## Backends numéricos (`BACKEND` en `.env`)

| Backend | Dónde | Qué usa |
|---|---|---|
| `numba` (por defecto) | Mac | CPU con `prange`, todos los núcleos del M1 |
| `mlx` | Mac (opcional) | GPU del chip Apple (float32) |
| `cupy` | Google Colab | GPU NVIDIA (`notebooks/colab_gpu.ipynb`, incluye un kernel Numba CUDA) |
| `numpy` | cualquiera | respaldo |

Si el backend pedido no está instalado se usa el siguiente disponible y se avisa. La primera llamada con `numba` tarda unos segundos en compilar. En la pestaña **Cómputo** del dashboard hay un benchmark que compara los backends.

## Estructura

```
src/config.py        lectura de .env            src/clean.py       limpieza y tabla por hora
src/ingest/          clientes async + demo      src/analysis.py    tablas del dashboard
src/storage.py       SQLite (caché, historial)  src/mpi_run.py     reparto con mpi4py
src/compute/         backends y kernels         src/app.py         dashboard Streamlit
tests/               pytest (90 pruebas)        docs/              diseño y plan
```

## Cosas importantes

- **Historial:** las APIs reales dan solo la foto del momento y cada ingesta se *suma* a la base sin duplicar. Para que las correlaciones clima-retrasos tengan sentido, corre la ingesta varias veces (por ejemplo cada hora durante un día). En modo demo ya hay 72 horas de datos.
- **Límites de los planes gratuitos:** cada ingesta hace 2 consultas a AirLabs y 1 a OpenWeather por aeropuerto. Las respuestas se guardan en caché (5 min vuelos, 15 min clima), y si una API falla o responde 429 se usa la última respuesta guardada.
- **Retraso:** un vuelo cuenta como retrasado con 15 minutos o más (`DELAY_THRESHOLD_MIN` en `src/clean.py`).
- **Aeropuertos:** `AIRPORTS=CUU,MEX,GDL,MTY,TIJ` en `.env`. Las coordenadas de OpenWeather salen de `AIRPORT_COORDS` en `src/ingest/demo.py` (agrega ahí los que uses fuera de la lista).

## Qué se probó y qué falta probar

Probado con `pytest` (90 pruebas, datos simulados y respuestas de API falsas): configuración, caché, normalización de respuestas, concurrencia, limpieza, análisis, kernels, reparto MPI (con un comunicador simulado) y respaldos.

**No se pudo probar en el entorno donde se escribió el código (sin red ni instalaciones), así que revísalo en tu Mac:**

1. Las llamadas reales a AirLabs y OpenWeather. Los endpoints están como constantes al inicio de `src/ingest/clients.py` por si hay que ajustarlos.
2. `mpirun` real, Numba, MLX y CuPy/CUDA (la lógica de los kernels sí se probó con numpy y como Python puro).
3. El dashboard de Streamlit (solo se verificó que compila).

Si algo falla, copia el mensaje de error completo y se corrige.
# analisis_de_trafico_aereo
