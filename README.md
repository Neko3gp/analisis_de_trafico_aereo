# Tráfico aéreo y condiciones ambientales

Proyecto de Cómputo Paralelo y Distribuido. Pregunta de fondo: **¿el clima se relaciona con los retrasos y la actividad aérea?**

Fuentes de datos: **AirLabs** (vuelos, retrasos, posiciones) y **OpenWeather** (clima de cada aeropuerto).

```
AirLabs + OpenWeather ──asyncio──▶ SQLite ──Pandas──▶ MPI (un grupo de aeropuertos por proceso) ──Numba──▶ Streamlit
```

| Tecnología | Para qué se usa | Archivo |
|---|---|---|
| asyncio (aiohttp) | pedir a las APIs de todos los aeropuertos a la vez | `src/ingest/clients.py` |
| Pandas | limpiar y unir vuelos con clima por aeropuerto y hora | `src/clean.py`, `src/analysis.py` |
| MPI4Py | repartir los aeropuertos entre procesos (`scatter`) y juntar resultados (`gather`) | `src/mpi_run.py` |
| Numba | media, correlación y rutas simuladas compiladas y en paralelo (se compara contra numpy) | `src/compute/` |
| Streamlit | dashboard con las respuestas | `src/app.py` |

## Instalación (Mac)

```bash
brew install open-mpi
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env           # y pega tus claves de AirLabs y OpenWeather
```

**Sin claves el proyecto corre en modo demo** con datos simulados (también con `DEMO_MODE=1` en `.env`).

## Cómo correrlo

```bash
python demo_paso_a_paso.py         # recorre los 6 pasos e imprime lo que produce cada uno
python demo_paso_a_paso.py --pausa # igual, pero espera Enter entre pasos (para las capturas)
```

O cada etapa por separado:

```bash
python -m src.ingest.run_ingest    # descarga y guarda en data/trafico.db (cada corrida suma al historial)
mpirun -n 4 python -m src.mpi_run  # procesa con 4 procesos MPI
streamlit run src/app.py           # dashboard
```

Con claves reales, corre la ingesta varias veces (por ejemplo cada hora durante un día): las APIs dan solo la foto del momento, y las correlaciones necesitan varias horas de datos.

## Preguntas que responde

1. ¿Hay relación entre clima y retrasos? → correlaciones por aeropuerto (MPI) y tabla general
2. ¿Aumentan los retrasos con lluvia o baja visibilidad? → tabla «Retrasos según el clima»
3. ¿Qué aeropuertos tienen más retrasos? → «Retrasos por aeropuerto»
4. ¿Cómo cambia la actividad durante el día? → «Actividad por hora del día»
5. ¿Qué condiciones coinciden con más tráfico? → correlación de cada variable con los vuelos por hora

Un vuelo cuenta como retrasado con 15 minutos o más (`DELAY_THRESHOLD_MIN` en `src/clean.py`).

## Estructura

```
demo_paso_a_paso.py   recorrido completo en 6 pasos
src/config.py         lee .env                     src/clean.py     limpieza y tabla por hora
src/ingest/           asyncio + datos demo         src/analysis.py  tablas del dashboard
src/storage.py        SQLite                       src/mpi_run.py   reparto con MPI
src/compute/          numpy y numba                src/app.py       dashboard Streamlit
tests/                pytest (83 pruebas)
```

## Notas

- Se usa Numba en CPU (todos los núcleos del Mac). No se usa GPU/CUDA.
- Aeropuertos: `AIRPORTS=CUU,MEX,GDL,MTY,TIJ` en `.env`. Las coordenadas para OpenWeather salen de `AIRPORT_COORDS` en `src/ingest/demo.py`.
- Si una API falla o llega al límite del plan gratuito, esa consulta se omite con un aviso y el resto sigue.
- Los endpoints de AirLabs y OpenWeather están como constantes al inicio de `src/ingest/clients.py`.
