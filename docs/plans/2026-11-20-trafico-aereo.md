# Tráfico aéreo y ambiente: plan de implementación

> **Para quien ejecute:** usar superpowers:executing-plans. Los pasos usan casillas `- [ ]`.

**Goal:** Pipeline local en Mac que lee APIs en paralelo, las guarda en caché, procesa con MPI y backends numéricos intercambiables y lo muestra en Streamlit.

**Architecture:** `ingest` (asyncio) → `storage` (SQLite) → `clean` (Pandas) → `mpi_run` (reparto por aeropuerto, `compute` con backend) → `analysis` → `app` (Streamlit). Modo demo genera datos sintéticos con la misma forma que las APIs reales.

**Tech Stack:** Python 3.11+, asyncio/aiohttp, mpi4py (OpenMPI), numpy, numba, mlx (opcional), cupy (solo Colab), pandas, SQLite, Streamlit, pytest.

**Spec:** `docs/specs/2026-11-20-design.md`

## Global Constraints

- Todo corre en local en Mac arm64, sin Docker.
- `BACKEND` en `.env`: `numba` (por defecto), `mlx`, `cupy`, `numpy`. Si el backend pedido no está instalado, cae a `numba` o `numpy` con un aviso, sin romper.
- Sin claves (`DEMO_MODE=1` o claves vacías) corre con datos simulados.
- SQLite en modo WAL; solo rank 0 escribe.
- Aeropuertos por defecto: CUU, MEX, GDL, MTY, TIJ.
- Si MPI4Py u OpenMPI no están, el procesamiento cae a `multiprocessing` con un aviso.
- No hay repositorio git: no hay pasos de commit.

## Review Focus

- Respuesta de API vacía o con campos faltantes (p. ej. vuelo sin `speed`): no truena, queda NaN y se excluye de promedios.
- Error HTTP, 429 o sin red: se usa el último dato en caché y se avisa.
- Más procesos MPI que aeropuertos: los ranks sin trabajo devuelven resultado vacío.
- Backend opcional no instalado: cae al siguiente disponible.
- Datos con un solo punto o varianza cero: la correlación devuelve NaN, no excepción.

---

### Task 1: Configuración y arranque

**Files:** Create `src/config.py`, `.env.example`, `requirements.txt`, `setup.sh`, `run_pipeline.sh`, `tests/test_config.py`

**Interfaces:**
- Produces: `@dataclass Config(airlabs_key, openweather_key, openaq_key, thingspeak_channel, thingspeak_read_key, backend, demo_mode, airports, db_path)`; `load_config(env_path: str = ".env") -> Config`. Sin claves, `demo_mode=True`.

- [ ] Test `test_demo_when_no_keys`, `test_defaults_airports == ["CUU","MEX","GDL","MTY","TIJ"]`, `test_backend_default_numba`
- [ ] Implementar `load_config` leyendo `.env` a mano (sin dependencias)
- [ ] `setup.sh` (venv + `pip install -r requirements.txt`, aviso de `brew install open-mpi`) y `run_pipeline.sh`
- [ ] Verificar: `pytest tests/test_config.py -v` pasa

### Task 2: Almacenamiento SQLite con caché

**Files:** Create `src/storage.py`, `tests/test_storage.py`

**Interfaces:**
- Produces: `class Storage(path)` con `get_cached(endpoint: str, params: dict, max_age_s: int) -> dict|list|None`, `put_cache(endpoint, params, payload) -> None`, `save_table(name: str, df: DataFrame) -> None`, `load_table(name: str) -> DataFrame` (vacío si no existe).

- [ ] Tests: `test_cache_hit`, `test_cache_expired_returns_none`, `test_params_order_irrelevant`, `test_table_roundtrip`
- [ ] Implementar con clave hash de `endpoint + params` ordenados, WAL activado
- [ ] Verificar: pytest pasa

### Task 3: Datos demo

**Files:** Create `src/ingest/demo.py`, `tests/test_demo.py`

**Interfaces:**
- Produces: `generate_demo(airports: list[str], hours: int = 72, seed: int = 0) -> dict[str, DataFrame]` con llaves `flights` (columnas: `flight_iata, airline_iata, dep_iata, arr_iata, lat, lng, alt, speed, status, dep_delayed, dep_time`), `weather` (`airport, ts, temp, humidity, pressure, wind_speed, rain_mm, visibility`), `air_quality` (`airport, ts, pm25, pm10`), `sensors` (`ts, temperature, humidity, pressure, pm25, pm10, light, wind_speed, wind_dir`).

- [ ] Tests: columnas exactas, determinista con misma semilla, hay retrasos mayores cuando `rain_mm > 0` (correlación positiva)
- [ ] Implementar con numpy

### Task 4: Clientes async de las APIs

**Files:** Create `src/ingest/clients.py`, `tests/test_clients.py`

**Interfaces:**
- Consumes: `Config`, `Storage`.
- Produces: `async fetch_json(session, url, params, headers, storage, ttl_s) -> dict|list` (usa caché; ante error devuelve caché vencida si existe); `async fetch_all(config, storage, session=None) -> dict[str, DataFrame]` con las mismas llaves y columnas que `generate_demo`; funciones `normalize_airlabs_flights/schedules`, `normalize_openweather`, `normalize_openaq`, `normalize_thingspeak` (dict → DataFrame).

- [ ] Tests con respuestas simuladas inyectadas: `test_normalize_handles_missing_fields`, `test_fetch_uses_cache`, `test_http_error_falls_back_to_cache`, `test_fetch_all_runs_concurrently` (`asyncio.gather`)
- [ ] Implementar `aiohttp` con import perezoso; en modo demo `fetch_all` devuelve `generate_demo`
- [ ] Endpoints (AirLabs `flights` y `schedules`, OpenWeather `weather` y `air_pollution`, OpenAQ v3, ThingSpeak `feeds.json`) como constantes al inicio del archivo, para corregirlos fácil al probar con claves reales

### Task 5: Limpieza con Pandas

**Files:** Create `src/clean.py`, `tests/test_clean.py`

**Interfaces:**
- Produces: `clean_flights(df) -> DataFrame` (tipos, duplicados fuera, `speed`/`alt` fuera de rango → NaN); `build_hourly(tables: dict[str, DataFrame]) -> DataFrame` con una fila por `airport, hour` y columnas `flights, delayed, delay_rate, temp, humidity, wind_speed, rain_mm, visibility, pm25, pm10`.

- [ ] Tests: duplicados, rangos, `test_build_hourly_one_row_per_airport_hour`, entradas vacías devuelven DataFrame vacío con las columnas
- [ ] Implementar

### Task 6: Backends numéricos

**Files:** Create `src/compute/backend.py`, `src/compute/kernels.py`, `tests/test_backend.py`

**Interfaces:**
- Produces: `get_backend(name: str) -> Backend` (con `.name` real tras el respaldo). Métodos: `mean_std(x) -> (float, float)`, `outlier_mask(x, z: float = 3.0) -> ndarray[bool]`, `pearson(x, y) -> float` (NaN si varianza cero o n<2), `haversine_km(lat1, lon1, lat2, lon2) -> ndarray`, `simulate_routes(lat1, lon1, lat2, lon2, speed_kmh, n_steps) -> ndarray` (tiempo estimado en horas). `benchmark(n: int) -> dict[str, float]` con tiempos por backend disponible.

- [ ] Tests (parametrizados por backend disponible): resultados de `numba`/`mlx` coinciden con `numpy` (`allclose`), `test_unavailable_backend_falls_back`, `test_pearson_zero_variance_nan`
- [ ] Implementar `numpy` siempre; `numba` con `@njit(parallel=True)` y `prange`; `mlx` y `cupy` con import perezoso; todo cálculo no soportado en un backend delega a `numpy`

### Task 7: Reparto con MPI

**Files:** Create `src/mpi_run.py`, `tests/test_mpi_run.py`

**Interfaces:**
- Consumes: `Backend`, `build_hourly`.
- Produces: `partition(items: list, n: int) -> list[list]`; `process_airport(df_airport: DataFrame, backend: Backend) -> dict` (`airport, mean_delay_rate, std, outliers, corr_rain_delay, corr_wind_delay, corr_pm25_flights, n_rows`); `run(config, storage) -> DataFrame` (resultados por aeropuerto, guardados en la tabla `results` solo por rank 0); ejecutable con `mpirun -n 4 python -m src.mpi_run`.

- [ ] Tests: `test_partition_more_ranks_than_items`, `test_process_airport_empty`, `test_run_without_mpi_uses_multiprocessing`
- [ ] Implementar con `mpi4py` perezoso: rank 0 reparte con `scatter`, todos procesan, `gather` en rank 0

### Task 8: Análisis y dashboard

**Files:** Create `src/analysis.py`, `src/app.py`, `tests/test_analysis.py`

**Interfaces:**
- Produces en `analysis.py`: `delays_by_airport(flights)`, `top_routes(flights, n=10)`, `flights_by_airline(flights)`, `activity_by_hour(flights)`, `speed_alt_summary(flights)`, `weather_delay_table(hourly)` (retrasos con y sin lluvia / baja visibilidad), `traffic_vs_environment(hourly)`; todas devuelven DataFrame.
- `app.py`: Streamlit delgado que llama a estas funciones; pestañas por pregunta del PDF; botón para correr el pipeline; muestra el backend usado y la comparación de `benchmark`.

- [ ] Tests con datos demo: tablas esperadas y `test_rain_raises_delay_rate`
- [ ] Implementar y verificar que `app.py` importa sin errores y compila (`python -m py_compile`)

### Task 9: Colab y README

**Files:** Create `notebooks/colab_gpu.ipynb`, `README.md`

- [ ] Notebook: clona o sube `src/`, pone `BACKEND=cupy`, corre `benchmark` y muestra la tabla de tiempos
- [ ] README en español: instalación en Mac (`brew install open-mpi`, `setup.sh`), dónde pegar las claves, cómo correr (`run_pipeline.sh`, `mpirun`, `streamlit run`), qué se probó aquí y qué falta probar con claves reales
