"""Aplicación web interactiva y Dashboard analítico con Streamlit.

Para iniciar la aplicación:
    streamlit run src/app.py

Estructura de la aplicación:
- Barra lateral: Controles de ejecución, selección de aeropuertos, trigger de ingesta
  y ejecución distribuida con MPI.
- Pestaña 1 (Resumen): Métricas clave de rendimiento (KPIs) y retrasos por aeropuerto.
- Pestaña 2 (Vuelos): Rutas más transitadas, volumen por aerolínea, distribución horaria,
  análisis de velocidad/altitud y mapa de posiciones en tiempo real.
- Pestaña 3 (Clima y retrasos): Impacto de lluvia y baja visibilidad en retrasos,
  gráfico de dispersión y matriz de correlaciones.
- Pestaña 4 (Cómputo): Resumen de los resultados distribuidos por proceso y benchmark
  interactivo en vivo comparando NumPy frente a Numba.
"""

import os
import shutil
import subprocess
import sys

import pandas as pd
import streamlit as st

# Permite ejecutar `streamlit run src/app.py` directamente desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis  # noqa: E402
from src.clean import build_hourly, clean_flights  # noqa: E402
from src.compute.backend import benchmark, get_backend  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ingest.run_ingest import run_ingest  # noqa: E402
from src.storage import Storage  # noqa: E402

# Configurar propiedades visuales de la página
st.set_page_config(page_title="Tráfico aéreo y ambiente", layout="wide")
cfg = load_config()


@st.cache_data(ttl=60, show_spinner=False)
def load_data(db_path: str) -> dict:
    """Carga y acondiciona todas las tablas de la base de datos con caché de 60 segundos.

    Si la tabla horaria aún no ha sido generada por el script MPI, la calcula al vuelo
    a partir de los vuelos limpios y el clima para que el dashboard siempre muestre información.

    Args:
        db_path (str): Ruta al archivo SQLite.

    Returns:
        dict: Diccionario con DataFrames {'flights': df, 'weather': df, 'hourly': df, 'results': df}.
    """
    storage = Storage(db_path)
    data = {
        name: storage.load_table(name)
        for name in ("flights", "weather", "hourly", "results")
    }

    # Aplicar validación y limpieza a los vuelos
    if not data["flights"].empty:
        data["flights"] = clean_flights(data["flights"])

    # Fallback: si aún no se procesó con MPI, armar tabla horaria preliminar
    if data["hourly"].empty and not data["flights"].empty:
        data["hourly"] = build_hourly({k: data[k] for k in ("flights", "weather")})

    # Convertir marcas temporales a formato datetime
    for name, col in (("weather", "ts"), ("hourly", "hour")):
        if not data[name].empty:
            data[name][col] = pd.to_datetime(data[name][col], errors="coerce")

    return data


def process_with_mpi(n_procs: int) -> str:
    """Invoca el procesamiento distribuido en un subproceso del sistema.

    Intenta utilizar `mpirun -n <n_procs>`; si el comando `mpirun` no existe o falla,
    ejecuta el módulo como script estándar para activar el fallback con multiprocessing.

    Args:
        n_procs (int): Cantidad de procesos solicitados.

    Returns:
        str: Registro de salida combinada (stdout + stderr) generado por la ejecución.
    """
    if shutil.which("mpirun"):
        cmd = ["mpirun", "-n", str(n_procs), sys.executable, "-m", "src.mpi_run"]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode == 0:
            return (out.stdout + out.stderr).strip()

    cmd = [sys.executable, "-m", "src.mpi_run"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


# ==============================================================================
# Barra Lateral: Controles y Disparadores de Pipeline
# ==============================================================================
st.sidebar.title("Controles")
st.sidebar.caption("Modo: " + ("DEMO (datos simulados)" if cfg.demo_mode else "APIs reales"))
st.sidebar.caption(f"Cálculo numérico: {get_backend(cfg.backend).name}")

# Selector de concurrencia: procesos MPI / núcleos de CPU
n_procs = st.sidebar.slider(
    "Procesos MPI",
    min_value=1,
    max_value=max(os.cpu_count() or 1, 2),
    value=min(4, os.cpu_count() or 1),
)

# Botón 1: Ingesta de datos (asíncrona con asyncio)
if st.sidebar.button("1. Actualizar datos (ingesta)"):
    with st.spinner("Descargando en paralelo..."):
        run_ingest()
    load_data.clear()
    st.sidebar.success("Datos actualizados")

# Botón 2: Procesamiento paralelo por aeropuerto
if st.sidebar.button("2. Procesar con MPI"):
    with st.spinner("Procesando por aeropuerto..."):
        log = process_with_mpi(n_procs)
    load_data.clear()
    st.sidebar.code(log or "listo")

# Cargar datos desde SQLite
data = load_data(cfg.db_path)

# Si la base de datos no contiene registros, guiar al usuario
if data["flights"].empty:
    st.title("Tráfico aéreo y condiciones ambientales")
    st.info("Aún no hay datos. Pulsa «1. Actualizar datos» y luego «2. Procesar con MPI».")
    st.stop()

# Filtro interactivo de aeropuertos en la barra lateral
airports = sorted(data["flights"]["dep_iata"].dropna().unique())
chosen = st.sidebar.multiselect("Aeropuertos", airports, default=airports)

# Filtrar DataFrames según la selección del usuario
flights = data["flights"][data["flights"]["dep_iata"].isin(chosen)]
hourly = (
    data["hourly"][data["hourly"]["airport"].isin(chosen)]
    if not data["hourly"].empty
    else data["hourly"]
)

# ==============================================================================
# Contenido Principal: Pestañas de Visualización y Análisis
# ==============================================================================
st.title("Tráfico aéreo y condiciones ambientales")
tab_res, tab_fl, tab_wx, tab_cpu = st.tabs([
    "Resumen",
    "Vuelos",
    "Clima y retrasos",
    "Cómputo",
])

# ------------------------------------------------------------------------------
# Pestaña 1: Resumen Ejecutivo y Métricas Principales
# ------------------------------------------------------------------------------
with tab_res:
    d = analysis.delays_by_airport(flights)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Vuelos", f"{len(flights):,}")
    c2.metric("Activos", int((flights["status"] == "active").sum()))
    c3.metric("Cancelados", int((flights["status"] == "cancelled").sum()))
    c4.metric("Retraso medio (min)", f"{flights['dep_delayed'].mean():.1f}")
    c5.metric("Retrasados (≥15 min)", f"{(flights['dep_delayed'] >= 15).mean() * 100:.1f}%")

    st.subheader("Retrasos por aeropuerto")
    st.dataframe(d, width="stretch", hide_index=True)
    if not d.empty:
        st.bar_chart(d.set_index("airport")["mean_delay_min"])

# ------------------------------------------------------------------------------
# Pestaña 2: Análisis Detallado de Operaciones Aéreas
# ------------------------------------------------------------------------------
with tab_fl:
    left, right = st.columns(2)
    with left:
        st.subheader("Rutas más frecuentes")
        routes = analysis.top_routes(flights, 10)
        st.bar_chart(routes.set_index("route")["flights"])
    with right:
        st.subheader("Vuelos por aerolínea")
        st.bar_chart(analysis.flights_by_airline(flights).set_index("airline_iata")["flights"])

    st.subheader("Actividad por hora del día")
    st.line_chart(analysis.activity_by_hour(flights).set_index("hour_of_day")["flights"])

    st.subheader("Velocidad y altitud promedio")
    st.dataframe(analysis.speed_alt_summary(flights), width="stretch", hide_index=True)

    # Identificación de velocidades anómalas mediante motor numérico
    speed = flights["speed"].to_numpy(dtype=float)
    st.caption(
        f"Velocidades atípicas (|z|>3, backend {get_backend(cfg.backend).name}): "
        f"{int(get_backend(cfg.backend).outlier_mask(speed).sum())}"
    )

    # Mapa geográfico de aeronaves con posiciones disponibles
    pos = flights.dropna(subset=["lat", "lng"]).rename(columns={"lng": "lon"})
    if not pos.empty:
        st.subheader("Posiciones")
        st.map(pos[["lat", "lon"]])

# ------------------------------------------------------------------------------
# Pestaña 3: Impacto Ambiental y Meteorológico
# ------------------------------------------------------------------------------
with tab_wx:
    st.subheader("Retrasos según el clima")
    table = analysis.weather_delay_table(hourly)
    st.dataframe(table, width="stretch", hide_index=True)
    if not table.empty:
        st.bar_chart(table.set_index("condition")["mean_delay_rate"])
        st.subheader("Lluvia vs tasa de retraso (por aeropuerto y hora)")
        st.scatter_chart(hourly.dropna(subset=["delay_rate"]), x="rain_mm", y="delay_rate")

    st.subheader("Tráfico y clima (correlaciones)")
    st.dataframe(analysis.traffic_vs_environment(hourly), width="stretch", hide_index=True)

# ------------------------------------------------------------------------------
# Pestaña 4: Resultados de Cómputo Distribuido y Benchmark
# ------------------------------------------------------------------------------
with tab_cpu:
    st.subheader("Resultados del procesamiento distribuido")
    if data["results"].empty:
        st.info("Pulsa «2. Procesar con MPI» en la barra lateral.")
    else:
        st.dataframe(data["results"].round(3), width="stretch", hide_index=True)

    st.subheader("Numpy vs Numba")
    n = st.number_input("Rutas a simular", 10_000, 2_000_000, 200_000, step=50_000)
    if st.button("Correr benchmark"):
        with st.spinner("Midiendo..."):
            times = benchmark(int(n))
        bench = pd.DataFrame({"backend": list(times), "segundos": list(times.values())})
        base = times.get("numpy")
        if base:
            bench["aceleración vs numpy"] = base / bench["segundos"]
        st.dataframe(bench, width="stretch", hide_index=True)
        st.bar_chart(bench.set_index("backend")["segundos"])
