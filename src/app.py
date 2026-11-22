"""Dashboard de Streamlit. Se lanza con: streamlit run src/app.py"""
import os
import shutil
import subprocess
import sys

import pandas as pd
import streamlit as st

# permite `streamlit run src/app.py` desde la raíz del proyecto
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis  # noqa: E402
from src.clean import build_hourly, clean_flights  # noqa: E402
from src.compute.backend import available_backends, benchmark, get_backend  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ingest.run_ingest import run_ingest  # noqa: E402
from src.storage import Storage  # noqa: E402

st.set_page_config(page_title="Tráfico aéreo y ambiente", layout="wide")
cfg = load_config()


@st.cache_data(ttl=60, show_spinner=False)
def load_data(db_path: str) -> dict:
    storage = Storage(db_path)
    data = {name: storage.load_table(name)
            for name in ("flights", "weather", "air_quality", "sensors", "hourly", "results")}
    if not data["flights"].empty:
        data["flights"] = clean_flights(data["flights"])
    if data["hourly"].empty and not data["flights"].empty:
        # aún no se ha procesado con MPI: se arma la tabla por hora aquí para no mostrar pestañas vacías
        data["hourly"] = build_hourly({k: data[k] for k in ("flights", "weather", "air_quality")})
    for name, col in (("weather", "ts"), ("air_quality", "ts"), ("sensors", "ts"), ("hourly", "hour")):
        if not data[name].empty:
            data[name][col] = pd.to_datetime(data[name][col], errors="coerce")
    return data


def process_with_mpi(n_procs: int) -> str:
    if shutil.which("mpirun"):
        cmd = ["mpirun", "-n", str(n_procs), sys.executable, "-m", "src.mpi_run"]
    else:
        cmd = [sys.executable, "-m", "src.mpi_run"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    return (out.stdout + out.stderr).strip()


# ------------------------------------------------------------------ barra lateral
st.sidebar.title("Controles")
st.sidebar.caption("Modo: " + ("DEMO (datos simulados)" if cfg.demo_mode else "APIs reales"))
st.sidebar.caption(f"Backend numérico pedido: {cfg.backend}")
st.sidebar.caption("Backends disponibles: " + ", ".join(available_backends()))
n_procs = st.sidebar.slider("Procesos MPI", 1, max(os.cpu_count() or 1, 2), min(4, os.cpu_count() or 1))

if st.sidebar.button("1. Actualizar datos (ingesta)"):
    with st.spinner("Descargando en paralelo..."):
        run_ingest()
    load_data.clear()
    st.sidebar.success("Datos actualizados")
if st.sidebar.button("2. Procesar con MPI"):
    with st.spinner("Procesando por aeropuerto..."):
        log = process_with_mpi(n_procs)
    load_data.clear()
    st.sidebar.code(log or "listo")

data = load_data(cfg.db_path)
if data["flights"].empty:
    st.title("Tráfico aéreo y condiciones ambientales")
    st.info("Aún no hay datos. Pulsa «1. Actualizar datos» y luego «2. Procesar con MPI».")
    st.stop()

airports = sorted(data["flights"]["dep_iata"].dropna().unique())
chosen = st.sidebar.multiselect("Aeropuertos", airports, default=airports)
flights = data["flights"][data["flights"]["dep_iata"].isin(chosen)]
hourly = data["hourly"][data["hourly"]["airport"].isin(chosen)] if not data["hourly"].empty else data["hourly"]

st.title("Tráfico aéreo y condiciones ambientales")
tab_res, tab_fl, tab_wx, tab_air, tab_cpu = st.tabs(
    ["Resumen", "Vuelos", "Clima y retrasos", "Calidad del aire y sensores", "Cómputo"])

with tab_res:
    d = analysis.delays_by_airport(flights)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vuelos", f"{len(flights):,}")
    c2.metric("Retraso medio (min)", f"{flights['dep_delayed'].mean():.1f}")
    c3.metric("Retrasados (≥15 min)", f"{(flights['dep_delayed'] >= 15).mean() * 100:.1f}%")
    c4.metric("Aeropuertos", len(chosen))
    st.subheader("Retrasos por aeropuerto")
    st.dataframe(d, use_container_width=True, hide_index=True)
    if not d.empty:
        st.bar_chart(d.set_index("airport")["mean_delay_min"])

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
    st.dataframe(analysis.speed_alt_summary(flights), use_container_width=True, hide_index=True)
    speed = flights["speed"].to_numpy(dtype=float)
    st.caption(f"Velocidades atípicas (|z|>3, backend {get_backend(cfg.backend).name}): "
               f"{int(get_backend(cfg.backend).outlier_mask(speed).sum())}")
    pos = flights.dropna(subset=["lat", "lng"]).rename(columns={"lng": "lon"})
    if not pos.empty:
        st.subheader("Posiciones")
        st.map(pos[["lat", "lon"]])

with tab_wx:
    st.subheader("Retrasos según el clima")
    table = analysis.weather_delay_table(hourly)
    st.dataframe(table, use_container_width=True, hide_index=True)
    if not table.empty:
        st.bar_chart(table.set_index("condition")["mean_delay_rate"])
        st.subheader("Lluvia vs tasa de retraso (por aeropuerto y hora)")
        st.scatter_chart(hourly.dropna(subset=["delay_rate"]), x="rain_mm", y="delay_rate")

with tab_air:
    st.subheader("Tráfico y condiciones ambientales (correlaciones)")
    st.dataframe(analysis.traffic_vs_environment(hourly), use_container_width=True, hide_index=True)
    aq = data["air_quality"]
    if not aq.empty:
        st.subheader("PM2.5 por aeropuerto")
        st.line_chart(aq[aq["airport"].isin(chosen)].pivot_table(
            index="ts", columns="airport", values="pm25"))
    sensors = data["sensors"]
    if not sensors.empty:
        st.subheader("Sensores (ThingSpeak)")
        st.line_chart(sensors.set_index("ts").dropna(axis=1, how="all"))

with tab_cpu:
    st.subheader("Resultados del procesamiento distribuido")
    if data["results"].empty:
        st.info("Pulsa «2. Procesar con MPI» en la barra lateral.")
    else:
        st.dataframe(data["results"].round(3), use_container_width=True, hide_index=True)
    st.subheader("Comparación de backends")
    n = st.number_input("Rutas a simular", 10_000, 2_000_000, 200_000, step=50_000)
    if st.button("Correr benchmark"):
        with st.spinner("Midiendo..."):
            times = benchmark(int(n))
        bench = pd.DataFrame({"backend": list(times), "segundos": list(times.values())})
        base = times.get("numpy")
        if base:
            bench["aceleración vs numpy"] = base / bench["segundos"]
        st.dataframe(bench, use_container_width=True, hide_index=True)
        st.bar_chart(bench.set_index("backend")["segundos"])
        st.caption("CuPy / CUDA solo aparece en Google Colab con GPU NVIDIA (notebooks/colab_gpu.ipynb).")
