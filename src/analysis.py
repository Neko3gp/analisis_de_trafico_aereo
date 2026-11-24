"""Módulo de análisis estadístico y agregaciones para el dashboard y reportes.

Provee funciones para examinar métricas clave de tráfico aéreo y condiciones climáticas:
- Retrasos por aeropuerto y aerolínea.
- Rutas más transitadas y patrones de tráfico aéreo por hora del día.
- Resúmenes de velocidades y altitudes operativas.
- Impacto de condiciones climáticas adversas (lluvia, baja visibilidad) sobre la puntualidad.
- Correlaciones bivariadas entre variables ambientales y flujo de vuelos / retrasos.

Todas las funciones están diseñadas para tolerar DataFrames vacíos o valores nulos.
"""

import numpy as np
import pandas as pd

from src.clean import DELAY_THRESHOLD_MIN

# Umbral de baja visibilidad meteorológica en metros (5,000 metros / 5 km)
LOW_VISIBILITY_M = 5000


def _flights(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara una copia segura del DataFrame de vuelos asegurando tipo datetime en 'dep_time'.

    Args:
        df (pd.DataFrame): DataFrame de vuelos de entrada.

    Returns:
        pd.DataFrame: Copia del DataFrame con formato de fecha validado, o DataFrame vacío.
    """
    if df is None:
        return pd.DataFrame()
    d = df.copy()
    if "dep_time" in d:
        d["dep_time"] = pd.to_datetime(d["dep_time"], errors="coerce")
    return d


def delays_by_airport(flights: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticas agregadas de retrasos agrupadas por aeropuerto de salida.

    Métricas calculadas por aeropuerto:
    - flights: Número total de vuelos despegados/programados.
    - mean_delay_min: Promedio de minutos de retraso.
    - delayed_pct: Porcentaje de vuelos que sufrieron retraso significativo (>= 15 min).

    Args:
        flights (pd.DataFrame): DataFrame de vuelos.

    Returns:
        pd.DataFrame: Tabla con columnas ['airport', 'flights', 'mean_delay_min', 'delayed_pct']
                      ordenada descendentemente por el retraso promedio.
    """
    cols = ["airport", "flights", "mean_delay_min", "delayed_pct"]
    d = _flights(flights)
    if d.empty:
        return pd.DataFrame(columns=cols)

    d["is_delayed"] = d["dep_delayed"] >= DELAY_THRESHOLD_MIN
    g = d.groupby("dep_iata", as_index=False).agg(
        flights=("flight_iata", "size"),
        mean_delay_min=("dep_delayed", "mean"),
        delayed_pct=("is_delayed", "mean"),
    )
    g["delayed_pct"] *= 100  # Convertir a porcentaje (0 a 100%)
    g = g.rename(columns={"dep_iata": "airport"})
    return g.sort_values("mean_delay_min", ascending=False).reset_index(drop=True)[cols]


def top_routes(flights: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Identifica las N rutas más frecuentes (pares origen -> destino).

    Args:
        flights (pd.DataFrame): DataFrame de vuelos.
        n (int): Cantidad máxima de rutas a retornar (por defecto: 10).

    Returns:
        pd.DataFrame: Rutas principales con columnas ['dep_iata', 'arr_iata', 'route', 'flights'].
    """
    cols = ["dep_iata", "arr_iata", "route", "flights"]
    d = _flights(flights)
    if d.empty:
        return pd.DataFrame(columns=cols)

    g = (
        d.dropna(subset=["dep_iata", "arr_iata"])
        .groupby(["dep_iata", "arr_iata"])
        .size()
        .reset_index(name="flights")
        .sort_values("flights", ascending=False)
        .head(n)
    )
    g["route"] = g["dep_iata"] + " -> " + g["arr_iata"]
    return g.reset_index(drop=True)[cols]


def flights_by_airline(flights: pd.DataFrame) -> pd.DataFrame:
    """Contabiliza el total de vuelos operados por cada aerolínea.

    Args:
        flights (pd.DataFrame): DataFrame de vuelos.

    Returns:
        pd.DataFrame: DataFrame con columnas ['airline_iata', 'flights'] ordenado descendentemente.
    """
    d = _flights(flights)
    if d.empty:
        return pd.DataFrame(columns=["airline_iata", "flights"])

    g = d.groupby("airline_iata").size().reset_index(name="flights")
    return g.sort_values("flights", ascending=False).reset_index(drop=True)


def activity_by_hour(flights: pd.DataFrame) -> pd.DataFrame:
    """Calcula la distribución horaria del tráfico aéreo durante las 24 horas del día.

    Garantiza que el resultado siempre contenga las 24 horas (0 a 23),
    asignando 0 vuelos a las horas sin operaciones registradas.

    Args:
        flights (pd.DataFrame): DataFrame de vuelos.

    Returns:
        pd.DataFrame: DataFrame con columnas ['hour_of_day', 'flights'] (24 filas).
    """
    d = _flights(flights)
    counts = pd.Series(0, index=range(24), dtype=int)
    if not d.empty:
        h = d["dep_time"].dt.hour.dropna().astype(int).value_counts()
        counts = counts.add(h, fill_value=0).astype(int)

    return pd.DataFrame({"hour_of_day": list(range(24)), "flights": counts.to_numpy()})


def speed_alt_summary(flights: pd.DataFrame) -> pd.DataFrame:
    """Calcula resúmenes de velocidad y altitud de vuelo por aeropuerto de salida.

    Args:
        flights (pd.DataFrame): DataFrame de vuelos.

    Returns:
        pd.DataFrame: DataFrame con columnas ['airport', 'mean_speed', 'max_speed', 'mean_alt'].
    """
    cols = ["airport", "mean_speed", "max_speed", "mean_alt"]
    d = _flights(flights)
    if d.empty:
        return pd.DataFrame(columns=cols)

    g = d.groupby("dep_iata", as_index=False).agg(
        mean_speed=("speed", "mean"),
        max_speed=("speed", "max"),
        mean_alt=("alt", "mean"),
    )
    return g.rename(columns={"dep_iata": "airport"})[cols]


def weather_delay_table(hourly: pd.DataFrame) -> pd.DataFrame:
    """Segmenta y compara la tasa promedio de retraso bajo diferentes condiciones meteorológicas.

    Categorías evaluadas:
    - Sin lluvia: precipitaciones iguales a 0 mm.
    - Con lluvia: precipitaciones estrictamente mayores a 0 mm.
    - Visibilidad baja: visibilidad menor a 5,000 metros.
    - Visibilidad normal: visibilidad igual o mayor a 5,000 metros.

    Args:
        hourly (pd.DataFrame): DataFrame con agregaciones horarias de tráfico y clima.

    Returns:
        pd.DataFrame: Resumen con columnas ['condition', 'hours', 'mean_delay_rate'].
    """
    cols = ["condition", "hours", "mean_delay_rate"]
    if hourly is None or hourly.empty:
        return pd.DataFrame(columns=cols)

    h = hourly.dropna(subset=["delay_rate"])
    groups = {
        "Sin lluvia": h["rain_mm"] == 0,
        "Con lluvia": h["rain_mm"] > 0,
        "Visibilidad baja (<5 km)": h["visibility"] < LOW_VISIBILITY_M,
        "Visibilidad normal": h["visibility"] >= LOW_VISIBILITY_M,
    }
    rows = [
        {
            "condition": name,
            "hours": int(mask.sum()),
            "mean_delay_rate": float(h.loc[mask, "delay_rate"].mean()) if mask.any() else np.nan,
        }
        for name, mask in groups.items()
    ]
    return pd.DataFrame(rows, columns=cols)


def traffic_vs_environment(hourly: pd.DataFrame) -> pd.DataFrame:
    """Calcula las correlaciones de Pearson entre variables ambientales y métricas de tráfico.

    Evalúa dos relaciones para cada variable climática (temp, humidity, wind_speed, rain_mm, visibility):
    - Correlación con el volumen de vuelos ('corr_with_flights').
    - Correlación con la proporción de retrasos ('corr_with_delay_rate').

    Args:
        hourly (pd.DataFrame): DataFrame con datos agrupados por aeropuerto y hora.

    Returns:
        pd.DataFrame: Tabla con columnas ['variable', 'corr_with_flights', 'corr_with_delay_rate'].
    """
    cols = ["variable", "corr_with_flights", "corr_with_delay_rate"]
    if hourly is None or hourly.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for var in ["temp", "humidity", "wind_speed", "rain_mm", "visibility"]:
        if var not in hourly:
            continue
        x = pd.to_numeric(hourly[var], errors="coerce")
        # Ignorar advertencias de división por cero o NaNs en series constantes
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append({
                "variable": var,
                "corr_with_flights": x.corr(hourly["flights"].astype(float)),
                "corr_with_delay_rate": x.corr(hourly["delay_rate"].astype(float)),
            })
    return pd.DataFrame(rows, columns=cols)
