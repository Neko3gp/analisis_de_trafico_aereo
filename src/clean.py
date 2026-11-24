"""Módulo de limpieza, validación y agregación de datos con Pandas.

Normaliza los datos crudos de vuelos y meteorología:
- Conversión y casteo estricto de tipos de datos.
- Filtrado de valores físicamente inverosímiles (velocidad y altitud fuera de rango).
- Eliminación de registros incompletos y duplicados.
- Agregación temporal por aeropuerto y hora para correlacionar tráfico con clima.
"""

import numpy as np
import pandas as pd

from src.schema import FLIGHT_COLS

# Estándar de la aviación comercial (FAA / OACI):
# Un vuelo se considera oficialmente con retraso si despega con 15 minutos o más de demora.
DELAY_THRESHOLD_MIN = 15

# Rangos físicos válidos para aeronaves comerciales:
# - Velocidad horizontal: 0 a 1300 km/h (límite subsónico comercial)
# - Altitud: 0 a 20,000 metros (~65,000 pies)
SPEED_RANGE = (0.0, 1300.0)
ALT_RANGE = (0.0, 20000.0)

# Esquema canónico para la tabla agregada por hora y aeropuerto
HOURLY_COLS = [
    "airport",
    "hour",
    "flights",
    "delayed",
    "delay_rate",
    "temp",
    "humidity",
    "wind_speed",
    "rain_mm",
    "visibility",
]

# Variables meteorológicas continuas a promediar en cada intervalo horario
_WEATHER_VARS = ["temp", "humidity", "wind_speed", "rain_mm", "visibility"]


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y valida un DataFrame de vuelos crudo.

    Operaciones de saneamiento aplicadas:
    1. Conversión de fechas ('dep_time') a objetos Timestamp de pandas.
    2. Conversión numérica segura de latitud, longitud, altitud, velocidad y retraso.
    3. Corrección de valores de retraso negativos (los vuelos adelantados se imputan con 0).
    4. Detección de valores anómalos: velocidades o altitudes fuera de rangos razonables
       se reemplazan con NaN para no distorsionar estadísticas posteriores.
    5. Eliminación de filas sin aeropuerto de origen o sin hora de salida.
    6. Deduplicación por combinación única (vuelo, aeropuerto de salida, hora de salida).

    Args:
        df (pd.DataFrame): DataFrame de vuelos crudos.

    Returns:
        pd.DataFrame: DataFrame limpio y ordenado según `FLIGHT_COLS`.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=FLIGHT_COLS)

    out = df.copy()

    # 1. Parsear fecha de salida
    out["dep_time"] = pd.to_datetime(out["dep_time"], errors="coerce")

    # 2. Conversión a numérico (valores inválidos se convierten en NaN)
    for col in ("lat", "lng", "alt", "speed", "dep_delayed"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    # 3. Retraso no negativo (un valor negativo indica salida anticipada)
    out["dep_delayed"] = out["dep_delayed"].fillna(0).clip(lower=0)

    # 4. Sustituir valores fuera de rango físico razonable por NaN
    out.loc[~out["speed"].between(*SPEED_RANGE), "speed"] = np.nan
    out.loc[~out["alt"].between(*ALT_RANGE), "alt"] = np.nan

    # 5. Descartar registros sin aeropuerto de origen o fecha de salida
    out = out.dropna(subset=["dep_iata", "dep_time"])

    # 6. Eliminar duplicados exactos del mismo vuelo en el mismo momento
    out = out.drop_duplicates(["flight_iata", "dep_iata", "dep_time"])

    return out.reset_index(drop=True)[FLIGHT_COLS]


def _hourly_mean(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Calcula el promedio horario de las variables climáticas por aeropuerto.

    Trunca las marcas de tiempo a la hora en punto mediante `dt.floor("h")`
    y calcula el promedio aritmético de las columnas indicadas.

    Args:
        df (pd.DataFrame): DataFrame con observaciones meteorológicas.
        cols (list): Lista de nombres de columnas a promediar.

    Returns:
        pd.DataFrame: DataFrame agrupado por ['airport', 'hour'] con los promedios.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["airport", "hour"] + cols)

    d = df.copy()
    # Redondear marca de tiempo a la hora correspondiente (ej. 14:35 -> 14:00)
    d["hour"] = pd.to_datetime(d["ts"], errors="coerce").dt.floor("h")
    return (
        d.dropna(subset=["hour"])
        .groupby(["airport", "hour"], as_index=False)[cols]
        .mean()
    )


def _hourly_flights(flights: pd.DataFrame) -> pd.DataFrame:
    """Agrega las métricas de vuelos por aeropuerto y por hora.

    Calcula:
    - flights: Número total de vuelos programados/despegados.
    - delayed: Cantidad de vuelos con retraso >= DELAY_THRESHOLD_MIN (15 minutos).

    Args:
        flights (pd.DataFrame): DataFrame limpio de vuelos.

    Returns:
        pd.DataFrame: DataFrame con columnas ['airport', 'hour', 'flights', 'delayed'].
    """
    if flights is None or flights.empty:
        return pd.DataFrame(columns=["airport", "hour", "flights", "delayed"])

    d = flights.copy()
    d["hour"] = pd.to_datetime(d["dep_time"]).dt.floor("h")
    d["is_delayed"] = d["dep_delayed"] >= DELAY_THRESHOLD_MIN

    # Agrupar y resumir conteo total y conteo de retrasados
    g = d.groupby(["dep_iata", "hour"], as_index=False).agg(
        flights=("flight_iata", "size"),
        delayed=("is_delayed", "sum"),
    )
    return g.rename(columns={"dep_iata": "airport"})


def build_hourly(tables: dict) -> pd.DataFrame:
    """Construye la tabla horaria consolidada uniendo vuelos y variables climáticas.

    Realiza una unión externa (outer merge) por aeropuerto y hora para conservar
    horas con información meteorológica aunque no haya habido vuelos (y viceversa).
    Calcula la tasa de retraso ('delay_rate') como la proporción de vuelos demorados.

    Args:
        tables (dict): Diccionario con DataFrames {'flights': df, 'weather': df}.

    Returns:
        pd.DataFrame: DataFrame consolidado con estructura correspondiente a `HOURLY_COLS`.
    """
    f = _hourly_flights(tables.get("flights"))
    w = _hourly_mean(tables.get("weather"), _WEATHER_VARS)

    # Combinación exterior para no perder intervalos horarios de ninguna de las dos fuentes
    out = f.merge(w, on=["airport", "hour"], how="outer")
    if out.empty:
        return pd.DataFrame(columns=HOURLY_COLS)

    # Rellenar valores nulos de vuelos con 0 en horas sin operaciones aéreas
    out["flights"] = out["flights"].fillna(0).astype(int)
    out["delayed"] = out["delayed"].fillna(0).astype(int)

    # Calcular tasa de retraso evitando divisiones por cero (NaN si flights == 0)
    out["delay_rate"] = np.where(
        out["flights"] > 0,
        out["delayed"] / out["flights"].where(out["flights"] > 0),
        np.nan,
    )

    return out.sort_values(["airport", "hour"]).reset_index(drop=True)[HOURLY_COLS]
