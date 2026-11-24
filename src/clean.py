"""Limpieza y unificación con Pandas."""
import numpy as np
import pandas as pd

from src.schema import FLIGHT_COLS

DELAY_THRESHOLD_MIN = 15   # un vuelo cuenta como retrasado con 15 min o más
SPEED_RANGE = (0.0, 1300.0)    # km/h
ALT_RANGE = (0.0, 20000.0)     # metros

HOURLY_COLS = ["airport", "hour", "flights", "delayed", "delay_rate", "temp", "humidity",
               "wind_speed", "rain_mm", "visibility"]
_WEATHER_VARS = ["temp", "humidity", "wind_speed", "rain_mm", "visibility"]


def clean_flights(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=FLIGHT_COLS)
    out = df.copy()
    out["dep_time"] = pd.to_datetime(out["dep_time"], errors="coerce")
    for col in ("lat", "lng", "alt", "speed", "dep_delayed"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["dep_delayed"] = out["dep_delayed"].fillna(0).clip(lower=0)
    out.loc[~out["speed"].between(*SPEED_RANGE), "speed"] = np.nan
    out.loc[~out["alt"].between(*ALT_RANGE), "alt"] = np.nan
    out = out.dropna(subset=["dep_iata", "dep_time"])
    out = out.drop_duplicates(["flight_iata", "dep_iata", "dep_time"])
    return out.reset_index(drop=True)[FLIGHT_COLS]


def _hourly_mean(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["airport", "hour"] + cols)
    d = df.copy()
    d["hour"] = pd.to_datetime(d["ts"], errors="coerce").dt.floor("h")
    return d.dropna(subset=["hour"]).groupby(["airport", "hour"], as_index=False)[cols].mean()


def _hourly_flights(flights: pd.DataFrame) -> pd.DataFrame:
    if flights is None or flights.empty:
        return pd.DataFrame(columns=["airport", "hour", "flights", "delayed"])
    d = flights.copy()
    d["hour"] = pd.to_datetime(d["dep_time"]).dt.floor("h")
    d["is_delayed"] = d["dep_delayed"] >= DELAY_THRESHOLD_MIN
    g = d.groupby(["dep_iata", "hour"], as_index=False).agg(
        flights=("flight_iata", "size"), delayed=("is_delayed", "sum"))
    return g.rename(columns={"dep_iata": "airport"})


def build_hourly(tables: dict) -> pd.DataFrame:
    """Una fila por aeropuerto y hora, uniendo vuelos y clima."""
    f = _hourly_flights(tables.get("flights"))
    w = _hourly_mean(tables.get("weather"), _WEATHER_VARS)

    out = f.merge(w, on=["airport", "hour"], how="outer")
    if out.empty:
        return pd.DataFrame(columns=HOURLY_COLS)

    out["flights"] = out["flights"].fillna(0).astype(int)
    out["delayed"] = out["delayed"].fillna(0).astype(int)
    out["delay_rate"] = np.where(out["flights"] > 0, out["delayed"] / out["flights"].where(out["flights"] > 0), np.nan)
    return out.sort_values(["airport", "hour"]).reset_index(drop=True)[HOURLY_COLS]
