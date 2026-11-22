"""Tablas de análisis para el dashboard (todas devuelven DataFrames y toleran entradas vacías)."""
import numpy as np
import pandas as pd

from src.clean import DELAY_THRESHOLD_MIN

LOW_VISIBILITY_M = 5000


def _flights(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    d = df.copy()
    if "dep_time" in d:
        d["dep_time"] = pd.to_datetime(d["dep_time"], errors="coerce")
    return d


def delays_by_airport(flights: pd.DataFrame) -> pd.DataFrame:
    cols = ["airport", "flights", "mean_delay_min", "delayed_pct"]
    d = _flights(flights)
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["is_delayed"] = d["dep_delayed"] >= DELAY_THRESHOLD_MIN
    g = d.groupby("dep_iata", as_index=False).agg(
        flights=("flight_iata", "size"), mean_delay_min=("dep_delayed", "mean"),
        delayed_pct=("is_delayed", "mean"))
    g["delayed_pct"] *= 100
    g = g.rename(columns={"dep_iata": "airport"})
    return g.sort_values("mean_delay_min", ascending=False).reset_index(drop=True)[cols]


def top_routes(flights: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    cols = ["dep_iata", "arr_iata", "route", "flights"]
    d = _flights(flights)
    if d.empty:
        return pd.DataFrame(columns=cols)
    g = (d.dropna(subset=["dep_iata", "arr_iata"]).groupby(["dep_iata", "arr_iata"])
         .size().reset_index(name="flights").sort_values("flights", ascending=False).head(n))
    g["route"] = g["dep_iata"] + " -> " + g["arr_iata"]
    return g.reset_index(drop=True)[cols]


def flights_by_airline(flights: pd.DataFrame) -> pd.DataFrame:
    d = _flights(flights)
    if d.empty:
        return pd.DataFrame(columns=["airline_iata", "flights"])
    g = d.groupby("airline_iata").size().reset_index(name="flights")
    return g.sort_values("flights", ascending=False).reset_index(drop=True)


def activity_by_hour(flights: pd.DataFrame) -> pd.DataFrame:
    d = _flights(flights)
    counts = pd.Series(0, index=range(24), dtype=int)
    if not d.empty:
        h = d["dep_time"].dt.hour.dropna().astype(int).value_counts()
        counts = counts.add(h, fill_value=0).astype(int)
    return pd.DataFrame({"hour_of_day": list(range(24)), "flights": counts.to_numpy()})


def speed_alt_summary(flights: pd.DataFrame) -> pd.DataFrame:
    cols = ["airport", "mean_speed", "max_speed", "mean_alt"]
    d = _flights(flights)
    if d.empty:
        return pd.DataFrame(columns=cols)
    g = d.groupby("dep_iata", as_index=False).agg(
        mean_speed=("speed", "mean"), max_speed=("speed", "max"), mean_alt=("alt", "mean"))
    return g.rename(columns={"dep_iata": "airport"})[cols]


def weather_delay_table(hourly: pd.DataFrame) -> pd.DataFrame:
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
    rows = [{"condition": name, "hours": int(mask.sum()),
             "mean_delay_rate": float(h.loc[mask, "delay_rate"].mean()) if mask.any() else np.nan}
            for name, mask in groups.items()]
    return pd.DataFrame(rows, columns=cols)


def traffic_vs_environment(hourly: pd.DataFrame) -> pd.DataFrame:
    cols = ["variable", "corr_with_flights", "corr_with_delay_rate"]
    if hourly is None or hourly.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for var in ["pm25", "pm10", "temp", "humidity", "wind_speed", "rain_mm", "visibility"]:
        if var not in hourly:
            continue
        x = pd.to_numeric(hourly[var], errors="coerce")
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append({"variable": var,
                         "corr_with_flights": x.corr(hourly["flights"].astype(float)),
                         "corr_with_delay_rate": x.corr(hourly["delay_rate"].astype(float))})
    return pd.DataFrame(rows, columns=cols)
