"""Clientes asíncronos de AirLabs (vuelos) y OpenWeather (clima).

Los endpoints están como constantes para corregirlos fácil al probar con claves reales.
"""
import asyncio
import sys

import numpy as np
import pandas as pd

from src.ingest.demo import AIRPORT_COORDS, generate_demo
from src.schema import FLIGHT_COLS, WEATHER_COLS

AIRLABS_FLIGHTS = "https://airlabs.co/api/v9/flights"
AIRLABS_SCHEDULES = "https://airlabs.co/api/v9/schedules"
OPENWEATHER_CURRENT = "https://api.openweathermap.org/data/2.5/weather"


def _warn(msg: str) -> None:
    print(f"[ingesta] {msg}", file=sys.stderr)


async def fetch_json(session, url, params):
    """GET asíncrono. Ante cualquier error (HTTP, red, límite de la API) avisa y devuelve None."""
    try:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            return await resp.json(content_type=None)
    except Exception as exc:  # noqa: BLE001 - un fallo de una API no debe tumbar a las demás
        _warn(f"{url} falló ({exc}); sin datos")
        return None


# ---------------------------------------------------------------- normalizadores
def _rows(payload, key):
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = payload.get(key) or []
    return payload if isinstance(payload, list) else []


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalize_airlabs_flights(payload) -> pd.DataFrame:
    """Posiciones en vivo: flight_iata, lat, lng, alt, speed."""
    cols = ["flight_iata", "lat", "lng", "alt", "speed"]
    rows = [{"flight_iata": r.get("flight_iata"), "lat": _num(r.get("lat")),
             "lng": _num(r.get("lng")), "alt": _num(r.get("alt")),
             "speed": _num(r.get("speed"))} for r in _rows(payload, "response")]
    return pd.DataFrame(rows, columns=cols)


def normalize_airlabs_schedules(payload, live: pd.DataFrame = None) -> pd.DataFrame:
    rows = [{
        "flight_iata": r.get("flight_iata"), "airline_iata": r.get("airline_iata"),
        "dep_iata": r.get("dep_iata"), "arr_iata": r.get("arr_iata"),
        "status": r.get("status"),
        "dep_delayed": _num(r.get("dep_delayed")) if r.get("dep_delayed") is not None else 0,
        "dep_time": r.get("dep_time"),
    } for r in _rows(payload, "response")]
    df = pd.DataFrame(rows, columns=["flight_iata", "airline_iata", "dep_iata", "arr_iata",
                                     "status", "dep_delayed", "dep_time"])
    if live is None or live.empty:
        for c in ("lat", "lng", "alt", "speed"):
            df[c] = np.nan
    else:
        live = live.drop_duplicates("flight_iata")
        df = df.merge(live, on="flight_iata", how="left")
    df["dep_time"] = pd.to_datetime(df["dep_time"], errors="coerce")
    return df[FLIGHT_COLS]


def normalize_openweather(airport: str, current) -> pd.DataFrame:
    if not current:
        return pd.DataFrame(columns=WEATHER_COLS)
    main = current.get("main") or {}
    row = {
        "airport": airport,
        "ts": pd.to_datetime(current.get("dt"), unit="s", errors="coerce"),
        "temp": _num(main.get("temp")),
        "humidity": _num(main.get("humidity")),
        "pressure": _num(main.get("pressure")),
        "wind_speed": _num((current.get("wind") or {}).get("speed")),
        "rain_mm": _num((current.get("rain") or {}).get("1h", 0.0)),
        "visibility": _num(current.get("visibility")),
    }
    return pd.DataFrame([row], columns=WEATHER_COLS)


# ---------------------------------------------------------------- ingesta completa
def _empty() -> dict:
    return {"flights": pd.DataFrame(columns=FLIGHT_COLS),
            "weather": pd.DataFrame(columns=WEATHER_COLS)}


async def _airport(session, cfg, ap: str):
    lat, lng = AIRPORT_COORDS.get(ap, (23.0, -102.0))
    key = {"api_key": cfg.airlabs_key}
    ow = {"lat": lat, "lon": lng, "appid": cfg.openweather_key, "units": "metric"}
    sched, live, cur = await asyncio.gather(
        fetch_json(session, AIRLABS_SCHEDULES, {**key, "dep_iata": ap}),
        fetch_json(session, AIRLABS_FLIGHTS, {**key, "dep_iata": ap}),
        fetch_json(session, OPENWEATHER_CURRENT, ow),
    )
    return (normalize_airlabs_schedules(sched, normalize_airlabs_flights(live)),
            normalize_openweather(ap, cur))


async def fetch_all(config, session=None) -> dict:
    """Descarga todo en paralelo (asyncio.gather). Demo: datos simulados, sin red."""
    if config.demo_mode:
        return generate_demo(config.airports)

    own_session = session is None
    if own_session:
        import aiohttp  # import perezoso: solo se necesita con claves reales
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    try:
        results = await asyncio.gather(*[_airport(session, config, ap) for ap in config.airports])
    finally:
        if own_session:
            await session.close()

    out = _empty()
    for key, idx in (("flights", 0), ("weather", 1)):
        frames = [r[idx] for r in results if not r[idx].empty]
        if frames:
            out[key] = pd.concat(frames, ignore_index=True)
    return out
