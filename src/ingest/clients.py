"""Clientes asíncronos de AirLabs, OpenWeather, OpenAQ y ThingSpeak, con caché en SQLite.

Los endpoints están como constantes para corregirlos fácil al probar con claves reales.
"""
import asyncio
import re
import sys

import numpy as np
import pandas as pd

from src.ingest.demo import AIRPORT_COORDS, generate_demo
from src.schema import AQ_COLS, FLIGHT_COLS, SENSOR_COLS, WEATHER_COLS

AIRLABS_FLIGHTS = "https://airlabs.co/api/v9/flights"
AIRLABS_SCHEDULES = "https://airlabs.co/api/v9/schedules"
OPENWEATHER_CURRENT = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHER_AIR = "https://api.openweathermap.org/data/2.5/air_pollution"
OPENAQ_LATEST = "https://api.openaq.org/v3/locations"  # sin verificar con clave real
THINGSPEAK_FEEDS = "https://api.thingspeak.com/channels/{channel}/feeds.json"

TTL_FLIGHTS = 300       # 5 min: los vuelos cambian rápido
TTL_WEATHER = 900       # 15 min
TTL_SENSORS = 300


def _warn(msg: str) -> None:
    print(f"[ingesta] {msg}", file=sys.stderr)


async def fetch_json(session, url, params, headers, storage, ttl_s):
    """GET con caché. Ante error HTTP o de red devuelve la caché vencida; si no hay, None."""
    cached = storage.get_cached(url, params, max_age_s=ttl_s)
    if cached is not None:
        return cached
    try:
        async with session.get(url, params=params, headers=headers) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            payload = await resp.json(content_type=None)
        storage.put_cache(url, params, payload)
        return payload
    except Exception as exc:  # noqa: BLE001 - cualquier fallo cae a la caché
        stale = storage.get_cached(url, params, max_age_s=None)
        _warn(f"{url} falló ({exc}); " + ("usando caché anterior" if stale is not None else "sin datos"))
        return stale


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


def normalize_openweather(airport: str, current, air=None) -> pd.DataFrame:
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


def normalize_openweather_air(airport: str, payload) -> pd.DataFrame:
    items = _rows(payload, "list")
    rows = [{"airport": airport,
             "ts": pd.to_datetime(i.get("dt"), unit="s", errors="coerce"),
             "pm25": _num((i.get("components") or {}).get("pm2_5")),
             "pm10": _num((i.get("components") or {}).get("pm10"))} for i in items]
    return pd.DataFrame(rows, columns=AQ_COLS)


def normalize_openaq(airport: str, payload) -> pd.DataFrame:
    """Une las mediciones pm25/pm10 de la respuesta en una fila por marca de tiempo."""
    values = {}
    for r in _rows(payload, "results"):
        name = ((r.get("parameter") or {}).get("name") or "").lower().replace(".", "")
        ts = ((r.get("datetime") or {}).get("utc"))
        if name in ("pm25", "pm10") and ts:
            values.setdefault(ts, {})[name] = _num(r.get("value"))
    rows = [{"airport": airport, "ts": pd.to_datetime(ts, errors="coerce", utc=True).tz_localize(None),
             "pm25": v.get("pm25", np.nan), "pm10": v.get("pm10", np.nan)}
            for ts, v in values.items()]
    return pd.DataFrame(rows, columns=AQ_COLS)


_SENSOR_ALIASES = {
    "temperature": "temperature", "temp": "temperature", "temperatura": "temperature",
    "humidity": "humidity", "humedad": "humidity",
    "pressure": "pressure", "presion": "pressure", "presión": "pressure",
    "pm25": "pm25", "pm10": "pm10",
    "light": "light", "luz": "light",
    "windspeed": "wind_speed", "wind_speed": "wind_speed", "viento": "wind_speed",
    "winddir": "wind_dir", "wind_dir": "wind_dir",
}


def _sensor_column(name: str):
    key = re.sub(r"[\s.\-]", "", (name or "").lower())
    return _SENSOR_ALIASES.get(key) or _SENSOR_ALIASES.get(key.replace("_", ""))


def normalize_thingspeak(payload) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame(columns=SENSOR_COLS)
    channel = payload.get("channel") or {}
    mapping = {f: _sensor_column(n) for f, n in channel.items()
               if f.startswith("field") and _sensor_column(n)}
    rows = []
    for feed in payload.get("feeds") or []:
        row = {c: np.nan for c in SENSOR_COLS}
        row["ts"] = pd.to_datetime(feed.get("created_at"), errors="coerce", utc=True)
        row["ts"] = row["ts"].tz_localize(None) if pd.notna(row["ts"]) else row["ts"]
        for field, col in mapping.items():
            row[col] = _num(feed.get(field))
        rows.append(row)
    return pd.DataFrame(rows, columns=SENSOR_COLS)


# ---------------------------------------------------------------- ingesta completa
def _empty() -> dict:
    return {"flights": pd.DataFrame(columns=FLIGHT_COLS),
            "weather": pd.DataFrame(columns=WEATHER_COLS),
            "air_quality": pd.DataFrame(columns=AQ_COLS),
            "sensors": pd.DataFrame(columns=SENSOR_COLS)}


async def _airport(session, cfg, storage, ap: str):
    lat, lng = AIRPORT_COORDS.get(ap, (23.0, -102.0))
    key = {"api_key": cfg.airlabs_key}
    ow = {"lat": lat, "lon": lng, "appid": cfg.openweather_key, "units": "metric"}
    tasks = [
        fetch_json(session, AIRLABS_SCHEDULES, {**key, "dep_iata": ap}, {}, storage, TTL_FLIGHTS),
        fetch_json(session, AIRLABS_FLIGHTS, {**key, "dep_iata": ap}, {}, storage, TTL_FLIGHTS),
        fetch_json(session, OPENWEATHER_CURRENT, ow, {}, storage, TTL_WEATHER),
        fetch_json(session, OPENWEATHER_AIR, ow, {}, storage, TTL_WEATHER),
    ]
    if cfg.openaq_key:
        tasks.append(fetch_json(
            session, OPENAQ_LATEST,
            {"coordinates": f"{lat},{lng}", "radius": 25000, "limit": 5},
            {"X-API-Key": cfg.openaq_key}, storage, TTL_WEATHER))
    sched, live, cur, air, *aq = await asyncio.gather(*tasks)

    air_df = normalize_openweather_air(ap, air)
    if aq:
        aq_df = normalize_openaq(ap, aq[0])
        if not aq_df.empty:
            air_df = pd.concat([air_df, aq_df], ignore_index=True)
    return (normalize_airlabs_schedules(sched, normalize_airlabs_flights(live)),
            normalize_openweather(ap, cur), air_df)


async def fetch_all(config, storage, session=None) -> dict:
    """Descarga todo en paralelo (asyncio.gather). Demo: datos simulados, sin red."""
    if config.demo_mode:
        return generate_demo(config.airports)

    own_session = session is None
    if own_session:
        import aiohttp  # import perezoso: solo se necesita con claves reales
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    try:
        jobs = [_airport(session, config, storage, ap) for ap in config.airports]
        if config.thingspeak_channel:
            jobs.append(fetch_json(
                session, THINGSPEAK_FEEDS.format(channel=config.thingspeak_channel),
                {"results": 200, **({"api_key": config.thingspeak_read_key}
                                    if config.thingspeak_read_key else {})},
                {}, storage, TTL_SENSORS))
        results = await asyncio.gather(*jobs)
    finally:
        if own_session:
            await session.close()

    out = _empty()
    airport_results = results[:len(config.airports)]
    for key, idx in (("flights", 0), ("weather", 1), ("air_quality", 2)):
        frames = [r[idx] for r in airport_results if not r[idx].empty]
        if frames:
            out[key] = pd.concat(frames, ignore_index=True)
    if config.thingspeak_channel:
        out["sensors"] = normalize_thingspeak(results[-1])
    return out
