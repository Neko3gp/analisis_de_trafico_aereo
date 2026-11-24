"""Clientes asíncronos para la ingesta de datos desde APIs externas.

Permite consultar de forma concurrente (usando aiohttp y asyncio):
1. AirLabs API:
   - Horarios y salidas programadas ('schedules').
   - Posiciones y telemetría en vivo ('flights').
2. OpenWeather API:
   - Condiciones meteorológicas actuales por aeropuerto.

Si el sistema corre en modo DEMO, delega automáticamente a `generate_demo`
sin realizar conexiones de red externas.
"""

import asyncio
import sys

import numpy as np
import pandas as pd

from src.ingest.demo import AIRPORT_COORDS, generate_demo
from src.schema import FLIGHT_COLS, WEATHER_COLS

# Endpoints oficiales de las APIs REST externas
AIRLABS_FLIGHTS = "https://airlabs.co/api/v9/flights"
AIRLABS_SCHEDULES = "https://airlabs.co/api/v9/schedules"
OPENWEATHER_CURRENT = "https://api.openweathermap.org/data/2.5/weather"


def _warn(msg: str) -> None:
    """Emite un mensaje de advertencia por stderr con el prefijo del módulo de ingesta.

    Args:
        msg (str): Mensaje descriptivo del error o advertencia.
    """
    print(f"[ingesta] {msg}", file=sys.stderr)


async def fetch_json(session, url: str, params: dict):
    """Realiza una petición HTTP GET asíncrona esperando una respuesta en formato JSON.

    Diseñado para tolerar fallos de red o errores HTTP sin interrumpir la ejecución
    del resto de llamadas concurrentes.

    Args:
        session: Instancia de aiohttp.ClientSession.
        url (str): URL de destino del endpoint.
        params (dict): Parámetros de consulta (query string).

    Returns:
        dict | list | None: Respuesta decodificada en JSON, o None en caso de error.
    """
    try:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
            # content_type=None permite decodificar JSON incluso con cabeceras no estándar
            return await resp.json(content_type=None)
    except Exception as exc:  # noqa: BLE001 - un fallo en una API no debe tumbar a las demás
        _warn(f"{url} falló ({exc}); sin datos")
        return None


# ------------------------------------------------------------------------------
# Funciones normalizadoras de payloads JSON a DataFrames de pandas
# ------------------------------------------------------------------------------

def _rows(payload, key: str) -> list:
    """Extrae de manera segura una lista de registros desde un payload JSON.

    Args:
        payload: Objeto o diccionario devuelto por la API.
        key (str): Clave que contiene la lista de elementos (ej. 'response').

    Returns:
        list: Lista de registros extraídos, o lista vacía si no existe o es inválido.
    """
    if payload is None:
        return []
    if isinstance(payload, dict):
        payload = payload.get(key) or []
    return payload if isinstance(payload, list) else []


def _num(value):
    """Convierte un valor a float de manera segura, retornando np.nan en caso de error.

    Args:
        value: Valor numérico o string representativo.

    Returns:
        float: Número flotante o np.nan.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalize_airlabs_flights(payload) -> pd.DataFrame:
    """Normaliza la respuesta de vuelos en vivo de AirLabs a un DataFrame de telemetría.

    Extrae las coordenadas geográficas, altitud y velocidad en tiempo real.

    Args:
        payload: Respuesta JSON de la API AirLabs (/flights).

    Returns:
        pd.DataFrame: DataFrame con columnas: flight_iata, lat, lng, alt, speed.
    """
    cols = ["flight_iata", "lat", "lng", "alt", "speed"]
    rows = [
        {
            "flight_iata": r.get("flight_iata"),
            "lat": _num(r.get("lat")),
            "lng": _num(r.get("lng")),
            "alt": _num(r.get("alt")),
            "speed": _num(r.get("speed")),
        }
        for r in _rows(payload, "response")
    ]
    return pd.DataFrame(rows, columns=cols)


def normalize_airlabs_schedules(payload, live: pd.DataFrame = None) -> pd.DataFrame:
    """Normaliza la respuesta de itinerarios de AirLabs y la une con la telemetría en vivo.

    Args:
        payload: Respuesta JSON de la API AirLabs (/schedules).
        live (pd.DataFrame, optional): DataFrame con posiciones en vivo proveniente de
                                       `normalize_airlabs_flights`.

    Returns:
        pd.DataFrame: DataFrame normalizado con todas las columnas de `FLIGHT_COLS`.
    """
    rows = [
        {
            "flight_iata": r.get("flight_iata"),
            "airline_iata": r.get("airline_iata"),
            "dep_iata": r.get("dep_iata"),
            "arr_iata": r.get("arr_iata"),
            "status": r.get("status"),
            "dep_delayed": _num(r.get("dep_delayed")) if r.get("dep_delayed") is not None else 0,
            "dep_time": r.get("dep_time"),
        }
        for r in _rows(payload, "response")
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "flight_iata",
            "airline_iata",
            "dep_iata",
            "arr_iata",
            "status",
            "dep_delayed",
            "dep_time",
        ],
    )

    # Si no hay telemetría en vivo, rellenar coordenadas, altitud y velocidad con NaN
    if live is None or live.empty:
        for c in ("lat", "lng", "alt", "speed"):
            df[c] = np.nan
    else:
        # Combinar mediante left join sobre el identificador del vuelo
        live = live.drop_duplicates("flight_iata")
        df = df.merge(live, on="flight_iata", how="left")

    df["dep_time"] = pd.to_datetime(df["dep_time"], errors="coerce")
    return df[FLIGHT_COLS]


def normalize_openweather(airport: str, current: dict) -> pd.DataFrame:
    """Normaliza la respuesta meteorológica actual de OpenWeather.

    Args:
        airport (str): Código IATA del aeropuerto consultado.
        current (dict): Respuesta JSON proveniente de OpenWeather (/weather).

    Returns:
        pd.DataFrame: DataFrame de una fila estructurado según `WEATHER_COLS`.
    """
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


# ------------------------------------------------------------------------------
# Ingesta completa y orquestación asíncrona
# ------------------------------------------------------------------------------

def _empty() -> dict:
    """Crea un diccionario con DataFrames vacíos para vuelos y clima."""
    return {
        "flights": pd.DataFrame(columns=FLIGHT_COLS),
        "weather": pd.DataFrame(columns=WEATHER_COLS),
    }


async def _airport(session, cfg, ap: str):
    """Consulta de forma concurrente todas las APIs necesarias para un aeropuerto individual.

    Lanza en paralelo con `asyncio.gather`:
    1. Itinerarios de vuelos de salida de AirLabs.
    2. Telemetría de vuelos activos de salida de AirLabs.
    3. Condiciones meteorológicas en las coordenadas del aeropuerto.

    Args:
        session: Sesión HTTP asíncrona (aiohttp.ClientSession).
        cfg: Objeto de configuración con credenciales.
        ap (str): Código IATA del aeropuerto.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (DataFrame de vuelos, DataFrame de clima).
    """
    lat, lng = AIRPORT_COORDS.get(ap, (23.0, -102.0))
    key = {"api_key": cfg.airlabs_key}
    ow = {"lat": lat, "lon": lng, "appid": cfg.openweather_key, "units": "metric"}

    sched, live, cur = await asyncio.gather(
        fetch_json(session, AIRLABS_SCHEDULES, {**key, "dep_iata": ap}),
        fetch_json(session, AIRLABS_FLIGHTS, {**key, "dep_iata": ap}),
        fetch_json(session, OPENWEATHER_CURRENT, ow),
    )
    return (
        normalize_airlabs_schedules(sched, normalize_airlabs_flights(live)),
        normalize_openweather(ap, cur),
    )


async def fetch_all(config, session=None) -> dict:
    """Descarga concurrentemente la información de todos los aeropuertos configurados.

    Si `config.demo_mode` está activo, retorna inmediatamente datos simulados
    sin realizar peticiones de red ni requerir aiohttp.

    Args:
        config: Objeto Config con claves y lista de aeropuertos.
        session (optional): Sesión aiohttp reutilizable; si es None, se crea y cierra una.

    Returns:
        dict: Diccionario {'flights': pd.DataFrame, 'weather': pd.DataFrame} con los datos consolidados.
    """
    if config.demo_mode:
        return generate_demo(config.airports)

    own_session = session is None
    if own_session:
        import aiohttp  # Importación perezosa: solo se requiere para llamadas reales con red
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

    try:
        # Consultar todos los aeropuertos en paralelo
        results = await asyncio.gather(*[_airport(session, config, ap) for ap in config.airports])
    finally:
        if own_session:
            await session.close()

    out = _empty()
    # Concatenar resultados de cada aeropuerto
    for key, idx in (("flights", 0), ("weather", 1)):
        frames = [r[idx] for r in results if not r[idx].empty]
        if frames:
            out[key] = pd.concat(frames, ignore_index=True)
    return out
