"""Configuración del proyecto: lee un archivo .env sin dependencias externas."""
import os
from dataclasses import dataclass, field

DEFAULT_AIRPORTS = ["CUU", "MEX", "GDL", "MTY", "TIJ"]


@dataclass
class Config:
    airlabs_key: str = ""
    openweather_key: str = ""
    openaq_key: str = ""
    thingspeak_channel: str = ""
    thingspeak_read_key: str = ""
    backend: str = "numba"
    demo_mode: bool = True
    airports: list = field(default_factory=lambda: list(DEFAULT_AIRPORTS))
    db_path: str = "data/trafico.db"


def _read_env_file(path: str) -> dict:
    values = {}
    if not os.path.exists(path):
        return values
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_config(env_path: str = ".env") -> Config:
    """Carga la configuración. Sin claves de AirLabs y OpenWeather corre en modo demo."""
    env = _read_env_file(env_path)

    def get(name: str, default: str = "") -> str:
        return env.get(name, default)

    airlabs = get("AIRLABS_API_KEY")
    openweather = get("OPENWEATHER_API_KEY")
    has_keys = bool(airlabs and openweather)

    demo_raw = get("DEMO_MODE", "").lower()
    if demo_raw in ("1", "true", "yes"):
        demo = True
    elif demo_raw in ("0", "false", "no"):
        demo = not has_keys  # sin claves no hay otra opción
    else:
        demo = not has_keys

    airports_raw = get("AIRPORTS")
    airports = (
        [a.strip().upper() for a in airports_raw.split(",") if a.strip()]
        if airports_raw
        else list(DEFAULT_AIRPORTS)
    )

    return Config(
        airlabs_key=airlabs,
        openweather_key=openweather,
        openaq_key=get("OPENAQ_API_KEY"),
        thingspeak_channel=get("THINGSPEAK_CHANNEL"),
        thingspeak_read_key=get("THINGSPEAK_READ_KEY"),
        backend=get("BACKEND", "numba").lower(),
        demo_mode=demo,
        airports=airports,
        db_path=get("DB_PATH", "data/trafico.db"),
    )
