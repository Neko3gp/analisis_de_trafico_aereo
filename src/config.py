"""Módulo de configuración del proyecto.

Lee y gestiona las variables de entorno desde un archivo .env sin requerir
dependencias externas (como python-dotenv). Provee valores por defecto y
determina si el sistema debe operar en modo demo (simulación sin red) o en modo
real consultando las APIs de AirLabs y OpenWeather.
"""

import os
from dataclasses import dataclass, field

# Aeropuertos principales de México monitoreados por defecto:
# CUU: Chihuahua, MEX: Ciudad de México, GDL: Guadalajara, MTY: Monterrey, TIJ: Tijuana
DEFAULT_AIRPORTS = ["CUU", "MEX", "GDL", "MTY", "TIJ"]


@dataclass
class Config:
    """Contenedor inmutable de la configuración de ejecución de la aplicación.

    Attributes:
        airlabs_key (str): Clave de API para el servicio AirLabs (vuelos en vivo).
        openweather_key (str): Clave de API para el servicio OpenWeather (meteorología).
        backend (str): Motor de cálculo numérico ('numba' o 'numpy').
        demo_mode (bool): Indica si se deben usar datos simulados (True) o APIs reales (False).
        airports (list): Lista de códigos IATA de los aeropuertos a monitorear.
        db_path (str): Ruta al archivo de base de datos SQLite.
    """

    airlabs_key: str = ""
    openweather_key: str = ""
    backend: str = "numba"
    demo_mode: bool = True
    airports: list = field(default_factory=lambda: list(DEFAULT_AIRPORTS))
    db_path: str = "data/trafico.db"


def _read_env_file(path: str) -> dict:
    """Parsea manualmente un archivo de variables de entorno (.env).

    Ignora líneas vacías, comentarios que inicien con '#' y elimina comillas
    simples o dobles que rodeen a los valores.

    Args:
        path (str): Ruta al archivo .env.

    Returns:
        dict: Diccionario clave-valor con las variables encontradas.
    """
    values = {}
    if not os.path.exists(path):
        return values

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Ignorar líneas vacías, comentarios o líneas sin el separador '='
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            # Limpiar espacios y comillas en los extremos del valor
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_config(env_path: str = ".env") -> Config:
    """Carga y valida la configuración del sistema desde el archivo .env especificado.

    Reglas de negocio aplicadas:
    - Si no se proporcionan claves válidas para AirLabs y OpenWeather, se fuerza
      el modo DEMO automáticamente para garantizar que la aplicación funcione offline.
    - La variable DEMO_MODE permite activar o desactivar la simulación explícitamente,
      pero si no hay claves de API, nunca podrá ser False.
    - La lista de aeropuertos se normaliza en mayúsculas y sin espacios en blanco.

    Args:
        env_path (str): Ruta al archivo .env (por defecto: '.env').

    Returns:
        Config: Instancia de Config lista para ser utilizada por el pipeline.
    """
    env = _read_env_file(env_path)

    def get(name: str, default: str = "") -> str:
        """Obtiene un valor del archivo .env o retorna el valor por defecto."""
        return env.get(name, default)

    airlabs = get("AIRLABS_API_KEY")
    openweather = get("OPENWEATHER_API_KEY")
    has_keys = bool(airlabs and openweather)

    # Determinar si se activa el modo DEMO
    demo_raw = get("DEMO_MODE", "").lower()
    if demo_raw in ("1", "true", "yes"):
        demo = True
    elif demo_raw in ("0", "false", "no"):
        # Si el usuario solicitó modo real pero faltan claves, se fuerza demo
        demo = not has_keys
    else:
        # Por omisión: modo demo si no hay claves disponibles
        demo = not has_keys

    # Procesar la lista de aeropuertos separados por coma (ej. "CUU,MEX,GDL")
    airports_raw = get("AIRPORTS")
    airports = (
        [a.strip().upper() for a in airports_raw.split(",") if a.strip()]
        if airports_raw
        else list(DEFAULT_AIRPORTS)
    )

    return Config(
        airlabs_key=airlabs,
        openweather_key=openweather,
        backend=get("BACKEND", "numba").lower(),
        demo_mode=demo,
        airports=airports,
        db_path=get("DB_PATH", "data/trafico.db"),
    )
