"""Módulo de generación de datos sintéticos (modo demo).

Genera series temporales simuladas de vuelos y condiciones meteorológicas con
la misma estructura y formato normalizado que las APIs reales (AirLabs y OpenWeather).
Permite desarrollar, evaluar y demostrar el proyecto de forma reproducible y offline,
modelando patrones realistas:
- Ciclos diurnos de temperatura y tráfico aéreo.
- Lluvia con distribución exponencial y su impacto en la visibilidad y retrasos.
- Distribuciones estadísticas realistas (Poisson para conteo de vuelos, Normal para velocidades).
- Casos límite y valores nulos para validar los filtros de limpieza.
"""

import numpy as np
import pandas as pd

# Coordenadas geográficas de referencia (latitud, longitud) para aeropuertos de México
AIRPORT_COORDS = {
    "CUU": (28.70, -105.96),  # Chihuahua (General Roberto Fierro Villalobos)
    "MEX": (19.44, -99.07),   # Ciudad de México (Benito Juárez)
    "GDL": (20.52, -103.31),  # Guadalajara (Miguel Hidalgo y Costilla)
    "MTY": (25.78, -100.11),  # Monterrey (General Mariano Escobedo)
    "TIJ": (32.54, -116.97),  # Tijuana (General Abelardo L. Rodríguez)
    "CUN": (21.04, -86.87),   # Cancún
    "SLP": (22.25, -100.93),  # San Luis Potosí
}

# Prefijos de aerolíneas habituales en el espacio aéreo mexicano
# AM: Aeroméxico, VB: VivaAerobus, Y4: Volaris, 4O: Interjet, AA: American, UA: United
AIRLINES = ["AM", "VB", "Y4", "4O", "AA", "UA"]

# Marca de tiempo de inicio predeterminada para la serie histórica simulada
DEFAULT_START = "2026-11-01 00:00"


def _coords(code: str) -> tuple:
    """Obtiene las coordenadas (lat, lng) de un aeropuerto o un punto central por defecto.

    Args:
        code (str): Código IATA del aeropuerto.

    Returns:
        tuple: Tupla (latitud, longitud).
    """
    return AIRPORT_COORDS.get(code, (23.0, -102.0))


def generate_demo(
    airports: list,
    hours: int = 72,
    seed: int = 0,
    start: str = DEFAULT_START,
) -> dict:
    """Genera datos sintéticos coherentes de vuelos y clima para un periodo de horas.

    Modela las siguientes dinámicas físicas y operativas:
    1. Meteorología:
       - Ciclo diurno térmico (curva senoidal con pico alrededor de las 15:00 hrs).
       - Probabilidad de lluvia del 25% (distribución exponencial).
       - Degradación de la visibilidad en función de la lluvia y ruido gaussiano.
       - Variación de humedad y viento correlacionadas con la precipitación.
    2. Vuelos y tráfico:
       - Conteo de vuelos por hora con distribución de Poisson y patrón diurno.
       - Duración de retrasos exponencialmente mayor ante lluvia o baja visibilidad.
       - Rutas aleatorias entre los aeropuertos configurados con interpolación lineal de posiciones.
       - Inyección de valores atípicos y nulos (~3-4%) para validar los módulos de limpieza.

    Args:
        airports (list): Lista de códigos IATA de aeropuertos a simular.
        hours (int): Número de horas a generar en la serie temporal (por defecto: 72 horas).
        seed (int): Semilla para el generador de números aleatorios (reproducibilidad).
        start (str): Fecha/hora inicial en formato ISO o compatible con pandas Timestamp.

    Returns:
        dict: Diccionario con dos DataFrames:
            - 'flights': Vuelos simulados con esquema idéntico a FLIGHT_COLS.
            - 'weather': Observaciones meteorológicas con esquema idéntico a WEATHER_COLS.
    """
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    # Generar marcas de tiempo horarias consecutivas
    stamps = [t0 + pd.Timedelta(hours=h) for h in range(hours)]

    weather_rows, flight_rows = [], []
    for ap in airports:
        base_lat, _ = _coords(ap)
        for ts in stamps:
            hour = ts.hour

            # Simular variables climáticas:
            # - Lluvia: 25% de probabilidad de tener lluvia > 0 mm
            rain = float(rng.exponential(1.5)) if rng.random() < 0.25 else 0.0
            # - Visibilidad: disminuye fuertemente con la lluvia (acotada entre 500 y 10000 m)
            visibility = float(
                np.clip(10000 - rain * 900 + rng.normal(0, 500), 500, 10000)
            )
            # - Temperatura: ciclo diurno senoidal ajustado por latitud
            temp = (
                20
                + 8 * np.sin((hour - 9) / 24 * 2 * np.pi)
                + (base_lat - 23) * -0.3
                + rng.normal(0, 1)
            )
            # - Viento y humedad
            wind = float(abs(rng.normal(4, 2)))
            weather_rows.append({
                "airport": ap,
                "ts": ts,
                "temp": round(float(temp), 1),
                "humidity": float(
                    np.clip(50 + rain * 8 + rng.normal(0, 8), 10, 100)
                ),
                "pressure": float(1013 + rng.normal(0, 4)),
                "wind_speed": round(wind, 1),
                "rain_mm": round(rain, 2),
                "visibility": round(visibility, 0),
            })

            # Simular volumen de tráfico con patrón diurno pico a mediodía
            traffic = 1.0 + 0.8 * np.sin((hour - 6) / 24 * 2 * np.pi)
            n_flights = int(rng.poisson(3 + 4 * max(traffic, 0)))

            for _ in range(n_flights):
                # Seleccionar destino aleatorio diferente al aeropuerto de origen
                dest = str(rng.choice([a for a in AIRPORT_COORDS if a != ap]))
                dlat, dlng = _coords(dest)
                slat, slng = _coords(ap)
                # Fracción recorrida en la trayectoria (posición en vuelo)
                frac = float(rng.random())

                # El retraso promedio aumenta sustancialmente si hay lluvia o baja visibilidad
                delay_mean = 6 + 18 * (rain > 0) + 10 * (visibility < 5000)
                delay = int(max(0, rng.exponential(delay_mean) - 4))

                # Distribución de estados del vuelo
                status = str(
                    rng.choice(
                        ["landed", "active", "scheduled", "cancelled"],
                        p=[0.55, 0.25, 0.14, 0.06],
                    )
                )

                # Velocidad y altitud de crucero
                speed = (
                    float(rng.normal(800, 60))
                    if status == "active"
                    else float(rng.normal(780, 70))
                )
                alt = float(rng.normal(10000, 1500))

                # Inyectar valores atípicos/nulos intencionalmente para evaluar la limpieza
                if rng.random() < 0.04:
                    speed = np.nan
                if rng.random() < 0.03:
                    alt = np.nan

                flight_rows.append({
                    "flight_iata": f"{rng.choice(AIRLINES)}{int(rng.integers(100, 999))}",
                    "airline_iata": None,  # Se derivará del prefijo del código IATA
                    "dep_iata": ap,
                    "arr_iata": dest,
                    "lat": slat + (dlat - slat) * frac,
                    "lng": slng + (dlng - slng) * frac,
                    "alt": alt,
                    "speed": speed,
                    "status": status,
                    "dep_delayed": delay,
                    "dep_time": ts + pd.Timedelta(minutes=int(rng.integers(0, 60))),
                })

    flights = pd.DataFrame(flight_rows)
    # Extraer el código IATA de la aerolínea a partir de los 2 primeros caracteres del vuelo
    flights["airline_iata"] = flights["flight_iata"].str[:2]
    # Reordenar las columnas al formato canónico
    flights = flights[[
        "flight_iata",
        "airline_iata",
        "dep_iata",
        "arr_iata",
        "lat",
        "lng",
        "alt",
        "speed",
        "status",
        "dep_delayed",
        "dep_time",
    ]]

    return {
        "flights": flights,
        "weather": pd.DataFrame(weather_rows),
    }
