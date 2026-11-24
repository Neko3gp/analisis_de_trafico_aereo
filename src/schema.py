"""Esquema y definición de columnas para las tablas del sistema.

Define la estructura canónica de los datos para garantizar coherencia
tanto al consultar las APIs externas reales (AirLabs y OpenWeather) como al
utilizar el generador de datos simulados (modo demo).
"""

# ==============================================================================
# Columnas normalizadas para la tabla de vuelos ('flights')
# ==============================================================================
# - flight_iata:   Código IATA del vuelo (ej. 'AM123', 'VB456')
# - airline_iata:  Código IATA de la aerolínea (ej. 'AM', 'VB')
# - dep_iata:      Código IATA del aeropuerto de salida (ej. 'CUU', 'MEX')
# - arr_iata:      Código IATA del aeropuerto de llegada (ej. 'GDL', 'MTY')
# - lat:           Latitud actual de la aeronave en grados decimales
# - lng:           Longitud actual de la aeronave en grados decimales
# - alt:           Altitud actual de la aeronave en metros
# - speed:         Velocidad horizontal de la aeronave en km/h
# - status:        Estado del vuelo ('scheduled', 'active', 'landed', 'cancelled')
# - dep_delayed:   Minutos de retraso en la salida (0 si salió a tiempo o adelantado)
# - dep_time:      Fecha y hora programada/real de salida (formato ISO / datetime)
FLIGHT_COLS = [
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
]

# ==============================================================================
# Columnas normalizadas para la tabla meteorológica ('weather')
# ==============================================================================
# - airport:       Código IATA del aeropuerto asociado a la medición
# - ts:            Marca de tiempo (timestamp UTC / datetime) de la observación
# - temp:          Temperatura ambiente en grados Celsius (°C)
# - humidity:      Porcentaje de humedad relativa (0 - 100%)
# - pressure:      Presión atmosférica a nivel del mar en hectopascales (hPa)
# - wind_speed:    Velocidad del viento en metros por segundo (m/s)
# - rain_mm:       Precipitación acumulada en la última hora en milímetros (mm)
# - visibility:    Visibilidad horizontal en metros (máximo típico: 10,000 m)
WEATHER_COLS = [
    "airport",
    "ts",
    "temp",
    "humidity",
    "pressure",
    "wind_speed",
    "rain_mm",
    "visibility",
]
