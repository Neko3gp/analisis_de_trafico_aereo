"""Columnas de las tablas normalizadas (las mismas para datos reales y demo)."""

FLIGHT_COLS = ["flight_iata", "airline_iata", "dep_iata", "arr_iata", "lat", "lng",
               "alt", "speed", "status", "dep_delayed", "dep_time"]
WEATHER_COLS = ["airport", "ts", "temp", "humidity", "pressure", "wind_speed",
                "rain_mm", "visibility"]
AQ_COLS = ["airport", "ts", "pm25", "pm10"]
SENSOR_COLS = ["ts", "temperature", "humidity", "pressure", "pm25", "pm10",
               "light", "wind_speed", "wind_dir"]
