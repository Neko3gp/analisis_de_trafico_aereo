"""Datos simulados con la misma forma que las respuestas normalizadas de las APIs."""
import numpy as np
import pandas as pd

AIRPORT_COORDS = {
    "CUU": (28.70, -105.96),
    "MEX": (19.44, -99.07),
    "GDL": (20.52, -103.31),
    "MTY": (25.78, -100.11),
    "TIJ": (32.54, -116.97),
    "CUN": (21.04, -86.87),
    "SLP": (22.25, -100.93),
}
AIRLINES = ["AM", "VB", "Y4", "4O", "AA", "UA"]
DEFAULT_START = "2026-11-01 00:00"


def _coords(code: str) -> tuple:
    return AIRPORT_COORDS.get(code, (23.0, -102.0))


def generate_demo(airports: list, hours: int = 72, seed: int = 0,
                  start: str = DEFAULT_START) -> dict:
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    stamps = [t0 + pd.Timedelta(hours=h) for h in range(hours)]

    weather_rows, aq_rows, flight_rows = [], [], []
    for ap in airports:
        base_lat, _ = _coords(ap)
        for ts in stamps:
            hour = ts.hour
            rain = float(rng.exponential(1.5)) if rng.random() < 0.25 else 0.0
            visibility = float(np.clip(10000 - rain * 900 + rng.normal(0, 500), 500, 10000))
            temp = 20 + 8 * np.sin((hour - 9) / 24 * 2 * np.pi) + (base_lat - 23) * -0.3 + rng.normal(0, 1)
            wind = float(abs(rng.normal(4, 2)))
            weather_rows.append({
                "airport": ap, "ts": ts, "temp": round(float(temp), 1),
                "humidity": float(np.clip(50 + rain * 8 + rng.normal(0, 8), 10, 100)),
                "pressure": float(1013 + rng.normal(0, 4)),
                "wind_speed": round(wind, 1), "rain_mm": round(rain, 2),
                "visibility": round(visibility, 0),
            })
            traffic = 1.0 + 0.8 * np.sin((hour - 6) / 24 * 2 * np.pi)
            pm25 = float(max(2, 18 + 6 * traffic + rng.normal(0, 4) - rain * 2))
            aq_rows.append({"airport": ap, "ts": ts, "pm25": round(pm25, 1),
                            "pm10": round(pm25 * 1.6 + float(rng.normal(0, 3)), 1)})

            n = int(rng.poisson(3 + 4 * max(traffic, 0)))
            for i in range(n):
                dest = str(rng.choice([a for a in AIRPORT_COORDS if a != ap]))
                dlat, dlng = _coords(dest)
                slat, slng = _coords(ap)
                frac = float(rng.random())
                delay_mean = 6 + 18 * (rain > 0) + 10 * (visibility < 5000)
                delay = int(max(0, rng.exponential(delay_mean) - 4))
                status = str(rng.choice(["landed", "active", "scheduled", "cancelled"],
                                        p=[0.55, 0.25, 0.14, 0.06]))
                speed = float(rng.normal(800, 60)) if status == "active" else float(rng.normal(780, 70))
                alt = float(rng.normal(10000, 1500))
                if rng.random() < 0.04:
                    speed = np.nan
                if rng.random() < 0.03:
                    alt = np.nan
                flight_rows.append({
                    "flight_iata": f"{rng.choice(AIRLINES)}{int(rng.integers(100, 999))}",
                    "airline_iata": None,
                    "dep_iata": ap, "arr_iata": dest,
                    "lat": slat + (dlat - slat) * frac,
                    "lng": slng + (dlng - slng) * frac,
                    "alt": alt, "speed": speed, "status": status,
                    "dep_delayed": delay,
                    "dep_time": ts + pd.Timedelta(minutes=int(rng.integers(0, 60))),
                })

    flights = pd.DataFrame(flight_rows)
    flights["airline_iata"] = flights["flight_iata"].str[:2]
    flights = flights[["flight_iata", "airline_iata", "dep_iata", "arr_iata", "lat", "lng",
                       "alt", "speed", "status", "dep_delayed", "dep_time"]]

    sensor_rows = []
    for ts in stamps:
        h = ts.hour
        sensor_rows.append({
            "ts": ts,
            "temperature": round(18 + 8 * np.sin((h - 9) / 24 * 2 * np.pi) + float(rng.normal(0, 1)), 1),
            "humidity": float(np.clip(55 + float(rng.normal(0, 8)), 10, 100)),
            "pressure": float(1013 + rng.normal(0, 3)),
            "pm25": round(float(max(2, 16 + float(rng.normal(0, 5)))), 1),
            "pm10": round(float(max(4, 28 + float(rng.normal(0, 8)))), 1),
            "light": float(max(0, 800 * np.sin((h - 6) / 12 * np.pi)) if 6 <= h <= 18 else 0.0),
            "wind_speed": round(float(abs(rng.normal(3, 1.5))), 1),
            "wind_dir": float(rng.uniform(0, 360)),
        })

    return {
        "flights": flights,
        "weather": pd.DataFrame(weather_rows),
        "air_quality": pd.DataFrame(aq_rows),
        "sensors": pd.DataFrame(sensor_rows),
    }
