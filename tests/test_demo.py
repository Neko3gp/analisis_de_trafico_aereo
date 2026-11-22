import pandas as pd

from src.ingest.demo import generate_demo

AIRPORTS = ["CUU", "MEX", "GDL"]

FLIGHT_COLS = ["flight_iata", "airline_iata", "dep_iata", "arr_iata", "lat", "lng",
               "alt", "speed", "status", "dep_delayed", "dep_time"]
WEATHER_COLS = ["airport", "ts", "temp", "humidity", "pressure", "wind_speed",
                "rain_mm", "visibility"]
AQ_COLS = ["airport", "ts", "pm25", "pm10"]
SENSOR_COLS = ["ts", "temperature", "humidity", "pressure", "pm25", "pm10",
               "light", "wind_speed", "wind_dir"]


def test_tables_and_columns():
    d = generate_demo(AIRPORTS, hours=24, seed=1)
    assert list(d["flights"].columns) == FLIGHT_COLS
    assert list(d["weather"].columns) == WEATHER_COLS
    assert list(d["air_quality"].columns) == AQ_COLS
    assert list(d["sensors"].columns) == SENSOR_COLS


def test_deterministic_with_same_seed():
    a = generate_demo(AIRPORTS, hours=24, seed=7)
    b = generate_demo(AIRPORTS, hours=24, seed=7)
    pd.testing.assert_frame_equal(a["flights"], b["flights"])


def test_different_seed_differs():
    a = generate_demo(AIRPORTS, hours=24, seed=1)
    b = generate_demo(AIRPORTS, hours=24, seed=2)
    assert not a["flights"].equals(b["flights"])


def test_weather_one_row_per_airport_hour():
    d = generate_demo(AIRPORTS, hours=24, seed=1)
    assert len(d["weather"]) == 24 * len(AIRPORTS)
    assert not d["weather"].duplicated(["airport", "ts"]).any()


def test_rain_raises_delays():
    d = generate_demo(AIRPORTS, hours=72, seed=3)
    f = d["flights"].copy()
    f["hour"] = pd.to_datetime(f["dep_time"]).dt.floor("h")
    w = d["weather"].copy()
    w["hour"] = pd.to_datetime(w["ts"]).dt.floor("h")
    m = f.merge(w, left_on=["dep_iata", "hour"], right_on=["airport", "hour"])
    rainy = m.loc[m["rain_mm"] > 0, "dep_delayed"].mean()
    dry = m.loc[m["rain_mm"] == 0, "dep_delayed"].mean()
    assert rainy > dry


def test_some_missing_values_exist():
    d = generate_demo(AIRPORTS, hours=72, seed=3)
    assert d["flights"]["speed"].isna().any()
