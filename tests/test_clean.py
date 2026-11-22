import numpy as np
import pandas as pd

from src.clean import DELAY_THRESHOLD_MIN, build_hourly, clean_flights
from src.ingest.demo import generate_demo
from src.schema import FLIGHT_COLS

HOURLY_COLS = ["airport", "hour", "flights", "delayed", "delay_rate", "temp", "humidity",
               "wind_speed", "rain_mm", "visibility", "pm25", "pm10"]


def _flights(rows):
    return pd.DataFrame(rows, columns=FLIGHT_COLS)


def _row(**kw):
    base = dict(flight_iata="AM1", airline_iata="AM", dep_iata="CUU", arr_iata="MEX",
                lat=25.0, lng=-103.0, alt=9000.0, speed=800.0, status="active",
                dep_delayed=0, dep_time="2026-11-20 10:15")
    base.update(kw)
    return base


def test_clean_removes_duplicates():
    df = clean_flights(_flights([_row(), _row()]))
    assert len(df) == 1


def test_clean_out_of_range_becomes_nan():
    df = clean_flights(_flights([_row(speed=5000.0, alt=-50.0)]))
    assert np.isnan(df.loc[0, "speed"]) and np.isnan(df.loc[0, "alt"])


def test_clean_types_and_negative_delay():
    df = clean_flights(_flights([_row(dep_delayed=-5, dep_time="2026-11-20 10:15")]))
    assert df.loc[0, "dep_delayed"] == 0
    assert pd.api.types.is_datetime64_any_dtype(df["dep_time"])


def test_clean_drops_rows_without_airport_or_time():
    df = clean_flights(_flights([_row(dep_iata=None), _row(flight_iata="X", dep_time=None)]))
    assert df.empty


def test_clean_empty_keeps_columns():
    df = clean_flights(pd.DataFrame(columns=FLIGHT_COLS))
    assert df.empty and list(df.columns) == FLIGHT_COLS


def test_build_hourly_one_row_per_airport_hour():
    t = generate_demo(["CUU", "MEX"], hours=24, seed=1)
    t["flights"] = clean_flights(t["flights"])
    h = build_hourly(t)
    assert list(h.columns) == HOURLY_COLS
    assert not h.duplicated(["airport", "hour"]).any()
    assert set(h["airport"]) == {"CUU", "MEX"}


def test_build_hourly_delay_rate_uses_threshold():
    flights = clean_flights(_flights([
        _row(flight_iata="A", dep_delayed=DELAY_THRESHOLD_MIN),
        _row(flight_iata="B", dep_delayed=DELAY_THRESHOLD_MIN - 1),
    ]))
    h = build_hourly({"flights": flights})
    assert h.loc[0, "flights"] == 2 and h.loc[0, "delayed"] == 1
    assert h.loc[0, "delay_rate"] == 0.5


def test_build_hourly_hour_without_flights_has_nan_rate():
    weather = pd.DataFrame([{"airport": "CUU", "ts": pd.Timestamp("2026-11-20 03:00"),
                             "temp": 15.0, "humidity": 40.0, "pressure": 1010.0,
                             "wind_speed": 2.0, "rain_mm": 0.0, "visibility": 9000.0}])
    h = build_hourly({"weather": weather})
    assert h.loc[0, "flights"] == 0 and np.isnan(h.loc[0, "delay_rate"])


def test_build_hourly_empty_inputs():
    h = build_hourly({})
    assert h.empty and list(h.columns) == HOURLY_COLS
    h2 = build_hourly({"flights": pd.DataFrame(columns=FLIGHT_COLS)})
    assert h2.empty and list(h2.columns) == HOURLY_COLS
