import numpy as np
import pandas as pd

from src import analysis
from src.clean import build_hourly, clean_flights
from src.ingest.demo import generate_demo
from src.schema import FLIGHT_COLS

T = generate_demo(["CUU", "MEX", "GDL"], hours=96, seed=4)
FLIGHTS = clean_flights(T["flights"])
HOURLY = build_hourly({**T, "flights": FLIGHTS})
EMPTY_F = pd.DataFrame(columns=FLIGHT_COLS)


def test_delays_by_airport_sorted_and_complete():
    d = analysis.delays_by_airport(FLIGHTS)
    assert list(d.columns) == ["airport", "flights", "mean_delay_min", "delayed_pct"]
    assert set(d["airport"]) == {"CUU", "MEX", "GDL"}
    assert d["mean_delay_min"].is_monotonic_decreasing
    assert d["delayed_pct"].between(0, 100).all()


def test_top_routes_limited_and_sorted():
    r = analysis.top_routes(FLIGHTS, n=5)
    assert len(r) <= 5 and r["flights"].is_monotonic_decreasing
    assert {"route", "flights"} <= set(r.columns)
    assert "->" in r["route"].iloc[0]


def test_flights_by_airline_counts_add_up():
    a = analysis.flights_by_airline(FLIGHTS)
    assert a["flights"].sum() == len(FLIGHTS)


def test_activity_by_hour_has_all_24_hours():
    a = analysis.activity_by_hour(FLIGHTS)
    assert list(a["hour_of_day"]) == list(range(24))
    assert a["flights"].sum() == len(FLIGHTS)


def test_activity_by_hour_accepts_string_times():
    f = FLIGHTS.copy()
    f["dep_time"] = f["dep_time"].astype(str)
    assert analysis.activity_by_hour(f)["flights"].sum() == len(FLIGHTS)


def test_speed_alt_summary_columns():
    s = analysis.speed_alt_summary(FLIGHTS)
    assert list(s.columns) == ["airport", "mean_speed", "max_speed", "mean_alt"]
    assert (s["mean_speed"] > 500).all()


def test_rain_raises_delay_rate():
    t = analysis.weather_delay_table(HOURLY).set_index("condition")
    assert t.loc["Con lluvia", "mean_delay_rate"] > t.loc["Sin lluvia", "mean_delay_rate"]
    assert t["hours"].sum() >= len(HOURLY.dropna(subset=["delay_rate"])) // 2


def test_low_visibility_row_present():
    t = analysis.weather_delay_table(HOURLY)
    assert {"Visibilidad baja (<5 km)", "Visibilidad normal"} <= set(t["condition"])


def test_traffic_vs_environment_table():
    t = analysis.traffic_vs_environment(HOURLY)
    assert list(t.columns) == ["variable", "corr_with_flights", "corr_with_delay_rate"]
    assert {"temp", "wind_speed", "rain_mm", "visibility"} <= set(t["variable"])


def test_all_functions_survive_empty_input():
    cols = {
        analysis.delays_by_airport: ["airport", "flights", "mean_delay_min", "delayed_pct"],
        analysis.top_routes: ["dep_iata", "arr_iata", "route", "flights"],
        analysis.flights_by_airline: ["airline_iata", "flights"],
        analysis.speed_alt_summary: ["airport", "mean_speed", "max_speed", "mean_alt"],
    }
    for fn, expected in cols.items():
        out = fn(EMPTY_F)
        assert out.empty and list(out.columns) == expected
    assert analysis.activity_by_hour(EMPTY_F)["flights"].sum() == 0
    assert analysis.weather_delay_table(pd.DataFrame()).empty
    assert analysis.traffic_vs_environment(pd.DataFrame()).empty


def test_traffic_vs_environment_constant_column_gives_nan_not_error():
    h = HOURLY.copy()
    h["temp"] = 10.0
    t = analysis.traffic_vs_environment(h).set_index("variable")
    assert np.isnan(t.loc["temp", "corr_with_flights"])
