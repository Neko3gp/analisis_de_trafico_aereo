import asyncio

import pandas as pd

from src.config import Config
from src.ingest import clients
from src.ingest.clients import (
    fetch_all, fetch_json, normalize_airlabs_flights, normalize_airlabs_schedules,
    normalize_openweather,
)
from src.schema import FLIGHT_COLS, WEATHER_COLS
from src.storage import Storage


class FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    async def json(self, content_type=None):
        return self._payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class FakeSession:
    """Imita aiohttp.ClientSession.get; responde según una función de la URL."""

    def __init__(self, handler, delay=0.0):
        self.handler = handler
        self.delay = delay
        self.calls = 0
        self.in_flight = 0
        self.max_in_flight = 0

    def get(self, url, params=None, headers=None):
        session = self

        class Ctx:
            async def __aenter__(self_inner):
                session.calls += 1
                session.in_flight += 1
                session.max_in_flight = max(session.max_in_flight, session.in_flight)
                await asyncio.sleep(session.delay)
                status, payload = session.handler(url, params or {})
                return FakeResponse(status, payload)

            async def __aexit__(self_inner, *a):
                session.in_flight -= 1
                return False

        return Ctx()


def run(coro):
    return asyncio.run(coro)


def test_normalize_airlabs_handles_missing_fields():
    payload = {"response": [{"flight_iata": "AM100", "dep_iata": "CUU"}]}
    df = normalize_airlabs_flights(payload)
    assert list(df.columns) == ["flight_iata", "lat", "lng", "alt", "speed"]
    assert pd.isna(df.loc[0, "speed"])


def test_normalize_airlabs_schedules_merges_live_and_matches_schema():
    sched = {"response": [
        {"flight_iata": "AM100", "airline_iata": "AM", "dep_iata": "CUU", "arr_iata": "MEX",
         "status": "active", "dep_delayed": 12, "dep_time": "2026-11-20 09:30"},
        {"flight_iata": "VB200", "dep_iata": "CUU", "arr_iata": "GDL"},
    ]}
    live = {"response": [{"flight_iata": "AM100", "lat": 25.0, "lng": -103.0,
                          "alt": 9000, "speed": 800}]}
    df = normalize_airlabs_schedules(sched, normalize_airlabs_flights(live))
    assert list(df.columns) == FLIGHT_COLS
    assert df.loc[df.flight_iata == "AM100", "speed"].iloc[0] == 800
    assert pd.isna(df.loc[df.flight_iata == "VB200", "speed"].iloc[0])
    assert df.loc[df.flight_iata == "VB200", "dep_delayed"].iloc[0] == 0


def test_normalize_none_returns_empty_with_columns():
    assert list(normalize_airlabs_schedules(None).columns) == FLIGHT_COLS
    assert list(normalize_openweather("CUU", None).columns) == WEATHER_COLS


def test_normalize_openweather_without_rain_or_visibility():
    cur = {"dt": 1763600000, "main": {"temp": 21.5, "humidity": 40, "pressure": 1015},
           "wind": {"speed": 3.2}}
    df = normalize_openweather("CUU", cur)
    assert list(df.columns) == WEATHER_COLS
    assert df.loc[0, "rain_mm"] == 0.0
    assert pd.isna(df.loc[0, "visibility"])


def test_normalize_openweather_with_rain_and_visibility():
    cur = {"dt": 1763600000, "main": {"temp": 21.5, "humidity": 40, "pressure": 1015},
           "wind": {"speed": 3.2}, "rain": {"1h": 2.0}, "visibility": 8000}
    df = normalize_openweather("CUU", cur)
    assert df.loc[0, "rain_mm"] == 2.0 and df.loc[0, "visibility"] == 8000


def test_fetch_uses_cache(tmp_path):
    st = Storage(str(tmp_path / "t.db"))
    s = FakeSession(lambda u, p: (200, {"ok": 1}))
    a = run(fetch_json(s, "http://x", {"a": 1}, {}, st, 60))
    b = run(fetch_json(s, "http://x", {"a": 1}, {}, st, 60))
    assert a == b == {"ok": 1} and s.calls == 1


def test_http_error_falls_back_to_stale_cache(tmp_path):
    st = Storage(str(tmp_path / "t.db"))
    st.put_cache("http://x", {"a": 1}, {"old": True})
    s = FakeSession(lambda u, p: (429, {"error": "rate"}))
    assert run(fetch_json(s, "http://x", {"a": 1}, {}, st, 0)) == {"old": True}


def test_http_error_without_cache_returns_none(tmp_path):
    st = Storage(str(tmp_path / "t.db"))
    s = FakeSession(lambda u, p: (500, {}))
    assert run(fetch_json(s, "http://x", {}, {}, st, 60)) is None


def test_network_exception_falls_back(tmp_path):
    st = Storage(str(tmp_path / "t.db"))
    st.put_cache("http://x", {}, [1])

    class Boom:
        def get(self, *a, **k):
            raise OSError("sin red")

    assert run(fetch_json(Boom(), "http://x", {}, {}, st, 0)) == [1]


def _handler(url, params):
    if "schedules" in url:
        return 200, {"response": [{"flight_iata": "AM1", "airline_iata": "AM",
                                   "dep_iata": params.get("dep_iata"), "arr_iata": "MEX",
                                   "dep_delayed": 5, "dep_time": "2026-11-20 10:00"}]}
    if "airlabs.co" in url:
        return 200, {"response": []}
    if "openweathermap" in url:
        return 200, {"dt": 1763600000, "main": {"temp": 20, "humidity": 30, "pressure": 1010},
                     "wind": {"speed": 2}}
    return 404, {}


def _cfg(tmp_path):
    return Config(airlabs_key="a", openweather_key="b",
                  demo_mode=False, airports=["CUU", "MEX"], db_path=str(tmp_path / "t.db"))


def test_fetch_all_runs_concurrently_and_returns_both_tables(tmp_path):
    cfg = _cfg(tmp_path)
    s = FakeSession(_handler, delay=0.05)
    out = run(fetch_all(cfg, Storage(cfg.db_path), session=s))
    assert set(out) == {"flights", "weather"}
    assert s.max_in_flight > 1
    assert set(out["weather"]["airport"]) == {"CUU", "MEX"}
    assert set(out["flights"]["dep_iata"]) == {"CUU", "MEX"}


def test_fetch_all_demo_mode_makes_no_requests(tmp_path):
    cfg = Config(demo_mode=True, airports=["CUU"], db_path=str(tmp_path / "t.db"))
    s = FakeSession(_handler)
    out = run(fetch_all(cfg, Storage(cfg.db_path), session=s))
    assert s.calls == 0 and not out["flights"].empty


def test_fetch_all_survives_total_failure(tmp_path):
    cfg = _cfg(tmp_path)
    s = FakeSession(lambda u, p: (500, {}))
    out = run(fetch_all(cfg, Storage(cfg.db_path), session=s))
    assert out["flights"].empty and list(out["flights"].columns) == FLIGHT_COLS


def test_endpoint_constants_exist():
    assert clients.AIRLABS_SCHEDULES.startswith("https://airlabs.co/")
    assert clients.OPENWEATHER_CURRENT.startswith("https://api.openweathermap.org/")
