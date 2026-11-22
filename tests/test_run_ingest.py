from src.ingest.run_ingest import run_ingest
from src.storage import Storage


def test_demo_ingest_saves_all_tables(tmp_path):
    env = tmp_path / ".env"
    env.write_text(f"DEMO_MODE=1\nAIRPORTS=CUU,MEX\nDB_PATH={tmp_path / 'x.db'}\n")
    tables = run_ingest(str(env))
    st = Storage(str(tmp_path / "x.db"))
    for name in ("flights", "weather", "air_quality", "sensors"):
        assert len(st.load_table(name)) == len(tables[name]) > 0


def _weather(ts):
    import pandas as pd
    from src.schema import WEATHER_COLS
    return pd.DataFrame([{"airport": "CUU", "ts": pd.Timestamp(ts), "temp": 20.0,
                          "humidity": 40.0, "pressure": 1010.0, "wind_speed": 2.0,
                          "rain_mm": 0.0, "visibility": 9000.0}])[WEATHER_COLS]


def _empty_tables():
    import pandas as pd
    from src.schema import AQ_COLS, FLIGHT_COLS, SENSOR_COLS
    return {"flights": pd.DataFrame(columns=FLIGHT_COLS), "air_quality": pd.DataFrame(columns=AQ_COLS),
            "sensors": pd.DataFrame(columns=SENSOR_COLS)}


def test_ingest_accumulates_history_and_skips_duplicates(tmp_path, monkeypatch):
    from src.ingest import run_ingest as mod

    env = tmp_path / ".env"
    env.write_text(f"DB_PATH={tmp_path / 'h.db'}\n")
    snapshots = iter([_weather("2026-11-20 10:00"), _weather("2026-11-20 11:00"),
                      _weather("2026-11-20 11:00")])

    async def fake_fetch_all(cfg, storage, session=None):
        return {**_empty_tables(), "weather": next(snapshots)}

    monkeypatch.setattr(mod, "fetch_all", fake_fetch_all)
    for _ in range(3):
        mod.run_ingest(str(env))
    w = Storage(str(tmp_path / "h.db")).load_table("weather")
    assert len(w) == 2  # dos horas distintas; la tercera corrida repetía la segunda
