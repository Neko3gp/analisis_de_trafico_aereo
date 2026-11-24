import numpy as np
import pandas as pd
import pytest

from src import mpi_run
from src.clean import HOURLY_COLS
from src.compute.backend import get_backend
from src.config import Config
from src.ingest.demo import generate_demo
from src.mpi_run import RESULT_COLS, partition, process_airport, run
from src.storage import Storage

AIRPORTS = ["CUU", "MEX", "GDL", "MTY", "TIJ"]


def test_partition_round_robin_sizes():
    parts = partition(list("abcde"), 3)
    assert [len(p) for p in parts] == [2, 2, 1]
    assert sorted(sum(parts, [])) == list("abcde")


def test_partition_more_ranks_than_items():
    parts = partition(["a", "b"], 5)
    assert len(parts) == 5 and [len(p) for p in parts] == [1, 1, 0, 0, 0]


def test_partition_rejects_zero_ranks():
    with pytest.raises(ValueError):
        partition([1], 0)


def _hourly(n=48, rain_effect=True, airport="CUU"):
    rng = np.random.default_rng(0)
    rain = np.where(rng.random(n) < 0.3, rng.exponential(2, n), 0.0)
    delay = 0.1 + (0.05 * rain if rain_effect else 0.0)
    return pd.DataFrame({
        "airport": airport, "hour": pd.date_range("2026-11-01", periods=n, freq="h"),
        "flights": 10, "delayed": 1, "delay_rate": delay, "temp": 20.0, "humidity": 40.0,
        "wind_speed": rng.uniform(0, 8, n), "rain_mm": rain, "visibility": 9000.0})[HOURLY_COLS]


def test_process_airport_empty():
    r = process_airport(pd.DataFrame(columns=HOURLY_COLS), get_backend("numpy"), airport="CUU")
    assert list(r) == RESULT_COLS and r["airport"] == "CUU" and r["n_rows"] == 0
    assert np.isnan(r["mean_delay_rate"]) and np.isnan(r["corr_rain_delay"])


def test_process_airport_finds_rain_delay_correlation():
    r = process_airport(_hourly(), get_backend("numpy"))
    assert r["airport"] == "CUU" and r["n_rows"] == 48
    assert r["corr_rain_delay"] > 0.99
    assert r["mean_delay_rate"] > 0.1


def test_process_airport_single_row_does_not_raise():
    r = process_airport(_hourly(n=1), get_backend("numpy"))
    assert r["n_rows"] == 1 and np.isnan(r["corr_rain_delay"])


def _stored(tmp_path, airports=AIRPORTS):
    st = Storage(str(tmp_path / "t.db"))
    for name, df in generate_demo(airports, hours=48, seed=2).items():
        st.save_table(name, df)
    cfg = Config(demo_mode=True, airports=airports, backend="numpy", db_path=st.path)
    return cfg, st


def test_run_without_mpi_uses_multiprocessing(tmp_path, monkeypatch):
    monkeypatch.setattr(mpi_run, "_get_comm", lambda: None)
    cfg, st = _stored(tmp_path)
    res = run(cfg, st)
    assert list(res.columns) == RESULT_COLS
    assert sorted(res["airport"]) == sorted(AIRPORTS)
    assert len(st.load_table("results")) == len(AIRPORTS)
    assert not st.load_table("hourly").empty


class FakeComm:
    """Simula la vista del rank dado: scatter/gather sin MPI real."""

    def __init__(self, size, rank=0, backend="numpy", mine=None):
        self.size, self.rank, self.backend, self.mine = size, rank, backend, mine
        self.chunks = None

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank

    def scatter(self, chunks, root=0):
        self.chunks = chunks
        return chunks[self.rank] if self.rank == 0 else self.mine

    def gather(self, data, root=0):
        if self.rank != 0:
            return None
        b = get_backend(self.backend)
        others = [[process_airport(df, b, airport=ap) for ap, df in self.chunks[r]]
                  for r in range(1, self.size)]
        return [data] + others


def test_run_with_comm_covers_all_airports_even_with_idle_ranks(tmp_path):
    cfg, st = _stored(tmp_path)
    comm = FakeComm(size=8)  # más ranks (8) que aeropuertos (5)
    res = run(cfg, st, comm=comm)
    assert len(comm.chunks) == 8 and sum(len(c) for c in comm.chunks) == 5
    assert sorted(res["airport"]) == sorted(AIRPORTS)


def test_non_root_rank_returns_none_and_does_not_write(tmp_path):
    cfg, st = _stored(tmp_path)
    assert run(cfg, st, comm=FakeComm(size=2, rank=1, mine=[])) is None
    assert st.load_table("results").empty


def test_run_without_data_returns_empty_results(tmp_path, monkeypatch):
    monkeypatch.setattr(mpi_run, "_get_comm", lambda: None)
    cfg = Config(backend="numpy", db_path=str(tmp_path / "vacio.db"))
    res = run(cfg, Storage(cfg.db_path))
    assert res.empty and list(res.columns) == RESULT_COLS
