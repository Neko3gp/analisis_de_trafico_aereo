import time

import pandas as pd

from src.storage import Storage


def make(tmp_path):
    return Storage(str(tmp_path / "t.db"))


def test_cache_hit(tmp_path):
    s = make(tmp_path)
    s.put_cache("flights", {"bbox": "1,2"}, {"a": 1})
    assert s.get_cached("flights", {"bbox": "1,2"}, max_age_s=60) == {"a": 1}


def test_cache_miss_returns_none(tmp_path):
    assert make(tmp_path).get_cached("flights", {}, max_age_s=60) is None


def test_cache_expired_returns_none(tmp_path):
    s = make(tmp_path)
    s.put_cache("flights", {}, [1, 2])
    time.sleep(1.1)
    assert s.get_cached("flights", {}, max_age_s=1) is None


def test_stale_cache_available_when_requested(tmp_path):
    s = make(tmp_path)
    s.put_cache("flights", {}, [1, 2])
    time.sleep(1.1)
    assert s.get_cached("flights", {}, max_age_s=None) == [1, 2]


def test_params_order_irrelevant(tmp_path):
    s = make(tmp_path)
    s.put_cache("w", {"a": 1, "b": 2}, "x")
    assert s.get_cached("w", {"b": 2, "a": 1}, max_age_s=60) == "x"


def test_table_roundtrip(tmp_path):
    s = make(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    s.save_table("t", df)
    pd.testing.assert_frame_equal(s.load_table("t"), df)


def test_missing_table_is_empty(tmp_path):
    assert make(tmp_path).load_table("nope").empty


def test_wal_enabled(tmp_path):
    s = make(tmp_path)
    assert s.journal_mode().lower() == "wal"
