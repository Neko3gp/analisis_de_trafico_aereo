import pandas as pd

from src.storage import Storage


def make(tmp_path):
    return Storage(str(tmp_path / "t.db"))


def test_table_roundtrip(tmp_path):
    s = make(tmp_path)
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    s.save_table("t", df)
    pd.testing.assert_frame_equal(s.load_table("t"), df)


def test_save_replaces_previous_table(tmp_path):
    s = make(tmp_path)
    s.save_table("t", pd.DataFrame({"a": [1, 2, 3]}))
    s.save_table("t", pd.DataFrame({"a": [9]}))
    assert s.load_table("t")["a"].tolist() == [9]


def test_missing_table_is_empty(tmp_path):
    assert make(tmp_path).load_table("nope").empty
