"""Almacenamiento en SQLite: caché de respuestas de APIs y tablas de resultados."""
import hashlib
import json
import os
import sqlite3
import time

import pandas as pd


class Storage:
    def __init__(self, path: str):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.path = path
        with self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS api_cache ("
                "key TEXT PRIMARY KEY, endpoint TEXT, params TEXT, "
                "payload TEXT, fetched_at REAL)"
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    @staticmethod
    def _key(endpoint: str, params: dict) -> str:
        raw = endpoint + json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def journal_mode(self) -> str:
        with self._connect() as con:
            return con.execute("PRAGMA journal_mode").fetchone()[0]

    def get_cached(self, endpoint: str, params: dict, max_age_s):
        """Devuelve el payload en caché, o None. max_age_s=None acepta cualquier antigüedad."""
        key = self._key(endpoint, params)
        with self._connect() as con:
            row = con.execute(
                "SELECT payload, fetched_at FROM api_cache WHERE key=?", (key,)
            ).fetchone()
        if row is None:
            return None
        payload, fetched_at = row
        if max_age_s is not None and time.time() - fetched_at > max_age_s:
            return None
        return json.loads(payload)

    def put_cache(self, endpoint: str, params: dict, payload) -> None:
        key = self._key(endpoint, params)
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO api_cache VALUES (?,?,?,?,?)",
                (
                    key,
                    endpoint,
                    json.dumps(params, sort_keys=True, default=str),
                    json.dumps(payload),
                    time.time(),
                ),
            )

    def save_table(self, name: str, df: pd.DataFrame) -> None:
        with self._connect() as con:
            df.to_sql(name, con, if_exists="replace", index=False)

    def load_table(self, name: str) -> pd.DataFrame:
        with self._connect() as con:
            exists = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            if not exists:
                return pd.DataFrame()
            return pd.read_sql_query(f'SELECT * FROM "{name}"', con)
