"""Almacenamiento en SQLite: guarda y lee tablas de pandas (vuelos, clima, resultados)."""
import os
import sqlite3
from contextlib import closing

import pandas as pd


class Storage:
    def __init__(self, path: str):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.path = path

    def save_table(self, name: str, df: pd.DataFrame) -> None:
        with closing(sqlite3.connect(self.path, timeout=30)) as con:
            df.to_sql(name, con, if_exists="replace", index=False)

    def load_table(self, name: str) -> pd.DataFrame:
        with closing(sqlite3.connect(self.path, timeout=30)) as con:
            exists = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
            if not exists:
                return pd.DataFrame()
            return pd.read_sql_query(f'SELECT * FROM "{name}"', con)
