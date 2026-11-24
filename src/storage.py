"""Módulo de almacenamiento persistente en SQLite.

Proporciona una interfaz simple para guardar y cargar DataFrames de pandas
en una base de datos SQLite local, creando automáticamente las carpetas
necesarias y asegurando el cierre adecuado de las conexiones.
"""

import os
import sqlite3
from contextlib import closing

import pandas as pd


class Storage:
    """Gestor de persistencia en SQLite para tablas de pandas.

    Administra la lectura y escritura de tablas relacionales en la base de datos,
    asegurando la existencia del directorio de destino y liberando recursos
    tras cada transacción mediante context managers.

    Attributes:
        path (str): Ruta al archivo de la base de datos SQLite (ej. 'data/trafico.db').
    """

    def __init__(self, path: str):
        """Inicializa el gestor de almacenamiento.

        Si la carpeta contenedora no existe, la crea automáticamente.

        Args:
            path (str): Ruta absoluta o relativa al archivo SQLite.
        """
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.path = path

    def save_table(self, name: str, df: pd.DataFrame) -> None:
        """Guarda un DataFrame de pandas en la base de datos reemplazando la tabla si existe.

        Utiliza un timeout de 30 segundos para evitar bloqueos concurrentes en SQLite.

        Args:
            name (str): Nombre de la tabla de destino.
            df (pd.DataFrame): Datos a persistir en la tabla.
        """
        with closing(sqlite3.connect(self.path, timeout=30)) as con:
            # if_exists="replace" sobrescribe la estructura y contenido de la tabla
            df.to_sql(name, con, if_exists="replace", index=False)

    def load_table(self, name: str) -> pd.DataFrame:
        """Carga una tabla completa de la base de datos como un DataFrame de pandas.

        Verifica previamente la existencia de la tabla en sqlite_master para evitar
        excepciones si la tabla aún no ha sido creada.

        Args:
            name (str): Nombre de la tabla a consultar.

        Returns:
            pd.DataFrame: DataFrame con los datos de la tabla, o un DataFrame vacío
                          si la tabla no existe en la base de datos.
        """
        with closing(sqlite3.connect(self.path, timeout=30)) as con:
            # Consultar el catálogo maestro para comprobar si la tabla existe
            exists = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            if not exists:
                return pd.DataFrame()
            # Cargar los registros usando pandas
            return pd.read_sql_query(f'SELECT * FROM "{name}"', con)
