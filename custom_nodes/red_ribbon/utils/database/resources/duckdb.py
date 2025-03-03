"""
A duckdb database API module.
"""
import logging
from typing import Any, Optional


import duckdb
from duckdb import DuckDBPyConnection


class DuckDB:
    """
    A duckdb database API class
    """
    def __init__(self, resources=None, configs=None):
        self.db_path = ":memory:" if not configs.paths.DB_PATH else configs.paths.DB_PATH
        self.logger = logging.getLogger(self.__class__.__name__)

        self.connection = None

    def _enter(self) -> DuckDBPyConnection:
        self.connection = duckdb.connect(self.db_path)
        return self

    def _exit(self) -> None:
        self.connection.close()

    def _execute(self, query: str, *args, **kwargs) -> Optional[Any]:
        # Ignore args because duckdb only supports kwargs
        if args:
            self.logger.warning("DuckDB only supports kwargs, ignoring args")
        return self.connection.execute(query,**kwargs)

