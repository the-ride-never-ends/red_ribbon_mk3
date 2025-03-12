"""
A duckdb database API module.
"""
import logging
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Self, TypeVar

Configs = TypeVar('Configs', dict[str, Callable], NamedTuple)


import duckdb
from duckdb import DuckDBPyConnection


class DuckDB:
    """
    A duckdb database API class
    """
    _CONNECTION_STRING = "duckdb:///{{os.path.expanduser('~')}}/red_ribbon_data/database.duckdb"

    def __init__(self, 
                 resources: Optional[dict[str, Callable]] = None, 
                 configs: Configs = None
                ):
        self.resources = resources
        self.configs = configs

        self._in_memory: bool = None
        self._db_path: str = None
        self._connection: DuckDBPyConnection = None
        
    def _enter(self, configs) -> Self:
    
        assert hasattr(configs, "database"), "configs must have a database attribute"
        self._in_memory:                bool = configs.database.in_memory or False
        self._db_path:                   str = ":memory:" if self._in_memory  else self._CONNECTION_STRING 
        self._connection: DuckDBPyConnection = duckdb.connect(self._db_path)
    
        return self

    def _exit(self) -> None:
        if isinstance(self._connection) is not None:
            self.connection.close()
        self._connection = None

    @classmethod
    def _set_configs(cls, configs: Configs) -> Self:
        return cls(configs)

    def _execute(self, query: str, *args, **kwargs) -> Optional[Any]:
        # Ignore args because duckdb only supports kwargs
        if args:
            print("DuckDB only supports kwargs, ignoring args")
        return self._connection.execute(query,**kwargs)

