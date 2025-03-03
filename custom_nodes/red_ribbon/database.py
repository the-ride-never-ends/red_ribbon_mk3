from dataclasses import dataclass, field, InitVar
from functools import cached_property
from typing import Callable, TypeVar, Optional


from .configs import Configs
from .utils.database.resources.duckdb import DuckDB
from .utils.instantiate import instantiate


from pydantic import BaseModel, Field

Class = TypeVar('Class')
ClassInstance = TypeVar('ClassInstance')


class DatabaseAPI:
    """
    Generic class for database operations.
    NOTE: This is a composition class and 
        must be assigned a supported database 
        resource to function properly.
    """

    def __init__(self, resources, configs) -> 'DatabaseAPI':
        self.configs = configs
        self.resources = resources or {}

        self._enter = self.resources["enter"]
        self._execute = self.resources["execute"]
        self._exit = self.resources["exit"]

    @classmethod
    def enter(cls, 
              resources: dict[str, Callable] = None, 
              configs: Configs = None
              ) -> 'DatabaseAPI':
        instance = cls(resources, configs)
        instance._enter()
        return instance

    def __enter__(self) -> 'DatabaseAPI':
        self.enter(self.resources, self.configs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return self._exit()

    def exit(self) -> None:
        return self._exit()

    def execute(self, statement: str, *args, **kwargs):
        self._execute(statement, *args, **kwargs)