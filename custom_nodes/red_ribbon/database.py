from dataclasses import dataclass, field, InitVar
from functools import cached_property
import logging
from typing import Callable, TypeVar, Optional


from .configs import Configs
from .utils.database.resources.duckdb import DuckDB
from .utils.main_.instantiate import instantiate


from pydantic import BaseModel, Field

Class = TypeVar('Class')
ClassInstance = TypeVar('ClassInstance')

class DatabaseApiError(Exception):
    """
    Custom exception for database API errors.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"DatabaseApiError: {self.message}"


class DatabaseAPI:
    """
    Generic class for database operations.
    NOTE: This is a composition class and 
        must be assigned a supported database 
        resource to function properly.
    """

    def __init__(self, resources, configs) -> 'DatabaseAPI':
        self.configs = configs
        self.resources = resources
        self.logger = self.resources['logger'] or logging.getLogger(self.__class__.__name__)

        self._dep_enter = self.resources["_enter"]
        self._execute = self.resources["_execute"]
        self._exit = self.resources["_exit"]

    @classmethod
    def enter(cls, 
              resources: dict[str, Callable] = None, 
              configs: Configs = None
              ) -> 'DatabaseAPI':
        instance = cls(resources, configs)
        instance._enter()
        return instance

    def _enter(self) -> None:
        try:
            self._dep_enter()
        except Exception as e:
            self.logger.error(f"Error entering database: {e}")
            raise DatabaseApiError from e
        return self

    def __enter__(self) -> 'DatabaseAPI':
        self._enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return self._exit()

    def exit(self) -> None:
        return self._exit()

    def execute(self, statement: str, *args, **kwargs):
        try:
            return self._execute(statement, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error executing statement: {e}")
            raise DatabaseApiError from e