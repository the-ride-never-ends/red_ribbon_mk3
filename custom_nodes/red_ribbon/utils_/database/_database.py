import logging
from typing import Callable, TypeVar


from ..configs import Configs, _configs


Class = TypeVar('Class')
ClassInstance = TypeVar('ClassInstance')


class InitializationError(Exception):
    """
    Custom exception for initialization errors.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)


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
        self.logger: logging.Logger = self.resources['logger']

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
        

def make_duckdb_database(
        resources: dict[str, Callable] = {}, 
        configs: Configs = _configs
        ) -> DatabaseAPI:
    """
    Factory function to create a DuckDB DatabaseAPI instance.
    
    Args:
        resources: Optional dictionary of resource overrides.
        configs: Configuration object for the database.

    Returns:
        An instance of DatabaseAPI configured for DuckDB.
    """
    from custom_nodes.red_ribbon.utils_.database.resources.duckdb import DuckDB

    try:
        resources = {
            "logger": resources.get("logger", logging.getLogger("DatabaseAPI")),
            "_enter": resources.get("_enter", DuckDB._enter),
            "_execute": resources.get("_execute", DuckDB._execute),
            "_exit": resources.get("_exit", DuckDB._exit),
        }
    except Exception as e:
        raise InitializationError(f"Failed to create DuckDB resources: {e}") from e

    try:
        return DatabaseAPI(resources=resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize DuckDB DatabaseAPI: {e}") from e