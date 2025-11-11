"""
A module to manage the database connection and operations.
This module provides a singleton instance of the Database class, which is read-only by default.
"""
from typing import Callable


from custom_nodes.red_ribbon.utils_.configs import configs as project_configs, Configs
from custom_nodes.red_ribbon.utils_.logger import logger as module_logger
from ._database import Database
from .dependencies.duckdb_database import DuckDbDatabase


class InitializationError(Exception):
    """
    Custom exception for initialization errors.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)


def make_duckdb_database(resources: dict[str, Callable] = {}, configs: Configs = project_configs) -> Database:
    """
    Factory function to create a new Database instance.
    
    This function initializes a new Database object with the provided configurations
    and DuckDB resources. It is intended to be used for creating database connections
    as needed.

    Args:
        resources (dict[str, Any], optional): A dictionary of callables to override injected defaults. Defaults to None.
        configs (Configs, optional): A Configs object to override default configurations. Defaults to None.

    Returns:
        Database: A new instance of the Database class.
    """
    assert project_configs is not None, "Global configs must be initialized."
    assert configs is not None, "Configs must be provided or available globally."
    # Export resources dictionary for use with Database class
    configs = configs.database

    _resources = resources
    db = DuckDbDatabase()
    db.logger = logger = _resources.pop("logger", module_logger)

    duckdb_resources = {
        "begin": _resources.pop("begin", db.begin),
        "close": _resources.pop("close", db.close),
        "commit": _resources.pop("commit", db.commit),
        "connect": _resources.pop("connect", db.connect),
        "create_function": _resources.pop("create_function", db.create_function),
        "create_index_if_not_exists": _resources.pop("create_index_if_not_exists", db.create_index_if_not_exists),
        "create_table_if_not_exists": _resources.pop("create_table_if_not_exists", db.create_table_if_not_exists),
        "execute": _resources.pop("execute", db.execute),
        "fetch": _resources.pop("fetch", db.fetch),
        "fetch_all": _resources.pop("fetch_all", db.fetch_all),
        "fetch_one": _resources.pop("fetch_one", db.fetch),
        "get_cursor": _resources.pop("get_cursor", db.get_cursor),
        "rollback": _resources.pop("rollback", db.rollback),
        "read_only": _resources.pop("read_only", True),  # Set read_only to True for read-only access
        "logger": logger,
    }

    for key in _resources.keys():
        if key not in duckdb_resources:
            raise KeyError(f"Unexpected resource key: {key}")

    try:
        return Database(configs=configs, resources=duckdb_resources)
    except Exception as e:
        logger.error(f"Error initializing Database: {e}")
        raise InitializationError(f"Failed to initialize Database: {e}") from e
