


import logging
from pathlib import Path
from typing import Any, Callable


def get_logger(name: str, configs: Any, implementation: str = "default", **kwargs) -> logging.Logger:
    """
    """
    name: str
    configs: Any
    implementation: str = "default"
    level: int = logging.DEBUG,
    log_dir: Path = Path("logs")
    stacklevel: int = 2

    match implementation:
        case "default":
            from .resources.python import PythonLogger
            resources = PythonLogger.__dict__
        case "pydantic":
            from .resources.pydantic import PydanticLogger
            resources = PydanticLogger.__dict__
        case _:
            raise ValueError(f"Invalid implementation: {implementation}")

    return Logger(resources,configs)



class Logger:

    def __init__(self, resources=None, configs=None):
        self.resources = resources
        self.configs = configs

        self.log_dir: str | Path = self.configs.get('log_dir', Path("logs"))
        self.logger_name = self.configs.get('name', 'default')
        self.level = self.configs.get('level', logging.DEBUG)

        self._setup_logger: Callable = self.resources['setup_logger']
        self._setup_file_handler: Callable = self.resources['setup_file_handler']
        self._debug: Callable = self.resources['debug']
        self._info: Callable = self.resources['info']
        self._warning: Callable = self.resources['warning']
        self._error: Callable = self.resources['error']
        self._critical: Callable = self.resources['critical']

        self.logger: logging.Logger = None
        self.log_file: Path = None

        self.file_handler = self.setup_file_handler(self.logger_name)
        self.logger = self.setup_logger(self.logger_name, self.level, self.file_handler)


    def setup_file_handler(self, logger_name: str) -> logging.FileHandler:
        """
        Set up and configure the file handler for the logger.
        
        Args:
        logger_name: Name of the logger to use in the log filename
        
        Returns:
        Configured file handler
        """
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / f"{logger_name}.log"

        if not self.log_file.exists():
            self.log_file.mkdir(parents=True, exist_ok=True)

        return self._setup_file_handler(self.log_file)
    
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        return file_handler

    def setup_logger(self, logger_name: str, level: int ) -> logging.Logger:
        """
        Set up the logger with the specified configuration.
        Creates the logger, sets the appropriate level, and configures handlers.
        """
        return self._setup_logger(logger_name, level)

        # Create logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Setup file handler
        self.logger.addHandler()
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
