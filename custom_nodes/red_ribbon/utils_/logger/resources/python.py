

import logging
from pathlib import Path


class PythonLogger:
    pass
    
    def __init__(self, resources=None, configs=None):
        self.resources = resources
        self.configs = configs

        self.logger: logging.Logger = None
        self.log_path: Path = None

        self.log_path = Path(self.configs.log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Python logger initialized")

    def __call__(self, resources=None, configs=None, *args, **kwargs):
        self.__init__(resources, configs)

    def make_logger():
        pass