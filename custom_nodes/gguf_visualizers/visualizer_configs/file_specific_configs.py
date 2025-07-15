import inspect
import os
from typing import Any

from ._get_config import get_config


class FileSpecificConfigs:

    def __init__(self, filename=None):
        # Get the name of the file that the method is called from
        # Remove the .py from the end, then make it UPPER_CASE
        self.filename: str = filename or os.path.splitext(os.path.basename(inspect.stack()[1].filename))[0].upper()

    def config(self, constant: str) -> Any:
        return get_config(self.filename, constant)


