

from typing import Any


from pydantic import BaseModel


def get_value_from_base_model(self: BaseModel, key: str) -> Any:
    try:
        return dict(self)[key]
    except KeyError:
        raise KeyError(f"Key '{key}' not found in {self.__class__.__qualname__}")