

from typing import Any


from pydantic import BaseModel


def get_value_with_default_from_base_model(self: BaseModel, key: str, default: Any) -> Any:
    try:
        return dict(self)[key]
    except KeyError:
        print(f"Key '{key}' not found in {self.__class__.__qualname__}. Returning default {default}")
        return default
