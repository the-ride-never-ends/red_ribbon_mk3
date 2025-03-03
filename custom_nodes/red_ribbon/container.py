
from dataclasses import dataclass, field
from dependency_injector import containers, providers
from pydantic import BaseModel, Field

from typing import Any


from .configs import Configs

class LlmResources(BaseModel):
    type: dict[str, Any] = None


class DatabaseResources(BaseModel):
    type: dict[str, Any] = None


class Container(containers.DeclarativeContainer):
    configs = providers.Configuration("configs")
    database_factory = providers.Factory(
        DatabaseResources,
        database=providers.Dict(
            duckdb=providers.Singleton(DatabaseResources, type="duckdb"),
        )
    )
    llm_factory = providers.Factory(
        LlmResources,
        openai=providers.Singleton(LlmResources, type="openai"),
    )


