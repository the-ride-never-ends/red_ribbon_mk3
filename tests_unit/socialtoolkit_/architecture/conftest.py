import logging
from unittest.mock import Mock, MagicMock, AsyncMock


import pytest


from custom_nodes.red_ribbon.socialtoolkit.architecture import (
    make_variable_codebook,
    make_document_retrieval_from_websites,
    make_document_storage,
    make_top10_document_retrieval,
    make_relevance_assessment,
)
from custom_nodes.red_ribbon.utils_.llm import LLM
from custom_nodes.red_ribbon.utils_.database import DatabaseAPI
from custom_nodes.red_ribbon.utils_.logger import logger
from custom_nodes.red_ribbon.utils_.configs import Configs, configs as real_configs


class FixtureError(Exception):
    """Custom exception for fixture errors."""
    def __init__(self, msg: str):
        super().__init__(msg)

def mock_logger():
    """Fixture providing a mocked logger for testing."""
    mock_logger = MagicMock(spec_set=logging.Logger)
    return mock_logger

@pytest.fixture
def mock_llm():
    """Fixture providing a mocked LLM instance for testing."""
    mock_llm = AsyncMock(spec=LLM)
    mock_llm.generate.return_value = "mocked response"
    return mock_llm


@pytest.fixture
def mock_database():
    """Fixture providing a mocked DatabaseAPI instance for testing."""
    mock_db = MagicMock(spec=DatabaseAPI)
    mock_db.query.return_value = [{"id": 1, "data": "mocked data"}]
    return mock_db


@pytest.fixture
def variable_codebook_fixture(mock_llm, mock_database):
    """Fixture providing a mocked VariableCodebook instance for testing."""

    resources = {
        "llm": mock_llm,
        "database": mock_database,
        "logger": logger,
    }
    try:
        return make_variable_codebook(
            resources=resources,
            configs=real_configs,
        )
    except Exception as e:
        raise FixtureError(f"Failed to create variable_codebook_fixture fixture: {e}") from e
