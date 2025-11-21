import logging
from pathlib import Path
import traceback
from unittest.mock import MagicMock, AsyncMock


import pytest
import networkx as nx


from custom_nodes.red_ribbon.socialtoolkit.architecture import (
    make_variable_codebook,
    make_document_retrieval_from_websites,
    make_document_storage,
    make_top10_document_retrieval,
    make_relevance_assessment,
)
from custom_nodes.red_ribbon.socialtoolkit.architecture.variable_codebook import Variable, VariableCodebookConfigs


from custom_nodes.red_ribbon.utils_ import LLM, logger, DatabaseAPI, Configs, configs as real_configs




class FixtureError(Exception):
    """Custom exception for fixture errors."""
    def __init__(self, msg: str):
        msg = f"{msg}\n*********\nTRACEBACK\n*********\n{traceback.format_exc()}"
        super().__init__(msg)


@pytest.fixture
def mock_logger():
    """Fixture providing a mocked logger for testing."""
    mock_logger = MagicMock(spec_set=logging.Logger)
    return mock_logger


def make_mock_llm(return_values: dict = None) -> AsyncMock:
    """Creates a mocked LLM instance for testing."""

    def _make_mock_llm():
        try:
            mock_llm = AsyncMock(spec=LLM)
            if return_values is None:
                mock_llm.generate = AsyncMock()
                mock_llm.generate.return_value = "mocked response"
            else:
                for attr, value in return_values.items():
                    setattr(mock_llm, attr, value)
            return mock_llm
        except Exception as e:
            raise FixtureError(f"Failed to create mock LLM: {e}") from e

    try:
        return _make_mock_llm
    except Exception as e:
        raise

def make_mock_db(attributes: dict = None) -> MagicMock:
    """Creates a mocked DatabaseAPI instance for testing."""

    def _make_mock_db():
        try:
            mock_database = MagicMock(spec=DatabaseAPI)
            if attributes is not None:
                for attr, value in attributes.items():
                    setattr(mock_database, attr, value)
            return mock_database
        except Exception as e:
            raise FixtureError(f"Failed to create mock DatabaseAPI: {e}") from e

    try:
        return _make_mock_db
    except Exception as e:
        raise


mock_llm = pytest.fixture(make_mock_llm())
mock_database = pytest.fixture(make_mock_db())

@pytest.fixture
def mock_storage_service():
    """Creates a mocked storage service for testing."""
    mock_storage_service = MagicMock()
    return mock_storage_service


@pytest.fixture
def mock_configs() -> Configs:
    """Creates a VariableCodebookConfigs instance for testing."""
    try:
        mock_configs = VariableCodebookConfigs()
        return mock_configs
    except Exception as e:
        raise FixtureError(f"Failed to create mock Configs: {e}") from e

@pytest.fixture
def mock_resources(mock_llm, mock_database, mock_logger, mock_storage_service):
    """Creates a mocked resources dictionary for testing."""
    return {
        "llm": mock_llm,
        "db": mock_database,
        "logger": mock_logger,
        "storage_service": mock_storage_service,
    }


@pytest.fixture
def saved_variable_file(variable_fixture, tmp_path) -> Path:
    """Fixture saving a Variable instance to a file for testing."""
    item_name = variable_fixture.item_name
    file_path = tmp_path / f"{item_name}.json"
    try:
        var_dict = variable_fixture.model_dump()
    except Exception as e:
        raise FixtureError(f"Failed to dump Variable fixture to dict: {e}") from e
    try:
        with open(file_path, 'w') as f:
            import json
            json.dump(var_dict, f)
    except Exception as e:
        raise FixtureError(f"Failed to save Variable fixture to file: {e}") from e
    else:
        if not file_path.exists():
            raise FixtureError(f"Variable file was not created at {file_path}")
        return file_path






@pytest.fixture
def mock_configs_with_variable_file(saved_variable_file, mock_configs):
    """Modifies mock_configs to point to the saved variable file."""
    configs = mock_configs.model_copy()
    configs.variable_codebook.load_from_file = True
    configs.variable_codebook.variable_file_paths = {
        f"{saved_variable_file.stem}": saved_variable_file.resolve(),
    }
    return configs


@pytest.fixture
def mock_configs_load_from_file_is_false(mock_configs):
    """Modifies mock_configs to set load_from_file to False."""
    configs = mock_configs.model_copy()
    configs.variable_codebook.load_from_file = False
    return configs


@pytest.fixture
def mock_configs_cache_enabled_is_true(mock_configs):
    """Modifies mock_configs to set cache_enabled to True."""
    configs = mock_configs.model_copy()
    configs.variable_codebook.cache_enabled = True
    return configs

def _make_variable_codebook_fixture(mock_resources, mock_configs):
    """Helper function to create a VariableCodebook fixture."""
    try:
        return make_variable_codebook(
            resources=mock_resources,
            configs=mock_configs,
        )
    except Exception as e:
        raise FixtureError(f"Failed to create variable_codebook_fixture fixture: {e}") from e

@pytest.fixture
def variable_codebook_fixture(mock_resources, mock_configs):
    """Fixture providing a mocked VariableCodebook instance for testing."""
    return _make_variable_codebook_fixture(mock_resources, mock_configs)

@pytest.fixture
def variable_codebook_load_from_file_is_false_fixture(mock_resources, mock_configs_load_from_file_is_false):
    """Fixture providing a mocked VariableCodebook instance for testing."""
    return _make_variable_codebook_fixture(mock_resources, mock_configs_load_from_file_is_false)

@pytest.fixture
def variable_codebook_cache_enabled_is_true_fixture(mock_resources, mock_configs_cache_enabled_is_true):
    """Fixture providing a mocked VariableCodebook instance for testing."""
    return _make_variable_codebook_fixture(mock_resources, mock_configs_cache_enabled_is_true)

