from datetime import datetime
import json


import pytest
import yaml


from tests_unit.socialtoolkit_.architecture.conftest import FixtureError
from custom_nodes.red_ribbon.socialtoolkit.socialtoolkit import (
    main,
    Configs
)

@pytest.fixture
def temp_dir(tmp_path):
    """Creates a temporary directory for testing."""
    return tmp_path


@pytest.fixture
def expected_timestamp():
    """Returns the expected timestamp format for testing."""
    return datetime.now().strftime("%Y%m%d")


@pytest.fixture
def expected_file_contents_happy_path(expected_timestamp):
    """Returns the expected input and output for testing."""
    return {
        "input": "What is the local sales tax in Cheyenne, WY?",
        "output": "6%",
        "citations": [
            {"Bluebook Citation 1": "..."},
            {"Bluebook Citation 2": "..."},
        ],
        "timestamp": expected_timestamp
    }


@pytest.fixture
def expected_output_string(expected_input_output):
    """Returns the expected output string for testing."""
    try:
        return json.dumps(expected_input_output)
    except Exception as e:
        raise FixtureError(f"Failed to create expected output string: '{e}'") from e


@pytest.fixture
def valid_config_values(tmp_path):
    """Returns a dictionary of valid configuration values for testing."""
    return {
        "INPUT_DATA_POINT": "What is the local sales tax in Cheyenne, WY?",
        "RETRIEVAL_COUNT": 10,
        "similarity_threshold": 0.6,
        "RANKING_METHOD": "cosine_similarity",
        "USE_FILTER": False,
        "FILTER_CRITERIA": {},
        "USE_RERANKING": False,
        "OPENAI_API_KEY": "test-api-key",
        "OPENAI_MODEL": "gpt-4o-mini",
        "OPENAI_SMALL_MODEL": "gpt-5-nano",
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        "OPENAI_MODERATION_MODEL": "omni-moderation-latest",
        "DEFAULT_SYSTEM_PROMPT": "You are a helpful assistant.",
        "EMBEDDING_DIMENSIONS": 1536,
        "TEMPERATURE": 0.0,
        "MAX_TOKENS": 4096,
        "LOG_LEVEL": 10,
        "SIMILARITY_SCORE_THRESHOLD": 0.4,
        "SEARCH_EMBEDDING_BATCH_SIZE": 10000,
        "OUTPUT_DIR": tmp_path, 
        "connection_string": None,
        "timeout": None
    }

@pytest.fixture
def invalid_config_values():
    return {
        "INPUT_DATA_POINT": "sales tax info",
        "RETRIEVAL_COUNT": "invalid",
        "similarity_threshold": "not_a_number",
        "RANKING_METHOD": 123,
        "USE_FILTER": "true",
        "FILTER_CRITERIA": "not_a_dict",
        "USE_RERANKING": "false",
        "OPENAI_API_KEY": "",
        "OPENAI_MODEL": None,
        "OPENAI_SMALL_MODEL": 42,
        "OPENAI_EMBEDDING_MODEL": [],
        "OPENAI_MODERATION_MODEL": {},
        "DEFAULT_SYSTEM_PROMPT": 123,
        "EMBEDDING_DIMENSIONS": "1536",
        "TEMPERATURE": "zero",
        "MAX_TOKENS": "4096",
        "LOG_LEVEL": "debug",
        "OUTPUT_DIR": 12345,
        "SIMILARITY_SCORE_THRESHOLD": "0.4",
        "SEARCH_EMBEDDING_BATCH_SIZE": "10000",
        "connection_string": 123,
        "timeout": "30"
    }

@pytest.fixture
def expected_output_file_path(tmp_path):
    return tmp_path / "socialtoolkit_output.jsonl"


@pytest.fixture
def valid_config_obj(valid_config_values):
    try:
        configs = Configs(**valid_config_values)
        return configs
    except Exception as e:
        raise FixtureError(f"Failed to create valid Configs object: '{e}'") from e

@pytest.fixture
def valid_config_file(valid_config_obj):
    temp_file = "valid_config_yaml_file.yaml"
    try:
        config_dict = valid_config_obj.model_dump()
        config_yaml = yaml.dump(config_dict)
        with open(temp_file, 'w') as f:
            f.write(config_yaml)
        return temp_file
    except Exception as e:
        raise FixtureError(f"Failed to create valid_config_file: '{e}'") from e

@pytest.fixture
def invalid_config_file(invalid_config_values):
    temp_file = "invalid_config_values.yaml"
    try:
        config_yaml = yaml.dump(invalid_config_values)
        with open(temp_file, 'w') as f:
            f.write(config_yaml)
        return temp_file
    except Exception as e:
        raise FixtureError(f"Failed to create invalid_config_file: '{e}'") from e


class TestMainHappyPath:

    def test_when_main_called_then_return_0(self, valid_config_file):
        """
        GIVEN a valid Socialtoolkit configuration
        WHEN main() is called with this configuration
        THEN the return should be 0
        """
        good_return_code = 0
        main_return = main()
        assert main_return == good_return_code, f"Expected return code '{good_return_code}', got '{main_return}'."

    def test_when_main_called_then_output_file_created(self, valid_config_file, expected_output_file_path):
        """
        GIVEN a valid Socialtoolkit pipeline configuration
        WHEN main() is called with this configuration
        THEN an output file should be created at the specified location
        """
        main_return = main()
        assert expected_output_file_path.exists(), f"Expected output file at '{expected_output_file_path}' does not exist."

    def test_when_main_called_then_output_file_not_empty(self, valid_config_file, expected_output_file_path):
        """
        GIVEN a valid Socialtoolkit pipeline configuration
        WHEN main() is called with this configuration
        THEN the output file should not be empty
        """
        zero = 0
        main_return = main()
        file_size = expected_output_file_path.stat().st_size
        assert file_size > zero, \
            f"Expected output file at '{expected_output_file_path}' to be >'{zero}', but got '{file_size}'."

    @pytest.mark.parameterize("expected_field", [
        "input", "output", "citations", "timestamp"
    ])
    def test_when_main_called_then_has_expected_fields_in_output(self, expected_field, valid_config_file, expected_output_file_path):
        """
        GIVEN a valid Socialtoolkit pipeline configuration
        WHEN main() is called with this configuration
        THEN the output file should contain expected fields
        """
        main_return = main()
        file_text = expected_output_file_path.read_text()
        assert expected_field in file_text, \
            f"Expected output file at '{expected_output_file_path}' to have field '{expected_field}', but got '{file_text}'."

    @pytest.mark.parameterize("expected_field", [
        "input", "output", "citations"
    ])
    def test_when_main_called_then_has_expected_values_in_output(
        self, expected_field, valid_config_file, expected_file_contents_happy_path, expected_output_file_path
    ):
        """
        GIVEN a valid Socialtoolkit pipeline configuration
        WHEN main() is called with this configuration
        THEN the output file should contain expected values
        """
        main_return = main()
        expected_value = expected_file_contents_happy_path[expected_field]
        file_text = expected_output_file_path.read_text()
        assert expected_value in file_text, \
            f"Expected output file at '{expected_output_file_path}' to contain value '{expected_value}', but got '{file_text}'."


class TestMainUnhappyPathBadConfig:

    def test_when_main_called_with_invalid_config_then_returns_1(self, invalid_config_file):
        """
        GIVEN an invalid Socialtoolkit pipeline configuration
        WHEN main() is called with this configuration
        THEN the return should be 1
        """
        bad_return_code = 1
        main_return = main()
        assert main_return == bad_return_code, f"Expected return code '{bad_return_code}', got '{main_return}'."

    def test_when_main_errors_then_no_output_file_created(self, invalid_config_file, expected_output_file_path):
        """
        GIVEN an invalid Socialtoolkit pipeline configuration
        WHEN main() is called with this configuration
        THEN no output file should be created
        """
        main_return = main()
        assert not expected_output_file_path.exists(), f"Did not expect output file at '{expected_output_file_path}', but it exists."

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
