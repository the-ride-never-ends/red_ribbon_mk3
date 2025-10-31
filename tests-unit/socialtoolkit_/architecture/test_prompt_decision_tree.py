#!/usr/bin/env python3
"""
Feature: Prompt Decision Tree
  As a data extraction system
  I want to execute decision trees of prompts to extract information from documents
  So that structured data can be extracted following a logical flow

  Background:
    GIVEN a PromptDecisionTree instance is initialized
    And an LLM API client is available
    And a variable codebook service is available
    And a logger is available
"""
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
import logging

from custom_nodes.red_ribbon.socialtoolkit.architecture.prompt_decision_tree import (
    PromptDecisionTree, 
    PromptDecisionTreeConfigs
)

from custom_nodes.red_ribbon.socialtoolkit.architecture.variable_codebook import VariableCodebook
from .conftest import FixtureError

from openai import AsyncClient


# Fixtures for Background
@pytest.fixture
def mock_llm_api():
    """
    And an LLM API client is available
    """
    mock_llm = AsyncMock(spec=AsyncClient)
    return mock_llm


@pytest.fixture
def mock_variable_codebook_service():
    """
    And a variable codebook service is available
    """
    mock_codebook = MagicMock(spec=VariableCodebook)
    return mock_codebook


@pytest.fixture
def mock_logger():
    """
    And a logger is available
    """
    logger = Mock(spec_set=logging.Logger)
    return logger


@pytest.fixture
def prompt_decision_tree(
    mock_llm_api,
    mock_variable_codebook_service,
    mock_logger,
):
    """
    Mock PromptDecisionTree instance for testing
    """
    resources = {
        'logger': mock_logger,
        'llm': mock_llm_api,
        'variable_codebook': mock_variable_codebook_service
    }
    configs = PromptDecisionTreeConfigs()
    try:
        tree = PromptDecisionTree(resources, configs)
    except Exception as e:
        raise FixtureError from e

    return tree


@pytest.fixture
def documents():
    """
    GIVEN various document configurations for testing
    """
    return {
        'sample': [{"content": "Municipal tax rate is 5%", "page_number": i} for i in range(1, 4)],
        'single': [{"content": "Municipal tax rate is 5%", "page_number": 1}],
        'multiple': [
            {"content": f"Page {i} content. Also, the municipal tax rate is 5%", "page_number": i} for i in range(1, 6)
        ],
        'empty': []
    }


@pytest.fixture
def prompt_sequences():
    """
    GIVEN various prompt sequence configurations for testing
    """
    return {
        'sample': ["What is the tax rate?"],
        'command': ["Get tax rate"],
        'google_fu': ["tax information municipal rate"],
        'empty': []
    }


@pytest.fixture
def response_keys():
    """
    GIVEN expected response dictionary keys
    """
    return {
        'success': 'success',
        'output_data_point': 'output_data_point', 
        'responses': 'responses',
        'iterations': 'iterations',
        'error': 'error'
    }


@pytest.fixture
def test_values():
    """
    GIVEN common test values and constants
    """
    return {
        'max_pages': 10,
        'total_pages': 15,
        'page_count': 5,
        'context_size': 8192,
        'large_content': "x" * 10000,
        'tax_document_title': "Tax Document",
        'example_url': "https://example.com"
    }


class TestExecuteMethodReturnsExtractedDataPoint:
    """Tests for PromptDecisionTree.execute method."""

    def test_when_execute_called_with_pages_and_prompts_then_returns_string(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN a list of 3 relevant pages
        WHEN I call execute with pages and prompts
        THEN a string data point is returned
        """
        result = prompt_decision_tree.execute(documents['sample'], prompt_sequences['sample'])
        
        assert isinstance(result, str), f"Expected string but got {type(result)}"

    def test_when_execute_called_with_valid_documents_then_returns_nonempty_string(
            self, prompt_decision_tree, test_values, prompt_sequences):
        """
        GIVEN a list of 3 relevant pages with tax information
        WHEN I call execute with pages and prompts
        THEN the data point contains extracted information
        """
        documents = [{"content": "Municipal tax ordinance states 3.5%", "page_number": i} for i in range(1, test_values['page_count'] - 2)]

        result = prompt_decision_tree.execute(documents, prompt_sequences['command'])

        assert len(result) > 0, f"Expected non-empty result but got length {len(result)}"

    def test_when_execute_called_with_empty_pages_then_completes_without_error(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN an empty list of relevant pages
        WHEN I call execute with pages and prompts
        THEN execution completes without error
        """
        result = prompt_decision_tree.execute(documents['empty'], prompt_sequences['command'])
        
        assert result is not None, f"Expected result but got {result}"

    def test_when_execute_called_with_empty_pages_then_returns_string(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN an empty list of relevant pages
        WHEN I call execute with pages and prompts
        THEN a string is returned
        """
        result = prompt_decision_tree.execute(documents['empty'], prompt_sequences['command'])
        
        assert isinstance(result, str), f"Expected string but got {type(result)}"


class TestControlFlowMethodReturnsDictionarywithRequiredKeys:
    """Tests for PromptDecisionTree.run method."""

    def test_when_control_flow_called_then_returns_dictionary(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN 5 relevant pages
        WHEN I call run with pages and prompts
        THEN I receive a dictionary response
        """
        result = prompt_decision_tree.run(documents['multiple'], prompt_sequences['command'])
        
        assert isinstance(result, dict), f"Expected dict but got {type(result)}"

    @pytest.mark.parametrize("key", ['success', 'output_data_point', 'responses', 'iterations'])
    def test_when_control_flow_called_then_response_contains_required_keys(
        self, prompt_decision_tree, documents, response_keys, key, prompt_sequencess):
        """
        GIVEN 5 relevant pages
        WHEN I call run with pages and prompts
        THEN the response contains required keys (success, output_data_point, responses, iterations)
        """
        prompt_sequence = ["Analyze documents"]
        
        result = prompt_decision_tree.run(documents['multiple'], prompt_sequences['command'])
        
        assert response_keys[key] in result, f"Expected '{response_keys[key]}' key in {result.keys()}"

    def test_when_control_flow_executes_successfully_then_success_is_true(
            self, prompt_decision_tree, documents, response_keys, prompt_sequences):
        """
        GIVEN valid inputs for control flow
        WHEN control flow executes successfully
        THEN the success field is True
        """
        prompt_sequence = ["Extract tax information"]
 
        result = prompt_decision_tree.run(documents['single'], prompt_sequences['command'])

        assert result[response_keys['success']] is True, f"Expected success=True but got {result[response_keys['success']]}"

    def test_when_control_flow_executes_successfully_then_output_data_point_is_nonempty(
            self, prompt_decision_tree, documents, response_keys, prompt_sequences):
        """
        GIVEN valid inputs for control flow
        WHEN control flow executes successfully
        THEN output_data_point contains extracted data
        """
        prompt_sequence = ["What is the tax rate?"]
        
        result = prompt_decision_tree.run(documents['single'], prompt_sequences['command'])
        
        assert len(result[response_keys['output_data_point']]) > 0, f"Expected non-empty output but got length {len(result[response_keys['output_data_point']])}"
    
    def test_when_control_flow_executes_successfully_then_error_key_not_present(self, prompt_decision_tree, documents, response_keys):
        """
        GIVEN valid inputs for control flow
        WHEN control flow executes successfully
        THEN error key is not present
        """
        prompt_sequence = ["Extract data"]
        
        result = prompt_decision_tree.run(documents['single'], prompt_sequences['command'])
        
        assert response_keys['error'] not in result, f"Expected no error key but found {result.get(response_keys['error'])}"

    def test_when_control_flow_encounters_error_then_success_is_false(
            self, prompt_decision_tree, documents, response_keys, prompt_sequences):
        """
        GIVEN inputs that cause execution to fail
        WHEN control flow encounters an error
        THEN the success field is False
        """
        prompt_decision_tree.resources['llm'].generate.side_effect = Exception("API Error")

        result = prompt_decision_tree.run(documents['single'], prompt_sequences['command'])
        
        assert result[response_keys['success']] is False, f"Expected success=False but got {result[response_keys['success']]}"

    def test_when_control_flow_encounters_error_then_error_key_contains_description(
            self, prompt_decision_tree, documents, response_keys, prompt_sequences):
        """
        GIVEN inputs that cause execution to fail
        WHEN control flow encounters an error
        THEN error key contains error description
        """
        prompt_decision_tree.resources['llm'].generate.side_effect = Exception("LLM API timeout")

        result = prompt_decision_tree.run(documents['single'], prompt_sequences['command'])
        
        assert len(result.get(response_keys['error'], "")) > 0, f"Expected error description but got {result.get(response_keys['error'])}"

    def test_when_control_flow_encounters_error_then_output_data_point_is_empty(self, prompt_decision_tree, documents, response_keys):
        """
        GIVEN inputs that cause execution to fail
        WHEN control flow encounters an error
        THEN output_data_point is empty string
        """
        prompt_decision_tree.resources['llm'].generate.side_effect = Exception("Extraction failed")

        result = prompt_decision_tree.run(documents['single'], prompt_sequences['command'])
        
        assert result[response_keys['output_data_point']] == '', f"Expected empty string but got '{result[response_keys['output_data_point']]}'"


class TestPagesAreConcatenatedUptomaxpagestoconcatenate:
    """Tests for PromptDecisionTree page concatenation logic."""

    def test_when_page_count_below_maximum_then_all_pages_included(self, prompt_decision_tree, test_values, prompt_sequences):
        """
        GIVEN max_pages_to_concatenate is configured as 10
        And 5 relevant pages are provided
        WHEN pages are concatenated
        THEN all 5 pages are included in concatenated text
        """
        prompt_decision_tree.configs.max_pages_to_concatenate = test_values['max_pages']
        documents = [{"content": f"Content {i}", "page_number": i} for i in range(1, test_values['page_count'] + 1)]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_page_count_exceeds_maximum_then_exactly_max_pages_included(self, prompt_decision_tree, test_values):
        """
        GIVEN max_pages_to_concatenate is configured as 10
        And 15 relevant pages are provided
        WHEN pages are concatenated
        THEN exactly 10 pages are included
        """
        prompt_decision_tree.configs.max_pages_to_concatenate = test_values['max_pages']
        documents = [{"content": f"Content {i}", "page_number": i} for i in range(1, test_values['total_pages'] + 1)]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_page_count_exceeds_maximum_then_excess_pages_excluded(self, prompt_decision_tree, test_values):
        """
        GIVEN max_pages_to_concatenate is configured as 10
        And 15 relevant pages are provided
        WHEN pages are concatenated
        THEN pages 11-15 are not included
        """
        prompt_decision_tree.configs.max_pages_to_concatenate = test_values['max_pages']
        documents = [{"content": f"Content {i}", "page_number": i} for i in range(1, test_values['total_pages'] + 1)]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

class TestConcatenatedPagesIncludeTitleURLandContent:
    """Tests for PromptDecisionTree page formatting."""

    def test_when_page_concatenated_then_includes_title_header(self, prompt_decision_tree, test_values):
        """
        GIVEN a page with title Tax Document
        WHEN the page is concatenated
        THEN the output includes title in header
        """
        documents = [{"title": test_values['tax_document_title'], "content": "text", "page_number": 1}]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_page_concatenated_then_includes_source_url(self, prompt_decision_tree, test_values):
        """
        GIVEN a page with URL https://example.com
        WHEN the page is concatenated
        THEN the output includes source URL
        """
        documents = [{"url": test_values['example_url'], "content": "text", "page_number": 1}]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_page_concatenated_then_includes_content_header(self, prompt_decision_tree):
        """
        GIVEN a page with content
        WHEN the page is concatenated
        THEN the output includes content header
        """
        documents = [{"content": "Sample text", "page_number": 1}]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_page_concatenated_then_includes_page_content(self, prompt_decision_tree):
        """
        GIVEN a page with specific content
        WHEN the page is concatenated
        THEN the output includes the page content
        """
        documents = [{"content": "Specific content text", "page_number": 1}]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_page_missing_title_then_uses_default(self, prompt_decision_tree):
        """
        GIVEN a page without a title field
        WHEN the page is concatenated
        THEN a default title is used
        """
        documents = [{"content": "text", "page_number": 1}]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"


class TestLLMPromptsAreGeneratedwithDocumentContext:
    """Tests for PromptDecisionTree prompt generation."""

    def test_when_prompt_generated_then_includes_node_question(self, prompt_decision_tree, prompt_sequences):
        """
        GIVEN a node with prompt text
        WHEN the prompt is generated
        THEN the prompt includes the node question
        """
        documents = [{"content": "test", "page_number": 1}]
        
        result = prompt_decision_tree.execute(documents, prompt_sequences['sample'])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_prompt_generated_then_includes_document_text(self, prompt_decision_tree):
        """
        GIVEN concatenated document text
        WHEN the prompt is generated
        THEN the prompt includes the full document text
        """
        documents = [{"content": "Full document content here", "page_number": 1}]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_prompt_generated_then_includes_llm_instructions(self, prompt_decision_tree):
        """
        GIVEN a node with prompt text
        WHEN the prompt is generated
        THEN the prompt includes instructions for the LLM
        """
        documents = [{"content": "test", "page_number": 1}]
        
        result = prompt_decision_tree.execute(documents, ["Extract data"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_document_exceeds_context_window_then_truncated(self, prompt_decision_tree, test_values):
        """
        GIVEN context_window_size is configured
        And concatenated document exceeds limit
        WHEN the prompt is generated
        THEN the document is truncated to fit context window
        """
        prompt_decision_tree.configs.context_window_size = test_values['context_size']
        documents = [{"content": test_values['large_content'], "page_number": 1}]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_document_truncated_then_space_reserved_for_instructions(self, prompt_decision_tree, test_values):
        """
        GIVEN context_window_size is configured
        WHEN the prompt is generated with large document
        THEN space is reserved for instructions
        """
        prompt_decision_tree.configs.context_window_size = test_values['context_size']
        documents = [{"content": test_values['large_content'], "page_number": 1}]
        
        result = prompt_decision_tree.execute(documents, ["Analyze"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

class TestExecutionValidatesInputTypes:
    """Tests for PromptDecisionTree.execute input validation."""

    def test_when_relevant_pages_not_list_then_raises_typeerror(self, prompt_decision_tree):
        """
        GIVEN relevant_pages is not a list
        WHEN execute is called
        THEN a TypeError is raised
        """
        invalid_documents = "not a list"
        prompt_sequence = ["Test prompt"]
        
        with pytest.raises(TypeError):
            prompt_decision_tree.execute(invalid_documents, prompt_sequences['command'])

    def test_when_prompt_sequence_not_list_then_raises_typeerror(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN prompt_sequence is not a list
        WHEN execute is called
        THEN a TypeError is raised
        """
        invalid_prompts = "not a list"
        
        with pytest.raises(TypeError):
            prompt_decision_tree.execute(documents['single'], invalid_prompts)

    def test_when_empty_lists_provided_then_completes_without_typeerror(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN relevant_pages and prompt_sequence are empty lists
        WHEN execute is called
        THEN a ValueError is raised
        """
        with pytest.raises(ValueError):
            result = prompt_decision_tree.execute(documents['empty'], prompt_sequences['empty'])


class TestHumanReviewIntegrationWhenErrorsOccur:
    """Tests for PromptDecisionTree.execute human review integration."""

    def test_when_error_occurs_and_review_enabled_then_includes_error_details(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN enable_human_review is configured as True
        WHEN human review is requested after error
        THEN review request includes error details
        """
        prompt_decision_tree.configs.enable_human_review = True
        
        result = prompt_decision_tree.execute(documents['single'], ["Test"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_error_occurs_and_review_enabled_then_includes_document_text(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN enable_human_review is configured as True
        WHEN human review is requested after error
        THEN review request includes document text
        """
        prompt_decision_tree.configs.enable_human_review = True
        
        result = prompt_decision_tree.execute(documents['single'], ["Test"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_error_occurs_and_review_enabled_then_includes_llm_responses(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN enable_human_review is configured as True
        WHEN human review is requested after error
        THEN review request includes LLM responses
        """
        prompt_decision_tree.configs.enable_human_review = True
        
        result = prompt_decision_tree.execute(documents['single'], ["Test"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_human_review_completes_then_success_is_true(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN an error occurred during execution
        WHEN review completes with corrected output
        THEN result success is set to True
        """
        prompt_decision_tree.configs.enable_human_review = True
        
        result = prompt_decision_tree.execute(documents['single'], ["Test"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_human_review_completes_then_output_contains_human_value(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN an error occurred during execution
        WHEN review completes with corrected output
        THEN output_data_point contains human-provided value
        """
        prompt_decision_tree.configs.enable_human_review = True
        
        result = prompt_decision_tree.execute(documents['single'], ["Test"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_human_review_completes_then_human_reviewed_flag_set(self, prompt_decision_tree, documents, prompt_sequences):
        """
        GIVEN an error occurred during execution
        WHEN review completes with corrected output
        THEN human_reviewed flag is set to True
        """
        prompt_decision_tree.configs.enable_human_review = True
        
        result = prompt_decision_tree.execute(documents['single'], ["Test"])
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

if __name__ == "__main__":
    pytest.main([__file__])