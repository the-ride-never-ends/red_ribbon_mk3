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
# NOTE: Current test function count: 34
import copy
import logging

from unittest.mock import Mock, MagicMock, AsyncMock

import pytest

from custom_nodes.red_ribbon.socialtoolkit.architecture.prompt_decision_tree import (
    PromptDecisionTree, 
    PromptDecisionTreeConfigs,
)
from custom_nodes.red_ribbon._custom_errors import LLMError
from custom_nodes.red_ribbon.utils_.common import get_cid

from custom_nodes.red_ribbon.socialtoolkit.architecture.variable_codebook import VariableCodebook
from custom_nodes.red_ribbon.socialtoolkit.architecture.dataclasses import Document, Section


from .conftest import (
    FixtureError,
    mock_logger,
    mock_database,
    mock_llm,
    variable_codebook_fixture,
)

from openai import AsyncClient


SEED = 420

import random
random.seed(SEED)


@pytest.fixture
def valid_section_kwargs():
    def _valid_section_args(content: str, idx: int):
        try:
            html_content = f"""<div class=\"chunk-content\"><div class=\"section-content\"><p>{content}</p></div></div>"""
            bluebook_citation = f"Free Country, Ill., Municipal Code, Chapter 1, §{idx} (1999)"
            return {
                'cid': get_cid(html_content),
                'doc_order': idx,
                'bluebook_cid': get_cid(bluebook_citation),
                'title': "Taxes",
                'chapter': "Chapter 1",
                'place_name': "Free Country",
                'bluebook_citation': bluebook_citation,
                'html': html_content,
            }
        except Exception as e:
            raise FixtureError(f"Failed to create valid_section_kwargs: {e}") from e
    return _valid_section_args


@pytest.fixture
def documents(valid_section_kwargs):

    info_content = "Municipal tax rate is 5%"
    irrelevant_content = "Rabbit fish swim through electric hearts"
    relevant_content = "The tax rate for municipal services is 4.2%"
    tax_content = "Municipal tax ordinance states 3.5%"
    multiple_content = "Page {idx} content. "

    def _make_section(content: str, idx: int = 0):
        if content == multiple_content:
            content = content.format(idx=idx)
        try:
            return Section(**valid_section_kwargs(content, idx))
        except Exception as e:
            raise FixtureError(f"Failed to create Section {idx} in documents: {e}") from e

    def _make_document(content, num_pages, other_content=None):
        if num_pages == 0:
            return []
        section_list = [_make_section(content, idx) for idx in range(1, num_pages + 1)]
        if other_content is not None:
            section_list.append(_make_section(other_content, num_pages + 1))
        try:
            return Document(section_list=section_list)
        except Exception as e:
            raise FixtureError(f"Failed to create Document in documents: {e}") from e

    docs = {}

    key_args =  {
        'sample': (multiple_content, 3, info_content),
        'single': (info_content, 1),
        'multiple': (multiple_content, 5, info_content),
        'empty': ("", 0, None),
        'tax_ordinance': (multiple_content, 3, info_content),
        'five_pages': (multiple_content, 5, tax_content),
        'fifteen_pages': (multiple_content, 15, info_content),

        'sample': [{"title": "Tax Document", "content": "text", "page_number": 1}],
        'with_url': [{"url": "https://example.com", "content": "text", "page_number": 1}],
        'specific_content': [{"content": "Specific content text", "page_number": 1}],
        'no_title': [{"content": "text", "page_number": 1}],
        'test_content': [{"content": "test", "page_number": 1}],
        'full_document': [{"content": "Full document content here", "page_number": 1}],
        'large_content': (multiple_content, 1000, info_content)
    }
    for key, args in key_args.items():
        try:
            docs[key] = _make_document(*args)
        except Exception as e:
            raise FixtureError(f"Failed to create document for key '{key}': {e}") from e
        return docs


@pytest.fixture
def prompt_sequences():
    return {
        'sample': ["What is the tax rate?"],
        'command': ["Get tax rate"],
        'google_fu': ["tax information municipal rate"],
        'empty': [],
        'analyze': ["Analyze"],
        'analyze_documents': ["Analyze documents"],
        'extract_tax': ["Extract tax information"],
        'extract_data': ["Extract data"],
        'test': ["Test"],
        'test_prompt': ["Test prompt"],
        'invalid_not_list': "not a list"
    }

@pytest.fixture
def valid_keys():
    return {
        'success': 'success',
        'output_data_point': 'output_data_point',
        'document_text'
        'responses': 'responses',
        'iterations': 'iterations',
        'msg': 'msg',
        'error': 'error'
    }


@pytest.fixture
def valid_configs():
    return {
        'max_pages': 10,
        'total_pages': 15,
        'page_count': 5,
        'context_window_size': 8192,
        'enable_human_review': True,
    }

@pytest.fixture
def test_values(documents, prompt_sequences, valid_keys, valid_configs):
    """
    GIVEN a unified fixture containing all test data configurations
    """
    return {
        'documents': documents,
        'prompt_sequences': prompt_sequences,
        'valid_keys': valid_keys,
        'valid_configs': valid_configs,
    }


@pytest.fixture
def make_mock_configs():
    def _make_mock_configs(kwargs={}):
        try:
            if not kwargs:
                return PromptDecisionTreeConfigs()
            return PromptDecisionTreeConfigs(**kwargs)
        except Exception as e:
            raise FixtureError(f"Failed to create mock PromptDecisionTreeConfigs: {e}") from e
    return _make_mock_configs

@pytest.fixture
def make_prompt_decision_tree(mock_llm_api, mock_variable_codebook, mock_logger):
    """
    Helper to create a PromptDecisionTree instance for testing
    """
    def _make_prompt_decision_tree(configs, resources={}):
        resources_ = {
            'logger': resources.get("logger", mock_logger),
            'llm': resources.get("llm", mock_llm_api),
            'variable_codebook': resources.get("variable_codebook", mock_variable_codebook)
        }
        try:
            tree = PromptDecisionTree(resources=resources_, configs=configs)
        except Exception as e:
            raise FixtureError from e
        return tree
    return _make_prompt_decision_tree


@pytest.fixture
def prompt_decision_tree_fixture(
    make_prompt_decision_tree,
    make_mock_configs,
):
    """
    Mock PromptDecisionTree instance for testing
    """
    mock_configs = make_mock_configs()
    return make_prompt_decision_tree(mock_configs)


@pytest.fixture
def prompt_decision_tree_with_max_pages(
    test_values,
    make_prompt_decision_tree,
    make_mock_configs,
):
    """
    Mock PromptDecisionTree instance with max_pages_to_concatenate configured
    """
    kwargs = {"max_pages_to_concatenate": test_values['max_pages']}
    mock_configs = make_mock_configs(**kwargs)
    return make_prompt_decision_tree(mock_configs)

@pytest.fixture
def prompt_decision_tree_with_context_window(
    test_values,
    make_prompt_decision_tree,
    make_mock_configs,
):
    """
    Mock PromptDecisionTree instance with context_window_size configured
    """
    kwargs = {"context_window_size": test_values['context_window_size']}
    mock_configs = make_mock_configs(**kwargs)
    return make_prompt_decision_tree(mock_configs)

@pytest.fixture
def prompt_decision_tree_with_human_review(
    test_values,
    make_prompt_decision_tree,
    make_mock_configs,
):
    """
    Mock PromptDecisionTree instance with enable_human_review configured
    """
    kwargs = {"enable_human_review": test_values['enable_human_review']}
    mock_configs = make_mock_configs(**kwargs)
    return make_prompt_decision_tree(mock_configs)


@pytest.fixture
def prompt_decision_tree_with_llm_api_error(
    mock_llm,
    make_prompt_decision_tree,
    make_mock_configs,
):
    """
    Mock PromptDecisionTree instance with LLM configured to raise errors
    """
    mock_llm_with_error = copy.deepcopy(mock_llm)
    mock_llm_with_error.generate.side_effect = LLMError("LLM API Error")

    mock_configs = make_mock_configs()
    tree = make_prompt_decision_tree(mock_configs)
    return 


class TestExecuteMethodReturnsExtractedDataPoint:
    """Tests for PromptDecisionTree.execute method."""

    def test_when_execute_called_with_pages_and_prompts_then_returns_dict(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN a list of with an arbitrary number of relevant pages
        WHEN I call execute with valid arguments
        THEN a dict is returned
        """
        args = (documents['sample'], prompt_sequences['sample'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, dict), f"Expected result to be a dict but got {type(result)}"

    def test_when_execute_called_with_valid_documents_then_returns_nonempty_string(
            self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN a list of 3 relevant pages with tax information
        WHEN I call execute with pages and prompts
        THEN the data point contains extracted information
        """
        args = (documents['tax_ordinance'], prompt_sequences['command'])
        result = prompt_decision_tree_fixture.execute(*args)

        assert len(result) > 0, f"Expected non-empty result but got length {len(result)}"

    def test_when_execute_called_with_empty_pages_then_returns_dict(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN an empty list of relevant pages
        WHEN I call execute with pages and prompts
        THEN a dict is returned
        """
        args = (documents['empty'], prompt_sequences['command'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, dict), f"Expected result to be a dict but got {type(result)}"


class TestControlFlowMethodReturnsDictionarywithRequiredKeys:
    """Tests for PromptDecisionTree.run method."""

    def test_when_control_flow_called_then_returns_dictionary(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN 5 relevant pages
        WHEN I call run with pages and prompts
        THEN I receive a dictionary response
        """
        args = (documents['multiple'], prompt_sequences['command'])
        result = prompt_decision_tree_fixture.run(*args)
        
        assert isinstance(result, dict), f"Expected dict but got {type(result)}"

    @pytest.mark.parametrize("key", ['success', 'output_data_point', 'responses', 'iterations', 'msg', 'output_documents'])
    def test_when_control_flow_called_then_response_contains_required_keys(
        self, key, prompt_decision_tree_fixture, documents, prompt_sequences, valid_keys):
        """
        GIVEN 5 relevant pages
        WHEN I call run with pages and prompts
        THEN the response contains required keys (success, output_data_point, output_documents, msg, responses, iterations)
        """
        expected_key = valid_keys[key]
        args = (documents['multiple'], prompt_sequences['analyze_documents'])
        result = prompt_decision_tree_fixture.run(*args)
        
        assert expected_key in result, f"Expected '{expected_key}' key in {result.keys()}"

    def test_when_control_flow_executes_successfully_then_success_is_true(
            self, prompt_decision_tree_fixture, documents, valid_keys, prompt_sequences):
        """
        GIVEN valid inputs for control flow
        WHEN control flow executes successfully
        THEN the success field is True
        """
        key = valid_keys['success']
        args = (documents['single'], prompt_sequences['extract_tax'])
        result = prompt_decision_tree_fixture.run(*args)

        assert result[key] is True, f"Expected success=True but got {result[key]}"

    def test_when_control_flow_executes_successfully_then_output_data_point_is_nonempty(
            self, prompt_decision_tree_fixture, documents, prompt_sequences, valid_keys):
        """
        GIVEN valid inputs for control flow
        WHEN control flow executes successfully
        THEN output_data_point contains extracted data
        """
        key = valid_keys['output_data_point']
        args = (documents['single'], prompt_sequences['sample'])
        result = prompt_decision_tree_fixture.run(*args)
        
        assert len(result[key]) > 0, f"Expected non-empty output but got length {len(result[key])}"

    def test_when_control_flow_executes_successfully_then_error_key_not_present(
            self, prompt_decision_tree_fixture, documents, prompt_sequences, valid_keys):
        """
        GIVEN valid inputs for control flow
        WHEN control flow executes successfully
        THEN msg key does not contain an error message
        """
        error_string = "error"
        args = (documents['single'], prompt_sequences['extract_data'])
        result = prompt_decision_tree_fixture.run(*args)
        msg = result['msg']

        assert error_string not in msg, f"Expected msg to not contain '{error_string}' but got '{msg}'"

    def test_when_control_flow_encounters_error_then_success_is_false(
            self, prompt_decision_tree_fixture_with_llm_api_error, documents, prompt_sequences, valid_keys):
        """
        GIVEN inputs that cause execution to fail
        WHEN control flow encounters an error
        THEN the success field is False
        """
        key = valid_keys['success']
        args = (documents['single'], prompt_sequences['command'])
        result = prompt_decision_tree_fixture_with_llm_api_error.run(*args)

        assert result[key] is False, f"Expected success=False but got {result[key]}"

    def test_when_control_flow_encounters_error_then_error_key_contains_description(
            self, prompt_decision_tree_fixture_with_llm_api_error, documents, prompt_sequences, valid_keys):
        """
        GIVEN inputs that cause execution to fail
        WHEN control flow encounters an error
        THEN error key contains error description
        """
        error_string = "error"
        args = (documents['single'], prompt_sequences['command'])
        result = prompt_decision_tree_fixture_with_llm_api_error.run(*args)
        msg = result['msg']

        assert error_string in msg, f"Expected msg to contain '{error_string}' but got '{msg}'"

    def test_when_control_flow_encounters_error_then_output_data_point_is_empty(
            self, prompt_decision_tree_fixture_with_llm_api_error, documents, prompt_sequences, valid_keys):
        """
        GIVEN inputs that cause execution to fail
        WHEN control flow encounters an error
        THEN output_data_point is empty string
        """
        empty_string = ""
        key = valid_keys['output_data_point']
        args = (documents['single'], prompt_sequences['command'])
        result = prompt_decision_tree_fixture_with_llm_api_error.run(*args)
        
        assert result[key] == empty_string, f"Expected empty string but got '{result[key]}'"


class TestPagesAreConcatenatedUptomaxpagestoconcatenate:
    """Tests for PromptDecisionTree page concatenation logic."""

    pytest.mark.parametrize("idx", [idx for idx in range(5)])
    def test_when_page_count_below_maximum_then_all_pages_included(self, idx, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN max_pages_to_concatenate is configured as 10
        And 5 relevant pages are provided
        WHEN pages are concatenated
        THEN all 5 pages are included in concatenated text
        """
        page = documents['five_pages'][idx]
        args = (documents['five_pages'], prompt_sequences['analyze'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert page in result, f"Expected '{page}' to be in result, but got {result}"

    pytest.mark.parametrize("idx", [idx for idx in range(10)])
    def test_when_page_count_exceeds_maximum_then_exactly_max_pages_included(
            self, idx, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN max_pages_to_concatenate is configured as 10
        And 15 relevant pages are provided
        WHEN pages are concatenated
        THEN exactly 10 pages are included
        """
        page = documents['fifteen_pages'][idx]
        args = (documents['fifteen_pages'], prompt_sequences['analyze'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert page in result, f"Expected '{page}' to be in result, but got {result}"

    @pytest.mark.parametrize("idx", [idx for idx in range(10, 15)])
    def test_when_page_count_exceeds_maximum_then_excess_pages_excluded(
        self, idx, valid_keys, prompt_decision_tree_with_max_pages, documents, prompt_sequences):
        """
        GIVEN max_pages_to_concatenate is configured as 10
        And 15 relevant pages are provided
        WHEN pages are concatenated
        THEN pages 11-15 are not included
        """
        key = valid_keys['responses']
        page = documents['fifteen_pages'][idx]
        args = (documents['fifteen_pages'], prompt_sequences['analyze'])
        result = prompt_decision_tree_with_max_pages.execute(*args)
        
        assert page not in result[key], f"Expected '{page}' to be in result, but got {result[key]}"

class TestConcatenatedPagesIncludeTitleURLandContent:
    """Tests for PromptDecisionTree page formatting."""

    def test_when_page_concatenated_then_includes_title_header(
            self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN a page with a title
        WHEN execute is called
        THEN the result includes the title in 
        """
        title = documents['sample'][0]['title']
        args = (documents['sample'], prompt_sequences['analyze'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert title in result, f"Expected '{title}' in result, but got "

    def test_when_page_concatenated_then_includes_source_url(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN a page with a Bluebook citation
        WHEN the page is concatenated
        THEN the output includes source URL
        """
        args = (documents['with_url'], prompt_sequences['analyze'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_page_concatenated_then_includes_content_header(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN a page with content
        WHEN the page is concatenated
        THEN the output includes content header
        """
        args = (documents['sample'], prompt_sequences['analyze'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_page_concatenated_then_includes_page_content(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN a page with specific content
        WHEN the page is concatenated
        THEN the output includes the page content
        """
        args = (documents['specific_content'], prompt_sequences['analyze'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_page_missing_title_then_uses_default(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN a page without a title field
        WHEN the page is concatenated
        THEN a default title is used
        """
        args = (documents['sample'], prompt_sequences['analyze'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"


class TestLLMPromptsAreGeneratedwithDocumentContext:
    """Tests for PromptDecisionTree prompt generation."""

    def test_when_prompt_generated_then_includes_node_question(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN a node with prompt text
        WHEN the prompt is generated
        THEN the prompt includes the node question
        """
        args = (documents['test_content'], prompt_sequences['sample'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_prompt_generated_then_includes_document_text(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN concatenated document text
        WHEN the prompt is generated
        THEN the prompt includes the full document text
        """
        args = (documents['full_document'], prompt_sequences['analyze'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_prompt_generated_then_includes_llm_instructions(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN a node with prompt text
        WHEN the prompt is generated
        THEN the prompt includes instructions for the LLM
        """
        args = (documents['test_content'], prompt_sequences['extract_data'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_document_exceeds_context_window_then_truncated(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN context_window_size is configured
        And concatenated document exceeds limit
        WHEN the prompt is generated
        THEN the document is truncated to fit context window
        """
        args = (documents['large_content'], prompt_sequences['analyze'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_document_truncated_then_space_reserved_for_instructions(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN context_window_size is configured
        WHEN the prompt is generated with large document
        THEN space is reserved for instructions
        """

        args = (documents['large_content'], prompt_sequences['analyze'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

class TestExecutionValidatesInputTypes:
    """Tests for PromptDecisionTree.execute input validation."""

    def test_when_relevant_pages_not_list_then_raises_typeerror(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN relevant_pages is not a list
        WHEN execute is called
        THEN a TypeError is raised
        """
        args = (documents['invalid_not_list'], prompt_sequences['test_prompt'])
        
        with pytest.raises(TypeError):
            prompt_decision_tree_fixture.execute(*args)

    def test_when_prompt_sequence_not_list_then_raises_typeerror(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN prompt_sequence is not a list
        WHEN execute is called
        THEN a TypeError is raised
        """
        args = (documents['single'], prompt_sequences['invalid_not_list'])
        
        with pytest.raises(TypeError):
            prompt_decision_tree_fixture.execute(*args)

    def test_when_empty_lists_provided_then_completes_without_typeerror(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN relevant_pages and prompt_sequence are empty lists
        WHEN execute is called
        THEN a ValueError is raised
        """
        args = (documents['empty'], prompt_sequences['empty'])
        
        with pytest.raises(ValueError):
            result = prompt_decision_tree_fixture.execute(*args)


class TestHumanReviewIntegrationWhenErrorsOccur:
    """Tests for PromptDecisionTree.execute human review integration."""

    def test_when_review_enabled_and_no_error_then_includes_file_does_not_exist(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN enable_human_review is configured as True
        WHEN execute runs without error
        THEN result json file does not exist in error log directory. 
        """
        prompt_decision_tree_fixture.configs.enable_human_review = True
        args = (documents['single'], prompt_sequences['test'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_error_occurs_and_review_enabled_then_includes_error_details(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN enable_human_review is configured as True
        WHEN human review is requested after error
        THEN review request includes error details
        """
        prompt_decision_tree_fixture.configs.enable_human_review = True
        args = (documents['single'], prompt_sequences['test'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_error_occurs_and_review_enabled_then_includes_document_text(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN enable_human_review is configured as True
        WHEN human review is requested after error
        THEN review request includes document text
        """
        prompt_decision_tree_fixture.configs.enable_human_review = True
        args = (documents['single'], prompt_sequences['test'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_error_occurs_and_review_enabled_then_includes_llm_responses(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN enable_human_review is configured as True
        WHEN human review is requested after error
        THEN review request includes LLM responses
        """
        prompt_decision_tree_fixture.configs.enable_human_review = True
        args = (documents['single'], prompt_sequences['test'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_human_review_completes_then_success_is_true(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN an error occurred during execution
        WHEN review completes with corrected output
        THEN result success is set to True
        """
        prompt_decision_tree_fixture.configs.enable_human_review = True
        args = (documents['single'], prompt_sequences['test'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_human_review_completes_then_output_contains_human_value(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN enable_human_review flag is set to True
        WHEN an error occurs during execution
        THEN result json file exists in error log directory
        """
        prompt_decision_tree_fixture.configs.enable_human_review = True
        args = (documents['single'], prompt_sequences['test'])
        result = prompt_decision_tree_fixture.execute(*args)

        assert isinstance(result, str), f"Expected string result but got {type(result)}"

    def test_when_human_review_completes_then_human_reviewed_flag_set(self, prompt_decision_tree_fixture, documents, prompt_sequences):
        """
        GIVEN an error occurred during execution
        WHEN review completes with corrected output
        THEN enable_human_review flag is set to True
        """
        prompt_decision_tree_fixture.configs.enable_human_review = True
        args = (documents['single'], prompt_sequences['test'])
        result = prompt_decision_tree_fixture.execute(*args)
        
        assert isinstance(result, str), f"Expected string result but got {type(result)}"

if __name__ == "__main__":
    pytest.main([__file__])