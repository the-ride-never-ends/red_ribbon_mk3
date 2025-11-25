#!/usr/bin/env python3
"""
Feature: Relevance Assessment
  As a document analysis system
  I want to assess and filter documents based on relevance to a variable definition
  So that only pertinent information is used for data extraction

  Background:
    Given a RelevanceAssessment instance is initialized
    And a variable codebook service is available
    And a top10 retrieval service is available
    And a cited page extractor service is available
    And a prompt decision tree service is available
    And an LLM API client is available
"""
from unittest.mock import Mock, AsyncMock


from pydantic import ValidationError
import pytest


from custom_nodes.red_ribbon.socialtoolkit.architecture.dataclasses import (
    Document, Section, make_document, make_section, CID
)
from custom_nodes.red_ribbon.socialtoolkit.architecture.variable_codebook import Variable
from custom_nodes.red_ribbon.socialtoolkit.architecture.prompt_decision_tree import PromptDecisionTree
from custom_nodes.red_ribbon.socialtoolkit.architecture.relevance_assessment import (
    RelevanceAssessment,
    make_relevance_assessment,
)
from custom_nodes.red_ribbon.utils_.common import get_cid
from .conftest import FixtureError



@pytest.fixture
def constants():
    """Provide standardized constants for test methods."""
    return {
        'NUM_TEST_DOCUMENTS': 10,
        'NUM_SMALL_TEST_DOCUMENTS': 5,
        'NUM_ONE_DOCUMENT': 1,
        'CRITERIA_THRESHOLD_HIGH': 0.7,
        'CRITERIA_THRESHOLD_MEDIUM': 0.5,
        'CRITERIA_THRESHOLD_VERY_HIGH': 0.9,
        'MAX_CITATION_LENGTH': 500,
        'CITATION_SHORT_LENGTH': 200,
        'CITATION_LONG_LENGTH': 1000,
        'MAX_RETRIES': 3,
        'EXPECTED_RELEVANT_COUNT_ZERO': 0,
        'EXPECTED_RELEVANT_COUNT_ONE': 1,
        'EXPECTED_RELEVANT_COUNT_TWO': 2,
        'EXPECTED_RELEVANT_COUNT_THREE': 3,
        'EXPECTED_RELEVANT_COUNT_FIVE': 5,
        'EXPECTED_RELEVANT_COUNT_SIX': 6,
        'EXPECTED_RELEVANT_COUNT_EIGHT': 8,
        'NUM_DOCUMENTS_FIFTEEN': 15,
        'CONFIDENCE_HIGH': 0.8,
        'CONFIDENCE_MEDIUM': 0.6,
        'CONFIDENCE_LOW': 0.3,
        'CONFIDENCE_VERY_LOW': 0.2,
        'CONFIDENCE_VERY_HIGH': 0.9,
        'CONFIDENCE_ZERO': 0.0,
        'CONFIDENCE_MAX': 1.0,
        'CONFIDENCE_MIN': 0.0,
        'TRUNCATION_INDICATOR': '...',
        'TRUNCATION_INDICATOR_LENGTH': 3,
        'EXCESS_CHARS': 100,
        'DOC_ID_ZERO': 'doc0',
        'DOC_ID_ONE': 'doc1',
        'DOC_ID_TWO': 'doc2',
        'VARIABLE_NAME': 'test_variable',
        'RESPONSE_RELEVANT_YES': 'RELEVANT: Yes\nCONFIDENCE: 0.8\nCITATION: Test citation\nREASONING: Test reasoning',
        'RESPONSE_RELEVANT_NO': 'RELEVANT: No\nCONFIDENCE: 0.3\nCITATION: Low relevance citation\nREASONING: Not relevant',
        'SHORT_CITATION': 'This is a short citation that fits within the limit.',
        'ERROR_MESSAGE_PERSISTENT': 'Persistent API Error',
        'ERROR_MESSAGE_API': 'API Error',
        'TYPE_ERROR_LIST_MESSAGE': 'list',
        'TYPE_ERROR_DICT_MESSAGE': 'dict',
        'NON_LIST_VALUE': 'not_a_list',
        'NON_DICT_VALUE': 'not_a_dict',
        'EMPTY_LIST': [],
        'EXPECTED_DICT_KEYS': ['relevant_pages', 'relevant_doc_ids', 'page_numbers', 'relevance_scores'],
    }


@pytest.fixture
def make_relevance_assessment_factory():
    """Factory for creating relevance assessment instances."""
    def _factory():
        resources = {
            "llm_api": AsyncMock(),
            "variable_codebook_service": Mock(),
            "top10_retrieval_service": Mock(),
            "cited_page_extractor_service": Mock(),
            "prompt_decision_tree_service": Mock(),
            "llm": Mock(),
        }
        try:
            return make_relevance_assessment(resources=resources)
        except Exception as e:
            raise FixtureError(f"Error creating RelevanceAssessment instance: {e}") from e
    return _factory


@pytest.fixture
def valid_variable_args():
    """Provide a valid variable definition dictionary."""
    return {
        "label": "test_variable",
        "item_name": "Test Variable",
        "description": "A test variable for relevance assessment.",
        "units": "units",
    }

@pytest.fixture
def valid_variable_args_with_optionals(valid_variable_args):
    """Provide a valid variable definition dictionary."""
    output_dict = {
        "assumptions": ["Some assumptions"],
        "prompt_decision_tree": {"step1": "What is the value?"},
    }
    output_dict.update(valid_variable_args)
    return output_dict


def make_variable_factory(overrides: dict = {}):
    """Factory for creating Variable instances with overrides."""
    @pytest.fixture
    def _make_variable(valid_variable_args) -> Variable:
        var_data = valid_variable_args.copy()

        if not isinstance(overrides, dict):
            raise FixtureError(f"Overrides must be a dictionary, got {type(overrides).__name__}.")
        var_data.update(overrides)

        try:
            variable = Variable(**var_data)
            return variable
        except Exception as e:
            raise FixtureError(f"Error creating Variable instance: {e}") from e
    return _make_variable


valid_variable = make_variable_factory()


@pytest.fixture
def make_expected_cid():
    def _make_expected_cid(string: str) -> str:
        """Generate expected CID for a given document index."""
        try:
            return get_cid(string)
        except ValidationError as e:
            raise FixtureError(f"Validation error generating expected CID for string '{string}': {e}") from e
        except Exception as e:
            raise FixtureError(f"Error generating expected CID for string '{string}': {e}") from e
    return _make_expected_cid


@pytest.fixture
def relevance_assessment(make_relevance_assessment_factory) -> RelevanceAssessment:
    """Provide a RelevanceAssessment instance."""
    return make_relevance_assessment_factory()


@pytest.fixture
def documents_small(constants):
    """Provide small set of test documents."""
    count = constants['NUM_SMALL_TEST_DOCUMENTS']
    return [
        {"id": f"doc{i}", "content": f"Content {i}"} for i in range(count)
    ]


@pytest.fixture
def variable_def(constants):
    """Provide test variable definition."""
    return {"name": constants['VARIABLE_NAME']}


@pytest.fixture
def mock_llm_high_relevance(constants):
    """Provide mock LLM that returns high relevance."""
    mock = Mock()
    mock.generate.return_value = constants['RESPONSE_RELEVANT_YES']
    return mock


@pytest.fixture
def mock_llm_low_relevance(constants):
    """Provide mock LLM that returns low relevance."""
    mock = Mock()
    mock.generate.return_value = constants['RESPONSE_RELEVANT_NO']
    return mock


@pytest.fixture
def mock_llm_very_high_confidence(constants):
    """Provide mock LLM that returns very high confidence."""
    mock = Mock()
    confidence = constants['CONFIDENCE_VERY_HIGH']
    mock.generate.return_value = f"RELEVANT: Yes\nCONFIDENCE: {confidence}\nCITATION: High relevance citation\nREASONING: Highly relevant"
    return mock


@pytest.fixture
def mock_llm_very_low_confidence(constants):
    """Provide mock LLM that returns very low confidence."""
    mock = Mock()
    confidence = constants['CONFIDENCE_VERY_LOW']
    mock.generate.return_value = f"RELEVANT: No\nCONFIDENCE: {confidence}\nCITATION: Low relevance citation\nREASONING: Not relevant"
    return mock


@pytest.fixture
def mock_llm_long_citation(constants):
    """Provide mock LLM that returns a citation exceeding max length."""
    mock = Mock()
    max_len = constants['MAX_CITATION_LENGTH']
    excess = constants['EXCESS_CHARS']
    long_citation = "A" * (max_len + excess)
    confidence = constants['CONFIDENCE_HIGH']
    mock.generate.return_value = f"RELEVANT: Yes\nCONFIDENCE: {confidence}\nCITATION: {long_citation}\nREASONING: Test reasoning"
    return mock


@pytest.fixture
def mock_llm_short_citation(constants):
    """Provide mock LLM that returns a short citation."""
    mock = Mock()
    short_citation = constants['SHORT_CITATION']
    confidence = constants['CONFIDENCE_HIGH']
    mock.generate.return_value = f"RELEVANT: Yes\nCONFIDENCE: {confidence}\nCITATION: {short_citation}\nREASONING: Test reasoning"
    return mock


@pytest.fixture
def llm_instances(
    mock_llm_high_relevance,
    mock_llm_low_relevance,
    mock_llm_very_high_confidence,
    mock_llm_very_low_confidence,
    mock_llm_long_citation,
    mock_llm_short_citation,
):
    """Provide dictionary of mock LLM instances for different scenarios."""
    return {
        "high_relevance": mock_llm_high_relevance,
        "low_relevance": mock_llm_low_relevance,
        "very_high_confidence": mock_llm_very_high_confidence,
        "very_low_confidence": mock_llm_very_low_confidence,
        "long_citation": mock_llm_long_citation,
        "short_citation": mock_llm_short_citation,
    }


class TestControlFlowMethodReturnsDictionarywithRequiredKeys:
    """Tests run method of RelevanceAssessment."""

    def test_when_calling_control_flow_then_returns_dict(
        self, relevance_assessment, documents_small, variable_def):
        """
        GIVEN a list of potentially relevant documents
        WHEN run is called with documents and variable definition
        THEN expect result to be a dictionary.
        """
        result = relevance_assessment.run(documents_small, variable_def)
        assert isinstance(result, dict), f"Expected result to be dict but got {type(result)}"

    def test_when_calling_control_flow_then_contains_required_keys(
        self, relevance_assessment, make_expected_cid, documents_small, variable_def):
        """
        GIVEN a list of potentially relevant documents
        WHEN run is called with documents and variable definition
        THEN expect response keys to be document CIDs
        """
        expected_cid = make_expected_cid()
        result = relevance_assessment.run(documents_small, variable_def)
        assert expected_cid in result, f"Expected CID '{expected_cid}' in result but got {list(result.keys())}"

    def test_when_calling_control_flow_then_contains_key_values_are_expected_types(
            self, relevance_assessment, make_expected_cid, documents_small, variable_def
    ):
        """
        GIVEN a list of potentially relevant documents
        WHEN run is called with documents and variable definition
        THEN expect the response values to be Document instances.
        """
        expected_cid = make_expected_cid()
        result = relevance_assessment.run(documents_small, variable_def)
        value = result[expected_cid]
        assert isinstance(value, Document), f"Expected value for key '{expected_cid}' to be Document, but got {type(value)}"



class TestRelevanceAssessmentFiltersDocumentsbyCriteriaThreshold:
    """Tests run method filters documents by criteria threshold."""


    def test_when_documents_above_threshold_then_marked_relevant(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance, constants):
        """
        GIVEN criteria_threshold configured and documents with varying relevance scores
        WHEN run assesses the documents
        THEN expect only documents above threshold in relevant_pages.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        expected_count = constants['NUM_SMALL_TEST_DOCUMENTS']
        assert len(result["relevant_pages"]) == expected_count, f"Expected {expected_count} relevant pages but got {len(result['relevant_pages'])}"


    def test_when_documents_above_threshold_then_scores_meet_criteria(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance, constants):
        """
        GIVEN criteria_threshold configured and documents with varying relevance scores
        WHEN run assesses the documents
        THEN expect all relevant document scores to meet or exceed threshold.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        threshold = constants['CRITERIA_THRESHOLD_HIGH']
        min_score = min(item["score"] for item in result["relevance_scores"])
        assert min_score >= threshold, f"Expected minimum score {min_score} to be >= threshold {threshold}"


    def test_when_documents_below_threshold_then_in_discarded_pages(
        self, relevance_assessment, documents_small, variable_def, mock_llm_low_relevance, constants):
        """
        GIVEN criteria_threshold configured and documents with low relevance scores
        WHEN run assesses the documents
        THEN expect documents below threshold in discarded pages.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_low_relevance)
        expected_count = constants['EXPECTED_RELEVANT_COUNT_ZERO']
        assert len(result["relevant_pages"]) == expected_count, f"Expected {expected_count} relevant pages but got {len(result['relevant_pages'])}"


    def test_when_all_documents_exceed_threshold_then_all_relevant(
        self, relevance_assessment, documents_small, variable_def, mock_llm_very_high_confidence, constants):
        """
        GIVEN criteria_threshold configured and all documents with high relevance scores
        WHEN run assesses the documents
        THEN expect all documents in relevant_pages.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_very_high_confidence)
        expected_count = constants['NUM_SMALL_TEST_DOCUMENTS']
        assert len(result["relevant_pages"]) == expected_count, f"Expected {expected_count} relevant pages but got {len(result['relevant_pages'])}"


    def test_when_all_documents_below_threshold_then_all_discarded(
        self, relevance_assessment, documents_small, variable_def, mock_llm_very_low_confidence, constants):
        """
        GIVEN criteria_threshold configured and all documents with low relevance scores
        WHEN run assesses the documents
        THEN expect all documents in discarded pages.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_very_low_confidence)
        expected_count = constants['EXPECTED_RELEVANT_COUNT_ZERO']
        assert len(result["relevant_pages"]) == expected_count, f"Expected {expected_count} relevant pages but got {len(result['relevant_pages'])}"


    def test_when_all_documents_exceed_threshold_then_none_discarded(
        self, relevance_assessment, documents_small, variable_def, mock_llm_very_high_confidence, constants):
        """
        GIVEN criteria_threshold configured and all documents with high relevance scores
        WHEN run assesses the documents
        THEN expect no documents to be discarded.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_very_high_confidence)
        threshold = constants['CRITERIA_THRESHOLD_HIGH']
        min_score = min(item["score"] for item in result["relevance_scores"])
        assert min_score >= threshold, f"Expected all scores >= {threshold} but got minimum {min_score}"

    def test_when_all_documents_below_threshold_then_relevant_pages_empty(
        self, relevance_assessment, documents_small, variable_def, mock_llm_very_low_confidence, constants):
        """
        GIVEN criteria_threshold configured and all documents with low relevance scores
        WHEN run assesses the documents
        THEN expect relevant_pages to be empty list.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_very_low_confidence)
        expected_list = constants['EMPTY_LIST']
        assert result["relevant_pages"] == expected_list, f"Expected empty list but got {result['relevant_pages']}"



class TestHallucinationFilterIsAppliedWhenEnabled:
    """Tests run method applies hallucination filter when enabled."""
    
    def test_when_hallucination_filter_enabled_then_removes_false_information(
        self, relevance_assessment, documents_small, variable_def, constants):
        """
        GIVEN use_hallucination_filter configured as True and results contain hallucinated content
        WHEN run processes the results
        THEN expect hallucinated content to be filtered out.
        """
        result = relevance_assessment.run(documents_small, variable_def)
        expected_count = constants['EXPECTED_RELEVANT_COUNT_ZERO']
        assert len(result["relevant_pages"]) == expected_count, f"Expected {expected_count} relevant pages but got {len(result['relevant_pages'])}"

    def test_when_hallucination_filter_enabled_then_only_verified_remains(
        self, relevance_assessment, documents_small, variable_def, constants):
        """
        GIVEN use_hallucination_filter configured as True and results contain hallucinated content
        WHEN run processes the results
        THEN expect only verified information to remain.
        """
        result = relevance_assessment.run(documents_small, variable_def)
        expected_count = constants['EXPECTED_RELEVANT_COUNT_ONE']
        assert len(result["relevant_pages"]) == expected_count, f"Expected {expected_count} relevant pages but got {len(result['relevant_pages'])}"

    def test_when_hallucination_filter_disabled_then_all_results_retained(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance, constants):
        """
        GIVEN use_hallucination_filter configured as False and results may contain hallucinations
        WHEN run processes the results
        THEN expect all assessment results to be retained.
        """
        relevance_assessment.configs.use_hallucination_filter = False
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        expected_score_count = constants['NUM_SMALL_TEST_DOCUMENTS']
        assert len(result["relevance_scores"]) == expected_score_count, f"Expected {expected_score_count} relevance scores but got {len(result['relevance_scores'])}"


class TestRelevanceScoresAreCalculatedforEachDocument:
    """Tests run method calculates relevance scores for each document."""
    
    def test_when_assessing_documents_then_each_receives_score(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance, constants):
        """
        GIVEN potentially relevant documents
        WHEN relevance is assessed
        THEN expect each document to receive a relevance score.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        expected_count = constants['NUM_SMALL_TEST_DOCUMENTS']
        assert len(result["relevance_scores"]) == expected_count, f"Expected {expected_count} relevance scores but got {len(result['relevance_scores'])}"

    def test_when_assessing_documents_then_scores_within_valid_range(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance, constants):
        """
        GIVEN potentially relevant documents
        WHEN relevance is assessed
        THEN expect each score to be between 0.0 and 1.0.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        max_score = max(item["score"] for item in result["relevance_scores"])
        max_allowed = constants['CONFIDENCE_MAX']
        assert max_score <= max_allowed, f"Expected maximum score {max_score} to be <= {max_allowed}"

    def test_when_assessing_documents_then_scores_reflect_relevance(
        self, relevance_assessment, documents_small, variable_def, constants):
        """
        GIVEN potentially relevant documents
        WHEN relevance is assessed
        THEN expect scores to reflect document relevance to variable.
        """
        mock_llm = Mock()
        mock_llm.generate.side_effect = [
            "RELEVANT: Yes\nCONFIDENCE: 0.9\nCITATION: High\nREASONING: Very relevant",
            "RELEVANT: Yes\nCONFIDENCE: 0.5\nCITATION: Medium\nREASONING: Somewhat relevant",
            "RELEVANT: No\nCONFIDENCE: 0.1\nCITATION: Low\nREASONING: Not relevant",
            "RELEVANT: No\nCONFIDENCE: 0.1\nCITATION: Low\nREASONING: Not relevant",
            "RELEVANT: No\nCONFIDENCE: 0.1\nCITATION: Low\nREASONING: Not relevant",
        ]
        result = relevance_assessment.run(documents_small, variable_def, mock_llm)
        doc0_score = next(s["score"] for s in result["relevance_scores"] if s["doc_id"] == constants['DOC_ID_ZERO'])
        doc1_score = next(s["score"] for s in result["relevance_scores"] if s["doc_id"] == constants['DOC_ID_ONE'])
        assert doc0_score > doc1_score, f"Expected doc0 score {doc0_score} > doc1 score {doc1_score}"

    def test_when_assessing_documents_then_scores_map_doc_ids(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance, constants):
        """
        GIVEN documents with specific IDs
        WHEN relevance is assessed
        THEN expect relevance_scores dictionary to have corresponding keys.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        scored_doc_ids = [score["doc_id"] for score in result["relevance_scores"]]
        expected_doc_id = constants['DOC_ID_ZERO']
        assert expected_doc_id in scored_doc_ids, f"Expected {expected_doc_id} in relevance_scores but got {scored_doc_ids}"

    def test_when_assessing_documents_then_scores_map_to_float_values(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance):
        """
        GIVEN documents with specific IDs
        WHEN relevance is assessed
        THEN expect each key to map to a float score value.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        first_score = result["relevance_scores"][0]["score"]
        assert isinstance(first_score, float), f"Expected score to be float but got {type(first_score)}"


class TestPageNumbersAreExtractedfromRelevantDocuments:
    """Tests run method extracts page numbers from relevant documents."""

    def test_when_relevant_documents_found_then_page_numbers_sorted(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance):
        """
        GIVEN relevant documents reference specific pages
        WHEN run completes
        THEN expect page numbers to be in ascending order.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        first_doc_pages = list(result["page_numbers"].values())[0]
        sorted_pages = sorted(first_doc_pages)
        assert first_doc_pages == sorted_pages, f"Expected page numbers to be sorted but got {first_doc_pages}"

    def test_when_no_relevant_documents_then_page_numbers_empty(
        self, relevance_assessment, documents_small, variable_def, mock_llm_low_relevance, constants):
        """
        GIVEN no documents pass the relevance threshold
        WHEN run completes
        THEN expect page_numbers to be an empty list.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_low_relevance)
        expected_dict = {}
        assert result["page_numbers"] == expected_dict, f"Expected empty page_numbers dict but got {result['page_numbers']}"


class TestCitedPagesAreExtractedbyPageNumbers:
    """Tests run method extracts cited pages by page numbers."""
    
    def test_when_page_numbers_identified_then_exact_page_count_extracted(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance):
        """
        GIVEN potentially relevant documents with pages and specific page numbers identified
        WHEN cited pages are extracted
        THEN expect exact count of pages to be returned.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        total_pages = sum(len(pages) for pages in result["page_numbers"].values())
        assert len(result["relevant_pages"]) == total_pages, f"Expected {total_pages} relevant pages but got {len(result['relevant_pages'])}"

    def test_when_page_numbers_identified_then_correct_pages_extracted(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance, constants):
        """
        GIVEN potentially relevant documents with pages and specific page numbers identified
        WHEN cited pages are extracted
        THEN expect pages to correspond to identified page numbers.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        first_page = result["relevant_pages"][0]
        doc_id = first_page["doc_id"]
        assert doc_id in result["page_numbers"], f"Expected doc_id {doc_id} in page_numbers but got {list(result['page_numbers'].keys())}"


    def test_when_page_numbers_identified_then_full_content_extracted(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance):
        """
        GIVEN potentially relevant documents with pages and specific page numbers identified
        WHEN cited pages are extracted
        THEN expect each page to contain full text content.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        first_page = result["relevant_pages"][0]
        assert "content" in first_page, f"Expected content key in page but got {list(first_page.keys())}"


class TestLLMAPIIsUsedforRelevanceAssessment:
    """Tests run method uses LLM API for relevance assessment."""

    def test_when_llm_persistently_fails_then_error_logged(
            self, relevance_assessment, documents, variable_def, mock_llm_high_relevance):
        """
        GIVEN max_retries configured and LLM API fails on all attempts
        WHEN relevance assessment is performed
        THEN expect an error to be logged.
        """
        key = "error"
        result = relevance_assessment.run(documents, variable_def, mock_llm_high_relevance)
        score_item = result["relevance_scores"][0]
        assert key in score_item, f"Expected error to be recorded but got {list(score_item.keys())}"

    def test_when_llm_persistently_fails_then_document_marked_unable(
        self, relevance_assessment, documents, variable_def, mock_llm_high_relevance):
        """
        GIVEN max_retries configured and LLM API fails on all attempts
        WHEN relevance assessment is performed
        THEN expect document to be marked as unable to assess.
        """
        result = relevance_assessment.run(documents, variable_def, mock_llm_high_relevance)
        expected_relevance = False
        actual_relevance = result["relevance_scores"][0]["relevant"]
        assert actual_relevance == expected_relevance, \
            f"Expected relevant to be {expected_relevance} but got {actual_relevance}"


class TestCitationsAreTruncatedtoMaximumLength:
    """Tests run method truncates citations to maximum length."""
    
    def test_when_citation_exceeds_max_then_truncated(
        self, relevance_assessment, documents_small, variable_def, mock_llm_long_citation, constants):
        """
        GIVEN max_citation_length configured and citation text exceeds limit
        WHEN citation is processed
        THEN expect citation to be truncated to max length.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_long_citation)
        first_page = result["relevant_pages"][0]
        citation = first_page["citation"]
        max_len = constants['MAX_CITATION_LENGTH']
        indicator_len = constants['TRUNCATION_INDICATOR_LENGTH']
        assert len(citation) <= max_len + indicator_len, f"Expected citation length <= {max_len + indicator_len} but got {len(citation)}"

    def test_when_citation_exceeds_max_then_indicator_added(
        self, relevance_assessment, documents_small, variable_def, mock_llm_long_citation, constants):
        """
        GIVEN max_citation_length configured and citation text exceeds limit
        WHEN citation is processed
        THEN expect truncation indicator to be added.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_long_citation)
        first_page = result["relevant_pages"][0]
        citation = first_page["citation"]
        indicator = constants['TRUNCATION_INDICATOR']
        assert citation.endswith(indicator), f"Expected truncation indicator in citation but got {citation[-10:]}"

    def test_when_citation_within_max_then_full_text_preserved(
        self, relevance_assessment, documents_small, variable_def, mock_llm_short_citation, constants):
        """
        GIVEN max_citation_length configured and citation text within limit
        WHEN citation is processed
        THEN expect full text to be preserved.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_short_citation)
        first_page = result["relevant_pages"][0]
        citation = first_page["citation"]
        expected_citation = constants['SHORT_CITATION']
        assert citation == expected_citation, f"Expected full citation to be preserved but got {citation}"

    def test_when_citation_within_max_then_no_truncation(
        self, relevance_assessment, documents_small, variable_def, mock_llm_short_citation, constants):
        """
        GIVEN max_citation_length configured and citation text within limit
        WHEN citation is processed
        THEN expect no truncation indicator to be added.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_short_citation)
        first_page = result["relevant_pages"][0]
        citation = first_page["citation"]
        indicator = constants['TRUNCATION_INDICATOR']
        assert not citation.endswith(indicator), f"Expected no truncation indicator but found one in {citation[-10:]}"


class TestAssessMethodProvidesPublicInterface:
    """Tests assess method provides public interface for RelevanceAssessment."""
    
    def test_when_calling_assess_then_performs_assessment(
        self, relevance_assessment, documents_small, mock_llm_high_relevance):
        """
        GIVEN documents and prompt sequence
        WHEN assess method is called
        THEN expect relevance assessment to be performed.
        """
        prompt_sequence = ["What is the tax rate?", "How is tax calculated?"]
        result = relevance_assessment.assess(documents_small, prompt_sequence, mock_llm_high_relevance)
        assert isinstance(result, list), f"Expected list but got {type(result)}"

    @pytest.mark.parametrize("llm_fixture,expected_count_key", [
        ("mock_llm_high_relevance", "EXPECTED_RELEVANT_COUNT_TWO"),
        ("mock_llm_low_relevance", "EXPECTED_RELEVANT_COUNT_THREE"),
        ("mock_llm_very_high_confidence", "EXPECTED_RELEVANT_COUNT_ONE"),
    ])
    def test_when_calling_assess_then_returns_expected_document_count(
        self, relevance_assessment, documents_small, constants, llm_fixture, expected_count_key, request):
        """
        GIVEN documents and prompt sequence
        WHEN assess method is called with different LLM configurations
        THEN expect appropriate number of relevant documents to be returned.
        """
        mock_llm = request.getfixturevalue(llm_fixture)
        prompt_sequence = ["What is the tax rate?", "How is tax calculated?"]
        result = relevance_assessment.assess(documents_small, prompt_sequence, mock_llm)
        expected_count = constants[expected_count_key]
        assert len(result) == expected_count, f"Expected {expected_count} relevant documents but got {len(result)}"


class TestControlFlowValidatesInputTypes:
    """Tests run method validates input types."""

    def test_when_documents_not_list_then_error_indicates_list_required(
        self, relevance_assessment, variable_def, constants, mock_llm_high_relevance):
        """
        GIVEN documents parameter that is not a list
        WHEN run is called
        THEN expect TypeError to be raised with message indicating list is required.
        """
        non_list_value = constants['NON_LIST_VALUE']
        expected_msg = constants['TYPE_ERROR_LIST_MESSAGE']
        with pytest.raises(TypeError, match=rf'{expected_msg}'):
            relevance_assessment.run(non_list_value, variable_def, mock_llm_high_relevance)

    def test_when_variable_definition_not_dict_then_error_indicates_dict_required(
        self, relevance_assessment, documents_small, constants, mock_llm_high_relevance):
        """
        GIVEN variable_definition parameter that is not a dict
        WHEN run is called
        THEN expect TypeError to be raised with message indicating dict is required.
        """
        non_dict_value = constants['NON_DICT_VALUE']
        expected_msg = constants['TYPE_ERROR_DICT_MESSAGE']
        with pytest.raises(TypeError, match=rf'{expected_msg}'):
            relevance_assessment.run(documents_small, non_dict_value, mock_llm_high_relevance)

    def test_when_empty_document_list_then_completes_without_error(
        self, relevance_assessment, variable_def, mock_llm_high_relevance, constants):
        """
        GIVEN empty document list
        WHEN run is called
        THEN expect to complete without error.
        """
        empty_list = constants['EMPTY_LIST']
        result = relevance_assessment.run(empty_list, variable_def, mock_llm_high_relevance)
        assert isinstance(result, dict), f"Expected dict but got {type(result)}"

    def test_when_empty_document_list_then_returns_empty_results(
        self, relevance_assessment, variable_def, mock_llm_high_relevance, constants):
        """
        GIVEN empty document list
        WHEN run is called
        THEN expect empty results to be returned.
        """
        empty_list = constants['EMPTY_LIST']
        result = relevance_assessment.run(empty_list, variable_def, mock_llm_high_relevance)
        expected_count = constants['EXPECTED_RELEVANT_COUNT_ZERO']
        assert len(result["relevant_pages"]) == expected_count, f"Expected {expected_count} relevant_pages but got {len(result['relevant_pages'])}"


class TestRelevantDocumentIDsAreTracked:
    """Tests run method tracks relevant document IDs."""
    
    def test_when_documents_assessed_then_ids_match_relevant_documents(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance):
        """
        GIVEN documents with specific IDs
        WHEN documents are assessed
        THEN expect relevant_doc_ids to match IDs of relevant documents.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        expected_ids = [page["doc_id"] for page in result["relevant_pages"]]
        assert set(result["relevant_doc_ids"]) == set(expected_ids), f"Expected {expected_ids} but got {result['relevant_doc_ids']}"

    def test_when_documents_assessed_then_ids_correspond_to_relevant_pages(
        self, relevance_assessment, documents_small, variable_def, mock_llm_high_relevance):
        """
        GIVEN documents with specific IDs
        WHEN documents are assessed
        THEN expect relevant_doc_ids to correspond to relevant_pages.
        """
        result = relevance_assessment.run(documents_small, variable_def, mock_llm_high_relevance)
        page_doc_ids = [page["doc_id"] for page in result["relevant_pages"]]
        first_id = result["relevant_doc_ids"][0]
        assert first_id in page_doc_ids, f"Expected doc_id {first_id} to have corresponding page in relevant_pages"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
