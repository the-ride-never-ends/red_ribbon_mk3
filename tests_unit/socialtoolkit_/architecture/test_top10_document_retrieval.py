#!/usr/bin/env python3
"""
Feature: Top-10 Document Retrieval
  As a data researcher
  I want to retrieve the top 10 most relevant documents based on a query
  So that I can find the most relevant information efficiently

  Background:
    Given a Top10DocumentRetrieval instance is initialized
    And a vector search engine is available
    And a document storage service is available
"""
# NOTE: Current test function count: 23
import pytest
from typing import Any
from unittest.mock import MagicMock

from custom_nodes.red_ribbon.socialtoolkit.architecture.top10_document_retrieval import (
    Top10DocumentRetrieval,
    make_top10_document_retrieval,
)
from custom_nodes.red_ribbon.utils_ import DatabaseAPI
from .conftest import FixtureError

@pytest.fixture
def constants() -> dict[str, Any]:
    """Fixture providing test constants for document retrieval tests."""
    return {
        "VALID_QUERY": "What is the sales tax rate?",
        "EMPTY_QUERY": "",
        "TAX_QUERY": "tax rates",
        "BUSINESS_QUERY": "business licenses",
        "TAX_INFO_QUERY": "tax information",
        "TEST_QUERY": "test query",
        "SINGLE_DOC_CONTENT": "Tax info",
        "DOC_ID_PREFIX": "doc",
        "CONTENT_PREFIX": "Content",
        "URL_PREFIX": "http://url",
        "TITLE_PREFIX": "Title",
        "EXPECTED_COUNT_TEN": 10,
        "EXPECTED_COUNT_FIVE": 5,
        "EXPECTED_COUNT_FIFTY": 50,
        "EXPECTED_COUNT_FOUR": 4,
        "EXPECTED_COUNT_THREE": 3,
        "EXPECTED_COUNT_TWO": 2,
        "EXPECTED_COUNT_TWENTY": 20,
        "THRESHOLD_POINT_SIX": 0.6,
        "THRESHOLD_POINT_NINE": 0.9,
        "SCORE_MIN": -1.0,
        "SCORE_MAX": 1.0,
        "SCORE_ZERO": 0,
        "VECTOR_DIM": 3,
        "RANKING_COSINE": "cosine_similarity",
        "RANKING_DOT": "dot_product",
        "RANKING_EUCLIDEAN": "euclidean",
        "KEY_RELEVANT_DOCS": "relevant_documents",
        "KEY_SCORES": "scores",
        "KEY_TOP_DOC_IDS": "top_doc_ids",
        "FIELD_ID": "id",
        "FIELD_CONTENT": "content",
        "FIELD_URL": "url",
        "FIELD_TITLE": "title",
        "INTEGER_QUERY": 12345,
    }


@pytest.fixture
def mock_db() -> MagicMock:
    return MagicMock(spec=DatabaseAPI)

@pytest.fixture
def mock_encode_query() -> MagicMock:
    """Mock encoder that returns a simple vector."""
    mock = MagicMock()
    mock.return_value = [0.1, 0.2, 0.3]  # Simple 3D vector
    return mock

@pytest.fixture
def make_top10_retrieval(mock_db, mock_encode_query):
    def _make_top10_retrieval():
        resources = {
            "database": mock_db,
            "_encode_query": mock_encode_query
        }
        try:
            return make_top10_document_retrieval(resources=resources)
        except Exception as e:
            raise FixtureError(f"Failed to create Top10DocumentRetrieval: {e}") from e
    return _make_top10_retrieval

@pytest.fixture
def top10_retrieval(make_top10_retrieval) -> Top10DocumentRetrieval:
    return make_top10_retrieval()

@pytest.fixture
def top10_retrieval_retrieval_count_is_ten(make_top10_retrieval, constants) -> Top10DocumentRetrieval:
    top10_retrieval = make_top10_retrieval()
    top10_retrieval.retrieval_count = constants["EXPECTED_COUNT_TEN"]
    return top10_retrieval

@pytest.fixture
def top10_retrieval_retrieval_count_is_five(make_top10_retrieval, constants) -> Top10DocumentRetrieval:
    top10_retrieval = make_top10_retrieval()
    top10_retrieval.retrieval_count = constants["EXPECTED_COUNT_FIVE"]
    return top10_retrieval

@pytest.fixture
def top10_retrieval_similarity_threshold_is_point_six(make_top10_retrieval, constants) -> Top10DocumentRetrieval:
    top10_retrieval = make_top10_retrieval()
    top10_retrieval.similarity_threshold = constants["THRESHOLD_POINT_SIX"]
    return top10_retrieval

def top10_retrieval_ranking_method_is_dot_product(make_top10_retrieval, constants) -> Top10DocumentRetrieval:
    top10_retrieval = make_top10_retrieval()
    top10_retrieval.ranking_method = constants["RANKING_DOT"]
    return top10_retrieval

def top10_retrieval_ranking_method_is_cosine(make_top10_retrieval, constants) -> Top10DocumentRetrieval:
    top10_retrieval = make_top10_retrieval()
    top10_retrieval.ranking_method = constants["RANKING_COSINE"]
    return top10_retrieval

def top10_retrieval_ranking_method_is_euclidean(make_top10_retrieval, constants) -> Top10DocumentRetrieval:
    top10_retrieval = make_top10_retrieval()
    top10_retrieval.ranking_method = constants["RANKING_EUCLIDEAN"]
    return top10_retrieval


@pytest.fixture
def retrieval_instances(
    top10_retrieval: Top10DocumentRetrieval,
    top10_retrieval_retrieval_count_is_ten: Top10DocumentRetrieval,
    top10_retrieval_retrieval_count_is_five: Top10DocumentRetrieval,
    top10_retrieval_ranking_method_is_dot_product: Top10DocumentRetrieval,
    top10_retrieval_ranking_method_is_cosine: Top10DocumentRetrieval,
    top10_retrieval_ranking_method_is_euclidean: Top10DocumentRetrieval,
    top10_retrieval_similarity_threshold_is_point_six: Top10DocumentRetrieval,
    ):
    """Factory function for creating document retrieval test objects."""
    return {
        "DEFAULT": top10_retrieval,
        "RETRIEVAL_COUNT_IS_TEN": top10_retrieval_retrieval_count_is_ten,
        "RETRIEVAL_COUNT_IS_FIVE": top10_retrieval_retrieval_count_is_five,
        "SIMILARITY_THRESHOLD_IS_POINT_SIX": top10_retrieval_similarity_threshold_is_point_six,
        "RANKING_METHOD_IS_DOT_PRODUCT": top10_retrieval_ranking_method_is_dot_product,
        "RANKING_METHOD_IS_COSINE": top10_retrieval_ranking_method_is_cosine,
        "RANKING_METHOD_IS_EUCLIDEAN": top10_retrieval_ranking_method_is_euclidean,
    }


@pytest.fixture
def make_documents(constants):
    """Factory fixture for creating document lists of any size."""
    def _make(count: int, with_metadata: bool = False):
        doc_id_prefix = constants["DOC_ID_PREFIX"]
        content_prefix = constants["CONTENT_PREFIX"]
        url_prefix = constants["URL_PREFIX"]
        title_prefix = constants["TITLE_PREFIX"]
        
        base_doc = {"id": f"{doc_id_prefix}{{i}}", "content": f"{content_prefix} {{i}}"}
        metadata_doc = {
            "id": f"{doc_id_prefix}{{i}}", 
            "content": f"{content_prefix} {{i}}",
            "url": f"{url_prefix}{{i}}",
            "title": f"{title_prefix} {{i}}"
        }
        
        template = metadata_doc if with_metadata else base_doc
        return [
            {k: v.format(i=i) for k, v in template.items()}
            for i in range(count)
        ]
    return _make

@pytest.fixture
def documents(make_documents):
    return {
        "EMPTY": [],
        "SINGLE": make_documents(1),
        "FIFTY": make_documents(50),
        "FIVE": make_documents(5),
        "FOUR": make_documents(4),
        "THREE": make_documents(3),
        "TWENTY": make_documents(20),
        "ONE_WITH_METADATA": make_documents(1, with_metadata=True),
        "TWO_WITH_METADATA": make_documents(2, with_metadata=True),
    }

@pytest.fixture
def three_vectors(constants):
    vector_dim = constants["VECTOR_DIM"]
    base_vector = [0.1, 0.2, 0.3]
    return [base_vector[:] for _ in range(vector_dim)]

@pytest.fixture
def vectors(three_vectors):
    base_vector = [0.1, 0.2, 0.3]
    return {
        "EMPTY": [],
        "SINGLE": [base_vector],
        "THREE": three_vectors,
        "FIVE": three_vectors + [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        "FIFTY": three_vectors * 17,
        "FOUR": three_vectors + [[0.4, 0.5, 0.6]],
        "TWENTY": three_vectors * 7,
    }

@pytest.fixture
def valid_args(documents, vectors, constants) ->  dict[str, Any]:
    return {
        "QUERY_ONLY": {
            "input_data_point": constants["VALID_QUERY"],
            "documents": None,
            "document_vectors": None,
        },
        "QUERY_AND_DOCUMENTS_ONLY": {
            "input_data_point": constants["VALID_QUERY"],
            "documents": documents["SINGLE"],
            "document_vectors": None,
        },
        "EMPTY": {
            "input_data_point": constants["VALID_QUERY"],
            "documents": documents["EMPTY"],
            "document_vectors": vectors["EMPTY"],
        },
        "SINGLE": {
            "input_data_point": constants["VALID_QUERY"],
            "documents": documents["SINGLE"],
            "document_vectors": vectors["SINGLE"],
        },
        "THREE": {
            "input_data_point": constants["TEST_QUERY"],
            "documents": documents["THREE"],
            "document_vectors": vectors["THREE"],
        },
        "FIVE": {
            "input_data_point": constants["TEST_QUERY"],
            "documents": documents["FIVE"],
            "document_vectors": vectors["FIVE"],
        },
        "FIFTY": {
            "input_data_point": constants["TEST_QUERY"],
            "documents": documents["FIFTY"],
            "document_vectors": vectors["FIFTY"],
        },
        "TWENTY": {
            "input_data_point": constants["TEST_QUERY"],
            "documents": documents["TWENTY"],
            "document_vectors": vectors["TWENTY"],
        },
    }


@pytest.mark.parametrize("num_docs", [
    "QUERY_ONLY", "QUERY_AND_DOCUMENTS_ONLY", "EMPTY", "SINGLE", "FIFTY", "FIVE", "FOUR", "THREE", "TWENTY"
])
class TestExecuteMethodAlwaysReturnsDictionarywithRequiredKeys:
    """Tests for Top10DocumentRetrieval execute method return structure."""

    # NOTE: Done
    def test_when_execute_is_called_with_valid_args_then_returns_dictionary(
        self, num_docs, top10_retrieval, valid_args):
        """
        GIVEN arguments with valid types and values
        WHEN execute is called with the query
        THEN expect result to be a dictionary.
        """
        result = top10_retrieval.execute(**valid_args[num_docs])
        assert isinstance(result, dict), f"Expected dict but got {type(result)}"

    @pytest.mark.parametrize("key", [
        "relevant_documents", "scores", "top_doc_ids"
    ]) # NOTE: Done
    def test_when_execute_is_called_with_valid_args_then_contains_required_key(
        self, num_docs, key, top10_retrieval, valid_args):
        """
        GIVEN arguments with valid types and values
        WHEN execute is called with the query
        THEN expect result to contain required key.
        """
        result = top10_retrieval.execute(**valid_args[num_docs])
        assert key in result, f"Missing {key} key in {list(result.keys())}"

    @pytest.mark.parametrize("key,expected_type", [
        ("relevant_documents", list),
        ("scores", dict),
        ("top_doc_ids", list)
    ]) # NOTE: Done
    def test_when_execute_is_called_with_valid_args_then_result_key_is_expected_type(
        self, num_docs, key, expected_type, top10_retrieval, valid_args):
        """
        GIVEN arguments with valid types and values
        WHEN execute is called with a query
        THEN expect result key to have the expected type.
        """
        result = top10_retrieval.execute(**valid_args[num_docs])
        assert isinstance(result[key], expected_type), \
            f"Expected {type(expected_type).__name__} but got {type(result[key]).__name__}"


@pytest.mark.parametrize("key,expected_value", [
    ("relevant_documents", []),
    ("scores", {}),
    ("top_doc_ids", [])
]) # NOTE: Done
def test_when_execute_is_called_with_empty_documents_then_result_key_is_empty(
    key, expected_value, top10_retrieval, constants):
    """
    GIVEN an empty document collection
    WHEN execute is called with a query
    THEN expect result key to have expected value.
    """
    result = top10_retrieval.execute(constants["TAX_QUERY"], documents=[])
    assert result[key] == expected_value, f"Expected {expected_value} but got {result[key]}"



class TestExecuteMethodReturnsAtMostNDocumentsWhereNisretrievalcount:
    """Tests for Top10DocumentRetrieval execute retrieval count limits."""

    @pytest.mark.parametrize("doc_fixture,query_key,expected_count,result_key", [
        ("fifty_documents", "BUSINESS_QUERY", 10, "relevant_documents"),
        ("fifty_documents", "BUSINESS_QUERY", 10, "top_doc_ids"),
        ("five_documents", "TAX_INFO_QUERY", 5, "relevant_documents"),
        ("five_documents", "TAX_INFO_QUERY", 5, "top_doc_ids"),
    ])
    def test_when_execute_is_called_with_docs_then_returns_expected_count(
        self, top10_retrieval, constants, doc_fixture, query_key, expected_count, result_key, request
    ): # NOTE: Done
        """
        GIVEN retrieval_count is configured and documents are available
        WHEN execute is called with a query
        THEN expect result key to have expected count.
        """
        documents = request.getfixturevalue(doc_fixture)
        result = top10_retrieval.execute(constants[query_key], documents=documents)
        actual_count = len(result[result_key])
        assert actual_count == expected_count, f"Expected {expected_count} but got {actual_count}"


@pytest.mark.parametrize("num_docs", [
    "QUERY_ONLY", "QUERY_AND_DOCUMENTS_ONLY", "EMPTY", "SINGLE", "FIFTY", "FIVE", "FOUR", "THREE", "TWENTY"
])
class TestDocumentsAreRankedbySimilarityScoreinDescendingOrder:
    """Tests for Top10DocumentRetrieval execute similarity score ranking."""

    # NOTE: Done
    def test_when_execute_is_called_with_multiple_documents_then_first_has_highest_score(
        self, num_docs, top10_retrieval: Top10DocumentRetrieval, documents, constants):
        """
        GIVEN multiple documents with varying similarity scores
        WHEN execute is called with a query
        THEN expect first document to have highest similarity score.
        """
        result = top10_retrieval.execute(constants["TEST_QUERY"], documents=documents[num_docs])

        first_doc_id = result[constants["KEY_TOP_DOC_IDS"]][0]
        first_score = result[constants["KEY_SCORES"]][first_doc_id]
        all_scores = tuple(result[constants["KEY_SCORES"]].values())

        assert first_score == max(all_scores), \
            f"First score {first_score} not highest among {all_scores}"

    # NOTE: Done
    def test_when_execute_is_called_with_multiple_documents_then_scores_descend(
        self, num_docs, top10_retrieval, documents, constants):
        """
        GIVEN multiple documents with varying similarity scores
        WHEN execute is called with a query
        THEN expect scores to be in descending order.
        """
        result = top10_retrieval.execute(constants["TEST_QUERY"], documents=documents[num_docs])

        scores = list(result[constants["KEY_SCORES"]].values())

        sorted_desc_scores = sorted(scores, reverse=True)

        assert scores == sorted_desc_scores, \
            f"Scores not in descending order: '{scores}' vs '{sorted_desc_scores}'"

    # NOTE: Done
    def test_when_execute_is_called_with_multiple_documents_then_all_doc_ids_have_scores(
        self, num_docs, top10_retrieval, documents, constants):
        """
        GIVEN multiple documents with varying similarity scores
        WHEN execute is called with a query
        THEN expect all document IDs to have corresponding scores.
        """
        result = top10_retrieval.execute(constants["TEST_QUERY"], documents=documents[num_docs])

        doc_ids_set = set(result[constants["KEY_TOP_DOC_IDS"]])
        scores_keys_set = set(result[constants["KEY_SCORES"]].keys())

        assert doc_ids_set == scores_keys_set, \
            f"Expected doc IDs '{doc_ids_set}' to match score keys '{scores_keys_set}'"


@pytest.mark.parametrize("num_docs", [
    "QUERY_ONLY", "QUERY_AND_DOCUMENTS_ONLY", "EMPTY", "SINGLE", "FIFTY", "FIVE", "FOUR", "THREE", "TWENTY"
])
class TestSimilarityScoresRespectConfiguredThreshold:
    """Tests for Top10DocumentRetrieval execute similarity threshold filtering."""

    # NOTE: Done
    def test_when_execute_is_called_with_threshold_configured_then_only_above_threshold_returned(
        self, num_docs, top10_retrieval_similarity_threshold_is_point_six, valid_args, constants):
        """
        GIVEN similarity_threshold is configured
        WHEN execute is called with a query
        THEN expect only documents with scores above threshold to be returned.
        """
        threshold = constants["THRESHOLD_POINT_SIX"]
        result = top10_retrieval_similarity_threshold_is_point_six.execute(**valid_args[num_docs])
        min_score = min(list(result[constants["KEY_SCORES"]].values()))

        assert min_score > threshold, \
            f"Expected min_score to be greater than '{threshold}' but got '{min_score}'"

    # NOTE: Done
    def test_when_execute_is_called_with_threshold_configured_then_at_least_minimum_documents_returned(
        self, num_docs, top10_retrieval_similarity_threshold_is_point_six, documents, constants):
        """
        GIVEN similarity_threshold is configured
        WHEN execute is called with a query
        THEN expect at least 10 documents in results.
        """
        # TODO: Find a way to get the minimum expected based on actual similarity scores
        result = top10_retrieval_similarity_threshold_is_point_six.execute(**valid_args[num_docs])
        actual_count = len(result[constants["KEY_RELEVANT_DOCS"]])
        minimum_expected = 10
        assert actual_count >= minimum_expected, \
            f"Expected at least {minimum_expected} but got {actual_count}"


@pytest.mark.parametrize("num_docs", [
    "QUERY_ONLY", "QUERY_AND_DOCUMENTS_ONLY", "EMPTY", "SINGLE", "FIFTY", "FIVE", "FOUR", "THREE", "TWENTY"
])
class TestExecuteValidatesInputTypes:
    """Tests for Top10DocumentRetrieval execute input validation."""

    # NOTE: Done
    @pytest.mark.parametrize("invalid_query", [12345, None, 12.34, [], {}, set()])
    def test_when_execute_is_called_with_invalid_query_type_then_raises_type_error(
        self, num_docs, invalid_query, top10_retrieval, valid_args):
        """
        GIVEN an invalid query type
        WHEN execute is called with the invalid query
        THEN expect TypeError to be raised.
        """
        invalid_args = valid_args[num_docs]
        invalid_args["input_data_point"] = invalid_query

        with pytest.raises(TypeError, match=r"input_data_point must be a string"):
            top10_retrieval.execute(**invalid_args)

    # NOTE: Done
    @pytest.mark.parametrize("invalid_docs", [12345, None, 12.34, "invalid", {}, set()])
    def test_when_execute_is_called_with_invalid_documents_type_then_raises_type_error(
        self, num_docs, invalid_docs, top10_retrieval, valid_args):
        """
        GIVEN an invalid document type
        WHEN execute is called with the invalid documents
        THEN expect TypeError to be raised.
        """
        invalid_args = valid_args[num_docs]
        invalid_args["documents"] = invalid_docs

        with pytest.raises(TypeError, match=r"documents must be a list"):
            top10_retrieval.execute(**invalid_args)

    # NOTE: Done
    @pytest.mark.parametrize("invalid_vectors", [12345, None, 12.34, "invalid", {}, set()])
    def test_when_execute_is_called_with_invalid_vectors_type_then_raises_type_error(
        self, num_docs, invalid_vectors, top10_retrieval, valid_args):
        """
        GIVEN an invalid vectors type
        WHEN execute is called with the invalid vectors
        THEN expect TypeError to be raised.
        """
        invalid_args = valid_args[num_docs]
        invalid_args["document_vectors"] = invalid_vectors

        with pytest.raises(TypeError, match=r"vectors must be a list of lists of floats"):
            top10_retrieval.execute(**invalid_args)


@pytest.mark.parametrize("num_docs", [
    "SINGLE", "FIFTY", "FIVE", "FOUR", "THREE", "TWENTY"
])
class TestExecuteHandlesDocumentandVectorParameters:
    """Tests for Top10DocumentRetrieval execute document and vector parameter handling."""

    # NOTE: Done
    def test_when_execute_is_called_with_explicit_documents_and_vectors_then_returns_limited_results(
        self, num_docs, top10_retrieval, valid_args, constants
    ):
        """
        GIVEN explicit documents list and vectors
        WHEN execute is called with documents and vectors parameters
        THEN expect at most configured number of documents in relevant_documents.
        """
        docs = valid_args[num_docs]["documents"]
        result = top10_retrieval.execute(**valid_args[num_docs])
        max_expected = len(docs)
        actual_count = len(result[constants["KEY_RELEVANT_DOCS"]])
        assert actual_count <= max_expected, f"Expected <= {max_expected} but got {actual_count}"

    # NOTE: Done
    def test_when_execute_is_called_with_explicit_documents_and_vectors_then_returns_list(
        self, num_docs, top10_retrieval, constants, valid_args
    ):
        """
        GIVEN explicit documents list and vectors
        WHEN execute is called with documents and vectors parameters
        THEN expect relevant_documents to be a list.
        """
        result = top10_retrieval.execute(**valid_args[num_docs])
        relevant_docs = result[constants["KEY_RELEVANT_DOCS"]]
        assert isinstance(relevant_docs, list), f"Expected list but got {type(relevant_docs).__name__}"

    # NOTE: Done
    def test_when_execute_is_called_with_documents_parameter_is_none_then_returns_dict(
        self, num_docs, top10_retrieval, constants):
        """
        GIVEN documents parameter is None
        WHEN execute is called with only the query
        THEN expect result to be a dictionary.
        """
        result = top10_retrieval.execute(constants["TEST_QUERY"])
        assert isinstance(result, dict), f"Expected dict but got {type(result)}"

    # NOTE: Done
    def test_when_execute_is_called_with_documents_parameter_is_none_then_contains_relevant_documents_key(
        self, num_docs, top10_retrieval, constants):
        """
        GIVEN documents parameter is None
        WHEN execute is called with only the query
        THEN expect result to contain relevant_documents key.
        """
        result = top10_retrieval.execute(constants["TEST_QUERY"])
        key = constants["KEY_RELEVANT_DOCS"]
        assert key in result, f"Missing {key} key in {list(result.keys())}"


@pytest.mark.parametrize("num_docs", [
    "QUERY_AND_DOCUMENTS_ONLY", "SINGLE", "FIFTY", "FIVE", "FOUR", "THREE", "TWENTY"
])
class TestRankingMethodConfigurationAffectsScoreCalculation:
    """Tests for Top10DocumentRetrieval execute ranking method configuration."""

    @pytest.mark.parametrize("ranking_method", [
        "RANKING_METHOD_IS_COSINE", "RANKING_METHOD_IS_DOT_PRODUCT", "RANKING_METHOD_IS_EUCLIDEAN"
    ]) # NOTE: Done
    def test_when_execute_is_called_with_ranking_method_configured_then_contains_scores(
        self, num_docs, ranking_method, retrieval_instances, constants, valid_args):
        """
        GIVEN ranking_method is configured
        WHEN execute is called with a query
        THEN expect result to contain scores key.
        """
        top10_retrieval = retrieval_instances[ranking_method]
        result = top10_retrieval.execute(**valid_args[num_docs])
        key = constants["KEY_SCORES"]
        assert key in result, f"Missing {key} key in {list(result.keys())}"

    # NOTE: Done
    def test_when_execute_is_called_with_cosine_similarity_then_scores_above_expected_min(
        self, num_docs, retrieval_instances, constants, valid_args):
        """
        GIVEN ranking_method is cosine_similarity
        WHEN execute is called with a query
        THEN expect the lowest score to be greater than or equal to the expected minimum score.
        """
        top10_retrieval = retrieval_instances["RANKING_METHOD_IS_COSINE"]
        result = top10_retrieval.execute(**valid_args[num_docs])
        expected_min = constants["SCORE_MIN"]
        min_score = min(list(result[constants["KEY_SCORES"]].values()))
        assert min_score >= expected_min, f"Expected min_score >= {expected_min} but got {min_score}."

    # NOTE: Done
    def test_when_execute_is_called_with_cosine_similarity_then_scores_below_expected_max(
        self, num_docs, retrieval_instances, constants, valid_args):
        """
        GIVEN ranking_method is cosine_similarity
        WHEN execute is called with a query
        THEN expect the highest score to be less than or equal to the expected maximum score.
        """
        top10_retrieval = retrieval_instances["RANKING_METHOD_IS_COSINE"]
        result = top10_retrieval.execute(**valid_args[num_docs])
        expected_max = constants["SCORE_MAX"]
        max_score = max(list(result[constants["KEY_SCORES"]].values()))
        assert max_score <= expected_max, f"Expected max_score <= {expected_max} but got {max_score}."

    # NOTE: Done
    def test_when_execute_is_called_with_ranking_method_doc_product_then_scores_is_dict(
        self, num_docs, constants, valid_args, retrieval_instances):
        """
        GIVEN ranking_method is dot_product
        WHEN execute is called with a query
        THEN expect scores to be a dictionary.
        """
        top10_retrieval = retrieval_instances["RANKING_METHOD_IS_DOT_PRODUCT"]
        result = top10_retrieval.execute(**valid_args[num_docs])
        key_scores = result[constants["KEY_SCORES"]]
        assert isinstance(key_scores, dict), f"Expected dict but got {type(key_scores).__name__}"

    # NOTE: Done
    def test_when_execute_is_called_with_ranking_method_euclidean_then_scores_are_positive(
        self, num_docs, retrieval_instances, constants, valid_args):
        """
        GIVEN ranking_method is euclidean
        WHEN execute is called with a query
        THEN expect all scores to be non-negative.
        """
        top10_retrieval = retrieval_instances["RANKING_METHOD_IS_EUCLIDEAN"]
        ZERO = 0
        result = top10_retrieval.execute(**valid_args[num_docs])
        min_score = min(list(result[constants["KEY_SCORES"]].values()))

        assert min_score > ZERO, f"Expected lowest score to be non-negative but got {min_score}" 


class TestResultDocumentsContainRequiredMetadata:
    """Tests for Top10DocumentRetrieval execute result document metadata."""

    @pytest.mark.parametrize("key", ["url", "title"])
    def test_when_execute_is_called_with_documents_with_metadata_then_all_have_required_field(
        self, key, top10_retrieval, constants, valid_args
    ): # NOTE: Done
        """
        GIVEN documents with metadata fields
        WHEN execute is called with a query
        THEN expect each document in relevant_documents to have required field.
        """
        kwargs = valid_args["QUERY_AND_DOCUMENTS_ONLY"]
        result = top10_retrieval.execute(**kwargs)
        doc = result[constants["KEY_RELEVANT_DOCS"]][0]
        assert key in doc, f"Missing {key} field in document {doc}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
