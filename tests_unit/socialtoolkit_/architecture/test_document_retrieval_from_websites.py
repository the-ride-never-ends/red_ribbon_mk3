#!/usr/bin/env python3
"""
Feature: Document Retrieval from Websites
  As a data collection system
  I want to retrieve and process documents from websites
  So that I can extract structured data for analysis

  Background:
    GIVEN a DocumentRetrievalFromWebsites instance is initialized
    And a static webpage parser is available
    And a dynamic webpage parser is available
    And a data extractor is available
    And a vector generator is available
    And a metadata generator is available
    And a document storage service is available
    And a URL path generator is available
"""
# NOTE: 36 test functions
import sqlite3
from unittest.mock import Mock, MagicMock
from datetime import datetime


import pytest
from requests.exceptions import HTTPError, Timeout


# Import the actual class being tested
from custom_nodes.red_ribbon.socialtoolkit.architecture.document_retrieval_from_websites import (
    DocumentRetrievalFromWebsites,
    DocumentRetrievalConfigs,
    WebpageType
)
from custom_nodes.red_ribbon.socialtoolkit.architecture.document_storage import DocumentStorage
from custom_nodes.red_ribbon.utils_ import get_cid

from .conftest import FixtureError

@pytest.fixture
def mock_get_cid() -> MagicMock:
    """Fixture providing a mock get_cid function"""
    mock_func = MagicMock(return_value="bafybeigdyrzt5s3z")
    return mock_func


@pytest.fixture
def valid_config_values():
    return {
        "timeout_seconds": 30,
        "max_retries": 3,
        "batch_size": 10,
        "follow_links": False,
        "max_depth": 1
    }


# Mock constants fixture to avoid hardcoded values
@pytest.fixture
def constants(valid_config_values):
    """Fixture providing mock constants for tests"""
    base_url = "https://example.com"
    return {
        # URLs
        "BASE_URL": base_url,
        "DYNAMIC_URL": f"{base_url}/app",
        "TEST_DOT_ORG_URL": "https://test.org",
        "STATIC_PAGE_URL": f"{base_url}/static-page.html",
        "PAGE1_URL": f"{base_url}/page1",
        "PAGE2_URL": f"{base_url}/page2",
        "TIMEOUT_URL": f"{base_url}/timeout",
        "SUCCESS_URL": f"{base_url}/success",
        "NOT_FOUND_URL": f"{base_url}/not-found",
        "ABOUT_URL": f"{base_url}/about",
        "INVALID_URL": "not-a-valid-url",
        "CUSTOM_USER_AGENT": "CustomBot/1.0",
        "SINGLE_TEXT_CONTENT": "Single text content",
        
        # Generated URLs for path generator
        "GENERATED_URLS": [
            base_url,
            f"{base_url}/page1",
            f"{base_url}/page2",
            f"{base_url}/about",
            f"{base_url}/contact"
        ],
        
        # HTML Content
        "STATIC_HTML_CONTENT": "<html><body><h1>Test Content</h1></body></html>",
        "DYNAMIC_HTML_CONTENT": "<html><body><div id='app'>Dynamic Content</div></body></html>",
        
        # HTTP Response data
        "HTTP_STATUS_OK": 200,
        "HTML_CONTENT_TYPE": "text/html",
        
        # Extracted text content
        "EXTRACTED_TEXT_CONTENT": [
            "Extracted text content 1",
            "Extracted text content 2",
            "Extracted text content 3"
        ],
        
        # Vector embeddings
        "PATTERN_REPETITIONS": 512,  # Number of times to repeat 3-element pattern
        "VECTOR_DIMENSION": 1536,  # Full embedding dimension (512 * 3 = 1536)
        "DOC_EMBEDDINGS": {
            "doc_1": [0.1, 0.2, 0.3],
            "doc_2": [0.4, 0.5, 0.6],
            "doc_3": [0.7, 0.8, 0.9]
        },
        
        # Document IDs
        "DOC_IDS": ["doc_1", "doc_2", "doc_3"],
        
        # Metadata
        "MOCK_TIMESTAMP": datetime(2023, 1, 1, 12, 0, 0),
        "CONTENT_LENGTH": 25,
        "SOURCE_DOMAIN": "example.com",
        
        # Configuration values
        "TIMEOUT_SECONDS": valid_config_values["timeout_seconds"],
        "MAX_RETRIES": valid_config_values["max_retries"],
        "BATCH_SIZE": valid_config_values["batch_size"],
        "MAX_DEPTH": valid_config_values["max_depth"],
        "FOLLOW_LINKS": valid_config_values["follow_links"]
    }

@pytest.fixture
def make_mock_configs(valid_config_values):
    def _make_configs(overrides={}):
        config_values = valid_config_values.copy()
        config_values.update(overrides)
        try:
            return DocumentRetrievalConfigs(**config_values)
        except Exception as e:
            raise FixtureError(f"Failed to create mock configs: {e}") from e
    return _make_configs

@pytest.fixture
def mock_configs(make_mock_configs):
    return make_mock_configs()


@pytest.fixture
def mock_timestamp_service(constants):
    mock_service = Mock()
    mock_service.now.return_value = constants["MOCK_TIMESTAMP"]
    return mock_service


@pytest.fixture
def mock_static_webpage_parser(constants):
    """
    And a static webpage parser is available
    """
    mock_parser = Mock()
    mock_parser.parse.return_value = {
        "url": constants["BASE_URL"],
        "html_content": constants["STATIC_HTML_CONTENT"],
        "status_code": constants["HTTP_STATUS_OK"],
        "headers": {"Content-Type": constants["HTML_CONTENT_TYPE"]}
    }
    return mock_parser


@pytest.fixture
def mock_dynamic_webpage_parser(constants):
    """
    And a dynamic webpage parser is available
    """
    mock_parser = Mock()
    mock_parser.parse.return_value = {
        "url": constants["DYNAMIC_URL"],
        "html_content": constants["DYNAMIC_HTML_CONTENT"],
        "status_code": constants["HTTP_STATUS_OK"],
        "headers": {"Content-Type": constants["HTML_CONTENT_TYPE"]}
    }
    return mock_parser


@pytest.fixture
def mock_data_extractor(constants):
    """
    And a data extractor is available
    """
    mock_extractor = Mock()
    mock_extractor.extract.return_value = constants["EXTRACTED_TEXT_CONTENT"]
    return mock_extractor


@pytest.fixture
def mock_vector_generator(constants):
    """
    And a vector generator is available
    """
    mock_embeddings = [
        {"embedding": constants["DOC_EMBEDDINGS"][f"doc_{idx}"] * constants["PATTERN_REPETITIONS"], "doc_id": f"doc_{idx}"} 
        for idx in range(1, 4)
    ]
    mock_generator = Mock()
    mock_generator.generate.return_value = mock_embeddings
    return mock_generator


@pytest.fixture
def mock_metadata_generator(constants):
    """
    And a metadata generator is available
    """
    mock_generator = Mock()
    mock_generator.generate.return_value = [
        {
            "doc_id": doc_id,
            "source_url": constants["BASE_URL"],
            "creation_time": constants["MOCK_TIMESTAMP"],
            "content_length": constants["CONTENT_LENGTH"],
            "source_domain": constants["SOURCE_DOMAIN"]
        }
        for doc_id in constants["DOC_IDS"]
    ]
    return mock_generator

@pytest.fixture
def mock_document_storage_store_method_is_mocked(mock_document_storage):
    """
    And the document storage's store method is mocked
    """
    mock_document_storage.store = Mock()
    return mock_document_storage

@pytest.fixture
def mock_resources(
    mock_timestamp_service,
    mock_static_webpage_parser,
    mock_dynamic_webpage_parser, 
    mock_data_extractor,
    mock_vector_generator,
    mock_metadata_generator,
    mock_document_storage_store_method_is_mocked,
    mock_url_path_generator,
    mock_logger,
    mock_db,
    mock_llm,
    mock_get_cid,
    ):
    return {
        "static_parser": mock_static_webpage_parser,
        "dynamic_parser": mock_dynamic_webpage_parser,
        "data_extractor": mock_data_extractor,
        "vector_generator": mock_vector_generator,
        "metadata_generator": mock_metadata_generator,
        "document_storage": mock_document_storage_store_method_is_mocked,
        "url_path_generator": mock_url_path_generator,
        "timestamp_service": mock_timestamp_service,
        "logger": mock_logger,
        "db": mock_db,
        "llm": mock_llm,
        "get_cid": mock_get_cid
    }


@pytest.fixture
def make_document_retrieval_fixture(mock_resources, mock_configs):
    def _make_fixture(custom_resources=None, custom_configs=None):
        resource_dict = mock_resources.copy()
        configs = mock_configs
        if custom_resources is not None:
            resource_dict.update(custom_resources)
        if custom_configs is not None:
            configs = custom_configs
        try:
            return DocumentRetrievalFromWebsites(resources=resource_dict, configs=configs)
        except Exception as e:
            raise FixtureError(f"Failed to create DocumentRetrievalFromWebsites fixture: {e}") from e
    return _make_fixture


# Main mock fixture
@pytest.fixture
def document_retrieval_fixture(make_document_retrieval_fixture):
    """
    GIVEN a DocumentRetrievalFromWebsites instance is initialized
    """
    return make_document_retrieval_fixture()


# @pytest.fixture
# def sql_statements():
#     """
#     Fixture providing SQL statements for database setup
#     """
#     return {
#         "CREATE_DOCUMENTS_TABLE": """
#             CREATE TABLE IF NOT EXISTS documents (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 doc_id TEXT UNIQUE,
#                 url TEXT,
#                 content TEXT
#             )
#         """,
#         "CREATE_METADATA_TABLE": """
#             CREATE TABLE IF NOT EXISTS metadata (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 doc_id TEXT,
#                 source_url TEXT,
#                 creation_time TEXT,
#                 content_length INTEGER,
#                 source_domain TEXT,
#                 FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
#             )
#         """,
#         "CREATE_VECTORS_TABLE": """
#             CREATE TABLE IF NOT EXISTS vectors (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 doc_id TEXT,
#                 embedding BLOB,
#                 FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
#             )
#         """
#     }


# @pytest.fixture
# def mock_document_storage(tmp_path, sql_statements):
#     """
#     And a document storage service is available
#     """
#     conn = None
#     db_path = None
#     try:
#         # Create a temporary SQLite database
#         db_path = tmp_path / "test_documents.db"
#         conn = sqlite3.connect(str(db_path))
#         cursor = conn.cursor()

#         # Create tables for documents, metadata, and vectors using SQL statements from fixture
#         cursor.execute(sql_statements["CREATE_DOCUMENTS_TABLE"])
#         cursor.execute(sql_statements["CREATE_METADATA_TABLE"])
#         cursor.execute(sql_statements["CREATE_VECTORS_TABLE"])

#         conn.commit()

#         yield conn
#     except Exception as e:
#         raise FixtureError(f"Error setting up document storage service fixture: {e}") from e
#     finally:
#         # Cleanup
#         if conn is not None:
#             conn.close()
#         if db_path is not None and db_path.exists():
#             db_path.unlink()


@pytest.fixture
def mock_url_path_generator(constants):
    """
    And a URL path generator is available
    """
    mock_generator = Mock()
    # Return the input URL if it's a specific page, otherwise return GENERATED_URLS
    def generate_urls(url):
        # If URL has a path component beyond the domain, return just that URL
        if url.count('/') > 2:  # More than just http://domain/
            return [url]
        return constants["GENERATED_URLS"]
    mock_generator.generate.side_effect = generate_urls
    return mock_generator


@pytest.fixture
def base_domain_url(constants):
    """Fixture providing base domain URL list for tests"""
    return [constants["BASE_URL"]]

@pytest.fixture
def two_domain_urls(constants):
    """Fixture providing two domain URLs for tests"""
    return [constants["BASE_URL"], constants["TEST_DOT_ORG_URL"]]

@pytest.fixture
def single_page_url(constants):
    """Fixture providing a single page URL"""
    return [constants["PAGE1_URL"]]

@pytest.fixture
def timeout_urls(constants):
    """Fixture providing timeout URL list"""
    return [constants["TIMEOUT_URL"]]

@pytest.fixture
def urls_max_depth_is_one(constants):
    """Fixture providing shallow URLs for testing max depth"""
    return [
        constants["BASE_URL"],  # depth 0
        constants["PAGE1_URL"],  # depth 1
        constants["ABOUT_URL"],  # depth 1
    ]


@pytest.fixture
def timeout_error():
    """Fixture providing Timeout exception"""
    return Timeout("Request timed out")


@pytest.fixture
def http_error_404():
    """Fixture providing 404 HTTPError exception"""
    return HTTPError("404 Not Found")


@pytest.fixture
def exceptions(timeout_error, http_error_404):
    """Fixture providing a list of URL errors for testing"""
    return {
        "TIMEOUT_ERROR": timeout_error,
        "HTTP_404_ERROR": http_error_404
    }



@pytest.fixture
def multiple_domain_urls(constants):
    """Fixture providing multiple domain URLs for tests"""
    return [constants["BASE_URL"], constants["TEST_DOT_ORG_URL"]]


@pytest.fixture
def empty_domain_urls():
    """Fixture providing an empty list of domain URLs"""
    return []


@pytest.fixture
def dynamic_url_list(constants):
    """Fixture providing a list with dynamic URL"""
    return [constants["DYNAMIC_URL"]]


@pytest.fixture
def mock_vectors_5_docs(constants):
    """Fixture providing 5 mock vectors for testing"""
    embedding = constants["VECTOR_DIMENSION"] * 3 * [0.1]
    return [
        {"embedding": embedding, "doc_id": f"doc_{idx}"} for idx in range(5)
    ]


@pytest.fixture
def document_retrieval_fixture_single_url_generated(make_document_retrieval_fixture, single_page_url):
    """Fixture to configure URL generator to return a single page URL"""
    document_retrieval_fixture = make_document_retrieval_fixture()
    document_retrieval_fixture.url_path_generator.generate.side_effect = lambda url: single_page_url
    return document_retrieval_fixture


@pytest.fixture
def document_retrieval_fixture_single_text_extraction(make_document_retrieval_fixture, constants):
    """Fixture to configure data extractor to return single text content"""
    document_retrieval_fixture = make_document_retrieval_fixture()
    document_retrieval_fixture.data_extractor.extract.return_value = [constants["SINGLE_TEXT_CONTENT"]]
    return document_retrieval_fixture


@pytest.fixture
def document_retrieval_fixture_no_text_extracted(make_document_retrieval_fixture, constants):
    """Fixture to configure data extractor to return single text content"""
    document_retrieval_fixture = make_document_retrieval_fixture()
    document_retrieval_fixture.data_extractor.extract.return_value = []
    return document_retrieval_fixture


@pytest.fixture
def document_retrieval_fixture_depth_1_urls(make_document_retrieval_fixture, make_mock_configs, urls_max_depth_is_one):
    """Fixture to configure URL generator to return depth 1 URLs"""
    mock_configs = make_mock_configs(overrides={"max_depth": 1})
    document_retrieval_fixture = make_document_retrieval_fixture(mock_configs)
    document_retrieval_fixture.url_path_generator.generate.side_effect = lambda url: urls_max_depth_is_one
    return document_retrieval_fixture


@pytest.fixture
def document_retrieval_fixture_follow_links_is_false(make_document_retrieval_fixture, make_mock_configs, single_page_url):
    """Fixture to configure URL generator to return single page URL with follow_links=False"""
    mock_configs = make_mock_configs(overrides={"follow_links": False})
    document_retrieval_fixture = make_document_retrieval_fixture(mock_configs)
    document_retrieval_fixture.url_path_generator.generate.side_effect = lambda url: single_page_url
    return document_retrieval_fixture


@pytest.fixture
def make_static_parser_with_error(make_document_retrieval_fixture, constants, exceptions):
    def _make_static_parser_with_error(exception_key):
        try:
            document_retrieval_fixture = make_document_retrieval_fixture()
            document_retrieval_fixture.url_path_generator.generate.return_value = [constants["STATIC_PAGE_URL"]]
            document_retrieval_fixture.static_parser.parse.side_effect = [exceptions[exception_key]]
            return document_retrieval_fixture
        except Exception as e:
            raise FixtureError(f"Failed to create static parser: {e}") from e
    return _make_static_parser_with_error



@pytest.fixture
def make_static_parser_with_error_then_success(make_document_retrieval_fixture, constants, exceptions):
    def _make_static_parser(exception_key):
        try:
            timeout_url = constants["TIMEOUT_URL"]
            success_url = constants["SUCCESS_URL"]
            document_retrieval_fixture = make_document_retrieval_fixture()
            document_retrieval_fixture.url_path_generator.generate.return_value = [timeout_url, success_url]
            document_retrieval_fixture.static_parser.parse.side_effect = [
                exceptions[exception_key],
                {"url": success_url, "html_content": constants["STATIC_HTML_CONTENT"], "status_code": constants["HTTP_STATUS_OK"], "headers": {}}
            ]
            return document_retrieval_fixture
        except Exception as e:
            raise FixtureError(f"Failed to create static parser with error: {e}") from e
    return _make_static_parser

@pytest.fixture
def document_retrieval_fixture_timeout_then_success(make_static_parser_with_error_then_success):
    """Fixture to configure parser with timeout then success response"""
    document_retrieval_fixture = make_static_parser_with_error_then_success('TIMEOUT_ERROR')
    return document_retrieval_fixture

@pytest.fixture
def document_retrieval_fixture_404_then_success(make_static_parser_with_error_then_success):
    """Fixture to configure parser with 404 then success response"""
    document_retrieval_fixture = make_static_parser_with_error_then_success('HTTP_404_ERROR')
    return document_retrieval_fixture

@pytest.fixture
def document_retrieval_fixture_404(make_static_parser_with_error):
    """Fixture to configure parser with 404 for single URL"""
    document_retrieval_fixture = make_static_parser_with_error('HTTP_404_ERROR')
    return document_retrieval_fixture

@pytest.fixture
def document_retrieval_fixture_timeout(make_static_parser_with_error):
    """Fixture to configure parser with timeout for single URL"""
    document_retrieval_fixture = make_static_parser_with_error('TIMEOUT_ERROR')
    return document_retrieval_fixture


@pytest.fixture
def document_retrieval_fixture_10_docs(make_document_retrieval_fixture, constants):
    """Fixture to configure document retrieval to generate 10 documents"""
    document_retrieval_fixture = make_document_retrieval_fixture()
    urls_to_return = [f"{constants['BASE_URL']}/page{i}" for i in range(10)]
    document_retrieval_fixture.url_path_generator.generate.side_effect = lambda url: urls_to_return
    document_retrieval_fixture.data_extractor.extract.return_value = [constants["SINGLE_TEXT_CONTENT"]]
    return document_retrieval_fixture


@pytest.fixture
def document_retrieval_fixture_5_vectors(document_retrieval_fixture, mock_vectors_5_docs, constants):
    """Fixture to configure document retrieval to generate 5 vectors"""
    document_retrieval_fixture.data_extractor.extract.return_value = ["Single text content"]
    # vector_generator.generate() is called once per URL with the documents list from that URL
    # Since we have 5 URLs each producing 1 document, it's called 5 times with 1 document each
    # Return one vector per call
    embedding = constants["VECTOR_DIMENSION"] * [0.1]
    document_retrieval_fixture.vector_generator.generate.side_effect = [
        [{"embedding": embedding, "doc_id": f"doc_{idx}"}] for idx in range(5)
    ]
    return document_retrieval_fixture






class TestExecuteMethodAcceptsListofDomainURLs:
    """
    Rule: execute Method Accepts List of Domain URLs
    """
    # NOTE: Done
    def test_when_execute_called_with_single_domain_url_then_returns_dict(self, document_retrieval_fixture, base_domain_url):
        """
        Scenario: Execute with single domain URL
          GIVEN a single domain URL "https://example.com"
          WHEN I call execute with the domain URL list
          THEN the execution returns a dictionary
        """
        result = document_retrieval_fixture.execute(base_domain_url)
        assert isinstance(result, dict), f"Expected execute to return a dict, but {type(result).__name__}"

    # NOTE: Done
    @pytest.mark.parametrize("expected_key", ["documents", "metadata", "vectors"])
    def test_when_execute_called_with_single_domain_url_then_dict_has_expected_keys(
        self, expected_key, document_retrieval_fixture, base_domain_url):
        """
        Scenario: Execute with single domain URL
          GIVEN a single domain URL "https://example.com"
          WHEN I call execute with the domain URL list
          THEN the dictionary has expected keys
        """
        result = document_retrieval_fixture.execute(base_domain_url)
        assert expected_key in result, f"Expected key '{expected_key}' to be in result, but got keys: {list(result.keys())}"
        """
        Scenario: Execute with single domain URL
          GIVEN a single domain URL "https://example.com"
          WHEN I call execute with the domain URL list
          THEN the returned dictionary has expected keys "documents", "metadata", and "vectors"
        """
        result = document_retrieval_fixture.execute(base_domain_url)
        assert expected_key in result, f"Expected key '{expected_key}' to be in result, but got keys: {list(result.keys())}"

    @pytest.mark.parametrize("key,expected_value", [
        ("documents", list), ("metadata", list), ("vectors", list)
    ]) # NOTE: Done
    def test_when_execute_called_with_single_domain_url_then_dict_has_expected_value_types(
        self, key, expected_value, document_retrieval_fixture, base_domain_url):
        """
        Scenario: Execute with single domain URL
          GIVEN a single domain URL "https://example.com"
          WHEN I call execute with the domain URL list
          THEN the dictionary keys have expected value types
        """
        result = document_retrieval_fixture.execute(base_domain_url)
        actual_value = result[key]
        assert isinstance(actual_value, expected_value), \
            f"Expected '{key}' to be of type {expected_value.__name__}, but got {type(actual_value).__name__}"

    # NOTE: Done
    def test_when_execute_called_with_single_domain_url_then_documents_retrieved(self, document_retrieval_fixture, constants, base_domain_url):
        """
        Scenario: Execute with single domain URL
          GIVEN a single domain URL "https://example.com"
          WHEN I call execute with the domain URL list
          THEN documents are retrieved from the domain
        """
        zero = 0
        result = document_retrieval_fixture.execute(base_domain_url)
        len_documents = len(result["documents"])
        assert  len_documents > zero, \
            f"Expected documents to be retrieved from '{base_domain_url}', but got {len_documents}"

    # NOTE: Done
    def test_when_execute_called_with_multiple_domain_urls_then_documents_retrieved_from_all(
            self, document_retrieval_fixture, two_domain_urls):
        """
        Scenario: Execute with multiple domain URLs
          GIVEN multiple domain URLs ["https://example.com", "https://test.org"]
          WHEN I call execute with the domain URL list
          THEN documents are retrieved from all domains
        """
        # Each domain generates 5 URLs, each URL extracts 3 text content items
        # So we expect 2 domains * 5 URLs * 3 documents = 30 documents
        expected_count = 30
        result = document_retrieval_fixture.execute(two_domain_urls)
        actual_count = len(result["documents"])
        assert actual_count == expected_count, \
            f"Expected {expected_count} documents from {len(two_domain_urls)} domains, but got {actual_count}"

    # NOTE: Done
    def test_when_execute_called_with_multiple_domain_urls_then_results_aggregated(
            self, document_retrieval_fixture, two_domain_urls):
        """
        Scenario: Execute with multiple domain URLs
          GIVEN multiple domain URLs ["https://example.com", "https://test.org"]
          WHEN I call execute with the domain URL list
          THEN results are aggregated across domains
        """
        result = document_retrieval_fixture.execute(two_domain_urls)
        doc_count = len(result["documents"])
        metadata_count = len(result["metadata"])
        vector_count = len(result["vectors"])
        assert doc_count == metadata_count == vector_count, \
            f"Expected equal counts for documents, metadata, and vectors, but got docs={doc_count}, metadata={metadata_count}, vectors={vector_count}"

    # NOTE: Done
    def test_when_execute_called_with_empty_url_list_then_no_documents_retrieved(self, document_retrieval_fixture, constants):
        """
        Scenario: Execute with empty URL list
          GIVEN an empty list of domain URLs
          WHEN I call execute with the empty list
          THEN no documents are retrieved
        """
        domain_urls = []
        expected_count = 0
        result = document_retrieval_fixture.execute(domain_urls)
        doc_count = len(result["documents"])
        assert doc_count == expected_count, f"Expected {expected_count} documents from empty URL list, but got {doc_count}"


class TestStaticandDynamicWebpagesAreParsedAppropriately:
    """
    Rule: Static and Dynamic Webpages Are Parsed Appropriately
    Tests for: DocumentRetrievalFromWebsites.execute() - webpage parsing logic
    """

    @pytest.mark.parametrize("webpage_type,url_key,content_key", [
        ("static", "BASE_URL", "STATIC_HTML_CONTENT"),
        ("dynamic", "DYNAMIC_URL", "DYNAMIC_HTML_CONTENT")
    ]) # NOTE: Done
    def test_when_webpage_processed_then_appropriate_content_extracted(self, webpage_type, url_key, content_key, document_retrieval_fixture, constants):
        """
        Scenario: Webpages are parsed with appropriate parsers
          GIVEN a URL identified as static or dynamic
          WHEN execute is called with that URL
          THEN expected content is extracted
        """
        domain_urls = [constants[url_key]]
        result = document_retrieval_fixture.execute(domain_urls)
        # The data extractor processes raw HTML and returns extracted text content
        # We verify that documents were created from the URL
        first_doc_content = result["documents"][0]["content"]
        first_doc_url = result["documents"][0]["url"]
        assert first_doc_url == constants[url_key], f"Expected document URL to be '{constants[url_key]}', but got '{first_doc_url}'"

class TestURLGenerationExpandsDomainURLstoPageURLs:
    """
    Rule: URL Generation Expands Domain URLs to Page URLs
    Tests for: DocumentRetrievalFromWebsites.execute() - URL generation logic
    """
    @pytest.mark.parametrize("idx", [idx for idx in range(5)]) # NOTE: Done
    def test_when_single_domain_url_processed_then_documents_retrieved_from_each_page(self, idx, document_retrieval_fixture, constants, base_domain_url):
        """
        Scenario: Single domain URL is expanded to multiple page URLs
          GIVEN domain URL "https://example.com"
          And URL path generator produces 5 page URLs
          WHEN execute processes the domain
          THEN documents are retrieved from each page
        """
        expected_url = constants["GENERATED_URLS"][idx]
        result = document_retrieval_fixture.execute(base_domain_url)
        # Each URL generates 3 documents, so document at idx*3 should have the expected URL
        actual_url = result["documents"][idx * 3]["url"]
        assert expected_url == actual_url, \
            f"Expected document URL '{expected_url}' to be in retrieved documents, but got '{actual_url}'."

    # NOTE: Done
    def test_when_max_depth_configured_then_only_depth_1_urls_processed(
            self, document_retrieval_fixture_depth_1_urls, urls_max_depth_is_one, base_domain_url):
        """
        Scenario: URL generator respects max_depth configuration
          GIVEN domain URL "https://example.com"
          And max_depth is configured as 1
          WHEN execute processes the domain
          THEN only URLs at depth 1 or less are processed
        """
        expected_call_count = len(urls_max_depth_is_one)

        result = document_retrieval_fixture_depth_1_urls.execute(base_domain_url)
        actual_call_count = document_retrieval_fixture_depth_1_urls.data_extractor.extract.call_count

        assert actual_call_count == expected_call_count, \
            f"Expected '{expected_call_count}' URLs processed at depth 1, but got '{actual_call_count}'"

    # NOTE: Done
    def test_when_max_depth_configured_then_deeper_urls_not_followed(
            self, document_retrieval_fixture_depth_1_urls, urls_max_depth_is_one, base_domain_url):
        """
        Scenario: URL generator respects max_depth configuration
          GIVEN domain URL "https://example.com"
          And max_depth is configured as 1
          WHEN execute processes the domain
          THEN deeper URLs are not followed
        """
        # Each URL generates 3 documents (from EXTRACTED_TEXT_CONTENT)
        expected_url_count = len(urls_max_depth_is_one) * 3
        result = document_retrieval_fixture_depth_1_urls.execute(base_domain_url)
        actual_url_count = len(result["documents"])

        assert expected_url_count == actual_url_count, \
            f"Expected '{expected_url_count}' URLs at depth <= 1, but got '{actual_url_count}'"

    # NOTE: Done
    def test_when_follow_links_false_then_only_provided_urls_processed(
            self, document_retrieval_fixture_follow_links_is_false, single_page_url, base_domain_url):
        """
        Scenario: URL generator respects follow_links configuration
          GIVEN domain URL "https://example.com/page1"
          And follow_links is configured as False
          WHEN execute processes the domain
          THEN only the provided URLs are processed
        """
        expected_call_count = 1
        result = document_retrieval_fixture_follow_links_is_false.execute(single_page_url)
        actual_call_count = document_retrieval_fixture_follow_links_is_false.data_extractor.extract.call_count
        assert actual_call_count == expected_call_count, \
            f"Expected {expected_call_count} URL processed with follow_links=False, but got {actual_call_count}"

    # NOTE: Done
    def test_when_follow_links_false_then_no_additional_links_followed(
            self, document_retrieval_fixture_follow_links_is_false, single_page_url, base_domain_url):
        """
        Scenario: URL generator respects follow_links configuration
          GIVEN domain URL "https://example.com/page1"
          And follow_links is configured as False
          WHEN execute processes the domain
          THEN no additional links are followed
        """
        expected_url = single_page_url[0]
        result = document_retrieval_fixture_follow_links_is_false.execute(single_page_url)
        actual_url = result["documents"][0]["url"]
        assert actual_url == expected_url, \
            f"Expected URL to be '{expected_url}', but got '{actual_url}'"


@pytest.mark.parametrize("idx", [idx for idx in range(3)])
class TestDataExtractionConvertsRawDatatoStructuredStrings:
    """
    Rule: Data Extraction Converts Raw Data to Structured Strings
    Tests for: DocumentRetrievalFromWebsites.execute() - data extraction logic
    """
    # NOTE: Done
    def test_when_raw_html_processed_then_structured_text_strings_returned(self, idx, document_retrieval_fixture, constants, base_domain_url):
        """
        Scenario: Raw HTML is extracted to text strings
          GIVEN raw HTML data from a webpage
          WHEN the data extractor processes the raw data
          THEN structured text strings are returned
        """
        result = document_retrieval_fixture.execute(base_domain_url)
        returned_content = result["documents"][idx]["content"]

        assert isinstance(returned_content, str), \
            f"Expected extracted text to be a string, but got '{type(returned_content).__name__}'"

    @pytest.mark.parametrize("html_tag", [
        "<div>", "</div>", "<p>", "</p>", "<span>", "</span>"
    ]) # NOTE: Done
    def test_when_raw_html_processed_then_html_tags_removed(self, idx, html_tag, document_retrieval_fixture, base_domain_url):
        """
        Scenario: Raw HTML is extracted to text strings
          GIVEN raw HTML data from a webpage
          WHEN the data extractor processes the raw data
          THEN HTML tags are removed
        """
        result = document_retrieval_fixture.execute(base_domain_url)
        returned_content = result["documents"][idx]["content"]

        assert html_tag not in returned_content, f"Expected '{html_tag}' to not be in returned content, but got '{returned_content}'"

    # NOTE: Done
    def test_when_raw_html_processed_then_text_content_preserved(self, idx, document_retrieval_fixture, constants, base_domain_url):
        """
        Scenario: Raw HTML is extracted to text strings
          GIVEN raw HTML data from a webpage
          WHEN the data extractor processes the raw data
          THEN text content is preserved
        """
        result = document_retrieval_fixture.execute(base_domain_url)
        returned_content = result["documents"][idx]["content"]
        expected_content = constants["EXTRACTED_TEXT_CONTENT"][idx]
        
        assert returned_content == expected_content, \
            f"Expected extracted text to be '{expected_content}', but found '{returned_content}'"


# NOTE: Done
def test_when_empty_raw_data_processed_then_empty_list_returned(document_retrieval_fixture_no_text_extracted, base_domain_url):
    """
    Scenario: Empty raw data results in empty extraction
        GIVEN raw data that is empty or None
        WHEN the data extractor processes the raw data
        THEN an empty list is returned
    """
    expected_count = 0
    result = document_retrieval_fixture_no_text_extracted.execute(base_domain_url)
    actual_count = len(result["documents"])
    assert actual_count == expected_count, f"Expected '{expected_count}' documents from empty extraction, but got '{actual_count}'"


@pytest.mark.parametrize("idx", [idx for idx in range(3)])
class TestDocumentsAreCreatedwithURLContext:
    """
    Rule: Documents Are Created with URL Context
    Tests for: DocumentRetrievalFromWebsites.execute() - document creation logic
    """

    # NOTE: Done
    def test_when_documents_created_then_url_matches_expected(
            self, idx, document_retrieval_fixture, constants, base_domain_url):
        """
        Scenario: Documents include source URL
          GIVEN text strings extracted from a URL
          WHEN execute is called
          THEN the URL in each document in result matches the source URL
        """
        expected_url = constants["GENERATED_URLS"][idx]
        result = document_retrieval_fixture.execute(base_domain_url)
        
        # Each URL generates 3 documents, so document at idx*3 should have the expected URL
        actual_url = result["documents"][idx * 3]["url"]

        assert actual_url == expected_url, f"Expected document {idx} to have URL '{expected_url}', but got '{actual_url}'"

    # NOTE: Done
    def test_when_multiple_strings_extracted_then_all_have_same_source_url(self, idx, document_retrieval_fixture, constants, base_domain_url):
        """
        Scenario: Multiple strings create multiple documents
          GIVEN 3 extracted text strings from a single URL
          WHEN execute is called
          THEN each document in result has the same source URL
        """
        expected_url = constants["GENERATED_URLS"][idx]
        result = document_retrieval_fixture.execute(base_domain_url)

        # Each URL generates 3 documents, check documents from this URL (idx*3 to idx*3+3)
        docs_for_url = result["documents"][idx * 3:(idx * 3) + 3]
        all_same_url = all(doc["url"] == expected_url for doc in docs_for_url)
        assert all_same_url, f"Expected all documents to have URL '{expected_url}', but found different URLs"



class TestVectorsAreGeneratedforAllDocuments:
    """
    Rule: Vectors Are Generated for All Documents
    Tests for: DocumentRetrievalFromWebsites.execute() - vector generation logic
    """
    # NOTE: Done
    def test_when_vector_generation_performed_then_exactly_5_vectors_generated(self, document_retrieval_fixture_5_vectors, base_domain_url):
        """
        Scenario: Vector generator creates embeddings
          GIVEN a URL that produces 5 documents
          WHEN execute is called
          THEN exactly 5 vectors are present in result
        """
        expected_vector_count = 5
        result = document_retrieval_fixture_5_vectors.execute(base_domain_url)
        print(f"result: {result}")
        actual_vector_count = len(result['vectors'])
        assert actual_vector_count == expected_vector_count, f"Expected {expected_vector_count} vectors, but got {actual_vector_count}"

    @pytest.mark.parametrize("idx", [idx for idx in range(5)]) # NOTE: Done
    def test_when_vector_generation_performed_then_each_vector_corresponds_to_document(
        self, idx, document_retrieval_fixture_5_vectors, base_domain_url):
        """
        Scenario: Vector generator creates embeddings
          GIVEN a URL that produces 5 documents
          WHEN execute is called
          THEN each vector in result corresponds to a document in result
        """
        expected_vector_id = f"doc_{idx}"
        result = document_retrieval_fixture_5_vectors.execute(base_domain_url)
        actual_vector_id = result["vectors"][idx]["doc_id"]

        assert actual_vector_id == expected_vector_id, \
            f"Expected vector {idx} to have doc_id '{expected_vector_id}', but got '{actual_vector_id}'"
        """
        Scenario: Vector generator creates embeddings
          GIVEN a URL that produces 5 documents
          WHEN execute is called
          THEN each vector in result corresponds to a document in result
        """
        expected_vector_id = f"doc_{idx}"
        result = document_retrieval_fixture_5_vectors.execute(base_domain_url)
        actual_vector_id = result["vectors"][idx]["doc_id"]

        assert actual_vector_id == expected_vector_id, \
            f"Expected vector {idx} to have doc_id '{expected_vector_id}', but got '{actual_vector_id}'"

    # NOTE: Done
    def test_when_vectors_generated_then_dimensions_match_configuration(self, document_retrieval_fixture, constants, base_domain_url):
        """
        Scenario: Vector dimensions match configuration
          GIVEN documents are ready for vectorization
          And vector_dim is configured as 1536
          WHEN vectors are generated
          THEN each vector in result has dimension 1536
        """
        expected_dimension = constants["VECTOR_DIMENSION"]
        result = document_retrieval_fixture.execute(base_domain_url)
        print(result)

        actual_dimension = len(result['vectors'][0]['embedding'])
        assert actual_dimension == expected_dimension, f"Expected vector dimension {expected_dimension}, but got {actual_dimension}"


@pytest.mark.parametrize("idx", [idx for idx in range(3)])
class TestMetadataIsGeneratedforAllDocuments:
    """
    Rule: Metadata Is Generated for All Documents
    Tests for: DocumentRetrievalFromWebsites.execute() - metadata generation logic
    """
    @pytest.mark.parametrize("expected_field", [
        ("creation_time"),
        ("content_length"),
        ("source_url")
    ]) # NOTE: Done
    def test_when_metadata_generated_then_includes_required_properties(
        self, idx, expected_field, document_retrieval_fixture_single_url_generated, constants, single_page_url, base_domain_url
    ):
        """
        Scenario: Metadata includes document properties
          GIVEN documents from URL "https://example.com/page"
          WHEN execute is called
          THEN metadata in result includes required fields
        """
        # Act
        result = document_retrieval_fixture_single_url_generated.execute(base_domain_url)
        metadata = result["metadata"][idx]
        print(f"Metadata: {metadata}")

        # Assert
        assert expected_field in metadata, f"Expected metadata to include field '{expected_field}', but got keys: {list(result['metadata'].keys())}"


    @pytest.mark.parametrize("field,expected_value_key", [
        ("source_url", "BASE_URL"),
        ("content_length", "CONTENT_LENGTH"),
        ("creation_time", "MOCK_TIMESTAMP")
    ]) # NOTE: Done
    def test_when_metadata_generated_then_fields_have_expected_values(self, idx, document_retrieval_fixture_single_url_generated, constants, single_page_url, base_domain_url, field, expected_value_key):
        """
        Scenario: Metadata includes document properties
          GIVEN documents from URL "https://example.com/page"
          WHEN execute is called
          THEN each field in metadata in result has expected values
        """
        result = document_retrieval_fixture_single_url_generated.execute(base_domain_url)

        expected_value = constants[expected_value_key]
        actual_field_value = result["metadata"][idx][field]
        assert actual_field_value == expected_value, \
            f"Expected metadata field '{field}' to be '{expected_value}', but got '{result['metadata'][field]}'"

    # NOTE: Done
    def test_when_metadata_generated_then_count_matches_document_count(self, idx, document_retrieval_fixture_single_url_generated, constants, single_page_url, base_domain_url):
        """
        Scenario: Metadata includes document properties
          GIVEN documents from URL "https://example.com/page"
          WHEN metadata is generated
          THEN metadata count matches document count
        """
        result = document_retrieval_fixture_single_url_generated.execute(base_domain_url)

        metadata_count = len(result["metadata"])
        document_count = len(result["documents"])
        assert metadata_count == document_count, \
            f"Expected '{document_count}' metadata records to match document count, but got '{metadata_count}'"



class TestDocumentsVectorsandMetadataAreStored:
    """
    Rule: Documents, Vectors, and Metadata Are Stored
    Tests for: DocumentRetrievalFromWebsites.execute() - storage logic
    """
    @pytest.mark.parametrize("key,idx", [
        ("documents", 0),
        ("metadata", 1), 
        ("vectors", 2)
    ]) # NOTE: Done
    def test_when_storage_executes_then_storage_gets_docs_vecs_and_metadata(
        self, key, idx, document_retrieval_fixture_10_docs, two_domain_urls):
        """
        Scenario: All data is persisted to storage
          GIVEN 10 documents, vectors, and metadata are generated
          WHEN the storage step is called during execute
          THEN storage service receives all documents, vectors, and metadata
        """
        expected_count = 10
        result = document_retrieval_fixture_10_docs.execute(two_domain_urls)

        storage_call_args = document_retrieval_fixture_10_docs.document_storage.store.call_args[0]
        actual_count = len(storage_call_args[idx])
        assert actual_count == expected_count, f"Expected storage to receive '{expected_count}' for '{key}', but got '{actual_count}'"

    # NOTE: Done
    def test_when_storage_executes_then_value_of_success_key_is_true(
            self, document_retrieval_fixture_10_docs, two_domain_urls):
        """
        Scenario: All data is persisted to storage
          GIVEN 10 documents, vectors, and metadata are generated
          WHEN the storage step is called during execute
          THEN storage operation returns a dictionary with "success"=True
        """
        expected_return = True
        result = document_retrieval_fixture_10_docs.execute(two_domain_urls)

        assert result['success'] == expected_return, \
            f"Expected storage operation to return {expected_return}, but got {result['success']}"


class TestExecuteHandlesHTTPRequestFailures:
    """
    Rule: Execute Handles HTTP Request Failures
    Tests for: DocumentRetrievalFromWebsites.execute() - error handling logic
    """
    # NOTE: Done
    def test_when_url_times_out_then_request_retried_max_times(self, document_retrieval_fixture_timeout, constants, base_domain_url):
        """
        Scenario: Timeout during webpage fetch is handled
          GIVEN a URL that times out after timeout_seconds
          WHEN execute is called with that URL
          THEN the request is retried up to max_retries times
        """
        expected_retries = constants["MAX_RETRIES"]
        result = document_retrieval_fixture_timeout.execute(base_domain_url)
        actual_retries = document_retrieval_fixture_timeout.static_parser.parse.call_count
        assert actual_retries == expected_retries, f"Expected {expected_retries} retry attempts, but got {actual_retries}"

    # NOTE: Done
    def test_when_url_times_out_and_retries_fail_then_url_skipped(self, document_retrieval_fixture_timeout, base_domain_url):
        """
        Scenario: Timeout during webpage fetch is handled
          GIVEN a URL that times out after timeout_seconds
          WHEN execute is called with that URL
          THEN the result contains no documents for that URL
        """
        expected_count = 0
        result = document_retrieval_fixture_timeout.execute(base_domain_url)

        actual_count = len(result["documents"])
        assert actual_count == expected_count, f"Expected {expected_count} documents after timeout failures, but got {actual_count}"

    # NOTE: Done
    def test_when_url_times_out_then_other_urls_still_processed(self, document_retrieval_fixture_timeout_then_success, constants):
        """
        Scenario: Timeout during webpage fetch is handled
          GIVEN a URL that times out after timeout_seconds
          WHEN execute is called with that URL
          THEN other urls are in the returned result
        """
        expected_min = 0
        result = document_retrieval_fixture_timeout_then_success.execute([constants["BASE_URL"]])

        actual_count = len(result["documents"])
        assert actual_count > expected_min, f"Expected more than {expected_min} documents after timeout, but got {actual_count}"

    # NOTE: Done
    def test_when_url_returns_404_then_no_documents_in_result(self, document_retrieval_fixture_404, constants):
        """
        Scenario: 404 Not Found error is handled gracefully
          GIVEN a URL that returns 404 status
          WHEN execute is called with that URL
          THEN result contains no documents for that URL
        """
        expected_count = 0
        result = document_retrieval_fixture_404.execute([constants["BASE_URL"]])
        actual_count = len(result["documents"])
        assert actual_count == expected_count, f"Expected {expected_count} documents for 404 URL, but got {actual_count}"

    # NOTE: Done
    def test_when_url_returns_404_then_processing_continues_with_remaining_urls(self, document_retrieval_fixture_404_then_success, constants):
        """
        Scenario: 404 Not Found error is handled gracefully
          GIVEN a URL that returns 404 status
          WHEN execute is called with that URL
          THEN other urls are in the returned result
        """
        expected_min = 0
        result = document_retrieval_fixture_404_then_success.execute([constants["BASE_URL"]])
        actual_count = len(result["documents"])
        assert actual_count > expected_min, f"Expected documents after 404 error, but got {actual_count}"


    @pytest.mark.parametrize("invalid_type", [
        None,
        123,
        45.67,
        {},
        (),
        set()
    ]) # NOTE: Done
    def test_when_invalid_url_container_processed_then_type_error_raised(self, invalid_type, document_retrieval_fixture, constants):
        """
        Scenario: Invalid URL format is rejected
          GIVEN an invalid URL "not-a-valid-url"
          WHEN execute is called with that URL
          THEN TypeError is raised
        """
        with pytest.raises(TypeError, match=r"domain_urls must be a list"):
            document_retrieval_fixture.execute(invalid_type)

    @pytest.mark.parametrize("invalid_url", [
        None,
        123,
        45.67,
        {},
        (),
        set()
    ]) # NOTE: Done
    def test_when_invalid_url_type_processed_then_type_error_raised(self, invalid_url, document_retrieval_fixture, constants):
        """
        Scenario: Invalid URL type is rejected
          GIVEN an invalid URL "not-a-valid-url"
          WHEN execute is called with that URL
          THEN TypeError is raised
        """
        with pytest.raises(TypeError, match=r"Each domain URL must be a string"):
            document_retrieval_fixture.execute([invalid_url])

    @pytest.mark.parametrize("invalid_url", [
        "not-a-valid-url",
        "http:/incomplete.com",
        "www..double-dot.com",
        "ftp://unsupported-protocol.com",
        "http://",
        "://missing-scheme.com",
        "http//missing-colon.com",
        "http://contains spaces.com",
    ]) # NOTE: Done
    def test_when_invalid_url_value_processed_then_value_error_raised(self, invalid_url, document_retrieval_fixture, constants):
        """
        Scenario: Invalid URL format is rejected
          GIVEN an invalid URL "not-a-valid-url"
          WHEN execute is called with that URL
          THEN ValueError is raised
        """
        with pytest.raises(ValueError, match=r"Invalid domain URLs provided"):
            document_retrieval_fixture.execute([invalid_url])


class TestUserAgentConfigurationIsApplied:
    """
    Rule: User Agent Configuration Is Applied
    Tests for: DocumentRetrievalFromWebsites.execute() - user agent configuration
    """
    # NOTE: Done
    def test_when_static_webpage_requested_then_custom_user_agent_included(
            self, document_retrieval_fixture, base_domain_url, constants, static_parser):
        """
        Scenario: Custom user agent is sent in HTTP requests
          GIVEN user_agent is configured as "CustomBot/1.0"
          WHEN a static webpage is requested during execute
          THEN the HTTP request includes User-Agent header "CustomBot/1.0"
        """
        expected_user_agent = constants["CUSTOM_USER_AGENT"]
        result = document_retrieval_fixture.execute(base_domain_url)
        call_args = static_parser.parse.call_args
        actual_user_agent = call_args[1]['headers']['User-Agent']
        assert actual_user_agent == expected_user_agent, f"Expected User-Agent '{expected_user_agent}', but got '{actual_user_agent}'"


if __name__ == "__main__":
    pytest.main()
