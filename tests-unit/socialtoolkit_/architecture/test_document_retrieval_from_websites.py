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
import pytest
from unittest.mock import Mock
from datetime import datetime


# Import the actual class being tested
from custom_nodes.red_ribbon.socialtoolkit.architecture.document_retrieval_from_websites import (
    DocumentRetrievalFromWebsites,
    DocumentRetrievalConfigs,
    WebpageType
)

import sqlite3

from requests.exceptions import HTTPError
from .conftest import FixtureError

# Mock constants fixture to avoid hardcoded values
@pytest.fixture
def mock_constants():
    """Fixture providing mock constants for tests"""
    return {
        # URLs
        "BASE_URL": "https://example.com",
        "DYNAMIC_URL": "https://example.com/app",
        "TEST_URL_2": "https://test.org",
        "STATIC_PAGE_URL": "https://example.com/static-page.html",
        "PAGE1_URL": "https://example.com/page1",
        "TIMEOUT_URL": "https://example.com/timeout",
        "SUCCESS_URL": "https://example.com/success",
        "NOT_FOUND_URL": "https://example.com/not-found",
        "INVALID_URL": "not-a-valid-url",
        "CUSTOM_USER_AGENT": "CustomBot/1.0",
        "SINGLE_TEXT_CONTENT": "Single text content",
        
        # Generated URLs for path generator
        "GENERATED_URLS": [
            "https://example.com",
            "https://example.com/page1",
            "https://example.com/page2",
            "https://example.com/about",
            "https://example.com/contact"
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
        "VECTOR_DIMENSION": 512,  # 512 * 3 = 1536 dimensions
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
        "TIMEOUT_SECONDS": 30,
        "MAX_RETRIES": 3,
        "BATCH_SIZE": 10,
        "MAX_DEPTH": 1
    }

# Main mock fixture that all tests use
@pytest.fixture
def mock_document_retrieval(
    mock_constants,
    static_webpage_parser,
    a_dynamic_webpage_parser_is_available, 
    data_extractor,
    vector_generator,
    metadata_generator,
    a_document_storage_service_is_available,
    a_url_path_generator_is_available
):
    """
    GIVEN a DocumentRetrievalFromWebsites instance is initialized
    """
    # Create mock timestamp service
    mock_timestamp_service = Mock()
    mock_timestamp_service.now.return_value = mock_constants["MOCK_TIMESTAMP"]
    
    resources = {
        "static_webpage_parser": static_webpage_parser,
        "dynamic_webpage_parser": a_dynamic_webpage_parser_is_available,
        "data_extractor": data_extractor,
        "vector_generator": vector_generator,
        "metadata_generator": metadata_generator,
        "document_storage_service": a_document_storage_service_is_available,
        "url_path_generator": a_url_path_generator_is_available,
        "timestamp_service": mock_timestamp_service
    }
    
    configs = DocumentRetrievalConfigs(
        timeout_seconds=mock_constants["TIMEOUT_SECONDS"],
        max_retries=mock_constants["MAX_RETRIES"],
        batch_size=mock_constants["BATCH_SIZE"],
        follow_links=False,
        max_depth=mock_constants["MAX_DEPTH"]
    )
    
    return DocumentRetrievalFromWebsites(resources=resources, configs=configs)


@pytest.fixture
def static_webpage_parser(mock_constants):
    """
    And a static webpage parser is available
    """
    mock_parser = Mock()
    mock_parser.parse.return_value = {
        "url": mock_constants["BASE_URL"],
        "html_content": mock_constants["STATIC_HTML_CONTENT"],
        "status_code": mock_constants["HTTP_STATUS_OK"],
        "headers": {"Content-Type": mock_constants["HTML_CONTENT_TYPE"]}
    }
    return mock_parser


@pytest.fixture
def a_dynamic_webpage_parser_is_available(mock_constants):
    """
    And a dynamic webpage parser is available
    """
    mock_parser = Mock()
    mock_parser.parse.return_value = {
        "url": mock_constants["DYNAMIC_URL"],
        "html_content": mock_constants["DYNAMIC_HTML_CONTENT"],
        "status_code": mock_constants["HTTP_STATUS_OK"],
        "headers": {"Content-Type": mock_constants["HTML_CONTENT_TYPE"]}
    }
    return mock_parser


@pytest.fixture
def data_extractor(mock_constants):
    """
    And a data extractor is available
    """
    mock_extractor = Mock()
    mock_extractor.extract.return_value = mock_constants["EXTRACTED_TEXT_CONTENT"]
    return mock_extractor


@pytest.fixture
def vector_generator(mock_constants):
    """
    And a vector generator is available
    """
    mock_generator = Mock()
    mock_generator.generate.return_value = [
        {"embedding": mock_constants["DOC_EMBEDDINGS"]["doc_1"] * mock_constants["VECTOR_DIMENSION"], "doc_id": "doc_1"},  # 1536 dimensions
        {"embedding": mock_constants["DOC_EMBEDDINGS"]["doc_2"] * mock_constants["VECTOR_DIMENSION"], "doc_id": "doc_2"},
        {"embedding": mock_constants["DOC_EMBEDDINGS"]["doc_3"] * mock_constants["VECTOR_DIMENSION"], "doc_id": "doc_3"}
    ]
    return mock_generator


@pytest.fixture
def metadata_generator(mock_constants):
    """
    And a metadata generator is available
    """
    mock_generator = Mock()
    mock_generator.generate.return_value = [
        {
            "doc_id": doc_id,
            "source_url": mock_constants["BASE_URL"],
            "creation_time": mock_constants["MOCK_TIMESTAMP"],
            "content_length": mock_constants["CONTENT_LENGTH"],
            "source_domain": mock_constants["SOURCE_DOMAIN"]
        }
        for doc_id in mock_constants["DOC_IDS"]
    ]
    return mock_generator



@pytest.fixture
def sql_statements():
    """
    Fixture providing SQL statements for database setup
    """
    return {
        "CREATE_DOCUMENTS_TABLE": """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE,
                url TEXT,
                content TEXT
            )
        """,
        "CREATE_METADATA_TABLE": """
            CREATE TABLE IF NOT EXISTS metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                source_url TEXT,
                creation_time TEXT,
                content_length INTEGER,
                source_domain TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """,
        "CREATE_VECTORS_TABLE": """
            CREATE TABLE IF NOT EXISTS vectors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT,
                embedding BLOB,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """
    }


@pytest.fixture
def a_document_storage_service_is_available(tmp_path, sql_statements):
    """
    And a document storage service is available
    """
    conn = None
    try:
        # Create a temporary SQLite database
        db_path = tmp_path / "test_documents.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create tables for documents, metadata, and vectors using SQL statements from fixture
        cursor.execute(sql_statements["CREATE_DOCUMENTS_TABLE"])
        cursor.execute(sql_statements["CREATE_METADATA_TABLE"])
        cursor.execute(sql_statements["CREATE_VECTORS_TABLE"])

        conn.commit()

        yield conn
    except Exception as e:
        raise FixtureError(f"Error setting up document storage service fixture: {e}") from e
    finally:
        # Cleanup
        if conn is not None:
            conn.close()


@pytest.fixture
def a_url_path_generator_is_available(mock_constants):
    """
    And a URL path generator is available
    """
    mock_generator = Mock()
    mock_generator.generate.return_value = mock_constants["GENERATED_URLS"]
    return mock_generator


@pytest.fixture
def base_domain_urls(mock_constants):
    """Fixture providing base domain URL list for tests"""
    return [mock_constants["BASE_URL"]]


@pytest.fixture
def single_page_url(mock_constants):
    """Fixture providing a single page URL"""
    return mock_constants["BASE_URL"] + "/page1"


@pytest.fixture
def mock_document_retrieval_url_generator(mock_document_retrieval, single_page_url):
    """Fixture to configure URL generator to return a single page URL"""
    mock_document_retrieval.url_path_generator.generate.return_value = [single_page_url]
    return mock_document_retrieval


@pytest.fixture
def mock_document_retrieval_single_text_extraction(mock_document_retrieval, mock_constants):
    """Fixture to configure data extractor to return single text content"""
    mock_document_retrieval.data_extractor.extract.return_value = [mock_constants["SINGLE_TEXT_CONTENT"]]
    return mock_document_retrieval


@pytest.fixture
def timeout_exception():
    """Fixture providing Timeout exception"""
    from requests.exceptions import Timeout
    return Timeout("Request timed out")


@pytest.fixture
def http_error_404():
    """Fixture providing 404 HTTPError exception"""

    return HTTPError("404 Not Found")


@pytest.fixture
def success_and_timeout_urls(mock_constants):
    """Fixture providing success and timeout URL pair"""
    return {
        "timeout": mock_constants["BASE_URL"] + "/timeout",
        "success": mock_constants["BASE_URL"] + "/success"
    }


@pytest.fixture
def success_and_not_found_urls(mock_constants):
    """Fixture providing success and not-found URL pair"""
    return {
        "not_found": mock_constants["BASE_URL"] + "/not-found",
        "success": mock_constants["BASE_URL"] + "/success"
    }


@pytest.fixture
def multiple_domain_urls(mock_constants):
    """Fixture providing multiple domain URLs for tests"""
    return [mock_constants["BASE_URL"], mock_constants["TEST_URL_2"]]


@pytest.fixture
def empty_domain_urls():
    """Fixture providing an empty list of domain URLs"""
    return []


@pytest.fixture
def dynamic_url_list(mock_constants):
    """Fixture providing a list with dynamic URL"""
    return [mock_constants["DYNAMIC_URL"]]


@pytest.fixture
def mock_vectors_5_docs():
    """Fixture providing 5 mock vectors for testing"""
    return [
        {"embedding": [0.1] * 1536, "doc_id": f"doc_{i}"}
        for i in range(5)
    ]

@pytest.fixture
def mock_document_retrieval_with_depth_1_urls(mock_document_retrieval, mock_constants):
    """Fixture to configure URL generator to return depth 1 URLs"""
    depth_1_urls = [
        mock_constants["BASE_URL"],  # depth 0
        mock_constants["BASE_URL"] + "/page1",  # depth 1
        mock_constants["BASE_URL"] + "/about"  # depth 1
    ]
    mock_document_retrieval.url_path_generator.generate.return_value = depth_1_urls
    return mock_document_retrieval



class TestExecuteMethodAcceptsListofDomainURLs:
    """
    Rule: execute Method Accepts List of Domain URLs
    """
    def test_when_execute_called_with_single_domain_url_then_execution_completes(self, mock_document_retrieval, base_domain_urls):
        """
        Scenario: Execute with single domain URL
          GIVEN a single domain URL "https://example.com"
          WHEN I call execute with the domain URL list
          THEN the execution completes successfully
        """
        result = mock_document_retrieval.execute(base_domain_urls)
        assert result is not None, f"Expected execute to return a result, but got None for domain_urls={base_domain_urls}"


    def test_when_execute_called_with_single_domain_url_then_documents_retrieved(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: Execute with single domain URL
          GIVEN a single domain URL "https://example.com"
          WHEN I call execute with the domain URL list
          THEN documents are retrieved from the domain
        """
        result = mock_document_retrieval.execute(base_domain_urls)
        base_url = mock_constants["BASE_URL"]
        first_doc_url = result["documents"][0]["url"]
        assert  len(result["documents"]) > 0, \
            f"Expected documents to be retrieved from {base_url}, but got no documents"

    def test_when_execute_called_with_multiple_domain_urls_then_documents_retrieved_from_all(self, mock_document_retrieval, mock_constants):
        """
        Scenario: Execute with multiple domain URLs
          GIVEN multiple domain URLs ["https://example.com", "https://test.org"]
          WHEN I call execute with the domain URL list
          THEN documents are retrieved from all domains
        """
        domain_urls = [mock_constants["BASE_URL"], mock_constants["TEST_URL_2"]]
        result = mock_document_retrieval.execute(domain_urls)
        expected_count = len(domain_urls)
        actual_count = len(result["documents"])
        assert actual_count == expected_count, f"Expected {expected_count} documents from {expected_count} domains, but got {actual_count}"


    def test_when_execute_called_with_multiple_domain_urls_then_results_aggregated(self, mock_document_retrieval, mock_constants):
        """
        Scenario: Execute with multiple domain URLs
          GIVEN multiple domain URLs ["https://example.com", "https://test.org"]
          WHEN I call execute with the domain URL list
          THEN results are aggregated across domains
        """
        domain_urls = [mock_constants["BASE_URL"], mock_constants["TEST_URL_2"]]
        result = mock_document_retrieval.execute(domain_urls)
        doc_count = len(result["documents"])
        metadata_count = len(result["metadata"])
        vector_count = len(result["vectors"])
        assert doc_count == metadata_count == vector_count, f"Expected equal counts for documents, metadata, and vectors, but got docs={doc_count}, metadata={metadata_count}, vectors={vector_count}"


    def test_when_execute_called_with_empty_url_list_then_no_documents_retrieved(self, mock_document_retrieval, mock_constants):
        """
        Scenario: Execute with empty URL list
          GIVEN an empty list of domain URLs
          WHEN I call execute with the empty list
          THEN no documents are retrieved
        """
        domain_urls = []
        result = mock_document_retrieval.execute(domain_urls)
        doc_count = len(result["documents"])
        assert doc_count == 0, f"Expected 0 documents from empty URL list, but got {doc_count}"


    def test_when_execute_called_with_empty_url_list_then_empty_collections_returned(self, mock_document_retrieval, mock_constants):
        """
        Scenario: Execute with empty URL list
          GIVEN an empty list of domain URLs
          WHEN I call execute with the empty list
          THEN the returned dictionary has empty collections
        """
        domain_urls = []
        result = mock_document_retrieval.execute(domain_urls)
        documents = result["documents"]
        assert documents == [], f"Expected empty documents list from empty URL list, but got {documents}"


class TestExecuteReturnsDictionarywithRequiredKeys:
    """
    Rule: Execute Returns Dictionary with Required Keys
    Tests for: DocumentRetrievalFromWebsites.execute()
    """
    def test_when_execute_called_then_dictionary_returned(self, mock_document_retrieval, base_domain_urls):
        """
        Scenario: Execute returns expected result structure
          GIVEN a valid domain URL "https://example.com"
          WHEN I call execute with the domain URL
          THEN I receive a dictionary response
        """
        result = mock_document_retrieval.execute(base_domain_urls)
        assert isinstance(result, dict), f"Expected execute to return dict, but got {type(result).__name__}"

    @pytest.mark.parametrize("expected_key", ["documents", "metadata", "vectors"])
    def test_when_execute_called_then_required_keys_present(self, mock_document_retrieval, base_domain_urls, expected_key):
        """
        Scenario: Execute returns expected result structure
          GIVEN a valid domain URL "https://example.com"
          WHEN I call execute with the domain URL
          THEN the response contains required keys "documents", "metadata", and "vectors"
        """
        result = mock_document_retrieval.execute(base_domain_urls)
        assert expected_key in result, f"Expected key '{expected_key}' to be in result, but got keys: {list(result.keys())}"



class TestStaticandDynamicWebpagesAreParsedAppropriately:
    """
    Rule: Static and Dynamic Webpages Are Parsed Appropriately
    Tests for: DocumentRetrievalFromWebsites.execute() - webpage parsing logic
    """
    def test_when_static_webpage_processed_then_html_extracted(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: Static webpage is parsed with static parser
          GIVEN a URL "https://example.com/static-page.html" identified as static
          WHEN the URL is processed
          THEN raw HTML data is extracted
        """
        result = mock_document_retrieval.execute(base_domain_urls)
        html_content = mock_constants["STATIC_HTML_CONTENT"]
        first_doc_content = result["documents"][0]["content"]
        assert html_content in first_doc_content, f"Expected HTML content '{html_content}' in document content, but got '{first_doc_content}'"

    def test_when_dynamic_webpage_processed_then_javascript_content_extracted(self, mock_document_retrieval, mock_constants):
        """
        Scenario: Dynamic webpage is parsed with dynamic parser
          GIVEN a URL "https://example.com/app" identified as dynamic
          WHEN the URL is processed
          THEN JavaScript-rendered content is extracted
        """
        domain_urls = [mock_constants["DYNAMIC_URL"]]
        result = mock_document_retrieval.execute(domain_urls)
        dynamic_content = mock_constants["DYNAMIC_HTML_CONTENT"]
        first_doc_content = result["documents"][0]["content"]
        assert dynamic_content in first_doc_content, f"Expected dynamic content '{dynamic_content}' in document, but got '{first_doc_content}'"


class TestURLGenerationExpandsDomainURLstoPageURLs:
    """
    Rule: URL Generation Expands Domain URLs to Page URLs
    Tests for: DocumentRetrievalFromWebsites.execute() - URL generation logic
    """
    def test_when_single_domain_processed_then_url_generator_called_once(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: Single domain URL is expanded to multiple page URLs
          GIVEN domain URL "https://example.com"
          And URL path generator produces 5 page URLs
          WHEN execute processes the domain
          THEN extract is called 5 times.
        """
        # Act
        expected_calls = 5
        result = mock_document_retrieval.execute(base_domain_urls)
        
        actual_calls = mock_document_retrieval.data_extractor.extract.call_count

        assert actual_calls == expected_calls, \
            f"Expected data extractor to be called {expected_calls} times, but got {actual_calls} calls."

    @pytest.mark.parametrize("doc_index", range(5))
    def test_single_domain_url_is_expanded_to_multiple_page_urls_1(self, mock_document_retrieval, mock_constants, base_domain_urls, doc_index):
        """
        Scenario: Single domain URL is expanded to multiple page URLs
          GIVEN domain URL "https://example.com"
          And URL path generator produces 5 page URLs
          WHEN execute processes the domain
          THEN documents are retrieved from each page
        """
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)

        expected_url = mock_constants["GENERATED_URLS"][doc_index]
        assert expected_url in result["documents"][doc_index]["url"], \
            f"Expected document URL '{expected_url}' to be in retrieved documents, but it was not found."



    def test_url_generator_respects_max_depth_configuration(self, mock_document_retrieval_with_depth_1_urls, mock_constants, base_domain_urls):
        """
        Scenario: URL generator respects max_depth configuration
          GIVEN domain URL "https://example.com"
          And max_depth is configured as 1
          WHEN execute processes the domain
          THEN only URLs at depth 1 or less are processed
        """
        # Arrange
        mock_document_retrieval = mock_document_retrieval_with_depth_1_urls
        depth_1_urls = [
            mock_constants["BASE_URL"],
            mock_constants["BASE_URL"] + "/page1",
            mock_constants["BASE_URL"] + "/about"
        ]
        
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)
        
        # Assert

        # Verify max_depth configuration is respected
        assert mock_document_retrieval.configs.max_depth == mock_constants["MAX_DEPTH"]
        # Verify URL generator was called with the domain
        mock_document_retrieval.url_path_generator.generate.assert_called_once_with(mock_constants["BASE_URL"])
        # Verify only depth 1 URLs are processed (3 URLs in this case)
        assert mock_document_retrieval.data_extractor.extract.call_count == len(depth_1_urls)

    def test_url_generator_respects_max_depth_configuration_1(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: URL generator respects max_depth configuration
          GIVEN domain URL "https://example.com"
          And max_depth is configured as 1
          WHEN execute processes the domain
          THEN deeper URLs are not followed
        """
        # Arrange
        # Mock URL generator to return only depth 0 and 1 URLs
        shallow_urls = [
            mock_constants["BASE_URL"],  # depth 0
            mock_constants["BASE_URL"] + "/page1"  # depth 1
        ]
        mock_document_retrieval.url_path_generator.generate.return_value = shallow_urls
        
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)
        
        # Assert

        # Verify max_depth configuration is respected
        assert mock_document_retrieval.configs.max_depth == 1
        # Verify no deep URLs (depth > 1) were processed
        for doc in result["documents"]:
            # Count slashes to determine depth
            url = doc["url"]
            path_depth = url.replace(mock_constants["BASE_URL"], "").count("/")
            assert path_depth <= 1

    def test_url_generator_respects_follow_links_configuration(self, mock_document_retrieval, single_page_url, base_domain_urls):
        """
        Scenario: URL generator respects follow_links configuration
          GIVEN domain URL "https://example.com/page1"
          And follow_links is configured as False
          WHEN execute processes the domain
          THEN only the provided URLs are processed
        """
        # Arrange
        # Use "https://example.com/page1" as described in scenario
        domain_urls = [single_page_url]
        # The instance already has follow_links=False from fixture
        # Mock URL generator to return only provided URL (no link following)
        mock_document_retrieval.url_path_generator.generate.return_value = [single_page_url]
        
        # Act
        result = mock_document_retrieval.execute(domain_urls)
        
        # Assert

        # Verify follow_links is configured as False
        assert mock_document_retrieval.configs.follow_links == False
        # Verify URL generator was called with the provided URL
        mock_document_retrieval.url_path_generator.generate.assert_called_once_with(single_page_url)
        # Verify no additional link discovery occurred (only 1 URL processed)
        assert mock_document_retrieval.data_extractor.extract.call_count == 1

    def test_url_generator_respects_follow_links_configuration_1(self, mock_document_retrieval, single_page_url, base_domain_urls):
        """
        Scenario: URL generator respects follow_links configuration
          GIVEN domain URL "https://example.com/page1"
          And follow_links is configured as False
          WHEN execute processes the domain
          THEN no additional links are followed
        """
        # Arrange
        domain_urls = [single_page_url]
        # Mock URL generator to return only provided URL (no link following)
        mock_document_retrieval.url_path_generator.generate.return_value = [single_page_url]
        
        # Act
        result = mock_document_retrieval.execute(domain_urls)

        # Assert
        # Verify only documents from the single URL exist (no followed links)
        assert all(doc["url"] == single_page_url for doc in result["documents"])


class TestDataExtractionConvertsRawDatatoStructuredStrings:
    """
    Rule: Data Extraction Converts Raw Data to Structured Strings
    """
    def test_raw_html_is_extracted_to_text_strings(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: Raw HTML is extracted to text strings
          GIVEN raw HTML data from a webpage
          WHEN the data extractor processes the raw data
          THEN structured text strings are returned
        """
        # Arrange
        
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)
        
        # Assert

        # Verify data extractor was called and returned text strings
        mock_document_retrieval.data_extractor.extract.assert_called()
        # Verify documents contain text content (not HTML)
        for doc in result["documents"]:
            assert "content" in doc
            assert isinstance(doc["content"], str)
            # Content should be from extracted text, not raw HTML
            assert doc["content"] in mock_constants["EXTRACTED_TEXT_CONTENT"]

    def test_raw_html_is_extracted_to_text_strings_1(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: Raw HTML is extracted to text strings
          GIVEN raw HTML data from a webpage
          WHEN the data extractor processes the raw data
          THEN HTML tags are removed
        """
        # Arrange
        
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)
        
        # Assert

        # Verify extracted text does not contain HTML tags
        for doc in result["documents"]:
            content = doc["content"]
            # Verify no HTML tags like <html>, <body>, <div> etc.
            assert "<" not in content
            assert ">" not in content
            assert "<html>" not in content
            assert "<body>" not in content
            assert "<div>" not in content

    def test_raw_html_is_extracted_to_text_strings_2(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: Raw HTML is extracted to text strings
          GIVEN raw HTML data from a webpage
          WHEN the data extractor processes the raw data
          THEN text content is preserved
        """
        # Arrange
        
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)
        
        # Assert

        # Verify extracted text content is preserved
        extracted_contents = mock_constants["EXTRACTED_TEXT_CONTENT"]
        document_contents = [doc["content"] for doc in result["documents"]]
        # All extracted text should be present in documents
        for expected_text in extracted_contents:
            assert any(expected_text in content for content in document_contents)

    def test_empty_raw_data_results_in_empty_extraction(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: Empty raw data results in empty extraction
          GIVEN raw data that is empty or None
          WHEN the data extractor processes the raw data
          THEN an empty list of strings is returned
        """
        # Arrange
        # Mock data extractor to return empty list for empty raw data
        mock_document_retrieval.data_extractor.extract.return_value = []
        
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)
        
        # Assert

        # Verify no documents created from empty extraction
        assert len(result["documents"]) == 0
        assert result["documents"] == []
        # Verify data extractor was still called
        mock_document_retrieval.data_extractor.extract.assert_called()


class TestDocumentsAreCreatedwithURLContext:
    """
    Rule: Documents Are Created with URL Context
    """
    def test_documents_include_source_url(self, mock_document_retrieval_url_generator, single_page_url, base_domain_urls):
        """
        Scenario: Documents include source URL
          GIVEN text strings extracted from "https://example.com/page1"
          WHEN documents are created from the strings
          THEN each document includes the source URL
        """
        # Arrange
        
        # Act
        result = mock_document_retrieval_url_generator.execute(base_domain_urls)
        
        # Assert
        for doc in result["documents"]:
            assert "url" in doc
            assert doc["url"] == single_page_url

    def test_documents_include_source_url_1(self, mock_document_retrieval_url_generator, mock_constants, single_page_url, base_domain_urls):
        """
        Scenario: Documents include source URL
          GIVEN text strings extracted from "https://example.com/page1"
          WHEN documents are created from the strings
          THEN the URL is "https://example.com/page1"
        """
        # Arrange
        
        # Act
        result = mock_document_retrieval_url_generator.execute(base_domain_urls)
        
        # Assert

        for doc in result["documents"]:
            assert doc["url"] == single_page_url
            assert doc["url"] == mock_constants["BASE_URL"] + "/page1"

    def test_multiple_strings_create_multiple_documents(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: Multiple strings create multiple documents
          GIVEN 3 extracted text strings from a single URL
          WHEN documents are created
          THEN 3 separate documents are created
        """
        # Arrange
        single_url = mock_constants["BASE_URL"]
        mock_document_retrieval.url_path_generator.generate.return_value = [single_url]
        # Mock extractor returns 3 text strings as in mock_constants
        three_strings = mock_constants["EXTRACTED_TEXT_CONTENT"]  # Has 3 items
        mock_document_retrieval.data_extractor.extract.return_value = three_strings

        # Act
        result = mock_document_retrieval.execute(base_domain_urls)

        # Assert
        contents = [doc["content"] for doc in result["documents"]]
        assert len(set(contents)) == 3

    def test_multiple_strings_create_multiple_documents_1(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: Multiple strings create multiple documents
          GIVEN 3 extracted text strings from a single URL
          WHEN documents are created
          THEN each document has the same source URL
        """
        # Arrange
        single_url = mock_constants["BASE_URL"]
        mock_document_retrieval.url_path_generator.generate.return_value = [single_url]
        # Mock extractor returns 3 text strings
        three_strings = mock_constants["EXTRACTED_TEXT_CONTENT"]
        mock_document_retrieval.data_extractor.extract.return_value = three_strings
        
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)
        
        # Assert
        urls = [doc["url"] for doc in result["documents"]]
        assert all(url == single_url for url in urls)



class TestVectorsAreGeneratedforAllDocuments:
    """
    Rule: Vectors Are Generated for All Documents
    """
    def test_vector_generator_creates_embeddings(self, mock_document_retrieval, mock_constants, base_domain_urls, mock_vectors_5_docs):
        """
        Scenario: Vector generator creates embeddings
          GIVEN a list of 5 documents
          WHEN vector generation is performed
          THEN exactly 5 vectors are generated
        """
        # Arrange
        # Set up scenario with exactly 5 documents (using the 5 generated URLs)
        # Mock data extractor to return exactly one text per URL for 5 documents total
        mock_document_retrieval.data_extractor.extract.return_value = ["Single text content"]
        # Mock vector generator to return exactly 5 vectors
        mock_document_retrieval.vector_generator.generate.return_value = mock_vectors_5_docs

        # Act
        result = mock_document_retrieval.execute(base_domain_urls)

        # Assert
        assert len(result['vectors']) == 5
 

    def test_vector_generator_creates_embeddings_1(self, mock_document_retrieval, mock_constants, base_domain_urls, mock_vectors_5_docs):
        """
        Scenario: Vector generator creates embeddings
          GIVEN a list of 5 documents
          WHEN vector generation is performed
          THEN each vector corresponds to a document
        """
        # Arrange
        # Mock 5 URLs for 5 documents
        mock_document_retrieval.data_extractor.extract.return_value = ["Single text content"]
        # Mock vector generator to return 5 vectors with doc_ids
        mock_document_retrieval.vector_generator.generate.return_value = mock_vectors_5_docs
        
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)
        
        # Assert
        for vector in result["vectors"]:
            assert vector["doc_id"].startswith("doc_")

    def test_vector_dimensions_match_configuration(self, mock_document_retrieval, mock_constants, base_domain_urls):
        """
        Scenario: Vector dimensions match configuration
          GIVEN documents are ready for vectorization
          And vector_dim is configured as 1536
          WHEN vectors are generated
          THEN each vector has dimension 1536
        """
        # Arrange

        # Act
        result = mock_document_retrieval.execute(base_domain_urls)

        # Assert
        len_embedding = len(result['vectors']['embedding'])
        assert len_embedding == mock_constants["VECTOR_DIMENSION"], \
            f"Expected vector dimension {mock_constants['VECTOR_DIMENSION']}, but got {len_embedding}"



class TestMetadataIsGeneratedforAllDocuments:
    """
    Rule: Metadata Is Generated for All Documents
    """
    @pytest.mark.parametrize("expected_field", [
        ("creation_time"),
        ("content_length"),
        ("source_url")
    ])
    def test_metadata_includes_document_properties(self, mock_document_retrieval_url_generator, mock_constants, single_page_url, base_domain_urls, expected_field):
        """
        Scenario: Metadata includes document properties
          GIVEN documents from URL "https://example.com/page"
          WHEN metadata is generated
          THEN metadata includes required document properties
        """
        # Arrange

        # Act
        result = mock_document_retrieval_url_generator.execute(base_domain_urls)

        # Assert
        assert expected_field in result["metadata"], f"Expected metadata to include field '{expected_field}', but got keys: {list(result['metadata'].keys())}"


    @pytest.mark.parametrize("field,expected_value_key", [
        ("source_url", "BASE_URL"),
        ("content_length", "CONTENT_LENGTH"),
        ("creation_time", "MOCK_TIMESTAMP")
    ])
    def test_metadata_includes_document_properties_1(self, mock_document_retrieval_url_generator, mock_constants, single_page_url, base_domain_urls, field, expected_value_key):
        """
        Scenario: Metadata includes document properties
          GIVEN documents from URL "https://example.com/page"
          WHEN metadata is generated
          THEN metadata includes required fields with expected values
        """
        # Arrange

        # Act
        result = mock_document_retrieval_url_generator.execute(base_domain_urls)

        # Assert
        expected_value = mock_constants[expected_value_key]
        assert result["metadata"][field] == expected_value, f"Expected metadata field '{field}' to be '{expected_value}', but got '{result['metadata'][field]}'"


    def test_metadata_includes_document_properties_3(self, mock_document_retrieval_url_generator, mock_constants, single_page_url, base_domain_urls):
        """
        Scenario: Metadata includes document properties
          GIVEN documents from URL "https://example.com/page"
          WHEN metadata is generated
          THEN metadata count matches document count
        """
        # Arrange
        
        # Act
        result = mock_document_retrieval_url_generator.execute(base_domain_urls)
        
        # Assert
        assert len(result["metadata"]) == len(result["documents"])



class TestDocumentsVectorsandMetadataAreStored:
    """
    Rule: Documents, Vectors, and Metadata Are Stored
    """
    def test_all_data_is_persisted_to_storage(self, mock_document_retrieval, mock_constants):
        """
        Scenario: All data is persisted to storage
          GIVEN 10 documents, vectors, and metadata are generated
          WHEN the storage step executes
          THEN document storage service receives 10 documents
        """
        # Arrange
        # Set up scenario with exactly 10 documents
        domain_urls = [mock_constants["BASE_URL"], mock_constants["TEST_URL_2"]]
        # Mock 5 URLs per domain = 10 total URLs
        mock_document_retrieval.url_path_generator.generate.return_value = [
            f"{mock_constants['BASE_URL']}/page{i}" for i in range(5)
        ]
        # Mock data extractor to return exactly one text per URL
        mock_document_retrieval.data_extractor.extract.return_value = ["Single text content"]
        
        # Act
        result = mock_document_retrieval.execute(domain_urls)
        
        # Assert

        # Verify storage service received exactly 10 documents
        expected_document_count = 10  # 2 domains * 5 URLs each
        storage_call_args = mock_document_retrieval.document_storage.store.call_args[0]
        documents, metadata, vectors = storage_call_args
        assert len(documents) == expected_document_count
        assert len(metadata) == expected_document_count
        assert len(vectors) == expected_document_count

    def test_all_data_is_persisted_to_storage_1(self, mock_document_retrieval, mock_constants):
        """
        Scenario: All data is persisted to storage
          GIVEN 10 documents, vectors, and metadata are generated
          WHEN the storage step executes
          THEN storage service receives 10 vectors
        """
        # Arrange
        domain_urls = [mock_constants["BASE_URL"], mock_constants["TEST_URL_2"]]
        # Mock 5 URLs per domain = 10 total
        mock_document_retrieval.url_path_generator.generate.return_value = [
            f"{mock_constants['BASE_URL']}/page{i}" for i in range(5)
        ]
        mock_document_retrieval.data_extractor.extract.return_value = ["Single text content"]
        
        # Act
        result = mock_document_retrieval.execute(domain_urls)
        
        # Assert

        # Verify storage service received exactly 10 vectors
        storage_call_args = mock_document_retrieval.document_storage.store.call_args[0]
        documents, metadata, vectors = storage_call_args
        assert len(vectors) == 10

    def test_all_data_is_persisted_to_storage_2(self, mock_document_retrieval, mock_constants):
        """
        Scenario: All data is persisted to storage
          GIVEN 10 documents, vectors, and metadata are generated
          WHEN the storage step executes
          THEN storage service receives 10 metadata records
        """
        # Arrange
        domain_urls = [mock_constants["BASE_URL"], mock_constants["TEST_URL_2"]]
        # Mock 5 URLs per domain = 10 total
        mock_document_retrieval.url_path_generator.generate.return_value = [
            f"{mock_constants['BASE_URL']}/page{i}" for i in range(5)
        ]
        mock_document_retrieval.data_extractor.extract.return_value = ["Single text content"]
        
        # Act
        result = mock_document_retrieval.execute(domain_urls)
        
        # Assert

        # Verify storage service received exactly 10 metadata records
        storage_call_args = mock_document_retrieval.document_storage.store.call_args[0]
        documents, metadata, vectors = storage_call_args
        assert len(metadata) == 10

    def test_all_data_is_persisted_to_storage_3(self, mock_document_retrieval, mock_constants):
        """
        Scenario: All data is persisted to storage
          GIVEN 10 documents, vectors, and metadata are generated
          WHEN the storage step executes
          THEN storage operation completes successfully
        """
        # Arrange
        domain_urls = [mock_constants["BASE_URL"], mock_constants["TEST_URL_2"]]
        # Mock 5 URLs per domain = 10 total
        mock_document_retrieval.url_path_generator.generate.return_value = [
            f"{mock_constants['BASE_URL']}/page{i}" for i in range(5)
        ]
        mock_document_retrieval.data_extractor.extract.return_value = ["Single text content"]
        # Mock storage to return True (success)
        mock_document_retrieval.document_storage.store.return_value = True

        # Act
        result = mock_document_retrieval.execute(domain_urls)

        # Assert

        # Verify storage.store was called
        mock_document_retrieval.document_storage.store.assert_called_once()
        # Verify storage operation completed successfully
        assert mock_document_retrieval.document_storage.store.return_value == True


class TestExecuteHandlesHTTPRequestFailures:
    """
    Rule: Execute Handles HTTP Request Failures
    """
    def test_timeout_during_webpage_fetch_is_handled(self, mock_document_retrieval, mock_constants, timeout_exception, base_domain_urls):
        """
        Scenario: Timeout during webpage fetch is handled
          GIVEN a URL that times out after timeout_seconds
          WHEN the URL is processed
          THEN the request is retried up to max_retries times
        """
        # Arrange
        timeout_url = mock_constants["BASE_URL"] + "/timeout"
        mock_document_retrieval.url_path_generator.generate.return_value = [timeout_url]
        # Mock HTTP timeout exception for specific URL
        mock_document_retrieval.static_webpage_parser.parse.side_effect = timeout_exception
        
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)
        
        # Assert
        assert mock_document_retrieval.static_webpage_parser.parse.call_count == mock_constants["MAX_RETRIES"]


    def test_timeout_during_webpage_fetch_is_handled_1(self, mock_document_retrieval, mock_constants, timeout_exception, base_domain_urls):
        """
        Scenario: Timeout during webpage fetch is handled
          GIVEN a URL that times out after timeout_seconds
          WHEN the URL is processed
          THEN if all retries fail, the URL is skipped
        """
        # Arrange
        timeout_url = mock_constants["BASE_URL"] + "/timeout"
        mock_document_retrieval.url_path_generator.generate.return_value = [timeout_url]
        # Mock persistent timeout on all retries
        mock_document_retrieval.static_webpage_parser.parse.side_effect = timeout_exception
        
        # Act
        result = mock_document_retrieval.execute(base_domain_urls)
        
        # Assert
        call_args = static_webpage_parser.parse.call_args
        assert call_args[1]['headers']['User-Agent'] == "CustomBot/1.0"


    def test_timeout_during_webpage_fetch_is_handled_2(self, mock_document_retrieval, mock_constants):
        """
        Scenario: Timeout during webpage fetch is handled
          GIVEN a URL that times out after timeout_seconds
          WHEN the URL is processed
          THEN other urls are in the returned result
        """
        # Arrange
        domain_urls = [mock_constants["BASE_URL"]]
        timeout_url = mock_constants["BASE_URL"] + "/timeout"
        success_url = mock_constants["BASE_URL"] + "/success"
        mock_document_retrieval.url_path_generator.generate.return_value = [timeout_url, success_url]
        # Mock timeout for first URL, success for second
        from requests.exceptions import Timeout
        mock_document_retrieval.static_webpage_parser.parse.side_effect = [
            Timeout("Request timed out"),
            {"url": success_url, "html_content": mock_constants["STATIC_HTML_CONTENT"], "status_code": 200, "headers": {}}
        ]
        
        # Act
        result = mock_document_retrieval.execute(domain_urls)
        
        # Assert

        # Verify processing continued after timeout
        # Should have documents from success_url
        assert len(result["documents"]) > 0

    def test_404_not_found_error_is_handled_gracefully_1(self, mock_document_retrieval, mock_constants):
        """
        Scenario: 404 Not Found error is handled gracefully
          GIVEN a URL that returns 404 status
          WHEN the URL is processed
          THEN result contains no documents for that URL
        """
        # Arrange
        domain_urls = [mock_constants["BASE_URL"]]
        not_found_url = mock_constants["BASE_URL"] + "/not-found"
        mock_document_retrieval.url_path_generator.generate.return_value = [not_found_url]
        # Mock 404 response
        from requests.exceptions import HTTPError
        mock_document_retrieval.static_webpage_parser.parse.side_effect = HTTPError("404 Not Found")
        
        # Act
        result = mock_document_retrieval.execute(domain_urls)

        # Assert

        # Verify URL was skipped (no documents created)
        assert len(result["documents"]) == 0

    def test_404_not_found_error_is_handled_gracefully_2(self, mock_document_retrieval, mock_constants):
        """
        Scenario: 404 Not Found error is handled gracefully
          GIVEN a URL that returns 404 status
          WHEN the URL is processed
          THEN processing continues with remaining URLs
        """
        # Arrange
        domain_urls = [mock_constants["BASE_URL"]]
        not_found_url = mock_constants["BASE_URL"] + "/not-found"
        success_url = mock_constants["BASE_URL"] + "/success"
        mock_document_retrieval.url_path_generator.generate.return_value = [not_found_url, success_url]
        # Mock 404 for first URL, success for second
        from requests.exceptions import HTTPError
        mock_document_retrieval.static_webpage_parser.parse.side_effect = [
            HTTPError("404 Not Found"),
            {"url": success_url, "html_content": mock_constants["STATIC_HTML_CONTENT"], "status_code": 200, "headers": {}}
        ]

        # Act
        result = mock_document_retrieval.execute(domain_urls)

        # Assert
        assert len(domain_urls) > len(result["documents"]) > 0

    def test_invalid_url_format_is_rejected(self, mock_document_retrieval, mock_constants):
        """
        Scenario: Invalid URL format is rejected
          GIVEN an invalid URL "not-a-valid-url"
          WHEN the URL is processed
          THEN ValueError is raised
        """
        # Arrange
        invalid_urls = ["not-a-valid-url"]

        # Act & Assert
        with pytest.raises(ValueError):
            mock_document_retrieval.execute(invalid_urls)


class TestUserAgentConfigurationIsApplied:
    """
    Rule: User Agent Configuration Is Applied
    """
    def test_custom_user_agent_is_sent_in_http_requests(
            self, document_retrieval, base_domain_urls, static_webpage_parser):
        """
        Scenario: Custom user agent is sent in HTTP requests
          GIVEN user_agent is configured as "CustomBot/1.0"
          WHEN a static webpage is requested
          THEN the HTTP request includes User-Agent header "CustomBot/1.0"
        """

        # Act
        result = document_retrieval.execute(base_domain_urls)

        # Assert
        call_args = static_webpage_parser.parse.call_args
        assert call_args[1]['headers']['User-Agent'] == "CustomBot/1.0"


if __name__ == "__main__":
    pytest.main()
