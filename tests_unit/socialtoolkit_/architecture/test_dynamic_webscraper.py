"""
Unit tests for DynamicWebscraper

Tests are based on the Gherkin feature file specification.
All test docstrings include the complete Given/When/Then structure.

Test Requirements:
- No conditional logic (including in assertions)
- Call the callable exactly once
- Exactly one assertion per test
- F-string assertion message with output value
- Use fixtures for test setups
- Output must be in the assertion
"""

import pytest
import logging
from typing import Dict, Any, List
from unittest.mock import Mock

from custom_nodes.red_ribbon.socialtoolkit.architecture.web_scraper.dynamic_webscraper import (
    DynamicWebscraper,
    DynamicWebscraperConfigs
)


# ============================================================================
# Fixtures - Based on Background and Given clauses
# ============================================================================

@pytest.fixture
def mock_logger():
    """Fixture providing a mock logger"""
    return Mock(spec=logging.Logger)


@pytest.fixture
def default_configs():
    """Fixture providing default DynamicWebscraperConfigs"""
    return DynamicWebscraperConfigs()


@pytest.fixture
def valid_resources(mock_logger):
    """Fixture providing valid resources dictionary"""
    return {
        "logger": mock_logger,
        "browser": Mock(),
        "parser": Mock(),
        "extractor": Mock()
    }


@pytest.fixture
def dynamic_webscraper_instance(valid_resources, default_configs):
    """Given a DynamicWebscraper instance with valid resources and configs"""
    return DynamicWebscraper(resources=valid_resources, configs=default_configs)


@pytest.fixture
def valid_url_dynamic_page():
    """Given a valid URL "https://example.com/dynamic-page" """
    return "https://example.com/dynamic-page"


@pytest.fixture
def valid_url():
    """Given a valid URL "https://example.com" """
    return "https://example.com"


@pytest.fixture
def valid_url_page1():
    """Given a valid URL "https://example.com/page1" """
    return "https://example.com/page1"


@pytest.fixture
def valid_url_test():
    """Given a valid URL "https://example.com/test" """
    return "https://example.com/test"


@pytest.fixture
def url_parameter_integer():
    """Given url parameter is an integer 12345"""
    return 12345


@pytest.fixture
def url_parameter_empty_string():
    """Given url parameter is an empty string "" """
    return ""


@pytest.fixture
def url_parameter_whitespace():
    """Given url parameter is "   " """
    return "   "


@pytest.fixture
def urls_two_pages():
    """Given URLs ["https://example.com/page1", "https://example.com/page2"]"""
    return ["https://example.com/page1", "https://example.com/page2"]


@pytest.fixture
def urls_with_failing():
    """Given URLs ["https://example.com/page1", "https://failing-url.com", "https://example.com/page2"]"""
    return ["https://example.com/page1", "https://failing-url.com", "https://example.com/page2"]


@pytest.fixture
def urls_with_one_that_fails():
    """Given URLs with one that fails"""
    return ["https://example.com/page1", "invalid-url", "https://example.com/page2"]


@pytest.fixture
def url_that_fails_to_scrape():
    """Given a URL that fails to scrape"""
    return "https://failing-url.com"


@pytest.fixture
def page_with_no_links():
    """Given a page with no links"""
    return "https://example.com/no-links"


@pytest.fixture
def page_with_10_links():
    """Given a page with 10 links"""
    return "https://example.com/ten-links"


@pytest.fixture
def url_https_example():
    """Given url "https://example.com" """
    return "https://example.com"


@pytest.fixture
def url_https_secure_with_path():
    """Given url "https://secure.example.com/path" """
    return "https://secure.example.com/path"


@pytest.fixture
def url_invalid_format():
    """Given url "not a url" """
    return "not a url"


@pytest.fixture
def url_empty_string():
    """Given url "" """
    return ""


@pytest.fixture
def url_parameter_not_string():
    """Given url parameter is not a string"""
    return 12345


@pytest.fixture
def urls_parameter_string():
    """Given urls parameter is a string "https://example.com" """
    return "https://example.com"


@pytest.fixture
def urls_parameter_empty_list():
    """Given urls parameter is an empty list []"""
    return []


@pytest.fixture
def urls_parameter_non_string_elements():
    """Given urls parameter contains [123, "https://example.com"]"""
    return [123, "https://example.com"]


@pytest.fixture
def selector_dynamic_content():
    """And selector "#dynamic-content" """
    return "#dynamic-content"


@pytest.fixture
def selector_parameter_empty_string():
    """And selector parameter is an empty string"""
    return ""


@pytest.fixture
def selector_parameter_not_string():
    """And selector parameter is not a string"""
    return 12345


@pytest.fixture
def timeout_10():
    """And timeout 10"""
    return 10


@pytest.fixture
def resources_parameter_not_dict():
    """Given resources parameter is not a dictionary"""
    return "not a dictionary"


@pytest.fixture
def configs_parameter_not_dynamicwebscraper_configs():
    """Given configs parameter is not a DynamicWebscraperConfigs instance"""
    return {"timeout": 30}


@pytest.fixture
def valid_resources_and_configs(valid_resources, default_configs):
    """Given valid resources and configs"""
    return (valid_resources, default_configs)


@pytest.fixture
def valid_resources_dictionary(valid_resources):
    """Given valid resources dictionary"""
    return valid_resources


# ============================================================================
# Test Class: Scrape Method Returns HTML Content
# ============================================================================

class TestScrapeMethodReturnsHtmlContent:
    """Rule: Scrape Method Returns HTML Content"""

    def test_scrape_returns_html_with_valid_structure(self, dynamic_webscraper_instance, valid_url_dynamic_page):
        """Scenario: Scrape returns HTML with valid structure
        Given a valid URL "https://example.com/dynamic-page"
        When scrape is called
        Then result["html"] contains "<html>"
        """
        result = dynamic_webscraper_instance.scrape(valid_url_dynamic_page)
        
        assert "<html>" in result["html"], f"Expected '<html>' in HTML output, got: {result['html']}"

    def test_scrape_returns_html_with_body_tag(self, dynamic_webscraper_instance, valid_url_dynamic_page):
        """Scenario: Scrape returns HTML with body tag
        Given a valid URL "https://example.com/dynamic-page"
        When scrape is called
        Then result["html"] contains "<body>"
        """
        result = dynamic_webscraper_instance.scrape(valid_url_dynamic_page)
        
        assert "<body>" in result["html"], f"Expected '<body>' in HTML output, got: {result['html']}"

    def test_scrape_returns_html_with_closing_tags(self, dynamic_webscraper_instance, valid_url_dynamic_page):
        """Scenario: Scrape returns HTML with closing tags
        Given a valid URL "https://example.com/dynamic-page"
        When scrape is called
        Then result["html"] contains "</html>"
        """
        result = dynamic_webscraper_instance.scrape(valid_url_dynamic_page)
        
        assert "</html>" in result["html"], f"Expected '</html>' in HTML output, got: {result['html']}"

    def test_scrape_returns_nonempty_html_string(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Scrape returns non-empty HTML string
        Given a valid URL "https://example.com"
        When scrape is called
        Then result["html"] is not empty
        """
        result = dynamic_webscraper_instance.scrape(valid_url)
        
        assert len(result["html"]) > 0, f"Expected non-empty HTML, got length: {len(result['html'])}"

    def test_scrape_returns_html_as_string_type(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Scrape returns HTML as string type
        Given a valid URL "https://example.com"
        When scrape is called
        """
        result = dynamic_webscraper_instance.scrape(valid_url)
        
        assert isinstance(result["html"], str), f"Expected HTML to be str, got: {type(result['html']).__name__}"


# ============================================================================
# Test Class: Scrape Method Returns Extracted Text Content
# ============================================================================

class TestScrapeMethodReturnsExtractedTextContent:
    """Rule: Scrape Method Returns Extracted Text Content"""

    def test_scrape_returns_extracted_content(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Scrape returns extracted content
        Given a valid URL "https://example.com"
        When scrape is called
        Then result["content"] is a string
        """
        result = dynamic_webscraper_instance.scrape(valid_url)
        
        assert isinstance(result["content"], str), f"Expected content to be str, got: {type(result['content']).__name__}"

    def test_scrape_content_references_the_url(self, dynamic_webscraper_instance, valid_url_page1):
        """Scenario: Scrape content references the URL
        Given a valid URL "https://example.com/page1"
        When scrape is called
        Then result["content"] contains "https://example.com/page1"
        """
        result = dynamic_webscraper_instance.scrape(valid_url_page1)
        
        assert valid_url_page1 in result["content"], f"Expected '{valid_url_page1}' in content, got: {result['content']}"

    def test_scrape_returns_nonempty_content(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Scrape returns non-empty content
        Given a valid URL "https://example.com"
        When scrape is called
        """
        result = dynamic_webscraper_instance.scrape(valid_url)
        
        assert len(result["content"]) > 0, f"Expected non-empty content, got length: {len(result['content'])}"


# ============================================================================
# Test Class: Scrape Method Returns Success Status
# ============================================================================

class TestScrapeMethodReturnsSuccessStatus:
    """Rule: Scrape Method Returns Success Status"""

    def test_scrape_with_valid_url_returns_success_true(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Scrape with valid URL returns success True
        Given a valid URL "https://example.com"
        When scrape is called
        Then result["success"] equals True
        """
        result = dynamic_webscraper_instance.scrape(valid_url)
        
        assert result["success"] is True, f"Expected success to be True, got: {result['success']}"

    def test_scrape_result_includes_requested_url(self, dynamic_webscraper_instance, valid_url_test):
        """Scenario: Scrape result includes requested URL
        Given a valid URL "https://example.com/test"
        When scrape is called
        Then result["url"] equals "https://example.com/test"
        """
        result = dynamic_webscraper_instance.scrape(valid_url_test)
        
        assert result["url"] == valid_url_test, f"Expected URL to be '{valid_url_test}', got: {result['url']}"


# ============================================================================
# Test Class: Scrape Method Returns Metadata Dictionary
# ============================================================================

class TestScrapeMethodReturnsMetadataDictionary:
    """Rule: Scrape Method Returns Metadata Dictionary"""

    def test_scrape_returns_metadata_with_title_key(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Scrape returns metadata with title key
        Given a valid URL "https://example.com"
        When scrape is called
        Then result["metadata"]["title"] exists
        """
        result = dynamic_webscraper_instance.scrape(valid_url)
        
        assert "title" in result["metadata"], f"Expected 'title' key in metadata, got keys: {list(result['metadata'].keys())}"

    def test_scrape_metadata_title_is_a_string(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Scrape metadata title is a string
        Given a valid URL "https://example.com"
        When scrape is called
        Then result["metadata"]["title"] is a string
        """
        result = dynamic_webscraper_instance.scrape(valid_url)
        
        assert isinstance(result["metadata"]["title"], str), f"Expected title to be str, got: {type(result['metadata']['title']).__name__}"

    def test_scrape_returns_metadata_with_description_key(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Scrape returns metadata with description key
        Given a valid URL "https://example.com"
        When scrape is called
        Then result["metadata"]["description"] exists
        """
        result = dynamic_webscraper_instance.scrape(valid_url)
        
        assert "description" in result["metadata"], f"Expected 'description' key in metadata, got keys: {list(result['metadata'].keys())}"

    def test_scrape_metadata_description_is_a_string(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Scrape metadata description is a string
        Given a valid URL "https://example.com"
        When scrape is called
        Then result["metadata"]["description"] is a string
        """
        result = dynamic_webscraper_instance.scrape(valid_url)
        
        assert isinstance(result["metadata"]["description"], str), f"Expected description to be str, got: {type(result['metadata']['description']).__name__}"


# ============================================================================
# Test Class: Scrape Method Validates Input Parameters
# ============================================================================

class TestScrapeMethodValidatesInputParameters:
    """Rule: Scrape Method Validates Input Parameters"""

    def test_scrape_with_nonstring_url_raises_typeerror(self, dynamic_webscraper_instance, url_parameter_integer):
        """Scenario: Scrape with non-string URL raises TypeError
        Given url parameter is an integer 12345
        When scrape is called
        Then a TypeError is raised
        """
        with pytest.raises(TypeError) as exc_info:
            dynamic_webscraper_instance.scrape(url_parameter_integer)
        
        assert exc_info.type is TypeError, f"Expected TypeError, got: {exc_info.type.__name__}"

    def test_scrape_with_empty_string_url_raises_valueerror(self, dynamic_webscraper_instance, url_parameter_empty_string):
        """Scenario: Scrape with empty string URL raises ValueError
        Given url parameter is an empty string ""
        When scrape is called
        Then a ValueError is raised
        """
        with pytest.raises(ValueError) as exc_info:
            dynamic_webscraper_instance.scrape(url_parameter_empty_string)
        
        assert exc_info.type is ValueError, f"Expected ValueError, got: {exc_info.type.__name__}"

    def test_scrape_with_whitespaceonly_url_raises_valueerror(self, dynamic_webscraper_instance, url_parameter_whitespace):
        """Scenario: Scrape with whitespace-only URL raises ValueError
        Given url parameter is "   "
        When scrape is called
        Then a ValueError is raised
        """
        with pytest.raises(ValueError) as exc_info:
            dynamic_webscraper_instance.scrape(url_parameter_whitespace)
        
        assert exc_info.type is ValueError, f"Expected ValueError, got: {exc_info.type.__name__}"

    def test_scrape_typeerror_message_indicates_type_requirement(self, dynamic_webscraper_instance, url_parameter_integer):
        """Scenario: Scrape TypeError message indicates type requirement
        Given url parameter is an integer 12345
        When scrape is called
        Then error message contains "url must be str"
        """
        with pytest.raises(TypeError) as exc_info:
            dynamic_webscraper_instance.scrape(url_parameter_integer)
        
        assert "url must be str" in str(exc_info.value), f"Expected 'url must be str' in error, got: {str(exc_info.value)}"

    def test_scrape_valueerror_message_indicates_empty_constraint(self, dynamic_webscraper_instance, url_parameter_empty_string):
        """Scenario: Scrape ValueError message indicates empty constraint
        Given url parameter is an empty string ""
        When scrape is called
        Then error message contains "url cannot be empty"
        """
        with pytest.raises(ValueError) as exc_info:
            dynamic_webscraper_instance.scrape(url_parameter_empty_string)
        
        assert "url cannot be empty" in str(exc_info.value), f"Expected 'url cannot be empty' in error, got: {str(exc_info.value)}"


# ============================================================================
# Test Class: Scrape Multiple Method Returns List of HTML Results
# ============================================================================

class TestScrapeMultipleMethodReturnsListOfHTMLResults:
    """Rule: Scrape Multiple Method Returns List of HTML Results"""

    def test_scrape_multiple_returns_list_of_dictionaries(self, dynamic_webscraper_instance, urls_two_pages):
        """Scenario: Scrape multiple returns list of dictionaries
        Given URLs ["https://example.com/page1", "https://example.com/page2"]
        When scrape_multiple is called
        Then a list of 2 dictionaries is returned
        """
        result = dynamic_webscraper_instance.scrape_multiple(urls_two_pages)
        
        assert len(result) == 2, f"Expected 2 dictionaries, got: {len(result)}"

    def test_scrape_multiple_first_result_contains_html(self, dynamic_webscraper_instance, urls_two_pages):
        """Scenario: Scrape multiple first result contains HTML
        Given URLs ["https://example.com/page1", "https://example.com/page2"]
        When scrape_multiple is called
        Then result[0]["html"] contains "<html>"
        """
        result = dynamic_webscraper_instance.scrape_multiple(urls_two_pages)
        
        assert "<html>" in result[0]["html"], f"Expected '<html>' in first result HTML, got: {result[0]['html']}"

    def test_scrape_multiple_second_result_contains_html(self, dynamic_webscraper_instance, urls_two_pages):
        """Scenario: Scrape multiple second result contains HTML
        Given URLs ["https://example.com/page1", "https://example.com/page2"]
        When scrape_multiple is called
        Then result[1]["html"] contains "<html>"
        """
        result = dynamic_webscraper_instance.scrape_multiple(urls_two_pages)
        
        assert "<html>" in result[1]["html"], f"Expected '<html>' in second result HTML, got: {result[1]['html']}"

    def test_scrape_multiple_each_result_has_unique_url(self, dynamic_webscraper_instance, urls_two_pages):
        """Scenario: Scrape multiple each result has unique URL
        Given URLs ["https://example.com/page1", "https://example.com/page2"]
        When scrape_multiple is called
        Then result[0]["url"] equals "https://example.com/page1"
        """
        result = dynamic_webscraper_instance.scrape_multiple(urls_two_pages)
        
        assert result[0]["url"] == urls_two_pages[0], f"Expected URL '{urls_two_pages[0]}', got: {result[0]['url']}"

    def test_scrape_multiple_second_result_has_correct_url(self, dynamic_webscraper_instance, urls_two_pages):
        """Scenario: Scrape multiple second result has correct URL
        Given URLs ["https://example.com/page1", "https://example.com/page2"]
        When scrape_multiple is called
        """
        result = dynamic_webscraper_instance.scrape_multiple(urls_two_pages)
        
        assert result[1]["url"] == urls_two_pages[1], f"Expected URL '{urls_two_pages[1]}', got: {result[1]['url']}"


# ============================================================================
# Test Class: Scrape Multiple Method Validates Input Parameters
# ============================================================================

class TestScrapeMultipleMethodValidatesInputParameters:
    """Rule: Scrape Multiple Method Validates Input Parameters"""

    def test_scrape_multiple_with_non_list_parameter_raises_typeerror(self, dynamic_webscraper_instance, urls_parameter_string):
        """Scenario: Scrape multiple with non-list parameter raises TypeError
        Given urls parameter is a string "https://example.com"
        When scrape_multiple is called
        Then a TypeError is raised
        """
        with pytest.raises(TypeError) as exc_info:
            dynamic_webscraper_instance.scrape_multiple(urls_parameter_string)
        
        assert exc_info.type is TypeError, f"Expected TypeError, got: {exc_info.type.__name__}"

    def test_scrape_multiple_with_empty_list_raises_valueerror(self, dynamic_webscraper_instance, urls_parameter_empty_list):
        """Scenario: Scrape multiple with empty list raises ValueError
        Given urls parameter is an empty list []
        When scrape_multiple is called
        Then a ValueError is raised
        """
        with pytest.raises(ValueError) as exc_info:
            dynamic_webscraper_instance.scrape_multiple(urls_parameter_empty_list)
        
        assert exc_info.type is ValueError, f"Expected ValueError, got: {exc_info.type.__name__}"

    def test_scrape_multiple_with_non_string_elements_raises_typeerror(self, dynamic_webscraper_instance, urls_parameter_non_string_elements):
        """Scenario: Scrape multiple with non-string elements raises TypeError
        Given urls parameter contains [123, "https://example.com"]
        When scrape_multiple is called
        Then a TypeError is raised
        """
        with pytest.raises(TypeError) as exc_info:
            dynamic_webscraper_instance.scrape_multiple(urls_parameter_non_string_elements)
        
        assert exc_info.type is TypeError, f"Expected TypeError, got: {exc_info.type.__name__}"

    def test_scrape_multiple_typeerror_message_indicates_list_requirement(self, dynamic_webscraper_instance, urls_parameter_string):
        """Scenario: Scrape multiple TypeError message indicates list requirement
        Given urls parameter is a string "https://example.com"
        When scrape_multiple is called
        Then error message contains "urls must be list"
        """
        with pytest.raises(TypeError) as exc_info:
            dynamic_webscraper_instance.scrape_multiple(urls_parameter_string)
        
        assert "urls must be list" in str(exc_info.value), f"Expected 'urls must be list' in error, got: {str(exc_info.value)}"

    def test_scrape_multiple_valueerror_message_indicates_empty_constraint(self, dynamic_webscraper_instance, urls_parameter_empty_list):
        """Scenario: Scrape multiple ValueError message indicates empty constraint
        Given urls parameter is an empty list []
        When scrape_multiple is called
        Then error message contains "urls cannot be empty"
        """
        with pytest.raises(ValueError) as exc_info:
            dynamic_webscraper_instance.scrape_multiple(urls_parameter_empty_list)
        
        assert "urls cannot be empty" in str(exc_info.value), f"Expected 'urls cannot be empty' in error, got: {str(exc_info.value)}"


# ============================================================================
# Test Class: Validate URL Method Checks Format
# ============================================================================

class TestValidateURLMethodChecksFormat:
    """Rule: Validate URL Method Checks Format"""

    def test_validate_url_with_valid_https_url_returns_true(self, dynamic_webscraper_instance, url_https_example):
        """Scenario: Validate URL with valid HTTPS URL returns True
        Given url "https://example.com"
        When validate_url is called
        Then True is returned
        """
        result = dynamic_webscraper_instance.validate_url(url_https_example)
        
        assert result is True, f"Expected True, got: {result}"

    def test_validate_url_with_path_returns_true(self, dynamic_webscraper_instance, url_https_secure_with_path):
        """Scenario: Validate URL with path returns True
        Given url "https://secure.example.com/path"
        When validate_url is called
        Then True is returned
        """
        result = dynamic_webscraper_instance.validate_url(url_https_secure_with_path)
        
        assert result is True, f"Expected True, got: {result}"

    def test_validate_url_with_invalid_format_returns_false(self, dynamic_webscraper_instance, url_invalid_format):
        """Scenario: Validate URL with invalid format returns False
        Given url "not a url"
        When validate_url is called
        Then False is returned
        """
        result = dynamic_webscraper_instance.validate_url(url_invalid_format)
        
        assert result is False, f"Expected False, got: {result}"

    def test_validate_url_with_non_string_parameter_returns_false(self, dynamic_webscraper_instance, url_parameter_not_string):
        """Scenario: Validate URL with non-string parameter returns False
        Given url parameter is not a string
        When validate_url is called
        Then False is returned
        """
        result = dynamic_webscraper_instance.validate_url(url_parameter_not_string)
        
        assert result is False, f"Expected False, got: {result}"

    def test_validate_url_with_empty_string_returns_false(self, dynamic_webscraper_instance, url_empty_string):
        """Scenario: Validate URL with empty string returns False
        Given url ""
        When validate_url is called
        Then False is returned
        """
        result = dynamic_webscraper_instance.validate_url(url_empty_string)
        
        assert result is False, f"Expected False, got: {result}"


# ============================================================================
# Test Class: DynamicWebscraper Initialization Validates Parameters
# ============================================================================

class TestDynamicwebscraperInitializationValidatesParameters:
    """Rule: DynamicWebscraper Initialization Validates Parameters"""

    def test_initialize_with_valid_resources_and_configs_succeeds(self, valid_resources_dictionary, default_configs):
        """Scenario: Initialize with valid resources and configs succeeds
        Given valid resources dictionary
        When DynamicWebscraper.__init__ is called with valid configs
        Then no exception is raised
        """
        result = DynamicWebscraper(resources=valid_resources_dictionary, configs=default_configs)
        
        assert isinstance(result, DynamicWebscraper), f"Expected DynamicWebscraper instance, got: {type(result).__name__}"

    def test_initialize_with_non_dict_resources_raises_typeerror(self, resources_parameter_not_dict, default_configs):
        """Scenario: Initialize with non-dict resources raises TypeError
        Given resources parameter is not a dictionary
        When DynamicWebscraper.__init__ is called
        Then a TypeError is raised
        """
        with pytest.raises(TypeError) as exc_info:
            DynamicWebscraper(resources=resources_parameter_not_dict, configs=default_configs)
        
        assert exc_info.type is TypeError, f"Expected TypeError, got: {exc_info.type.__name__}"

    def test_initialize_with_invalid_configs_raises_typeerror(self, valid_resources_dictionary, configs_parameter_not_dynamicwebscraper_configs):
        """Scenario: Initialize with invalid configs raises TypeError
        Given configs parameter is not a DynamicWebscraperConfigs instance
        When DynamicWebscraper.__init__ is called
        Then a TypeError is raised
        """
        with pytest.raises(TypeError) as exc_info:
            DynamicWebscraper(resources=valid_resources_dictionary, configs=configs_parameter_not_dynamicwebscraper_configs)
        
        assert exc_info.type is TypeError, f"Expected TypeError, got: {exc_info.type.__name__}"

    def test_initialize_resources_typeerror_message_indicates_dict_requirement(self, resources_parameter_not_dict, default_configs):
        """Scenario: Initialize resources TypeError message indicates dict requirement
        Given resources parameter is not a dictionary
        When DynamicWebscraper.__init__ is called
        Then error message contains "resources must be dict"
        """
        with pytest.raises(TypeError) as exc_info:
            DynamicWebscraper(resources=resources_parameter_not_dict, configs=default_configs)
        
        assert "resources must be dict" in str(exc_info.value), f"Expected 'resources must be dict' in error, got: {str(exc_info.value)}"

    def test_initialize_configs_typeerror_message_indicates_type_requirement(self, valid_resources_dictionary, configs_parameter_not_dynamicwebscraper_configs):
        """Scenario: Initialize configs TypeError message indicates type requirement
        Given configs parameter is not a DynamicWebscraperConfigs instance
        When DynamicWebscraper.__init__ is called
        Then error message contains "configs must be DynamicWebscraperConfigs"
        """
        with pytest.raises(TypeError) as exc_info:
            DynamicWebscraper(resources=valid_resources_dictionary, configs=configs_parameter_not_dynamicwebscraper_configs)
        
        assert "configs must be DynamicWebscraperConfigs" in str(exc_info.value), f"Expected 'configs must be DynamicWebscraperConfigs' in error, got: {str(exc_info.value)}"


# ============================================================================
# Test Class: DynamicWebscraper Logs Operations
# ============================================================================

class TestDynamicWebscraperLogsOperations:
    """Rule: DynamicWebscraper Logs Operations"""

    def test_initialize_logs_initialization_message(self, mock_logger, default_configs):
        """Scenario: Initialize logs initialization message
        Given valid resources and configs
        When DynamicWebscraper.__init__ is called
        Then "DynamicWebscraper initialized" is logged at INFO level
        """
        resources = {"logger": mock_logger}
        DynamicWebscraper(resources=resources, configs=default_configs)
        
        assert mock_logger.info.called, f"Expected logger.info to be called, got: {mock_logger.info.called}"

    def test_scrape_logs_webpage_url(self, mock_logger, default_configs, valid_url):
        """Scenario: Scrape logs webpage URL
        Given a valid URL "https://example.com"
        When scrape is called
        Then "Scraping dynamic webpage: https://example.com" is logged at INFO level
        """
        resources = {"logger": mock_logger}
        scraper = DynamicWebscraper(resources=resources, configs=default_configs)
        scraper.scrape(valid_url)
        
        assert mock_logger.info.call_count >= 2, f"Expected at least 2 info calls, got: {mock_logger.info.call_count}"

    def test_scrape_multiple_logs_url_count(self, mock_logger, default_configs, urls_two_pages):
        """Scenario: Scrape multiple logs URL count
        Given URLs ["https://example.com/page1", "https://example.com/page2"]
        When scrape_multiple is called
        Then "Scraping 2 dynamic webpages" is logged at INFO level
        """
        resources = {"logger": mock_logger}
        scraper = DynamicWebscraper(resources=resources, configs=default_configs)
        scraper.scrape_multiple(urls_two_pages)
        
        assert mock_logger.info.called, f"Expected logger.info to be called, got: {mock_logger.info.called}"

    def test_extract_links_logs_url(self, mock_logger, default_configs, valid_url):
        """Scenario: Extract links logs URL
        Given a valid URL "https://example.com"
        When extract_links is called
        Then "Extracting links from: https://example.com" is logged at INFO level
        """
        resources = {"logger": mock_logger}
        scraper = DynamicWebscraper(resources=resources, configs=default_configs)
        scraper.extract_links(valid_url)
        
        assert mock_logger.info.called, f"Expected logger.info to be called, got: {mock_logger.info.called}"

    def test_wait_for_element_logs_selector_and_timeout(self, mock_logger, default_configs, url_https_example, selector_dynamic_content):
        """Scenario: Wait for element logs selector and timeout
        Given url "https://example.com"
        When wait_for_element is called with selector "#content"
        Then log message contains "Waiting for element"
        """
        resources = {"logger": mock_logger}
        scraper = DynamicWebscraper(resources=resources, configs=default_configs)
        scraper.wait_for_element(url_https_example, selector_dynamic_content)
        
        assert mock_logger.info.called, f"Expected logger.info to be called, got: {mock_logger.info.called}"


# ============================================================================
# Test Class: Get Page Title Method Extracts Title  
# ============================================================================

class TestGetPageTitleMethodExtractsTitle:
    """Rule: Get Page Title Method Extracts Title"""

    def test_get_page_title_returns_string(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Get page title returns string
        Given a valid URL "https://example.com"
        When get_page_title is called
        Then a string is returned
        """
        result = dynamic_webscraper_instance.get_page_title(valid_url)
        
        assert isinstance(result, str), f"Expected str, got: {type(result).__name__}"

    def test_get_page_title_with_non_string_url_raises_typeerror(self, dynamic_webscraper_instance, url_parameter_not_string):
        """Scenario: Get page title with non-string URL raises TypeError
        Given url parameter is not a string
        When get_page_title is called
        Then a TypeError is raised
        """
        with pytest.raises(TypeError) as exc_info:
            dynamic_webscraper_instance.get_page_title(url_parameter_not_string)
        
        assert exc_info.type is TypeError, f"Expected TypeError, got: {exc_info.type.__name__}"

    def test_get_page_title_with_empty_url_raises_valueerror(self, dynamic_webscraper_instance, url_parameter_empty_string):
        """Scenario: Get page title with empty URL raises ValueError
        Given url parameter is an empty string
        When get_page_title is called
        Then a ValueError is raised
        """
        with pytest.raises(ValueError) as exc_info:
            dynamic_webscraper_instance.get_page_title(url_parameter_empty_string)
        
        assert exc_info.type is ValueError, f"Expected ValueError, got: {exc_info.type.__name__}"


# ============================================================================
# Test Class: Extract Links Method Returns URL List
# ============================================================================

class TestExtractLinksMethodReturnsURLList:
    """Rule: Extract Links Method Returns URL List"""

    def test_extract_links_returns_list_of_strings(self, dynamic_webscraper_instance, valid_url):
        """Scenario: Extract links returns list of strings
        Given a valid URL "https://example.com"
        When extract_links is called
        Then a list is returned
        """
        result = dynamic_webscraper_instance.extract_links(valid_url)
        
        assert isinstance(result, list), f"Expected list, got: {type(result).__name__}"

    def test_extract_links_with_non_string_url_raises_typeerror(self, dynamic_webscraper_instance, url_parameter_not_string):
        """Scenario: Extract links with non-string URL raises TypeError
        Given url parameter is not a string
        When extract_links is called
        Then a TypeError is raised
        """
        with pytest.raises(TypeError) as exc_info:
            dynamic_webscraper_instance.extract_links(url_parameter_not_string)
        
        assert exc_info.type is TypeError, f"Expected TypeError, got: {exc_info.type.__name__}"

    def test_extract_links_with_empty_url_raises_valueerror(self, dynamic_webscraper_instance, url_parameter_empty_string):
        """Scenario: Extract links with empty URL raises ValueError
        Given url parameter is an empty string
        When extract_links is called
        Then a ValueError is raised
        """
        with pytest.raises(ValueError) as exc_info:
            dynamic_webscraper_instance.extract_links(url_parameter_empty_string)
        
        assert exc_info.type is ValueError, f"Expected ValueError, got: {exc_info.type.__name__}"


# ============================================================================
# Test Class: Wait For Element Method Returns Boolean
# ============================================================================

class TestWaitForElementMethodReturnsBoolean:
    """Rule: Wait For Element Method Returns Boolean"""

    def test_wait_for_element_that_appears_returns_true(self, dynamic_webscraper_instance, url_https_example, selector_dynamic_content):
        """Scenario: Wait for element that appears returns True
        Given url "https://example.com"
        When wait_for_element is called with selector "#content"
        Then True is returned
        """
        result = dynamic_webscraper_instance.wait_for_element(url_https_example, "#content")
        
        assert result is True, f"Expected True, got: {result}"

    def test_wait_for_element_with_non_string_url_raises_typeerror(self, dynamic_webscraper_instance, url_parameter_not_string, selector_dynamic_content):
        """Scenario: Wait for element with non-string URL raises TypeError
        Given url parameter is not a string
        When wait_for_element is called with selector "#content"
        Then a TypeError is raised
        """
        with pytest.raises(TypeError) as exc_info:
            dynamic_webscraper_instance.wait_for_element(url_parameter_not_string, selector_dynamic_content)
        
        assert exc_info.type is TypeError, f"Expected TypeError, got: {exc_info.type.__name__}"

    def test_wait_for_element_with_non_string_selector_raises_typeerror(self, dynamic_webscraper_instance, url_https_example, selector_parameter_not_string):
        """Scenario: Wait for element with non-string selector raises TypeError
        Given url "https://example.com"
        And selector parameter is not a string
        When wait_for_element is called
        Then a TypeError is raised
        """
        with pytest.raises(TypeError) as exc_info:
            dynamic_webscraper_instance.wait_for_element(url_https_example, selector_parameter_not_string)
        
        assert exc_info.type is TypeError, f"Expected TypeError, got: {exc_info.type.__name__}"

    def test_wait_for_element_with_empty_url_raises_valueerror(self, dynamic_webscraper_instance, url_parameter_empty_string, selector_dynamic_content):
        """Scenario: Wait for element with empty URL raises ValueError
        Given url parameter is an empty string
        When wait_for_element is called with selector "#content"
        Then a ValueError is raised
        """
        with pytest.raises(ValueError) as exc_info:
            dynamic_webscraper_instance.wait_for_element(url_parameter_empty_string, selector_dynamic_content)
        
        assert exc_info.type is ValueError, f"Expected ValueError, got: {exc_info.type.__name__}"

    def test_wait_for_element_with_empty_selector_raises_valueerror(self, dynamic_webscraper_instance, url_https_example, selector_parameter_empty_string):
        """Scenario: Wait for element with empty selector raises ValueError
        Given url "https://example.com"
        And selector parameter is an empty string
        When wait_for_element is called
        Then a ValueError is raised
        """
        with pytest.raises(ValueError) as exc_info:
            dynamic_webscraper_instance.wait_for_element(url_https_example, selector_parameter_empty_string)
        
        assert exc_info.type is ValueError, f"Expected ValueError, got: {exc_info.type.__name__}"
