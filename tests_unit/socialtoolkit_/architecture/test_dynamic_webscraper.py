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
