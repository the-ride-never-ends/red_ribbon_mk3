Feature: Dynamic Webscraper
  As a data collection system
  I want to scrape dynamic webpages that require JavaScript rendering
  So that I can extract structured data from modern web applications

  Background:
    Given a DynamicWebscraper instance is initialized
    And browser automation service is available
    And HTML parser service is available
    And data extractor service is available

  Rule: Initialization Requires Resources and Configs

    Scenario: Initialize with valid resources and configs
      Given valid resources dictionary with required services
      And valid DynamicWebscraperConfigs instance
      When DynamicWebscraper is initialized
      Then initialization completes successfully
      And logger is available

    Scenario: Initialize with invalid resources type
      Given resources is not a dictionary
      When DynamicWebscraper initialization is attempted
      Then a TypeError is raised
      And error message indicates "resources must be dict"

    Scenario: Initialize with invalid configs type
      Given configs is not a DynamicWebscraperConfigs instance
      When DynamicWebscraper initialization is attempted
      Then a TypeError is raised
      And error message indicates "configs must be DynamicWebscraperConfigs"

  Rule: Scrape Method Accepts String URL

    Scenario: Scrape with valid URL
      Given a valid URL "https://example.com/dynamic-page"
      When scrape is called with the URL
      Then a dictionary is returned
      And the dictionary contains key "success"
      And success value is True

    Scenario: Scrape with non-string URL
      Given url parameter is an integer 12345
      When scrape is called
      Then a TypeError is raised
      And error message indicates "url must be str"

    Scenario: Scrape with empty string URL
      Given url parameter is an empty string ""
      When scrape is called
      Then a ValueError is raised
      And error message indicates "url cannot be empty"

    Scenario: Scrape with whitespace-only URL
      Given url parameter is "   "
      When scrape is called
      Then a ValueError is raised
      And error message indicates "url cannot be empty"

  Rule: Scrape Returns Dictionary with Required Keys

    Scenario: Scrape result contains success flag
      Given a valid URL "https://example.com"
      When scrape is called
      Then the result contains key "success"
      And success is a boolean

    Scenario: Scrape result contains url
      Given a valid URL "https://example.com/page"
      When scrape is called
      Then the result contains key "url"
      And url is a string
      And url value is "https://example.com/page"

    Scenario: Scrape result contains content
      Given a valid URL "https://example.com"
      When scrape is called
      Then the result contains key "content"
      And content is a string

    Scenario: Scrape result contains html
      Given a valid URL "https://example.com"
      When scrape is called
      Then the result contains key "html"
      And html is a string

    Scenario: Scrape result contains metadata
      Given a valid URL "https://example.com"
      When scrape is called
      Then the result contains key "metadata"
      And metadata is a dictionary

    Scenario: Scrape result contains timestamp
      Given a valid URL "https://example.com"
      When scrape is called
      Then the result contains key "timestamp"

  Rule: Metadata Contains Page Information

    Scenario: Metadata contains title
      Given a valid URL "https://example.com"
      When scrape is called
      Then metadata contains key "title"
      And title is a string

    Scenario: Metadata contains description
      Given a valid URL "https://example.com"
      When scrape is called
      Then metadata contains key "description"
      And description is a string

  Rule: Scrape Multiple Accepts List of URLs

    Scenario: Scrape multiple with list of URLs
      Given a list of URLs ["https://example.com/page1", "https://example.com/page2"]
      When scrape_multiple is called with the URL list
      Then a list is returned
      And the list contains 2 dictionaries
      And each dictionary has the same structure as scrape result

    Scenario: Scrape multiple with non-list parameter
      Given urls parameter is a string "https://example.com"
      When scrape_multiple is called
      Then a TypeError is raised
      And error message indicates "urls must be list"

    Scenario: Scrape multiple with empty list
      Given an empty list of URLs []
      When scrape_multiple is called
      Then a ValueError is raised
      And error message indicates "urls cannot be empty"

    Scenario: Scrape multiple with non-string elements
      Given a list containing non-string [123, "https://example.com"]
      When scrape_multiple is called
      Then a TypeError is raised
      And error message indicates "all elements in urls must be strings"

  Rule: Scrape Multiple Returns List of Results

    Scenario: Scrape multiple returns result for each URL
      Given URLs ["https://example.com/page1", "https://example.com/page2", "https://example.com/page3"]
      When scrape_multiple is called
      Then a list of 3 dictionaries is returned
      And each dictionary corresponds to one URL

    Scenario: Scrape multiple handles individual failures
      Given URLs including one that fails to scrape
      When scrape_multiple is called
      Then a list is returned
      And failed URL result has success = False
      And failed URL result contains "error" key
      And other URLs are processed normally

  Rule: Validate URL Checks URL Format

    Scenario: Validate URL with valid HTTP URL
      Given url "https://example.com"
      When validate_url is called
      Then True is returned

    Scenario: Validate URL with valid HTTPS URL
      Given url "https://secure.example.com/path"
      When validate_url is called
      Then True is returned

    Scenario: Validate URL with invalid format
      Given url "not a url"
      When validate_url is called
      Then False is returned

    Scenario: Validate URL with non-string parameter
      Given url parameter is an integer 12345
      When validate_url is called
      Then False is returned

    Scenario: Validate URL with empty string
      Given url ""
      When validate_url is called
      Then False is returned

  Rule: Get Page Title Extracts Title from Webpage

    Scenario: Get page title with valid URL
      Given a valid URL "https://example.com"
      When get_page_title is called
      Then a string is returned
      And the string contains the page title

    Scenario: Get page title with non-string URL
      Given url parameter is not a string
      When get_page_title is called
      Then a TypeError is raised
      And error message indicates "url must be str"

    Scenario: Get page title with empty URL
      Given url parameter is an empty string
      When get_page_title is called
      Then a ValueError is raised
      And error message indicates "url cannot be empty"

    Scenario: Get page title when scraping fails
      Given a URL that fails to scrape
      When get_page_title is called
      Then a RuntimeError is raised

  Rule: Extract Links Gets All Links from Page

    Scenario: Extract links from page
      Given a valid URL "https://example.com"
      When extract_links is called
      Then a list of strings is returned
      And each string is a URL

    Scenario: Extract links with non-string URL
      Given url parameter is not a string
      When extract_links is called
      Then a TypeError is raised
      And error message indicates "url must be str"

    Scenario: Extract links with empty URL
      Given url parameter is an empty string
      When extract_links is called
      Then a ValueError is raised
      And error message indicates "url cannot be empty"

    Scenario: Extract links from page with no links
      Given a page with no links
      When extract_links is called
      Then an empty list is returned

    Scenario: Extract links from page with multiple links
      Given a page with 10 links
      When extract_links is called
      Then a list of 10 URL strings is returned

  Rule: Wait For Element Waits for Selector

    Scenario: Wait for element that appears
      Given url "https://example.com"
      And selector "#content"
      When wait_for_element is called
      Then True is returned

    Scenario: Wait for element with custom timeout
      Given url "https://example.com"
      And selector "#dynamic-content"
      And timeout 10
      When wait_for_element is called with timeout
      Then the element is waited for up to 10 seconds
      And a boolean is returned

    Scenario: Wait for element with non-string URL
      Given url parameter is not a string
      And selector "#content"
      When wait_for_element is called
      Then a TypeError is raised
      And error message indicates "url must be str"

    Scenario: Wait for element with non-string selector
      Given url "https://example.com"
      And selector parameter is not a string
      When wait_for_element is called
      Then a TypeError is raised
      And error message indicates "selector must be str"

    Scenario: Wait for element with empty URL
      Given url parameter is an empty string
      And selector "#content"
      When wait_for_element is called
      Then a ValueError is raised
      And error message indicates "url cannot be empty"

    Scenario: Wait for element with empty selector
      Given url "https://example.com"
      And selector parameter is an empty string
      When wait_for_element is called
      Then a ValueError is raised
      And error message indicates "selector cannot be empty"

    Scenario: Wait for element uses default timeout when not provided
      Given url "https://example.com"
      And selector "#content"
      And timeout is not provided
      When wait_for_element is called
      Then default timeout from configs is used

  Rule: Configuration Controls Scraping Behavior

    Scenario: Timeout seconds configuration affects wait time
      Given configs with timeout_seconds = 30
      When scraper operates
      Then operations timeout after 30 seconds

    Scenario: Max retries configuration affects retry behavior
      Given configs with max_retries = 3
      When a scrape operation fails
      Then the operation is retried up to 3 times

    Scenario: Wait for render configuration affects page load
      Given configs with wait_for_render = 5
      When a dynamic page is scraped
      Then scraper waits 5 seconds for JavaScript to render

    Scenario: Headless mode configuration affects browser visibility
      Given configs with headless = True
      When browser is launched
      Then browser runs in headless mode

    Scenario: User agent configuration sets browser identity
      Given configs with user_agent = "Custom User Agent"
      When requests are made
      Then User-Agent header is "Custom User Agent"

    Scenario: Screenshot on error saves debugging images
      Given configs with screenshot_on_error = True
      When a scrape operation fails
      Then a screenshot is saved for debugging

    Scenario: JavaScript enabled controls script execution
      Given configs with javascript_enabled = True
      When pages are loaded
      Then JavaScript is executed

  Rule: Logger Records Scraping Operations

    Scenario: Logger records initialization
      When DynamicWebscraper is initialized
      Then "DynamicWebscraper initialized" is logged at INFO level

    Scenario: Logger records scrape operations
      Given a URL to scrape
      When scrape is called
      Then "Scraping dynamic webpage" message is logged at INFO level

    Scenario: Logger records multiple scrape operations
      Given multiple URLs to scrape
      When scrape_multiple is called
      Then "Scraping N dynamic webpages" is logged at INFO level

    Scenario: Logger records failed scrapes
      Given a URL that fails to scrape
      When scrape_multiple processes the URL
      Then "Failed to scrape" message is logged at ERROR level

    Scenario: Logger records link extraction
      Given a URL for link extraction
      When extract_links is called
      Then "Extracting links from" is logged at INFO level

    Scenario: Logger records element waiting
      Given a URL and selector
      When wait_for_element is called
      Then "Waiting for element" message is logged at INFO level
