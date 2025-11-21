Feature: Dynamic Webscraper
  As a data collection system
  I want to scrape dynamic webpages that require JavaScript rendering
  So that I can extract structured data from modern web applications

  Background:
    Given a DynamicWebscraper instance with valid resources and configs

  Rule: Scrape Method Returns HTML Content

    Scenario: Scrape returns HTML with valid structure
      Given a valid URL "https://example.com/dynamic-page"
      When scrape is called
      Then result["html"] contains "<html>"

    Scenario: Scrape returns HTML with body tag
      Given a valid URL "https://example.com/dynamic-page"
      When scrape is called
      Then result["html"] contains "<body>"

    Scenario: Scrape returns HTML with closing tags
      Given a valid URL "https://example.com/dynamic-page"
      When scrape is called
      Then result["html"] contains "</html>"

    Scenario: Scrape returns non-empty HTML string
      Given a valid URL "https://example.com"
      When scrape is called
      Then result["html"] is not empty

    Scenario: Scrape returns HTML as string type
      Given a valid URL "https://example.com"
      When scrape is called
      Then result["html"] is a string

  Rule: Scrape Method Returns Extracted Text Content

    Scenario: Scrape returns extracted content
      Given a valid URL "https://example.com"
      When scrape is called
      Then result["content"] is a string

    Scenario: Scrape content references the URL
      Given a valid URL "https://example.com/page1"
      When scrape is called
      Then result["content"] contains "https://example.com/page1"

    Scenario: Scrape returns non-empty content
      Given a valid URL "https://example.com"
      When scrape is called
      Then result["content"] is not empty

  Rule: Scrape Method Returns Success Status

    Scenario: Scrape with valid URL returns success True
      Given a valid URL "https://example.com"
      When scrape is called
      Then result["success"] equals True

    Scenario: Scrape result includes requested URL
      Given a valid URL "https://example.com/test"
      When scrape is called
      Then result["url"] equals "https://example.com/test"

  Rule: Scrape Method Returns Metadata Dictionary

    Scenario: Scrape returns metadata with title key
      Given a valid URL "https://example.com"
      When scrape is called
      Then result["metadata"]["title"] exists

    Scenario: Scrape metadata title is a string
      Given a valid URL "https://example.com"
      When scrape is called
      Then result["metadata"]["title"] is a string

    Scenario: Scrape returns metadata with description key
      Given a valid URL "https://example.com"
      When scrape is called
      Then result["metadata"]["description"] exists

    Scenario: Scrape metadata description is a string
      Given a valid URL "https://example.com"
      When scrape is called
      Then result["metadata"]["description"] is a string

  Rule: Scrape Method Validates Input Parameters

    Scenario: Scrape with non-string URL raises TypeError
      Given url parameter is an integer 12345
      When scrape is called
      Then a TypeError is raised

    Scenario: Scrape with empty string URL raises ValueError
      Given url parameter is an empty string ""
      When scrape is called
      Then a ValueError is raised

    Scenario: Scrape with whitespace-only URL raises ValueError
      Given url parameter is "   "
      When scrape is called
      Then a ValueError is raised

    Scenario: Scrape TypeError message indicates type requirement
      Given url parameter is an integer 12345
      When scrape is called
      Then error message contains "url must be str"

    Scenario: Scrape ValueError message indicates empty constraint
      Given url parameter is an empty string ""
      When scrape is called
      Then error message contains "url cannot be empty"

  Rule: Scrape Multiple Method Returns List of HTML Results

    Scenario: Scrape multiple returns list of dictionaries
      Given URLs ["https://example.com/page1", "https://example.com/page2"]
      When scrape_multiple is called
      Then a list of 2 dictionaries is returned

    Scenario: Scrape multiple first result contains HTML
      Given URLs ["https://example.com/page1", "https://example.com/page2"]
      When scrape_multiple is called
      Then result[0]["html"] contains "<html>"

    Scenario: Scrape multiple second result contains HTML
      Given URLs ["https://example.com/page1", "https://example.com/page2"]
      When scrape_multiple is called
      Then result[1]["html"] contains "<html>"

    Scenario: Scrape multiple each result has unique URL
      Given URLs ["https://example.com/page1", "https://example.com/page2"]
      When scrape_multiple is called
      Then result[0]["url"] equals "https://example.com/page1"

    Scenario: Scrape multiple second result has correct URL
      Given URLs ["https://example.com/page1", "https://example.com/page2"]
      When scrape_multiple is called
      Then result[1]["url"] equals "https://example.com/page2"

  Rule: Scrape Multiple Method Handles Failures

    Scenario: Scrape multiple with failing URL returns list
      Given URLs ["https://example.com/page1", "https://failing-url.com", "https://example.com/page2"]
      When scrape_multiple is called
      Then a list of 3 dictionaries is returned

    Scenario: Scrape multiple failing URL has success False
      Given URLs ["https://example.com/page1", "https://failing-url.com", "https://example.com/page2"]
      When scrape_multiple is called
      Then result[1]["success"] equals False

    Scenario: Scrape multiple failing URL includes error key
      Given URLs ["https://example.com/page1", "https://failing-url.com", "https://example.com/page2"]
      When scrape_multiple is called
      Then result[1]["error"] exists

    Scenario: Scrape multiple successful URLs have success True
      Given URLs ["https://example.com/page1", "https://failing-url.com", "https://example.com/page2"]
      When scrape_multiple is called
      Then result[0]["success"] equals True

    Scenario: Scrape multiple third URL succeeds despite earlier failure
      Given URLs ["https://example.com/page1", "https://failing-url.com", "https://example.com/page2"]
      When scrape_multiple is called
      Then result[2]["success"] equals True

  Rule: Scrape Multiple Method Validates Input Parameters

    Scenario: Scrape multiple with non-list parameter raises TypeError
      Given urls parameter is a string "https://example.com"
      When scrape_multiple is called
      Then a TypeError is raised

    Scenario: Scrape multiple with empty list raises ValueError
      Given urls parameter is an empty list []
      When scrape_multiple is called
      Then a ValueError is raised

    Scenario: Scrape multiple with non-string elements raises TypeError
      Given urls parameter contains [123, "https://example.com"]
      When scrape_multiple is called
      Then a TypeError is raised

    Scenario: Scrape multiple TypeError message indicates list requirement
      Given urls parameter is a string "https://example.com"
      When scrape_multiple is called
      Then error message contains "urls must be list"

    Scenario: Scrape multiple ValueError message indicates empty constraint
      Given urls parameter is an empty list []
      When scrape_multiple is called
      Then error message contains "urls cannot be empty"

  Rule: Validate URL Method Checks Format

    Scenario: Validate URL with valid HTTPS URL returns True
      Given url "https://example.com"
      When validate_url is called
      Then True is returned

    Scenario: Validate URL with path returns True
      Given url "https://secure.example.com/path"
      When validate_url is called
      Then True is returned

    Scenario: Validate URL with invalid format returns False
      Given url "not a url"
      When validate_url is called
      Then False is returned

    Scenario: Validate URL with non-string parameter returns False
      Given url parameter is an integer 12345
      When validate_url is called
      Then False is returned

    Scenario: Validate URL with empty string returns False
      Given url ""
      When validate_url is called
      Then False is returned

  Rule: Get Page Title Method Extracts Title

    Scenario: Get page title returns string
      Given a valid URL "https://example.com"
      When get_page_title is called
      Then a string is returned

    Scenario: Get page title with non-string URL raises TypeError
      Given url parameter is not a string
      When get_page_title is called
      Then a TypeError is raised

    Scenario: Get page title with empty URL raises ValueError
      Given url parameter is an empty string
      When get_page_title is called
      Then a ValueError is raised

    Scenario: Get page title when scraping fails raises RuntimeError
      Given a URL that fails to scrape
      When get_page_title is called
      Then a RuntimeError is raised

    Scenario: Get page title TypeError message indicates type requirement
      Given url parameter is not a string
      When get_page_title is called
      Then error message contains "url must be str"

    Scenario: Get page title ValueError message indicates empty constraint
      Given url parameter is an empty string
      When get_page_title is called
      Then error message contains "url cannot be empty"

  Rule: Extract Links Method Returns URL List

    Scenario: Extract links returns list of strings
      Given a valid URL "https://example.com"
      When extract_links is called
      Then a list is returned

    Scenario: Extract links from page with no links returns empty list
      Given a page with no links
      When extract_links is called
      Then an empty list is returned

    Scenario: Extract links from page with 10 links returns 10 URLs
      Given a page with 10 links
      When extract_links is called
      Then a list of 10 strings is returned

    Scenario: Extract links with non-string URL raises TypeError
      Given url parameter is not a string
      When extract_links is called
      Then a TypeError is raised

    Scenario: Extract links with empty URL raises ValueError
      Given url parameter is an empty string
      When extract_links is called
      Then a ValueError is raised

    Scenario: Extract links TypeError message indicates type requirement
      Given url parameter is not a string
      When extract_links is called
      Then error message contains "url must be str"

    Scenario: Extract links ValueError message indicates empty constraint
      Given url parameter is an empty string
      When extract_links is called
      Then error message contains "url cannot be empty"

  Rule: Wait For Element Method Returns Boolean

    Scenario: Wait for element that appears returns True
      Given url "https://example.com"
      When wait_for_element is called with selector "#content"
      Then True is returned

    Scenario: Wait for element with custom timeout logs timeout value
      Given url "https://example.com"
      And selector "#dynamic-content"
      And timeout 10
      When wait_for_element is called
      Then log message contains "(timeout: 10s)"

    Scenario: Wait for element without timeout uses default
      Given url "https://example.com"
      When wait_for_element is called with selector "#content"
      Then log message contains "(timeout: 30s)"

    Scenario: Wait for element with non-string URL raises TypeError
      Given url parameter is not a string
      When wait_for_element is called with selector "#content"
      Then a TypeError is raised

    Scenario: Wait for element with non-string selector raises TypeError
      Given url "https://example.com"
      And selector parameter is not a string
      When wait_for_element is called
      Then a TypeError is raised

    Scenario: Wait for element with empty URL raises ValueError
      Given url parameter is an empty string
      When wait_for_element is called with selector "#content"
      Then a ValueError is raised

    Scenario: Wait for element with empty selector raises ValueError
      Given url "https://example.com"
      And selector parameter is an empty string
      When wait_for_element is called
      Then a ValueError is raised

    Scenario: Wait for element URL TypeError message indicates type requirement
      Given url parameter is not a string
      When wait_for_element is called with selector "#content"
      Then error message contains "url must be str"

    Scenario: Wait for element selector TypeError message indicates type requirement
      Given url "https://example.com"
      And selector parameter is not a string
      When wait_for_element is called
      Then error message contains "selector must be str"

    Scenario: Wait for element URL ValueError message indicates empty constraint
      Given url parameter is an empty string
      When wait_for_element is called with selector "#content"
      Then error message contains "url cannot be empty"

    Scenario: Wait for element selector ValueError message indicates empty constraint
      Given url "https://example.com"
      And selector parameter is an empty string
      When wait_for_element is called
      Then error message contains "selector cannot be empty"

  Rule: DynamicWebscraper Initialization Validates Parameters

    Scenario: Initialize with valid resources and configs succeeds
      Given valid resources dictionary
      When DynamicWebscraper.__init__ is called with valid configs
      Then no exception is raised

    Scenario: Initialize with non-dict resources raises TypeError
      Given resources parameter is not a dictionary
      When DynamicWebscraper.__init__ is called
      Then a TypeError is raised

    Scenario: Initialize with invalid configs raises TypeError
      Given configs parameter is not a DynamicWebscraperConfigs instance
      When DynamicWebscraper.__init__ is called
      Then a TypeError is raised

    Scenario: Initialize resources TypeError message indicates dict requirement
      Given resources parameter is not a dictionary
      When DynamicWebscraper.__init__ is called
      Then error message contains "resources must be dict"

    Scenario: Initialize configs TypeError message indicates type requirement
      Given configs parameter is not a DynamicWebscraperConfigs instance
      When DynamicWebscraper.__init__ is called
      Then error message contains "configs must be DynamicWebscraperConfigs"

  Rule: DynamicWebscraper Logs Operations

    Scenario: Initialize logs initialization message
      Given valid resources and configs
      When DynamicWebscraper.__init__ is called
      Then "DynamicWebscraper initialized" is logged at INFO level

    Scenario: Scrape logs webpage URL
      Given a valid URL "https://example.com"
      When scrape is called
      Then "Scraping dynamic webpage: https://example.com" is logged at INFO level

    Scenario: Scrape multiple logs URL count
      Given URLs ["https://example.com/page1", "https://example.com/page2"]
      When scrape_multiple is called
      Then "Scraping 2 dynamic webpages" is logged at INFO level

    Scenario: Scrape multiple logs individual failure
      Given URLs with one that fails
      When scrape_multiple is called
      Then "Failed to scrape" is logged at ERROR level

    Scenario: Extract links logs URL
      Given a valid URL "https://example.com"
      When extract_links is called
      Then "Extracting links from: https://example.com" is logged at INFO level

    Scenario: Wait for element logs selector and timeout
      Given url "https://example.com"
      When wait_for_element is called with selector "#content"
      Then log message contains "Waiting for element '#content'"
