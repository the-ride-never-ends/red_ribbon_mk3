# Web Scraper Module

## Overview

The Web Scraper module provides architecture-level components for extracting content from dynamic webpages that require JavaScript rendering.

## Components

### DynamicWebscraper

Architecture component for scraping modern web applications that use JavaScript frameworks like React, Angular, or Vue.

**Purpose:**
- Load dynamic webpages with browser automation
- Wait for JavaScript to render content
- Extract structured data from rendered pages
- Handle errors and retries

## Public API

### Initialization

```python
from custom_nodes.red_ribbon.socialtoolkit.architecture.web_scraper import DynamicWebscraper
from custom_nodes.red_ribbon.socialtoolkit.architecture.web_scraper.dynamic_webscraper import DynamicWebscraperConfigs

configs = DynamicWebscraperConfigs(
    timeout_seconds=30,
    max_retries=3,
    wait_for_render=5,
    headless=True
)

resources = {
    "logger": logger_instance,
    "browser": browser_service,
    "parser": html_parser,
    "extractor": data_extractor
}

scraper = DynamicWebscraper(resources=resources, configs=configs)
```

### Methods

#### scrape(url: str) -> Dict[str, Any]
Scrape a single dynamic webpage.

**Returns:** Dictionary with keys:
- `success`: bool
- `url`: str
- `content`: str (extracted text)
- `html`: str (rendered HTML)
- `metadata`: dict (title, description, etc.)
- `timestamp`: datetime

#### scrape_multiple(urls: List[str]) -> List[Dict[str, Any]]
Scrape multiple dynamic webpages.

**Returns:** List of result dictionaries (same structure as `scrape()`)

#### validate_url(url: str) -> bool
Validate if a URL is properly formatted.

**Returns:** True if valid, False otherwise

#### get_page_title(url: str) -> str
Extract the title from a webpage.

**Returns:** Page title as string

#### extract_links(url: str) -> List[str]
Extract all links from a webpage.

**Returns:** List of URL strings

#### wait_for_element(url: str, selector: str, timeout: Optional[int] = None) -> bool
Wait for a specific element to appear on the page.

**Parameters:**
- `url`: URL of the webpage
- `selector`: CSS selector for the element
- `timeout`: Optional timeout in seconds (uses config default if not provided)

**Returns:** True if element appeared, False if timeout

## Configuration

### DynamicWebscraperConfigs

- `timeout_seconds` (int, default=30): Maximum wait time for operations
- `max_retries` (int, default=3): Number of retry attempts on failure
- `wait_for_render` (int, default=5): Seconds to wait for JavaScript rendering
- `headless` (bool, default=True): Run browser in headless mode
- `user_agent` (str): User agent string for requests
- `screenshot_on_error` (bool, default=False): Save screenshot on errors
- `javascript_enabled` (bool, default=True): Enable JavaScript execution

## Testing

### Unit Tests
Test file: `tests_unit/socialtoolkit_/architecture/test_dynamic_webscraper.py`

### Gherkin Documentation
Feature file: `tests_unit/socialtoolkit_/architecture/dynamic_webscraper.feature`

The Gherkin file contains 55 scenarios organized into 12 rules covering:
- Initialization validation
- Method parameter validation
- Return type specifications
- Error handling
- Configuration behavior
- Logging behavior

## Design Patterns

Follows SocialToolkit architecture patterns:
- Dependency injection via `resources` dictionary
- Pydantic-based configuration
- Typed method signatures
- Comprehensive error handling
- Logging at appropriate levels

## See Also

- `DocumentRetrievalFromWebsites`: Uses webpage parsers to retrieve documents
- `DynamicWebpageParser`: Resource-level parser for dynamic pages
