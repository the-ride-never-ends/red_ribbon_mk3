Feature: Document Retrieval from Websites
  As a data collection system
  I want to retrieve and process documents from websites
  So that I can extract structured data for analysis

  Background:
    Given a DocumentRetrievalFromWebsites instance is initialized
    And a static webpage parser is available
    And a dynamic webpage parser is available
    And a data extractor is available
    And a vector generator is available
    And a metadata generator is available
    And a document storage service is available
    And a URL path generator is available

  Rule: Execute Method Accepts List of Domain URLs

    Scenario: Execute with single domain URL
      Given a single domain URL "https://example.com"
      When I call execute with the domain URL list
      Then the execution completes successfully
      And documents are retrieved from the domain

    Scenario: Execute with multiple domain URLs
      Given multiple domain URLs ["https://example.com", "https://test.org"]
      When I call execute with the domain URL list
      Then documents are retrieved from all domains
      And results are aggregated across domains

    Scenario: Execute with empty URL list
      Given an empty list of domain URLs
      When I call execute with the empty list
      Then no documents are retrieved
      And the returned dictionary has empty collections

  Rule: Execute Returns Dictionary with Required Keys

    Scenario: Execute returns expected result structure
      Given a valid domain URL "https://example.com"
      When I call execute with the domain URL
      Then I receive a dictionary response
      And the response contains key "documents"
      And the response contains key "metadata"
      And the response contains key "vectors"

  Rule: Static and Dynamic Webpages Are Parsed Appropriately

    Scenario: Static webpage is parsed with static parser
      Given a URL "https://example.com/static-page.html" identified as static
      When the URL is processed
      Then the static webpage parser is called
      And the dynamic webpage parser is not called
      And raw HTML data is extracted

    Scenario: Dynamic webpage is parsed with dynamic parser
      Given a URL "https://example.com/app" identified as dynamic
      When the URL is processed
      Then the dynamic webpage parser is called with the URL
      And the static webpage parser is not called
      And JavaScript-rendered content is extracted

  Rule: URL Generation Expands Domain URLs to Page URLs

    Scenario: Single domain URL is expanded to multiple page URLs
      Given domain URL "https://example.com"
      And URL path generator produces 5 page URLs
      When execute processes the domain
      Then all 5 page URLs are processed
      And documents are retrieved from each page

    Scenario: URL generator respects max_depth configuration
      Given domain URL "https://example.com"
      And max_depth is configured as 1
      When execute processes the domain
      Then only URLs at depth 1 or less are processed
      And deeper URLs are not followed

    Scenario: URL generator respects follow_links configuration
      Given domain URL "https://example.com/page1"
      And follow_links is configured as False
      When execute processes the domain
      Then only the provided URLs are processed
      And no additional links are followed

  Rule: Data Extraction Converts Raw Data to Structured Strings

    Scenario: Raw HTML is extracted to text strings
      Given raw HTML data from a webpage
      When the data extractor processes the raw data
      Then structured text strings are returned
      And HTML tags are removed
      And text content is preserved

    Scenario: Empty raw data results in empty extraction
      Given raw data that is empty or None
      When the data extractor processes the raw data
      Then an empty list of strings is returned

  Rule: Documents Are Created with URL Context

    Scenario: Documents include source URL
      Given text strings extracted from "https://example.com/page1"
      When documents are created from the strings
      Then each document includes the source URL
      And the URL is "https://example.com/page1"

    Scenario: Multiple strings create multiple documents
      Given 3 extracted text strings from a single URL
      When documents are created
      Then 3 separate documents are created
      And each document has the same source URL

  Rule: Vectors Are Generated for All Documents

    Scenario: Vector generator creates embeddings
      Given a list of 5 documents
      When vector generation is performed
      Then exactly 5 vectors are generated
      And each vector corresponds to a document

    Scenario: Vector dimensions match configuration
      Given documents are ready for vectorization
      And vector_dim is configured as 1536
      When vectors are generated
      Then each vector has dimension 1536

  Rule: Metadata Is Generated for All Documents

    Scenario: Metadata includes document properties
      Given documents from URL "https://example.com/page"
      When metadata is generated
      Then metadata includes document creation time
      And metadata includes source URL
      And metadata includes document length
      And metadata count matches document count

  Rule: Documents, Vectors, and Metadata Are Stored

    Scenario: All data is persisted to storage
      Given 10 documents, vectors, and metadata are generated
      When the storage step executes
      Then document storage service receives 10 documents
      And storage service receives 10 vectors
      And storage service receives 10 metadata records
      And storage operation completes successfully

  Rule: Batch Processing Configuration Is Respected

    Scenario: Large document sets are processed in batches
      Given 100 documents are retrieved
      And batch_size is configured as 10
      When storage operation executes
      Then documents are stored in 10 batches
      And each batch contains 10 or fewer documents

  Rule: Execute Handles HTTP Request Failures

    Scenario: Timeout during webpage fetch is handled
      Given a URL that times out after timeout_seconds
      When the URL is processed
      Then the request is retried up to max_retries times
      And if all retries fail, the URL is skipped
      And processing continues with remaining URLs

    Scenario: 404 Not Found error is handled gracefully
      Given a URL that returns 404 status
      When the URL is processed
      Then the error is logged
      And the URL is skipped
      And processing continues with remaining URLs

    Scenario: Invalid URL format is rejected
      Given an invalid URL "not-a-valid-url"
      When the URL is processed
      Then a validation error is raised
      And the invalid URL is not processed

  Rule: User Agent Configuration Is Applied

    Scenario: Custom user agent is sent in HTTP requests
      Given user_agent is configured as "CustomBot/1.0"
      When a static webpage is requested
      Then the HTTP request includes User-Agent header "CustomBot/1.0"

  Rule: Execute Logs Progress and Completion

    Scenario: Execute logs domain processing start
      Given 2 domain URLs to process
      When execute is called
      Then a log message indicates "Starting document retrieval from 2 domains"

    Scenario: Execute logs document counts
      Given document retrieval completes with 50 documents
      When execute finishes
      Then a log message indicates the number of documents retrieved
