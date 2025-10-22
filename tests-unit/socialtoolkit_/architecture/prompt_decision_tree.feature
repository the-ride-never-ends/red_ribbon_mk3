Feature: Prompt Decision Tree
  As a data extraction system
  I want to execute decision trees of prompts to extract information from documents
  So that structured data can be extracted following a logical flow

  Background:
    Given a PromptDecisionTree instance is initialized
    And an LLM API client is available
    And a variable codebook service is available
    And a logger is available

  Rule: Execute Method Returns Extracted Data Point

    Scenario: Execute completes successfully
      Given a list of 3 relevant pages
      And a prompt sequence with 2 prompts
      When I call execute with pages and prompts
      Then a string data point is returned
      And the data point contains extracted information

    Scenario: Execute handles empty relevant pages
      Given an empty list of relevant pages
      And a prompt sequence
      When I call execute with pages and prompts
      Then execution completes without error
      And an empty or default data point is returned

  Rule: Control Flow Method Returns Dictionary with Required Keys

    Scenario: Control flow returns expected result structure
      Given 5 relevant pages
      And a prompt sequence
      When I call control_flow with pages and prompts
      Then I receive a dictionary response
      And the response contains key "success"
      And the response contains key "output_data_point"
      And the response contains key "responses"
      And the response contains key "iterations"

    Scenario: Successful control flow has success True
      Given valid inputs for control flow
      When control flow executes successfully
      Then the "success" field is True
      And "output_data_point" contains extracted data
      And "error" key is not present

    Scenario: Failed control flow has success False
      Given inputs that cause execution to fail
      When control flow encounters an error
      Then the "success" field is False
      And "error" key contains error description
      And "output_data_point" is empty string

  Rule: Pages Are Concatenated Up to max_pages_to_concatenate

    Scenario: All pages used when count is below maximum
      Given max_pages_to_concatenate is configured as 10
      And 5 relevant pages are provided
      When pages are concatenated
      Then all 5 pages are included in concatenated text

    Scenario: Pages limited when count exceeds maximum
      Given max_pages_to_concatenate is configured as 10
      And 15 relevant pages are provided
      When pages are concatenated
      Then exactly 10 pages are included
      And pages 11-15 are not included

  Rule: Concatenated Pages Include Title, URL, and Content

    Scenario: Each page is formatted with metadata
      Given a page with title "Tax Document", URL "https://example.com", and content "text"
      When the page is concatenated
      Then the output includes "# Tax Document"
      And the output includes "## Source: https://example.com"
      And the output includes "## Content:"
      And the output includes the page content

    Scenario: Missing title uses default
      Given a page without a title field
      When the page is concatenated
      Then a default title like "Document 1" is used

  Rule: Decision Tree Is Executed with Node Traversal

    Scenario: Tree execution starts at first node
      Given a decision tree with 3 nodes
      When execution begins
      Then the first node is processed first
      And its prompt is sent to the LLM

    Scenario: Tree execution follows edges based on responses
      Given a decision tree with conditional edges
      And an LLM response that matches an edge condition
      When the response is evaluated
      Then the next node follows the matching edge
      And that node is processed next

    Scenario: Tree execution stops at final node
      Given a decision tree where node 3 is marked as final
      When execution reaches node 3
      Then execution stops
      And the final response is processed

    Scenario: Tree execution stops at max iterations
      Given max_iterations is configured as 5
      And a decision tree that could run longer
      When 5 iterations complete
      Then execution stops
      And results from 5 iterations are returned

  Rule: LLM Prompts Are Generated with Document Context

    Scenario: Prompt includes question and document text
      Given a node with prompt "What is the tax rate?"
      And concatenated document text
      When the prompt is generated
      Then the prompt includes the node's question
      And the prompt includes the full document text
      And the prompt includes instructions for the LLM

    Scenario: Long documents are truncated to context window
      Given context_window_size is configured as 8192
      And concatenated document has 10000 characters
      When the prompt is generated
      Then the document is truncated to fit context window
      And approximately 500 characters reserved for instructions

  Rule: LLM Responses Are Collected for Each Node

    Scenario: Each node execution records response
      Given a tree with 3 nodes executes
      When all nodes are processed
      Then exactly 3 responses are recorded
      And each response includes "node_id", "prompt", and "response"

    Scenario: Response order matches execution order
      Given nodes execute in order node_0, node_1, node_2
      When responses are examined
      Then responses[0] is from node_0
      And responses[1] is from node_1
      And responses[2] is from node_2

  Rule: Output Data Point Is Extracted from Final Response

    Scenario: Percentage pattern is extracted
      Given final LLM response is "The rate is 5.5%"
      When output data point is extracted
      Then the data point is "5.5%"

    Scenario: Percentage word format is extracted
      Given final LLM response is "The rate is 7 percent"
      When output data point is extracted
      Then the data point is "7%"

    Scenario: Rate statement is extracted
      Given final LLM response is "The rate is 3.25 based on ordinance"
      When output data point is extracted
      Then the data point is "3.25%"

    Scenario: Long response is truncated
      Given final LLM response with 500 characters
      And no specific patterns match
      When output data point is extracted
      Then the data point is truncated to 100 characters
      And "..." is appended

  Rule: NgramValidator Provides Text Analysis Utilities

    Scenario: text_to_ngrams extracts word sequences
      Given text "The quick brown fox jumps"
      And n is 2
      When text_to_ngrams is called
      Then bigrams are extracted from the text
      And stop words and punctuation are filtered

    Scenario: text_to_ngrams rejects invalid input types
      Given text parameter is not a string
      When text_to_ngrams is called
      Then a TypeError is raised
      And error indicates text must be a string

    Scenario: text_to_ngrams rejects invalid n values
      Given n is 0 or negative
      When text_to_ngrams is called
      Then a ValueError is raised
      And error indicates n must be positive

  Rule: sentence_ngram_fraction Calculates Text Overlap

    Scenario: Full overlap returns 1.0
      Given original_text contains all ngrams from test_text
      When sentence_ngram_fraction is called
      Then the fraction is 1.0 or True

    Scenario: Partial overlap returns fractional value
      Given test_text has 4 ngrams
      And 2 of them appear in original_text
      When sentence_ngram_fraction is called
      Then the fraction is 0.5

    Scenario: No overlap returns 0.0
      Given original_text shares no ngrams with test_text
      When sentence_ngram_fraction is called
      Then the fraction is 0.0

    Scenario: Empty test_text returns True
      Given test_text has no ngrams
      When sentence_ngram_fraction is called
      Then the result is True

    Scenario: Invalid parameter types raise TypeError
      Given original_text or test_text is not a string
      When sentence_ngram_fraction is called
      Then a TypeError is raised

  Rule: Decision Tree Node Creation from Prompt Sequence

    Scenario: Linear sequence creates sequential nodes
      Given a prompt sequence with 3 prompts
      When decision tree is created
      Then 3 nodes are created
      And each node has ID "node_0", "node_1", "node_2"
      And each node contains one prompt

    Scenario: Last node is marked as final
      Given a prompt sequence with 4 prompts
      When decision tree is created
      Then node_3 has is_final set to True
      And nodes 0-2 have is_final set to False

    Scenario: Sequential nodes have default edges
      Given a prompt sequence with 3 prompts
      When decision tree is created
      Then node_0 has edge to node_1
      And node_1 has edge to node_2
      And node_2 has no edges (final node)

  Rule: Execution Validates Input Types

    Scenario: Execute rejects non-list relevant_pages
      Given relevant_pages is not a list
      When execute is called
      Then a TypeError is raised

    Scenario: Execute rejects non-list prompt_sequence
      Given prompt_sequence is not a list
      When execute is called
      Then a TypeError is raised

    Scenario: Execute accepts empty lists
      Given relevant_pages and prompt_sequence are empty lists
      When execute is called
      Then execution completes without TypeError

  Rule: Human Review Integration When Errors Occur

    Scenario: Human review is requested for errors when enabled
      Given enable_human_review is configured as True
      And control flow encounters an error
      When human review is requested
      Then review request includes error details
      And review request includes document text
      And review request includes LLM responses

    Scenario: Human review can override error results
      Given an error occurred during execution
      And human review provides corrected output
      When review completes
      Then result "success" is set to True
      And "output_data_point" contains human-provided value
      And "human_reviewed" flag is set to True

  Rule: Logging Tracks Execution Progress

    Scenario: Execute logs start with page count
      Given 7 relevant pages
      When execute is called
      Then a log message indicates "Starting prompt decision tree with 7 pages"

    Scenario: Execute logs completion
      Given execution completes successfully
      Then a log message indicates "Completed prompt decision tree execution"

    Scenario: Errors during execution are logged
      Given an exception occurs during tree execution
      When the error is caught
      Then the error is logged with logger.error
      And error message includes exception details



