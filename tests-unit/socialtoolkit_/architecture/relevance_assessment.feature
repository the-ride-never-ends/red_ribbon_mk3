Feature: Relevance Assessment
  As a document analysis system
  I want to assess and filter documents based on relevance to a variable definition
  So that only pertinent information is used for data extraction

  Background:
    Given a RelevanceAssessment instance is initialized
    And a variable codebook service is available
    And a top10 retrieval service is available
    And a cited page extractor service is available
    And a prompt decision tree service is available
    And an LLM API client is available

  Rule: Control Flow Method Returns Dictionary with Required Keys

    Scenario: Control flow returns expected result structure
      Given a list of 10 potentially relevant documents
      And a variable definition for "sales_tax_rate"
      When I call control_flow with documents and variable definition
      Then I receive a dictionary response
      And the response contains key "relevant_pages"
      And the response contains key "relevant_doc_ids"
      And the response contains key "page_numbers"
      And the response contains key "relevance_scores"

  Rule: Relevance Assessment Filters Documents by Criteria Threshold

    Scenario: Only documents above threshold are marked relevant
      Given criteria_threshold is configured as 0.7
      And documents with relevance scores [0.9, 0.8, 0.6, 0.4]
      When control_flow assesses the documents
      Then exactly 2 documents are in "relevant_pages"
      And the relevant documents have scores >= 0.7
      And documents with scores < 0.7 are in discarded pages

    Scenario: All documents pass when all exceed threshold
      Given criteria_threshold is configured as 0.5
      And all documents have relevance scores >= 0.5
      When control_flow assesses the documents
      Then all documents are in "relevant_pages"
      And no documents are discarded

    Scenario: No documents pass when all below threshold
      Given criteria_threshold is configured as 0.9
      And all documents have relevance scores < 0.9
      When control_flow assesses the documents
      Then "relevant_pages" is an empty list
      And all documents are in discarded pages

  Rule: Hallucination Filter Is Applied When Enabled

    Scenario: Hallucination filter removes false information
      Given use_hallucination_filter is configured as True
      And assessment results contain hallucinated content
      When control_flow processes the results
      Then hallucinated content is filtered out
      And only verified information remains

    Scenario: Hallucination filter is skipped when disabled
      Given use_hallucination_filter is configured as False
      And assessment results may contain hallucinations
      When control_flow processes the results
      Then no hallucination filtering is performed
      And all assessment results are retained

  Rule: Relevance Scores Are Calculated for Each Document

    Scenario: Each document receives a relevance score
      Given 5 potentially relevant documents
      When relevance is assessed
      Then exactly 5 relevance scores are calculated
      And each score is between 0.0 and 1.0
      And scores reflect document relevance to variable

    Scenario: Relevance scores dictionary maps doc IDs to scores
      Given documents with IDs ["doc1", "doc2", "doc3"]
      When relevance is assessed
      Then the "relevance_scores" dictionary has keys ["doc1", "doc2", "doc3"]
      And each key maps to a float score value

  Rule: Page Numbers Are Extracted from Relevant Documents

    Scenario: Page numbers are extracted for relevant documents
      Given relevant documents reference pages [1, 3, 5, 7]
      When control_flow completes
      Then "page_numbers" contains [1, 3, 5, 7]
      And page numbers are in ascending order

    Scenario: No page numbers when no relevant documents
      Given no documents pass the relevance threshold
      When control_flow completes
      Then "page_numbers" is an empty list

  Rule: Cited Pages Are Extracted by Page Numbers

    Scenario: Full page content is extracted for cited pages
      Given potentially relevant documents with pages 1-10
      And page numbers [2, 5, 8] are identified as relevant
      When cited pages are extracted
      Then exactly 3 pages of content are returned
      And the pages correspond to page numbers 2, 5, and 8
      And each page contains full text content

  Rule: LLM API Is Used for Relevance Assessment

    Scenario: LLM API receives document and variable definition
      Given a document and variable definition
      When relevance is assessed
      Then the LLM API is called with both inputs
      And the API returns a relevance judgment

    Scenario: Max retries is respected for LLM API failures
      Given max_retries is configured as 3
      And the LLM API fails on first 2 attempts
      When relevance assessment is performed
      Then the API is retried up to 3 times
      And assessment completes on third attempt

    Scenario: Persistent LLM API failure is handled
      Given max_retries is configured as 3
      And the LLM API fails on all attempts
      When relevance assessment is performed
      Then an error is logged
      And the document is marked as unable to assess

  Rule: Citations Are Truncated to Maximum Length

    Scenario: Long citations are truncated
      Given max_citation_length is configured as 500
      And a citation text with 1000 characters
      When the citation is processed
      Then the citation is truncated to 500 characters
      And a truncation indicator is added

    Scenario: Short citations are not truncated
      Given max_citation_length is configured as 500
      And a citation text with 200 characters
      When the citation is processed
      Then the full 200 character text is preserved
      And no truncation occurs

  Rule: Assess Method Provides Public Interface

    Scenario: Assess method accepts documents and prompt sequence
      Given 5 potentially relevant documents
      And a prompt sequence for assessment
      When I call assess with documents and prompts
      Then relevance assessment is performed
      And a list of relevant documents is returned

    Scenario: Assess method returns filtered document list
      Given 10 input documents
      And 6 documents are assessed as relevant
      When I call assess
      Then exactly 6 documents are returned
      And all returned documents exceeded threshold

  Rule: Control Flow Validates Input Types

    Scenario: Control flow rejects non-list documents parameter
      Given a documents parameter that is not a list
      When I call control_flow
      Then a TypeError is raised
      And the error indicates documents must be a list

    Scenario: Control flow rejects invalid variable definition
      Given a variable_definition that is not a dictionary
      When I call control_flow
      Then a TypeError is raised
      And the error indicates variable_definition must be a dict

    Scenario: Control flow accepts empty document list
      Given an empty list of potentially relevant documents
      When I call control_flow
      Then execution completes without error
      And empty results are returned

  Rule: Relevant Document IDs Are Tracked

    Scenario: Relevant doc IDs match relevant documents
      Given documents with IDs ["A", "B", "C", "D", "E"]
      And documents "B" and "D" are assessed as relevant
      When control_flow completes
      Then "relevant_doc_ids" contains ["B", "D"]
      And the IDs correspond to documents in "relevant_pages"

  Rule: Assessment Logs Progress and Completion

    Scenario: Control flow logs start with document count
      Given 15 potentially relevant documents
      When control_flow is called
      Then a log message indicates "Starting relevance assessment for 15 documents"

    Scenario: Control flow logs completion with relevant count
      Given relevance assessment identifies 8 relevant pages
      When control_flow completes
      Then a log message indicates "Completed relevance assessment: 8 relevant pages"
