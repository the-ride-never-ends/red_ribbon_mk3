Feature: Query Processor
  As a search system
  I want to process raw search queries into structured components
  So that queries can be efficiently matched against documents

  Background:
    Given a QueryProcessor instance is initialized

  Rule: Process Method Accepts String Query

    Scenario: Process with valid query string
      Given a query string "What is the sales tax rate?"
      When I call process with the query
      Then a dictionary is returned
      And the dictionary contains processed query components

    Scenario: Process with empty string
      Given an empty query string ""
      When I call process
      Then a dictionary is returned
      And normalized_query is an empty string
      And tokens list is empty

  Rule: Process Returns Dictionary with Required Keys

    Scenario: Processed query contains original_query
      Given query "Sales Tax Information"
      When process is called
      Then the result contains key "original_query"
      And original_query value is "Sales Tax Information"

    Scenario: Processed query contains normalized_query
      Given query "SALES TAX"
      When process is called
      Then the result contains key "normalized_query"
      And normalized_query is lowercased and stripped

    Scenario: Processed query contains tokens
      Given query "sales tax rate"
      When process is called
      Then the result contains key "tokens"
      And tokens is a list of words

    Scenario: Processed query contains keywords
      Given query "what is the sales tax rate"
      When process is called
      Then the result contains key "keywords"
      And keywords is a filtered list of significant words

  Rule: Normalization Converts Query to Lowercase

    Scenario: Uppercase query is normalized to lowercase
      Given query "SALES TAX RATE"
      When process is called
      Then normalized_query is "sales tax rate"

    Scenario: Mixed case query is normalized to lowercase
      Given query "Sales Tax Rate"
      When process is called
      Then normalized_query is "sales tax rate"

  Rule: Normalization Strips Whitespace

    Scenario: Leading whitespace is removed
      Given query "  sales tax"
      When process is called
      Then normalized_query starts with "sales"

    Scenario: Trailing whitespace is removed
      Given query "sales tax  "
      When process is called
      Then normalized_query ends with "tax"

    Scenario: Multiple spaces are preserved in middle
      Given query "sales    tax"
      When process is called
      Then normalized_query is "sales    tax"

  Rule: Tokenization Splits Query into Words

    Scenario: Query is split on whitespace
      Given query "sales tax rate"
      When process is called
      Then tokens is ["sales", "tax", "rate"]

    Scenario: Single word query produces single token
      Given query "sales"
      When process is called
      Then tokens is ["sales"]

    Scenario: Query with punctuation includes punctuation in tokens
      Given query "What's the rate?"
      When process is called
      Then tokens includes words with punctuation preserved

  Rule: Keywords Filter Extracts Significant Words

    Scenario: Words longer than 3 characters are keywords
      Given query "what is the sales tax rate"
      When process is called
      Then keywords includes "what", "sales", "rate"
      And keywords does not include "is" or "the" (length <= 3)

    Scenario: Query with all short words has empty keywords
      Given query "is it at no"
      When process is called
      Then keywords is an empty list

    Scenario: Query with all long words includes all as keywords
      Given query "sales property income"
      When process is called
      Then keywords is ["sales", "property", "income"]

  Rule: Original Query Is Preserved Unchanged

    Scenario: Original query retains original case
      Given query "Sales TAX Rate"
      When process is called
      Then original_query is exactly "Sales TAX Rate"

    Scenario: Original query retains whitespace
      Given query "  sales  tax  "
      When process is called
      Then original_query is exactly "  sales  tax  "

  Rule: Process Handles Special Characters

    Scenario: Query with punctuation is processed
      Given query "What's the sales tax rate?"
      When process is called
      Then the query is tokenized with punctuation
      And processing completes successfully

    Scenario: Query with numbers is processed
      Given query "sales tax 2024"
      When process is called
      Then tokens includes "sales", "tax", "2024"
      And "2024" is included as a keyword

  Rule: Process Logs Query Processing

    Scenario: Process logs the query being processed
      Given query "business license requirements"
      When process is called
      Then a log message indicates "Processing query: business license requirements"

  Rule: Empty or Whitespace-Only Queries Are Handled

    Scenario: Empty string produces empty components
      Given query ""
      When process is called
      Then normalized_query is ""
      And tokens is []
      And keywords is []

    Scenario: Whitespace-only query produces empty normalized query
      Given query "   "
      When process is called
      Then normalized_query is ""
      And tokens is []

  Rule: Unicode and Non-ASCII Characters Are Handled

    Scenario: Query with accented characters is processed
      Given query "café tax"
      When process is called
      Then the query is processed without error
      And tokens includes the words with accents

    Scenario: Query with emoji or special unicode is processed
      Given query "sales 💰 tax"
      When process is called
      Then the query is processed without error
      And tokens may include emoji character
