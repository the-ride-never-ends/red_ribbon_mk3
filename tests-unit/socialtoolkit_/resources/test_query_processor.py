"""
Feature: Query Processor
  As a search system
  I want to process raw search queries into structured components
  So that queries can be efficiently matched against documents

  Background:
    Given a QueryProcessor instance is initialized
"""

import pytest
from custom_nodes.red_ribbon.socialtoolkit.resources.top10_document_retrieval.query_processor import QueryProcessor

# Fixtures for Background

@pytest.fixture
def query_processor():
    """
    Given a QueryProcessor instance is initialized
    """
    return QueryProcessor()
class TestProcessMethodAcceptsStringQuery:
    """
    Rule: Process Method Accepts String Query
    """
    def test_process_with_valid_query_string(self, query_processor):
        """
        Scenario: Process with valid query string
          Given a query string "What is the sales tax rate?"
          When I call process with the query
          Then a dictionary is returned
        """
        # Arrange
        query = "What is the sales tax rate?"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert isinstance(result, dict)

    def test_process_with_valid_query_string_1(self, query_processor):
        """
        Scenario: Process with valid query string
          Given a query string "What is the sales tax rate?"
          When I call process with the query
          Then the dictionary contains processed query components
        """
        # Arrange
        query = "What is the sales tax rate?"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert "original_query" in result
        assert "normalized_query" in result
        assert "tokens" in result
        assert "keywords" in result

    def test_process_with_empty_string(self, query_processor):
        """
        Scenario: Process with empty string
          Given an empty query string ""
          When I call process
          Then a dictionary is returned
        """
        # Arrange
        query = ""
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert isinstance(result, dict)

    def test_process_with_empty_string_1(self, query_processor):
        """
        Scenario: Process with empty string
          Given an empty query string ""
          When I call process
          Then normalized_query is an empty string
        """
        # Arrange
        query = ""
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["normalized_query"] == ""

    def test_process_with_empty_string_2(self, query_processor):
        """
        Scenario: Process with empty string
          Given an empty query string ""
          When I call process
          Then tokens list is empty
        """
        # Arrange
        query = ""
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["tokens"] == []

class TestProcessReturnsDictionarywithRequiredKeys:
    """
    Rule: Process Returns Dictionary with Required Keys
    """
    def test_processed_query_contains_original_query(self, query_processor):
        """
        Scenario: Processed query contains original_query
          Given query "Sales Tax Information"
          When process is called
          Then the result contains key "original_query"
        """
        # Arrange
        query = "Sales Tax Information"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert "original_query" in result

    def test_processed_query_contains_original_query_1(self, query_processor):
        """
        Scenario: Processed query contains original_query
          Given query "Sales Tax Information"
          When process is called
          Then original_query value is "Sales Tax Information"
        """
        # Arrange
        query = "Sales Tax Information"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["original_query"] == "Sales Tax Information"

    def test_processed_query_contains_normalized_query(self, query_processor):
        """
        Scenario: Processed query contains normalized_query
          Given query "SALES TAX"
          When process is called
          Then the result contains key "normalized_query"
        """
        # Arrange
        query = "SALES TAX"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert "normalized_query" in result

    def test_processed_query_contains_normalized_query_1(self, query_processor):
        """
        Scenario: Processed query contains normalized_query
          Given query "SALES TAX"
          When process is called
          Then normalized_query is lowercased and stripped
        """
        # Arrange
        query = "SALES TAX"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["normalized_query"] == "sales tax"

    def test_processed_query_contains_tokens(self, query_processor):
        """
        Scenario: Processed query contains tokens
          Given query "sales tax rate"
          When process is called
          Then the result contains key "tokens"
        """
        # Arrange
        query = "sales tax rate"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert "tokens" in result

    def test_processed_query_contains_tokens_1(self, query_processor):
        """
        Scenario: Processed query contains tokens
          Given query "sales tax rate"
          When process is called
          Then tokens is a list of words
        """
        # Arrange
        query = "sales tax rate"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert isinstance(result["tokens"], list)
        assert result["tokens"] == ["sales", "tax", "rate"]

    def test_processed_query_contains_keywords(self, query_processor):
        """
        Scenario: Processed query contains keywords
          Given query "what is the sales tax rate"
          When process is called
          Then the result contains key "keywords"
        """
        # Arrange
        query = "what is the sales tax rate"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert "keywords" in result

    def test_processed_query_contains_keywords_1(self, query_processor):
        """
        Scenario: Processed query contains keywords
          Given query "what is the sales tax rate"
          When process is called
          Then keywords is a filtered list of significant words
        """
        # Arrange
        query = "what is the sales tax rate"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert isinstance(result["keywords"], list)
        # Keywords filter: words with length > 3
        assert "what" in result["keywords"]
        assert "sales" in result["keywords"]
        assert "rate" in result["keywords"]
        assert "is" not in result["keywords"]  # Length <= 3
        assert "the" not in result["keywords"]  # Length <= 3


class TestNormalizationConvertsQuerytoLowercase:
    """
    Rule: Normalization Converts Query to Lowercase
    """
    def test_uppercase_query_is_normalized_to_lowercase(self, query_processor):
        """
        Scenario: Uppercase query is normalized to lowercase
          Given query "SALES TAX RATE"
          When process is called
          Then normalized_query is "sales tax rate"
        """
        # Arrange
        query = "SALES TAX RATE"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["normalized_query"] == "sales tax rate"

    def test_mixed_case_query_is_normalized_to_lowercase(self, query_processor):
        """
        Scenario: Mixed case query is normalized to lowercase
          Given query "Sales Tax Rate"
          When process is called
          Then normalized_query is "sales tax rate"
        """
        # Arrange
        query = "Sales Tax Rate"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["normalized_query"] == "sales tax rate"


class TestNormalizationStripsWhitespace:
    """
    Rule: Normalization Strips Whitespace
    """
    def test_leading_whitespace_is_removed(self, query_processor):
        """
        Scenario: Leading whitespace is removed
          Given query "  sales tax"
          When process is called
          Then normalized_query starts with "sales"
        """
        # Arrange
        query = "  sales tax"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["normalized_query"].startswith("sales")

    def test_trailing_whitespace_is_removed(self, query_processor):
        """
        Scenario: Trailing whitespace is removed
          Given query "sales tax  "
          When process is called
          Then normalized_query ends with "tax"
        """
        # Arrange
        query = "sales tax  "
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["normalized_query"].endswith("tax")

    def test_multiple_spaces_are_preserved_in_middle(self, query_processor):
        """
        Scenario: Multiple spaces are preserved in middle
          Given query "sales    tax"
          When process is called
          Then normalized_query is "sales    tax"
        """
        # Arrange
        query = "sales    tax"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["normalized_query"] == "sales    tax"


class TestTokenizationSplitsQueryintoWords:
    """
    Rule: Tokenization Splits Query into Words
    """
    def test_query_is_split_on_whitespace(self, query_processor):
        """
        Scenario: Query is split on whitespace
          Given query "sales tax rate"
          When process is called
          Then tokens is ["sales", "tax", "rate"]
        """
        # Arrange
        query = "sales tax rate"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["tokens"] == ["sales", "tax", "rate"]

    def test_single_word_query_produces_single_token(self, query_processor):
        """
        Scenario: Single word query produces single token
          Given query "sales"
          When process is called
          Then tokens is ["sales"]
        """
        # Arrange
        query = "sales"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["tokens"] == ["sales"]

    def test_query_with_punctuation_includes_punctuation_in_tokens(self, query_processor):
        """
        Scenario: Query with punctuation includes punctuation in tokens
          Given query "What's the rate?"
          When process is called
          Then tokens includes words with punctuation preserved
        """
        # Arrange
        query = "What's the rate?"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        # split() preserves punctuation attached to words
        assert "what's" in result["tokens"]
        assert "rate?" in result["tokens"]


class TestKeywordsFilterExtractsSignificantWords:
    """
    Rule: Keywords Filter Extracts Significant Words
    """
    def test_words_longer_than_3_characters_are_keywords(self, query_processor):
        """
        Scenario: Words longer than 3 characters are keywords
          Given query "what is the sales tax rate"
          When process is called
          Then keywords includes "what", "sales", "rate"
        """
        # Arrange
        query = "what is the sales tax rate"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert "what" in result["keywords"]
        assert "sales" in result["keywords"]
        assert "rate" in result["keywords"]

    def test_words_longer_than_3_characters_are_keywords_1(self, query_processor):
        """
        Scenario: Words longer than 3 characters are keywords
          Given query "what is the sales tax rate"
          When process is called
          Then keywords does not include "is" or "the" (length <= 3)
        """
        # Arrange
        query = "what is the sales tax rate"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert "is" not in result["keywords"]
        assert "the" not in result["keywords"]
        assert "tax" not in result["keywords"]  # length == 3

    def test_query_with_all_short_words_has_empty_keywords(self, query_processor):
        """
        Scenario: Query with all short words has empty keywords
          Given query "is it at no"
          When process is called
          Then keywords is an empty list
        """
        # Arrange
        query = "is it at no"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["keywords"] == []

    def test_query_with_all_long_words_includes_all_as_keywords(self, query_processor):
        """
        Scenario: Query with all long words includes all as keywords
          Given query "sales property income"
          When process is called
          Then keywords is ["sales", "property", "income"]
        """
        # Arrange
        query = "sales property income"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["keywords"] == ["sales", "property", "income"]


class TestOriginalQueryIsPreservedUnchanged:
    """
    Rule: Original Query Is Preserved Unchanged
    """
    def test_original_query_retains_original_case(self, query_processor):
        """
        Scenario: Original query retains original case
          Given query "Sales TAX Rate"
          When process is called
          Then original_query is exactly "Sales TAX Rate"
        """
        # Arrange
        query = "Sales TAX Rate"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["original_query"] == "Sales TAX Rate"

    def test_original_query_retains_whitespace(self, query_processor):
        """
        Scenario: Original query retains whitespace
          Given query "  sales  tax  "
          When process is called
          Then original_query is exactly "  sales  tax  "
        """
        # Arrange
        query = "  sales  tax  "
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["original_query"] == "  sales  tax  "


class TestProcessHandlesSpecialCharacters:
    """
    Rule: Process Handles Special Characters
    """
    def test_query_with_punctuation_is_processed(self, query_processor):
        """
        Scenario: Query with punctuation is processed
          Given query "What's the sales tax rate?"
          When process is called
          Then the query is tokenized with punctuation
        """
        # Arrange
        query = "What's the sales tax rate?"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert isinstance(result["tokens"], list)
        assert len(result["tokens"]) > 0

    def test_query_with_punctuation_is_processed_1(self, query_processor):
        """
        Scenario: Query with punctuation is processed
          Given query "What's the sales tax rate?"
          When process is called
          Then processing completes successfully
        """
        # Arrange
        query = "What's the sales tax rate?"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result is not None
        assert isinstance(result, dict)

    def test_query_with_numbers_is_processed(self, query_processor):
        """
        Scenario: Query with numbers is processed
          Given query "sales tax 2024"
          When process is called
          Then tokens includes "sales", "tax", "2024"
        """
        # Arrange
        query = "sales tax 2024"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert "sales" in result["tokens"]
        assert "tax" in result["tokens"]
        assert "2024" in result["tokens"]

    def test_query_with_numbers_is_processed_1(self, query_processor):
        """
        Scenario: Query with numbers is processed
          Given query "sales tax 2024"
          When process is called
          Then "2024" is included as a keyword
        """
        # Arrange
        query = "sales tax 2024"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert "2024" in result["keywords"]


class TestProcessLogsQueryProcessing:
    """
    Rule: Process Logs Query Processing
    """
    def test_process_logs_the_query_being_processed(self, query_processor, caplog):
        """
        Scenario: Process logs the query being processed
          Given query "business license requirements"
          When process is called
          Then a log message indicates "Processing query: business license requirements"
        """
        # Arrange
        query = "business license requirements"
        
        # Act
        with caplog.at_level("INFO"):
            result = query_processor.process(query)
        
        # Assert
        assert any("Processing query" in record.message for record in caplog.records)


class TestEmptyorWhitespaceOnlyQueriesAreHandled:
    """
    Rule: Empty or Whitespace-Only Queries Are Handled
    """
    def test_empty_string_produces_empty_components(self, query_processor):
        """
        Scenario: Empty string produces empty components
          Given query ""
          When process is called
          Then normalized_query is ""
        """
        # Arrange
        query = ""
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["normalized_query"] == ""

    def test_empty_string_produces_empty_components_1(self, query_processor):
        """
        Scenario: Empty string produces empty components
          Given query ""
          When process is called
          Then tokens is []
        """
        # Arrange
        query = ""
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["tokens"] == []

    def test_empty_string_produces_empty_components_2(self, query_processor):
        """
        Scenario: Empty string produces empty components
          Given query ""
          When process is called
          Then keywords is []
        """
        # Arrange
        query = ""
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["keywords"] == []

    def test_whitespace_only_query_produces_empty_normalized_query(self, query_processor):
        """
        Scenario: Whitespace-only query produces empty normalized query
          Given query "   "
          When process is called
          Then normalized_query is ""
        """
        # Arrange
        query = "   "
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["normalized_query"] == ""

    def test_whitespace_only_query_produces_empty_normalized_query_1(self, query_processor):
        """
        Scenario: Whitespace-only query produces empty normalized query
          Given query "   "
          When process is called
          Then tokens is []
        """
        # Arrange
        query = "   "
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result["tokens"] == []


class TestUnicodeandNonASCIICharactersAreHandled:
    """
    Rule: Unicode and Non-ASCII Characters Are Handled
    """
    def test_query_with_accented_characters_is_processed(self, query_processor):
        """
        Scenario: Query with accented characters is processed
          Given query "café tax"
          When process is called
          Then the query is processed without error
        """
        # Arrange
        query = "café tax"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result is not None
        assert isinstance(result, dict)

    def test_query_with_accented_characters_is_processed_1(self, query_processor):
        """
        Scenario: Query with accented characters is processed
          Given query "café tax"
          When process is called
          Then tokens includes the words with accents
        """
        # Arrange
        query = "café tax"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert "café" in result["tokens"]
        assert "tax" in result["tokens"]

    def test_query_with_emoji_or_special_unicode_is_processed(self, query_processor):
        """
        Scenario: Query with emoji or special unicode is processed
          Given query "sales 💰 tax"
          When process is called
          Then the query is processed without error
        """
        # Arrange
        query = "sales 💰 tax"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        assert result is not None
        assert isinstance(result, dict)

    def test_query_with_emoji_or_special_unicode_is_processed_1(self, query_processor):
        """
        Scenario: Query with emoji or special unicode is processed
          Given query "sales 💰 tax"
          When process is called
          Then tokens may include emoji character
        """
        # Arrange
        query = "sales 💰 tax"
        
        # Act
        result = query_processor.process(query)
        
        # Assert
        # Emoji may be in tokens depending on implementation
        assert isinstance(result["tokens"], list)
        assert "sales" in result["tokens"]
        assert "tax" in result["tokens"]

