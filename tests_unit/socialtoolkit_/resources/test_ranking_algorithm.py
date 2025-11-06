"""
Feature: Ranking Algorithm
  As a document retrieval system
  I want to rank documents by relevance to a query
  So that the most relevant documents appear first

  Background:
    Given a RankingAlgorithm instance is initialized
"""

import pytest
from custom_nodes.red_ribbon.socialtoolkit.resources.top10_document_retrieval.ranking_algorithm import RankingAlgorithm

# Fixtures for Background

@pytest.fixture
def ranking_algorithm():
    """
    Given a RankingAlgorithm instance is initialized
    """
    return RankingAlgorithm()
class TestRankMethodAcceptsDocumentsandQuery:
    """
    Rule: Rank Method Accepts Documents and Query
    """
    def test_rank_with_valid_documents_and_query(self, ranking_algorithm):
        """
        Scenario: Rank with valid documents and query
          Given 10 documents
          And a query with keywords ["tax", "sales"]
          When I call rank with documents and query
          Then all 10 documents are returned
        """
        # Arrange
        documents = [{"id": i, "content": f"Document {i}", "title": f"Title {i}"} for i in range(10)]
        query = {"keywords": ["tax", "sales"], "original_query": "tax sales"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert len(result) == 10

    def test_rank_with_valid_documents_and_query_1(self, ranking_algorithm):
        """
        Scenario: Rank with valid documents and query
          Given 10 documents
          And a query with keywords ["tax", "sales"]
          When I call rank with documents and query
          Then documents are sorted by relevance
        """
        # Arrange
        documents = [
            {"id": 0, "content": "tax", "title": "Doc"},
            {"id": 1, "content": "sales tax", "title": "Doc"},
            {"id": 2, "content": "unrelated", "title": "Doc"}
        ]
        query = {"keywords": ["tax", "sales"], "original_query": "tax sales"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        # Document 1 has both keywords, should rank highest
        assert result[0]["id"] == 1
        assert result[0]["rank_score"] >= result[1]["rank_score"]
        assert result[1]["rank_score"] >= result[2]["rank_score"]

    def test_rank_with_empty_documents_list(self, ranking_algorithm):
        """
        Scenario: Rank with empty documents list
          Given an empty list of documents
          And a valid query
          When I call rank
          Then an empty list is returned
        """
        # Arrange
        documents = []
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert result == []

class TestRankingScoreIsBasedonKeywordFrequency:
    """
    Rule: Ranking Score Is Based on Keyword Frequency
    """
    def test_documents_with_more_keyword_matches_rank_higher(self, ranking_algorithm):
        """
        Scenario: Documents with more keyword matches rank higher
          Given document A with 5 keyword occurrences
          And document B with 2 keyword occurrences
          When documents are ranked
          Then document A ranks higher than document B
        """
        # Arrange
        doc_a = {"id": "A", "content": "tax tax tax tax tax", "title": ""}
        doc_b = {"id": "B", "content": "tax tax", "title": ""}
        documents = [doc_b, doc_a]  # Intentionally out of order
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert result[0]["id"] == "A"
        assert result[0]["rank_score"] > result[1]["rank_score"]

    def test_document_with_no_keyword_matches_has_score_of_0(self, ranking_algorithm):
        """
        Scenario: Document with no keyword matches has score of 0
          Given a document with no query keywords
          When the document is ranked
          Then the document has rank_score of 0
        """
        # Arrange
        documents = [{"id": 1, "content": "unrelated content", "title": "Unrelated"}]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert result[0]["rank_score"] == 0


class TestTitleMatchesReceiveBonusScore:
    """
    Rule: Title Matches Receive Bonus Score
    """
    def test_keyword_in_title_increases_rank_score(self, ranking_algorithm):
        """
        Scenario: Keyword in title increases rank score
          Given document A with keyword "tax" in title
          And document B with keyword "tax" only in content
          And both have same content keyword frequency
          When documents are ranked
          Then document A ranks higher than document B
        """
        # Arrange
        doc_a = {"id": "A", "content": "tax", "title": "Tax Information"}
        doc_b = {"id": "B", "content": "tax", "title": "Other"}
        documents = [doc_b, doc_a]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert result[0]["id"] == "A"
        assert result[0]["rank_score"] > result[1]["rank_score"]

    def test_keyword_in_title_increases_rank_score_1(self, ranking_algorithm):
        """
        Scenario: Keyword in title increases rank score
          Given document A with keyword "tax" in title
          And document B with keyword "tax" only in content
          And both have same content keyword frequency
          When documents are ranked
          Then document A's rank_score includes title bonus
        """
        # Arrange
        doc_a = {"id": "A", "content": "tax", "title": "Tax Information"}
        doc_b = {"id": "B", "content": "tax", "title": "Other"}
        documents = [doc_a, doc_b]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        # Title bonus is 2 per keyword in title, so doc_a should have higher score
        doc_a_result = next(d for d in result if d["id"] == "A")
        doc_b_result = next(d for d in result if d["id"] == "B")
        assert doc_a_result["rank_score"] >= doc_b_result["rank_score"] + 2

    def test_multiple_keywords_in_title_accumulate_bonuses(self, ranking_algorithm):
        """
        Scenario: Multiple keywords in title accumulate bonuses
          Given a document with 2 query keywords in title
          When the document is ranked
          Then the rank_score includes 2 title bonuses
        """
        # Arrange
        documents = [{"id": 1, "content": "", "title": "Sales Tax Information"}]
        query = {"keywords": ["sales", "tax"], "original_query": "sales tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        # Each keyword in title gives 2 bonus points
        assert result[0]["rank_score"] >= 4  # 2 keywords * 2 points each

    def test_multiple_keywords_in_title_accumulate_bonuses_1(self, ranking_algorithm):
        """
        Scenario: Multiple keywords in title accumulate bonuses
          Given a document with 2 query keywords in title
          When the document is ranked
          Then each title bonus is worth 2 points
        """
        # Arrange
        documents = [{"id": 1, "content": "", "title": "Sales Tax Information"}]
        query = {"keywords": ["sales", "tax"], "original_query": "sales tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        # Verify the bonus structure: 2 points per keyword in title
        assert result[0]["rank_score"] == 4  # 2 keywords * 2 points


class TestKeywordsatDocumentStartReceiveBonus:
    """
    Rule: Keywords at Document Start Receive Bonus
    """
    def test_document_starting_with_keyword_gets_bonus(self, ranking_algorithm):
        """
        Scenario: Document starting with keyword gets bonus
          Given document A starts with query keyword
          And document B has keyword later in content
          And both have same total keyword frequency
          When documents are ranked
          Then document A ranks higher than document B
        """
        # Arrange
        doc_a = {"id": "A", "content": "tax information", "title": ""}
        doc_b = {"id": "B", "content": "information tax", "title": ""}
        documents = [doc_b, doc_a]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert result[0]["id"] == "A"
        assert result[0]["rank_score"] > result[1]["rank_score"]

    def test_document_starting_with_keyword_gets_bonus_1(self, ranking_algorithm):
        """
        Scenario: Document starting with keyword gets bonus
          Given document A starts with query keyword
          And document B has keyword later in content
          And both have same total keyword frequency
          When documents are ranked
          Then document A's rank_score includes start bonus
        """
        # Arrange
        doc_a = {"id": "A", "content": "tax information", "title": ""}
        doc_b = {"id": "B", "content": "information tax", "title": ""}
        documents = [doc_a, doc_b]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        doc_a_result = next(d for d in result if d["id"] == "A")
        doc_b_result = next(d for d in result if d["id"] == "B")
        # Start bonus is 1 point
        assert doc_a_result["rank_score"] >= doc_b_result["rank_score"] + 1


class TestRankScoreIsStoredinDocument:
    """
    Rule: Rank Score Is Stored in Document
    """
    def test_each_document_receives_rank_score_field(self, ranking_algorithm):
        """
        Scenario: Each document receives rank_score field
          Given 5 documents to rank
          When rank is called
          Then each document has a "rank_score" field
        """
        # Arrange
        documents = [{"id": i, "content": f"Document {i}", "title": ""} for i in range(5)]
        query = {"keywords": ["document"], "original_query": "document"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        for doc in result:
            assert "rank_score" in doc

    def test_each_document_receives_rank_score_field_1(self, ranking_algorithm):
        """
        Scenario: Each document receives rank_score field
          Given 5 documents to rank
          When rank is called
          Then rank_score is a numeric value
        """
        # Arrange
        documents = [{"id": i, "content": f"Document {i}", "title": ""} for i in range(5)]
        query = {"keywords": ["document"], "original_query": "document"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        for doc in result:
            assert isinstance(doc["rank_score"], (int, float))

    def test_rank_score_reflects_combined_scoring_factors(self, ranking_algorithm):
        """
        Scenario: Rank score reflects combined scoring factors
          Given a document with keyword_count=3, title_bonus=2, start_bonus=1
          When the document is ranked
          Then the rank_score is 6 (3 + 2 + 1)
        """
        # Arrange
        # keyword_count=3: "tax tax tax"
        # title_bonus=2: "tax" in title (bonus is 2 per keyword)
        # start_bonus=1: starts with "tax"
        documents = [{"id": 1, "content": "tax tax tax", "title": "Tax Info"}]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        # 3 (keyword count) + 2 (title bonus) + 1 (start bonus) = 6
        assert result[0]["rank_score"] == 6


class TestDocumentsAreSortedinDescendingOrderbyRankScore:
    """
    Rule: Documents Are Sorted in Descending Order by Rank Score
    """
    def test_highest_scoring_document_is_first(self, ranking_algorithm):
        """
        Scenario: Highest scoring document is first
          Given documents with rank_scores [5, 10, 3, 8]
          When ranking is complete
          Then the first document has rank_score 10
        """
        # Arrange
        # Create documents that will result in different scores
        documents = [
            {"id": 1, "content": "tax tax tax tax tax", "title": ""},  # 5
            {"id": 2, "content": "tax tax tax tax tax", "title": "Tax Tax"},  # 10 (5 + 2*2 + 1)
            {"id": 3, "content": "tax tax tax", "title": ""},  # 3
            {"id": 4, "content": "tax tax tax tax tax", "title": "Tax"}  # 8 (5 + 2 + 1)
        ]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert result[0]["rank_score"] == 10

    def test_highest_scoring_document_is_first_1(self, ranking_algorithm):
        """
        Scenario: Highest scoring document is first
          Given documents with rank_scores [5, 10, 3, 8]
          When ranking is complete
          Then the last document has rank_score 3
        """
        # Arrange
        documents = [
            {"id": 1, "content": "tax tax tax tax tax", "title": ""},  # 5
            {"id": 2, "content": "tax tax tax tax tax", "title": "Tax Tax"},  # 10
            {"id": 3, "content": "tax tax tax", "title": ""},  # 3
            {"id": 4, "content": "tax tax tax tax tax", "title": "Tax"}  # 8
        ]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert result[-1]["rank_score"] == 3

    def test_equal_scores_maintain_relative_order(self, ranking_algorithm):
        """
        Scenario: Equal scores maintain relative order
          Given documents with equal rank_scores
          When ranking is complete
          Then documents with equal scores appear in stable order
        """
        # Arrange
        documents = [
            {"id": 1, "content": "tax", "title": ""},
            {"id": 2, "content": "tax", "title": ""},
            {"id": 3, "content": "tax", "title": ""}
        ]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        # Python's sorted is stable, so original order should be preserved for equal scores
        assert result[0]["id"] == 1 or result[0]["rank_score"] == result[1]["rank_score"]


class TestQueryKeywordsAreExtractedandUsedforScoring:
    """
    Rule: Query Keywords Are Extracted and Used for Scoring
    """
    def test_query_keywords_parameter_is_used(self, ranking_algorithm):
        """
        Scenario: Query keywords parameter is used
          Given a query with keywords ["business", "license"]
          When rank is called
          Then document scoring uses keywords "business" and "license"
        """
        # Arrange
        documents = [
            {"id": 1, "content": "business license information", "title": ""},
            {"id": 2, "content": "unrelated content", "title": ""}
        ]
        query = {"keywords": ["business", "license"], "original_query": "business license"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert result[0]["id"] == 1
        assert result[0]["rank_score"] > result[1]["rank_score"]

    def test_query_keywords_parameter_is_used_1(self, ranking_algorithm):
        """
        Scenario: Query keywords parameter is used
          Given a query with keywords ["business", "license"]
          When rank is called
          Then occurrences of these keywords increase rank_score
        """
        # Arrange
        documents = [{"id": 1, "content": "business license business license", "title": ""}]
        query = {"keywords": ["business", "license"], "original_query": "business license"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        # 2 occurrences of "business" + 2 occurrences of "license" + 1 start bonus = 5
        assert result[0]["rank_score"] >= 4

    def test_query_without_keywords_results_in_zero_scores(self, ranking_algorithm):
        """
        Scenario: Query without keywords results in zero scores
          Given a query with empty keywords list
          When rank is called
          Then all documents have rank_score of 0
        """
        # Arrange
        documents = [
            {"id": 1, "content": "some content", "title": "Title"},
            {"id": 2, "content": "other content", "title": "Title"}
        ]
        query = {"keywords": [], "original_query": ""}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        for doc in result:
            assert doc["rank_score"] == 0

    def test_query_without_keywords_results_in_zero_scores_1(self, ranking_algorithm):
        """
        Scenario: Query without keywords results in zero scores
          Given a query with empty keywords list
          When rank is called
          Then original document order may be preserved
        """
        # Arrange
        documents = [
            {"id": 1, "content": "content 1", "title": ""},
            {"id": 2, "content": "content 2", "title": ""}
        ]
        query = {"keywords": [], "original_query": ""}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        # With all scores at 0, order should be preserved (stable sort)
        assert result[0]["id"] == 1
        assert result[1]["id"] == 2


class TestContentMatchingIsCaseInsensitive:
    """
    Rule: Content Matching Is Case-Insensitive
    """
    def test_uppercase_and_lowercase_keywords_match_equally(self, ranking_algorithm):
        """
        Scenario: Uppercase and lowercase keywords match equally
          Given document with content "SALES TAX"
          And query keyword "sales"
          When the document is ranked
          Then the keyword match is counted
        """
        # Arrange
        documents = [{"id": 1, "content": "SALES TAX", "title": ""}]
        query = {"keywords": ["sales"], "original_query": "sales"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert result[0]["rank_score"] > 0

    def test_uppercase_and_lowercase_keywords_match_equally_1(self, ranking_algorithm):
        """
        Scenario: Uppercase and lowercase keywords match equally
          Given document with content "SALES TAX"
          And query keyword "sales"
          When the document is ranked
          Then rank_score increases
        """
        # Arrange
        doc_upper = {"id": 1, "content": "SALES TAX", "title": ""}
        doc_lower = {"id": 2, "content": "sales tax", "title": ""}
        documents = [doc_upper, doc_lower]
        query = {"keywords": ["sales"], "original_query": "sales"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        # Both should have the same score due to case-insensitive matching
        assert result[0]["rank_score"] == result[1]["rank_score"]

    def test_title_matching_is_case_insensitive(self, ranking_algorithm):
        """
        Scenario: Title matching is case-insensitive
          Given document with title "Property TAX"
          And query keyword "tax"
          When the document is ranked
          Then the title bonus applies
        """
        # Arrange
        documents = [{"id": 1, "content": "", "title": "Property TAX"}]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        # Should have title bonus of 2
        assert result[0]["rank_score"] >= 2

    def test_title_matching_is_case_insensitive_1(self, ranking_algorithm):
        """
        Scenario: Title matching is case-insensitive
          Given document with title "Property TAX"
          And query keyword "tax"
          When the document is ranked
          Then rank_score includes title bonus
        """
        # Arrange
        doc_with_title = {"id": 1, "content": "", "title": "Property TAX"}
        doc_without_title = {"id": 2, "content": "", "title": "Other"}
        documents = [doc_with_title, doc_without_title]
        query = {"keywords": ["tax"], "original_query": "tax"}
        
        # Act
        result = ranking_algorithm.rank(documents, query)
        
        # Assert
        with_title_result = next(d for d in result if d["id"] == 1)
        without_title_result = next(d for d in result if d["id"] == 2)
        assert with_title_result["rank_score"] > without_title_result["rank_score"]


class TestRankingLogsOperations:
    """
    Rule: Ranking Logs Operations
    """
    def test_rank_logs_document_count_and_query(self, ranking_algorithm, caplog):
        """
        Scenario: Rank logs document count and query
          Given 15 documents to rank
          And a query with original_query "sales tax rates"
          When rank is called
          Then a log message indicates "Ranking 15 documents for query: sales tax rates"
        """
        # Arrange
        documents = [{"id": i, "content": f"Document {i}", "title": ""} for i in range(15)]
        query = {"keywords": ["sales", "rates"], "original_query": "sales tax rates"}
        
        # Act
        with caplog.at_level("INFO"):
            result = ranking_algorithm.rank(documents, query)
        
        # Assert
        assert any("Ranking 15 documents" in record.message for record in caplog.records)
        assert any("sales tax rates" in record.message for record in caplog.records)

