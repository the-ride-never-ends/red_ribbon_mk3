Feature: Ranking Algorithm
  As a document retrieval system
  I want to rank documents by relevance to a query
  So that the most relevant documents appear first

  Background:
    Given a RankingAlgorithm instance is initialized

  Rule: Rank Method Accepts Documents and Query

    Scenario: Rank with valid documents and query
      Given 10 documents
      And a query with keywords ["tax", "sales"]
      When I call rank with documents and query
      Then all 10 documents are returned
      And documents are sorted by relevance

    Scenario: Rank with empty documents list
      Given an empty list of documents
      And a valid query
      When I call rank
      Then an empty list is returned

  Rule: Ranking Score Is Based on Keyword Frequency

    Scenario: Documents with more keyword matches rank higher
      Given document A with 5 keyword occurrences
      And document B with 2 keyword occurrences
      When documents are ranked
      Then document A ranks higher than document B

    Scenario: Document with no keyword matches has score of 0
      Given a document with no query keywords
      When the document is ranked
      Then the document has rank_score of 0

  Rule: Title Matches Receive Bonus Score

    Scenario: Keyword in title increases rank score
      Given document A with keyword "tax" in title
      And document B with keyword "tax" only in content
      And both have same content keyword frequency
      When documents are ranked
      Then document A ranks higher than document B
      And document A's rank_score includes title bonus

    Scenario: Multiple keywords in title accumulate bonuses
      Given a document with 2 query keywords in title
      When the document is ranked
      Then the rank_score includes 2 title bonuses
      And each title bonus is worth 2 points

  Rule: Keywords at Document Start Receive Bonus

    Scenario: Document starting with keyword gets bonus
      Given document A starts with query keyword
      And document B has keyword later in content
      And both have same total keyword frequency
      When documents are ranked
      Then document A ranks higher than document B
      And document A's rank_score includes start bonus

  Rule: Rank Score Is Stored in Document

    Scenario: Each document receives rank_score field
      Given 5 documents to rank
      When rank is called
      Then each document has a "rank_score" field
      And rank_score is a numeric value

    Scenario: Rank score reflects combined scoring factors
      Given a document with keyword_count=3, title_bonus=2, start_bonus=1
      When the document is ranked
      Then the rank_score is 6 (3 + 2 + 1)

  Rule: Documents Are Sorted in Descending Order by Rank Score

    Scenario: Highest scoring document is first
      Given documents with rank_scores [5, 10, 3, 8]
      When ranking is complete
      Then the first document has rank_score 10
      And the last document has rank_score 3

    Scenario: Equal scores maintain relative order
      Given documents with equal rank_scores
      When ranking is complete
      Then documents with equal scores appear in stable order

  Rule: Query Keywords Are Extracted and Used for Scoring

    Scenario: Query keywords parameter is used
      Given a query with keywords ["business", "license"]
      When rank is called
      Then document scoring uses keywords "business" and "license"
      And occurrences of these keywords increase rank_score

    Scenario: Query without keywords results in zero scores
      Given a query with empty keywords list
      When rank is called
      Then all documents have rank_score of 0
      And original document order may be preserved

  Rule: Content Matching Is Case-Insensitive

    Scenario: Uppercase and lowercase keywords match equally
      Given document with content "SALES TAX"
      And query keyword "sales"
      When the document is ranked
      Then the keyword match is counted
      And rank_score increases

    Scenario: Title matching is case-insensitive
      Given document with title "Property TAX"
      And query keyword "tax"
      When the document is ranked
      Then the title bonus applies
      And rank_score includes title bonus

  Rule: Ranking Logs Operations

    Scenario: Rank logs document count and query
      Given 15 documents to rank
      And a query with original_query "sales tax rates"
      When rank is called
      Then a log message indicates "Ranking 15 documents for query: sales tax rates"
