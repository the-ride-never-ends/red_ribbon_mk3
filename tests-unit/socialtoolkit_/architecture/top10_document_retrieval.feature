Feature: Top-10 Document Retrieval
  As a data researcher
  I want to retrieve the top 10 most relevant documents based on a query
  So that I can find the most relevant information efficiently

  Background:
    Given a Top10DocumentRetrieval instance is initialized
    And a vector search engine is available
    And a document storage service is available

  Rule: Execute Method Always Returns Dictionary with Required Keys

    Scenario: Execute with valid query returns dictionary with expected keys
      Given a valid query string "What is the sales tax rate?"
      And a collection of documents with embeddings is available
      When I call execute with the query
      Then I receive a dictionary response
      And the response contains key "relevant_documents"
      And the response contains key "scores"
      And the response contains key "top_doc_ids"

    Scenario: Execute with empty document collection returns empty results
      Given a valid query string "tax rates"
      And an empty document collection
      When I call execute with the query
      Then I receive a dictionary with empty "relevant_documents" list
      And the "scores" dictionary is empty
      And the "top_doc_ids" list is empty

  Rule: Execute Method Returns At Most N Documents Where N is retrieval_count

    Scenario: Execute retrieves exactly 10 documents when available
      Given retrieval_count is configured as 10
      And 50 documents are available in storage
      When I call execute with query "business licenses"
      Then I receive exactly 10 documents in "relevant_documents"
      And exactly 10 document IDs in "top_doc_ids"

    Scenario: Execute retrieves all documents when fewer than N available
      Given retrieval_count is configured as 10
      And 5 documents are available in storage
      When I call execute with query "tax information"
      Then I receive exactly 5 documents in "relevant_documents"
      And exactly 5 document IDs in "top_doc_ids"

  Rule: Documents Are Ranked by Similarity Score in Descending Order

    Scenario: Documents are returned in descending order of relevance
      Given multiple documents with varying similarity scores
      When I call execute with a query
      Then the first document has the highest similarity score
      And each subsequent document has a score less than or equal to the previous
      And the scores dictionary matches the document order

  Rule: Similarity Scores Respect Configured Threshold

    Scenario: Documents below similarity threshold are filtered out
      Given similarity_threshold is configured as 0.6
      And documents with scores [0.9, 0.7, 0.5, 0.3] are available
      When I call execute with a query
      Then only documents with scores >= 0.6 are returned
      And exactly 2 documents are in the results

    Scenario: All documents are filtered when none meet threshold
      Given similarity_threshold is configured as 0.8
      And all documents have similarity scores below 0.8
      When I call execute with a query
      Then I receive an empty "relevant_documents" list

  Rule: Execute Validates Input Types

    Scenario: Execute rejects non-string query
      Given an integer query value 12345
      When I call execute with the invalid query
      Then a TypeError is raised
      And the error message indicates "input_data_point must be a string"

    Scenario: Execute rejects None as query
      Given a None value as query
      When I call execute with the invalid query
      Then a TypeError is raised

    Scenario: Execute accepts empty string query
      Given an empty string "" as query
      When I call execute with the query
      Then the execution completes without error
      And results may be empty or contain default documents

  Rule: Execute Handles Document and Vector Parameters

    Scenario: Execute uses provided documents and vectors
      Given explicit documents list with 3 documents
      And explicit document_vectors list with 3 vectors
      When I call execute with documents and vectors parameters
      Then the storage service is not queried for documents
      And the provided documents are used for search

    Scenario: Execute fetches from storage when documents not provided
      Given documents parameter is None
      And document_vectors parameter is None
      When I call execute with only the query
      Then documents and vectors are fetched from storage
      And the fetched documents are used for search

  Rule: Ranking Method Configuration Affects Score Calculation

    Scenario: Cosine similarity ranking is applied
      Given ranking_method is configured as "cosine_similarity"
      When I call execute with a query
      Then similarity scores are calculated using cosine similarity
      And scores are between -1.0 and 1.0

    Scenario: Dot product ranking is applied
      Given ranking_method is configured as "dot_product"
      When I call execute with a query
      Then similarity scores are calculated using dot product
      And scores reflect the dot product values

    Scenario: Euclidean distance ranking is applied
      Given ranking_method is configured as "euclidean"
      When I call execute with a query
      Then similarity scores are inverse of euclidean distance
      And scores are calculated as 1.0 / (1.0 + distance)

  Rule: Result Documents Contain Required Metadata

    Scenario: Each returned document contains standard fields
      Given documents in storage have id, content, url, and title
      When I call execute with a query
      Then each document in "relevant_documents" has an "id" field
      And each document has a "content" field
      And each document has a "url" field
      And each document has a "title" field

  Rule: Execute Logs Retrieval Operations

    Scenario: Execute logs start and completion
      Given logging is enabled
      When I call execute with query "business regulations"
      Then a log message indicates "Starting top-10 document retrieval"
      And a log message indicates "Retrieved N potentially relevant documents"
