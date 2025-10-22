Feature: Vector Search Engine
  As a document retrieval system
  I want to search for documents using vector similarity
  So that semantically similar documents can be found efficiently

  Background:
    Given a VectorSearchEngine instance is initialized

  Rule: Add Vectors Method Stores Documents and Embeddings

    Scenario: Add vectors for multiple documents
      Given 5 documents with corresponding vector embeddings
      When I call add_vectors with documents and vectors
      Then all 5 document-vector pairs are stored
      And each document ID maps to its vector and document data

    Scenario: Add vectors handles empty lists
      Given empty lists for documents and vectors
      When I call add_vectors
      Then no vectors are stored
      And the operation completes without error

    Scenario: Add vectors overwrites existing document IDs
      Given a document with ID "doc1" already exists
      And a new document with the same ID "doc1"
      When I call add_vectors with the new document
      Then the old document is replaced
      And the new vector and document data are stored

  Rule: Search Method Returns Top-K Similar Documents

    Scenario: Search returns requested number of documents
      Given 20 documents are stored with vectors
      And top_k is 10
      When I call search with a query vector
      Then exactly 10 documents are returned
      And documents are ordered by similarity

    Scenario: Search returns all documents when fewer than top_k available
      Given 5 documents are stored with vectors
      And top_k is 10
      When I call search with a query vector
      Then exactly 5 documents are returned

    Scenario: Search returns empty list when no documents stored
      Given no documents are stored
      And top_k is 10
      When I call search with a query vector
      Then an empty list is returned

    Scenario: Search with top_k of 1 returns single document
      Given 10 documents are stored
      And top_k is 1
      When I call search
      Then exactly 1 document is returned

  Rule: Search Results Include Required Document Fields

    Scenario: Each result contains document metadata
      Given documents with id, content, url, and title fields
      When search is executed
      Then each result has an "id" field
      And each result has a "content" field
      And each result has a "url" field
      And each result has a "title" field
      And each result has a "similarity_score" field

    Scenario: Missing title uses default value
      Given a document without a title field
      When the document is included in search results
      Then the title is "Document {doc_id}"

  Rule: Search Results Are Ordered by Similarity Score

    Scenario: Results are in descending similarity order
      Given multiple documents with similarity scores
      When search returns results
      Then the first result has the highest similarity score
      And each subsequent result has equal or lower score

    Scenario: Similarity scores are in valid range
      Given search results are returned
      When similarity scores are examined
      Then all scores are between 0.0 and 1.0

  Rule: Query Vector Parameter Is Required

    Scenario: Search with valid query vector
      Given a query vector with valid dimensions
      When search is called with the vector
      Then the search completes successfully

    Scenario: Search accepts query vector as list of floats
      Given query_vector is [0.1, 0.2, 0.3, ...]
      When search is called
      Then the vector is processed correctly

  Rule: Search Logs Operations

    Scenario: Add vectors logs document count
      Given 15 documents to add
      When add_vectors is called
      Then a log message indicates "Adding vectors for 15 documents"

    Scenario: Search logs top_k value
      Given top_k is 5
      When search is called
      Then a log message indicates "Searching for top 5 similar documents"
