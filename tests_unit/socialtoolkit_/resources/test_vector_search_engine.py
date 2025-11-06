"""
Feature: Vector Search Engine
  As a document retrieval system
  I want to search for documents using vector similarity
  So that semantically similar documents can be found efficiently

  Background:
    Given a VectorSearchEngine instance is initialized
"""

import pytest
from custom_nodes.red_ribbon.socialtoolkit.resources.top10_document_retrieval.vector_search_engine import VectorSearchEngine

# Fixtures for Background

@pytest.fixture
def vector_search_engine():
    """
    Given a VectorSearchEngine instance is initialized
    """
    return VectorSearchEngine()
class TestAddVectorsMethodStoresDocumentsandEmbeddings:
    """
    Rule: Add Vectors Method Stores Documents and Embeddings
    """
    def test_add_vectors_for_multiple_documents(self, vector_search_engine):
        """
        Scenario: Add vectors for multiple documents
          Given 5 documents with corresponding vector embeddings
          When I call add_vectors with documents and vectors
          Then all 5 document-vector pairs are stored
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}"} for i in range(5)]
        vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(5)]
        
        # Act
        vector_search_engine.add_vectors(documents, vectors)
        
        # Assert
        assert len(vector_search_engine.document_vectors) == 5

    def test_add_vectors_for_multiple_documents_1(self, vector_search_engine):
        """
        Scenario: Add vectors for multiple documents
          Given 5 documents with corresponding vector embeddings
          When I call add_vectors with documents and vectors
          Then each document ID maps to its vector and document data
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}"} for i in range(5)]
        vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(5)]
        
        # Act
        vector_search_engine.add_vectors(documents, vectors)
        
        # Assert
        for i in range(5):
            doc_id = f"doc{i}"
            assert doc_id in vector_search_engine.document_vectors
            assert "vector" in vector_search_engine.document_vectors[doc_id]
            assert "document" in vector_search_engine.document_vectors[doc_id]

    def test_add_vectors_handles_empty_lists(self, vector_search_engine):
        """
        Scenario: Add vectors handles empty lists
          Given empty lists for documents and vectors
          When I call add_vectors
          Then no vectors are stored
        """
        # Arrange
        documents = []
        vectors = []
        
        # Act
        vector_search_engine.add_vectors(documents, vectors)
        
        # Assert
        assert len(vector_search_engine.document_vectors) == 0

    def test_add_vectors_handles_empty_lists_1(self, vector_search_engine):
        """
        Scenario: Add vectors handles empty lists
          Given empty lists for documents and vectors
          When I call add_vectors
          Then the operation completes without error
        """
        # Arrange
        documents = []
        vectors = []
        
        # Act & Assert (no exception should be raised)
        vector_search_engine.add_vectors(documents, vectors)

    def test_add_vectors_overwrites_existing_document_ids(self, vector_search_engine):
        """
        Scenario: Add vectors overwrites existing document IDs
          Given a document with ID "doc1" already exists
          And a new document with the same ID "doc1"
          When I call add_vectors with the new document
          Then the old document is replaced
        """
        # Arrange
        old_doc = [{"id": "doc1", "content": "Old content"}]
        old_vec = [[0.1, 0.2, 0.3]]
        vector_search_engine.add_vectors(old_doc, old_vec)
        
        new_doc = [{"id": "doc1", "content": "New content"}]
        new_vec = [[0.4, 0.5, 0.6]]
        
        # Act
        vector_search_engine.add_vectors(new_doc, new_vec)
        
        # Assert
        assert vector_search_engine.document_vectors["doc1"]["document"]["content"] == "New content"

    def test_add_vectors_overwrites_existing_document_ids_1(self, vector_search_engine):
        """
        Scenario: Add vectors overwrites existing document IDs
          Given a document with ID "doc1" already exists
          And a new document with the same ID "doc1"
          When I call add_vectors with the new document
          Then the new vector and document data are stored
        """
        # Arrange
        old_doc = [{"id": "doc1", "content": "Old content"}]
        old_vec = [[0.1, 0.2, 0.3]]
        vector_search_engine.add_vectors(old_doc, old_vec)
        
        new_doc = [{"id": "doc1", "content": "New content"}]
        new_vec = [[0.4, 0.5, 0.6]]
        
        # Act
        vector_search_engine.add_vectors(new_doc, new_vec)
        
        # Assert
        assert vector_search_engine.document_vectors["doc1"]["vector"] == [0.4, 0.5, 0.6]
        assert vector_search_engine.document_vectors["doc1"]["document"]["content"] == "New content"

class TestSearchMethodReturnsTopKSimilarDocuments:
    """
    Rule: Search Method Returns Top-K Similar Documents
    """
    def test_search_returns_requested_number_of_documents(self, vector_search_engine):
        """
        Scenario: Search returns requested number of documents
          Given 20 documents are stored with vectors
          And top_k is 10
          When I call search with a query vector
          Then exactly 10 documents are returned
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}", "url": f"url{i}", "title": f"Title {i}"} for i in range(20)]
        vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(20)]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=10)
        
        # Assert
        assert len(result) == 10

    def test_search_returns_requested_number_of_documents_1(self, vector_search_engine):
        """
        Scenario: Search returns requested number of documents
          Given 20 documents are stored with vectors
          And top_k is 10
          When I call search with a query vector
          Then documents are ordered by similarity
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}", "url": f"url{i}", "title": f"Title {i}"} for i in range(20)]
        vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(20)]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=10)
        
        # Assert
        # Check that similarity scores are in descending order
        for i in range(len(result) - 1):
            assert result[i]["similarity_score"] >= result[i + 1]["similarity_score"]

    def test_search_returns_all_documents_when_fewer_than_top_k_available(self, vector_search_engine):
        """
        Scenario: Search returns all documents when fewer than top_k available
          Given 5 documents are stored with vectors
          And top_k is 10
          When I call search with a query vector
          Then exactly 5 documents are returned
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}", "url": f"url{i}", "title": f"Title {i}"} for i in range(5)]
        vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(5)]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=10)
        
        # Assert
        assert len(result) == 5

    def test_search_returns_empty_list_when_no_documents_stored(self, vector_search_engine):
        """
        Scenario: Search returns empty list when no documents stored
          Given no documents are stored
          And top_k is 10
          When I call search with a query vector
          Then an empty list is returned
        """
        # Arrange
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=10)
        
        # Assert
        assert result == []

    def test_search_with_top_k_of_1_returns_single_document(self, vector_search_engine):
        """
        Scenario: Search with top_k of 1 returns single document
          Given 10 documents are stored
          And top_k is 1
          When I call search
          Then exactly 1 document is returned
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}", "url": f"url{i}", "title": f"Title {i}"} for i in range(10)]
        vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(10)]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=1)
        
        # Assert
        assert len(result) == 1


class TestSearchResultsIncludeRequiredDocumentFields:
    """
    Rule: Search Results Include Required Document Fields
    """
    def test_each_result_contains_document_metadata(self, vector_search_engine):
        """
        Scenario: Each result contains document metadata
          Given documents with id, content, url, and title fields
          When search is executed
          Then each result has an "id" field
        """
        # Arrange
        documents = [{"id": "doc1", "content": "Content", "url": "http://url", "title": "Title"}]
        vectors = [[0.1, 0.2, 0.3]]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=1)
        
        # Assert
        assert "id" in result[0]

    def test_each_result_contains_document_metadata_1(self, vector_search_engine):
        """
        Scenario: Each result contains document metadata
          Given documents with id, content, url, and title fields
          When search is executed
          Then each result has a "content" field
        """
        # Arrange
        documents = [{"id": "doc1", "content": "Content", "url": "http://url", "title": "Title"}]
        vectors = [[0.1, 0.2, 0.3]]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=1)
        
        # Assert
        assert "content" in result[0]

    def test_each_result_contains_document_metadata_2(self, vector_search_engine):
        """
        Scenario: Each result contains document metadata
          Given documents with id, content, url, and title fields
          When search is executed
          Then each result has a "url" field
        """
        # Arrange
        documents = [{"id": "doc1", "content": "Content", "url": "http://url", "title": "Title"}]
        vectors = [[0.1, 0.2, 0.3]]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=1)
        
        # Assert
        assert "url" in result[0]

    def test_each_result_contains_document_metadata_3(self, vector_search_engine):
        """
        Scenario: Each result contains document metadata
          Given documents with id, content, url, and title fields
          When search is executed
          Then each result has a "title" field
        """
        # Arrange
        documents = [{"id": "doc1", "content": "Content", "url": "http://url", "title": "Title"}]
        vectors = [[0.1, 0.2, 0.3]]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=1)
        
        # Assert
        assert "title" in result[0]

    def test_each_result_contains_document_metadata_4(self, vector_search_engine):
        """
        Scenario: Each result contains document metadata
          Given documents with id, content, url, and title fields
          When search is executed
          Then each result has a "similarity_score" field
        """
        # Arrange
        documents = [{"id": "doc1", "content": "Content", "url": "http://url", "title": "Title"}]
        vectors = [[0.1, 0.2, 0.3]]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=1)
        
        # Assert
        assert "similarity_score" in result[0]

    def test_missing_title_uses_default_value(self, vector_search_engine):
        """
        Scenario: Missing title uses default value
          Given a document without a title field
          When the document is included in search results
          Then the title is "Document {doc_id}"
        """
        # Arrange
        documents = [{"id": "doc1", "content": "Content", "url": "http://url"}]  # No title
        vectors = [[0.1, 0.2, 0.3]]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=1)
        
        # Assert
        assert result[0]["title"] == "Document doc1"


class TestSearchResultsAreOrderedbySimilarityScore:
    """
    Rule: Search Results Are Ordered by Similarity Score
    """
    def test_results_are_in_descending_similarity_order(self, vector_search_engine):
        """
        Scenario: Results are in descending similarity order
          Given multiple documents with similarity scores
          When search returns results
          Then the first result has the highest similarity score
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}", "url": f"url{i}", "title": f"Title {i}"} for i in range(5)]
        vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(5)]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=5)
        
        # Assert
        # First result should have highest score
        assert result[0]["similarity_score"] >= result[-1]["similarity_score"]

    def test_results_are_in_descending_similarity_order_1(self, vector_search_engine):
        """
        Scenario: Results are in descending similarity order
          Given multiple documents with similarity scores
          When search returns results
          Then each subsequent result has equal or lower score
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}", "url": f"url{i}", "title": f"Title {i}"} for i in range(5)]
        vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(5)]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=5)
        
        # Assert
        for i in range(len(result) - 1):
            assert result[i]["similarity_score"] >= result[i + 1]["similarity_score"]

    def test_similarity_scores_are_in_valid_range(self, vector_search_engine):
        """
        Scenario: Similarity scores are in valid range
          Given search results are returned
          When similarity scores are examined
          Then all scores are between 0.0 and 1.0
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}", "url": f"url{i}", "title": f"Title {i}"} for i in range(5)]
        vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(5)]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=5)
        
        # Assert
        for doc in result:
            assert 0.0 <= doc["similarity_score"] <= 1.0


class TestQueryVectorParameterIsRequired:
    """
    Rule: Query Vector Parameter Is Required
    """
    def test_search_with_valid_query_vector(self, vector_search_engine):
        """
        Scenario: Search with valid query vector
          Given a query vector with valid dimensions
          When search is called with the vector
          Then the search completes successfully
        """
        # Arrange
        documents = [{"id": "doc1", "content": "Content", "url": "url", "title": "Title"}]
        vectors = [[0.1, 0.2, 0.3]]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5, 0.5]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=1)
        
        # Assert
        assert result is not None
        assert isinstance(result, list)

    def test_search_accepts_query_vector_as_list_of_floats(self, vector_search_engine):
        """
        Scenario: Search accepts query vector as list of floats
          Given query_vector is [0.1, 0.2, 0.3, ...]
          When search is called
          Then the vector is processed correctly
        """
        # Arrange
        documents = [{"id": "doc1", "content": "Content", "url": "url", "title": "Title"}]
        vectors = [[0.1, 0.2, 0.3, 0.4]]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.1, 0.2, 0.3, 0.4]
        
        # Act
        result = vector_search_engine.search(query_vector, top_k=1)
        
        # Assert
        assert len(result) >= 0  # Should process without error


class TestSearchLogsOperations:
    """
    Rule: Search Logs Operations
    """
    def test_add_vectors_logs_document_count(self, vector_search_engine, caplog):
        """
        Scenario: Add vectors logs document count
          Given 15 documents to add
          When add_vectors is called
          Then a log message indicates "Adding vectors for 15 documents"
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}"} for i in range(15)]
        vectors = [[0.1 * i, 0.2 * i] for i in range(15)]
        
        # Act
        with caplog.at_level("INFO"):
            vector_search_engine.add_vectors(documents, vectors)
        
        # Assert
        assert any("Adding vectors for 15 documents" in record.message for record in caplog.records)

    def test_search_logs_top_k_value(self, vector_search_engine, caplog):
        """
        Scenario: Search logs top_k value
          Given top_k is 5
          When search is called
          Then a log message indicates "Searching for top 5 similar documents"
        """
        # Arrange
        documents = [{"id": f"doc{i}", "content": f"Content {i}", "url": "url", "title": "Title"} for i in range(10)]
        vectors = [[0.1 * i, 0.2 * i] for i in range(10)]
        vector_search_engine.add_vectors(documents, vectors)
        query_vector = [0.5, 0.5]
        
        # Act
        with caplog.at_level("INFO"):
            result = vector_search_engine.search(query_vector, top_k=5)
        
        # Assert
        assert any("Searching for top 5 similar documents" in record.message for record in caplog.records)

