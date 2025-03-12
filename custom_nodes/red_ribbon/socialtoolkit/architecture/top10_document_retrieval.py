from pydantic import BaseModel
from typing import Any, Callable, Optional
import logging
import numpy as np


logger = logging.getLogger(__name__)


from ..resources.top10_document_retrieval._cosine_similarity import cosine_similarity
from ..resources.top10_document_retrieval._dot_product import dot_product
from ..resources.top10_document_retrieval._euclidean_distance import euclidean_distance


class Top10DocumentRetrievalConfigs(BaseModel):
    """Configuration for Top-10 Document Retrieval workflow"""
    retrieval_count: int = 10  # Number of documents to retrieve
    similarity_threshold: float = 0.6  # Minimum similarity score
    ranking_method: str = "cosine_similarity"  # Options: cosine_similarity, dot_product, euclidean
    use_filter: bool = False  # Whether to filter results
    filter_criteria: dict[str, Any] = {}
    use_reranking: bool = False  # Whether to use reranking


class Top10DocumentRetrieval:
    """
    Top-10 Document Retrieval system based on mermaid chart in README.md
    Performs vector search to find the most relevant documents
    """
    
    def __init__(self, resources: dict[str, Any], configs: Top10DocumentRetrievalConfigs):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including search services
            configs: Configuration for Top-10 Document Retrieval
        """
        self.resources = resources
        self.configs = configs
        assert self.configs.__qualname__ == "Top10DocumentRetrievalConfigs", "configs must be of type Top10DocumentRetrievalConfigs"

        # Attributes
        self.use_filter = self.configs.use_filter
        self.use_reranking = self.configs.use_reranking
        self.ranking_method = self.configs.ranking_method
        self.similarity_threshold = self.configs.similarity_threshold
        self.retrieval_count = self.configs.retrieval_count

        # External dependencies
        self.llm = self.resources["llm"]
        self.logger: logging.Logger = self.resources["logger"]
        self.db = self.resources["database"]
        self.vector_db = self.resources["vector_db"]

        # Extract needed services from resources
        self.encoder_service = resources["encoder_service"]
        self.similarity_search_service = resources["similarity_search_service"]
        self.document_storage = resources["document_storage_service"]
        
        # Methods
        self._encode_query = self.resources["_encode_query"]
        self._similarity_search = self.resources["_similarity_search"] or None
        self._retrieve_top_documents = self.resources["_retrieve_top_documents"]
        self._cosine_similarity = self.resources["_cosine_similarity"] or cosine_similarity
        self._dot_product = self.resources["_dot_product"] or dot_product
        self._euclidean_distance = self.resources["_euclidean_distance"] or euclidean_distance


        self.logger.info("Top10DocumentRetrieval initialized with services")

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()

    def execute(self, 
                input_data_point: str, 
                documents: list[Any] = None, 
                document_vectors: list[Any] = None
                ) -> dict[str, Any]:
        """
        Execute the document retrieval flow.
        
        Args:
            input_data_point: The query or information request
            documents: Optional list of documents to search
            document_vectors: Optional list of document vectors to search
            
        Returns:
            Dictionary of documents containing potentially relevant documents, along with potentially relevant metadata.
        """
        self.logger.info(f"Starting top-10 document retrieval for: {input_data_point}")
        
        # Step 1: Encode the query
        encoded_query = self._encode_query(input_data_point)
        
        # Step 2: Get vector embeddings and document IDs from storage if not provided
        if documents is None or document_vectors is None:
            documents, document_vectors = self._get_documents_and_vectors()
        
        # Step 3: Perform similarity search
        similarity_scores, doc_ids = self.similarity_search(
            encoded_query, 
            document_vectors, 
            [doc("id") for doc in documents]
        )
        
        # Step 4: Rank and sort results
        ranked_results = self.rank_and_sort_results(similarity_scores, doc_ids)
        
        # Step 5: Filter to top-N results
        top_doc_ids = self.filter_to_top_n(ranked_results, self.retre)
        
        # Step 6: Retrieve potentially relevant documents
        potentially_relevant_docs = self.retrieve_relevant_documents(documents, top_doc_ids)
        
        self.logger.info(f"Retrieved {len(potentially_relevant_docs)} potentially relevant documents")
        return {
            "relevant_documents": potentially_relevant_docs,
            "scores": {doc_id: score for doc_id, score in ranked_results},
            "top_doc_ids": top_doc_ids
        }

    def retrieve_top_documents(self, input_data_point: str, documents: list[Any], document_vectors: list[Any]) -> list[Any]:
        """
        Public method to retrieve top documents for an input query
        
        Args:
            input_data_point: The query to search for
            documents: Documents to search
            document_vectors: Vectors for the documents

        Returns:
            List of potentially relevant documents
        """
        with self.db.enter() as db:
            result = self._retrieve_top_documents(input_data_point, documents, document_vectors, db)
        
        self.control_flow(input_data_point, documents, document_vectors)
        return result["relevant_documents"]
    
    def _encode_query(self, input_data_point: str) -> Any:
        """
        Encode the input query into a vector representation
        
        Args:
            input_data_point: The query to encode
            
        Returns:
            Vector representation of the query
        """
        self.logger.debug(f"Encoding query: {input_data_point}")
        return self._encode_query(input_data_point)
    
    def _get_documents_and_vectors(self) -> tuple[list[Any], list[Any]]:
        """
        Get all documents and their vectors from storage
        
        Returns:
            tuple of (documents, document_vectors)
        """
        self.logger.debug("Getting documents and vectors from storage")
        return self.document_storage.get_documents_and_vectors()

    def _run_ranking_method(self, encoded_query: list[float], vector: dict[str, list[float]]) -> Callable:
        match self.ranking_method:
                # Calculate similarity score based on the configured method
            case "cosine_similarity":
                return self.cosine_similarity(encoded_query, vector["embedding"])
            case "dot_product":
                return self.dot_product(encoded_query, vector["embedding"])
            case "euclidean":
                # Convert distance to similarity score (higher is more similar)
                return 1.0 / (1.0 + self.euclidean_distance(encoded_query, vector["embedding"]))
            case _:
                return 0.0

    def similarity_search(self, 
                          encoded_query: Any, 
                          document_vectors: list[Any], 
                          doc_ids: list[str]
                          ) -> tuple[list[float], list[str]]:
        """
        Perform similarity search between the query and document vectors
        
        Args:
            encoded_query: Vector representation of the query
            document_vectors: List of document vector embeddings
            doc_ids: List of document IDs corresponding to the vectors
            
        Returns:
            tuple of (similarity_scores, document_ids)
        """
        self.logger.debug("Performing similarity search")

        # Elements necessary for efficient vector search:
        # 1. Vector database/index (e.g., FAISS, Annoy, Milvus, Pinecone)
        # 2. Dimensionality reduction techniques (PCA, t-SNE if needed)
        # 3. Approximate Nearest Neighbor (ANN) algorithms
        # 4. Indexing structures (e.g., hierarchical navigable small worlds)
        # 5. Quantization to reduce memory footprint
        # 6. Partitioning/clustering for large datasets
        # 7. Caching mechanisms for frequent queries
        # 8. Batch processing for multiple queries
        
        # Currently falling back to brute force comparison:

        similarity_scores = [
            self._run_ranking_method(encoded_query, vector) for vector in document_vectors
        ]

        # If the similarity search service is available, use it instead
        if self._similarity_search:
            return self._similarity_search(
                encoded_query, document_vectors, doc_ids
            )
            
        return similarity_scores, doc_ids
    
    def rank_and_sort_results(self, 
                              similarity_scores: list[float], 
                              doc_ids: list[str]
                              ) -> list[tuple[str, float]]:
        """
        Rank and sort results by similarity score
        
        Args:
            similarity_scores: List of similarity scores
            doc_ids: List of document IDs
            
        Returns:
            List of (document_id, score) tuples sorted by score
        """
        self.logger.debug("Ranking and sorting results")
        
        # Create a list of (document_id, score) tuples
        result_tuples = list(zip(doc_ids, similarity_scores))
        
        # Sort by score in descending order
        return sorted(result_tuples, key=lambda x: x[1], reverse=True)

    def filter_to_top_n(self, ranked_results: list[tuple[str, float]], n_results: int = 10) -> list[str]:
        """
        Filter to top N results
        
        Args:
            ranked_results: List of (document_id, score) tuples
            
        Returns:
            List of top N document IDs
        """
        self.logger.debug(f"Filtering to top {n_results} results...")
        
        # Apply threshold filter if configured
        if self.use_filter:
            filtered_results = [
                doc_id for doc_id, score in ranked_results 
                if score >= self.similarity_threshold
            ]
        else:
            filtered_results = [doc_id for doc_id, _ in ranked_results]
        
        # Return top N results
        return filtered_results[:n_results]
    
    def retrieve_relevant_documents(self, documents: list[Any], top_doc_ids: list[str]) -> list[Any]:
        """
        Retrieve potentially relevant documents
        
        Args:
            documents: List of all documents
            top_doc_ids: List of top document IDs
            
        Returns:
            List of potentially relevant documents
        """
        self.logger.debug("Retrieving potentially relevant documents")
        
        # Create a map of document ID to document for faster lookup
        doc_map = {doc("id"): doc for doc in documents}
        
        # Retrieve documents by ID
        return [
            doc_map[doc_id] for doc_id in top_doc_ids 
            if doc_id in doc_map
        ]

    def _vector_functions(self, vec1: list[float], vec2: list[float], func: Callable) -> list[float]:
        if not vec1 or not vec2:
            return 0.0
        try:
            return func(vec1, vec2)
        except Exception as e:
            self.logger.error(f"Error calculating {func.__repr__()}: {e}")
            return 0.0

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        return self._vector_functions(vec1, vec2, self._cosine_similarity)

    def dot_product(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate dot product between two vectors"""
        return self._vector_functions(vec1, vec2, self._dot_product)

    def euclidean_distance(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate Euclidean distance between two vectors"""
        return self._vector_functions(vec1, vec2, self._euclidean_distance)
