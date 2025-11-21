"""
Embeddings utilities for working with the American Law dataset.
Provides tools for loading, processing, and using embeddings from parquet files.
"""
import logging
import os
from pathlib import Path
from typing import Any, Optional
import sqlite3


import pandas as pd
import numpy as np


# Set up logging
from custom_nodes.red_ribbon.utils_ import Configs


def _validate_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    value = value.strip()
    if not value:
        raise ValueError(f"{name} cannot be empty or whitespace")
    return value


class EmbeddingsManager:
    """
    Manager for embeddings data in the American Law dataset.
    Handles loading, processing, and searching embeddings in parquet files.
    """
    
    def __init__(self, *, 
                 configs: Configs,
                 resources: dict[str, Any]
                 ) -> None:
        """
        Initialize the embeddings manager.
        
        Args:
            configs: Configuration object.
            resources: Dictionary of resources.
        """
        self.configs = configs
        self.resources = resources

        self.logger: logging.Logger = resources["logger"]
        self.data_path: Path = self.configs.paths.AMERICAN_LAW_DATA_DIR
        self.db_path: Path = self.configs.paths.AMERICAN_LAW_DB_PATH

        # Cache for recently used embeddings
        self._embedding_cache: dict[str, pd.DataFrame] = {}

    def list_embedding_files(self) -> list[str]:
        """
        List all embedding files in the data path.
        
        Returns:
            List of embedding file paths
        """
        if not self.data_path or not self.data_path.exists():
            return []

        embedding_files = list(Path(self.data_path).glob("*_embeddings.parquet"))
        return [str(file) for file in embedding_files]

    def load_embeddings(self, file_id: str) -> pd.DataFrame:
        """
        Load embeddings from a parquet file.
        
        Args:
            file_id: ID of the file to load
            
        Returns:
            DataFrame with embeddings
        """
        _validate_string(file_id, "file_id")

        # Check if embeddings are already in cache
        if file_id in self._embedding_cache:
            return self._embedding_cache[file_id]

        # Load embeddings from file
        embedding_file: Path = self.data_path / f"{file_id}_embeddings.parquet"

        if not embedding_file.exists():
            self.logger.error(f"Embedding file not found: {embedding_file}. Returning empty DataFrame.")
            return pd.DataFrame()

        try:
            embedding_df = pd.read_parquet(embedding_file)

            # Cache embeddings for future use (limit cache size)
            if len(self._embedding_cache) > 10:
                # Remove oldest entry if cache is too large
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]

            self._embedding_cache[file_id] = embedding_df
            return embedding_df

        except Exception as e:
            self.logger.error(f"Error loading embeddings from {embedding_file}: {e}. Returning empty DataFrame.")
            return pd.DataFrame()
    
    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        
        dot_product = np.dot(vec1_array, vec2_array)
        norm1 = np.linalg.norm(vec1_array)
        norm2 = np.linalg.norm(vec2_array)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))
    
    def search_embeddings_in_file(
        self, 
        query_embedding: list[float], 
        file_id: str, 
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Search for similar embeddings in a specific file.
        
        Args:
            query_embedding: Query embedding vector
            file_id: ID of the file to search in
            top_k: Number of top results to return
            
        Returns:
            List of documents with similarity scores
        """
        embedding_df = self.load_embeddings(file_id)
        
        if embedding_df.empty:
            return []
            
        # Calculate similarity scores
        results = []
        
        for _, row in embedding_df.iterrows():
            similarity = self.cosine_similarity(query_embedding, row['embedding'])
            
            results.append({
                'cid': row['cid'],
                'similarity_score': similarity
            })
            
        # Sort by similarity and get top_k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]
    
    def get_document_metadata(self, cid: str, file_id: str) -> dict[str, Any]:
        """
        Get metadata for a document from citation and HTML files.
        
        Args:
            cid: Content ID
            file_id: File ID
            
        Returns:
            Document metadata
        """
        _validate_string(cid, "cid")
        _validate_string(file_id, "file_id")

        citation_file: Path = self.data_path / f"{file_id}_citation.parquet"
        html_file: Path = self.data_path / f"{file_id}_html.parquet"
        
        if not citation_file.exists() or html_file.exists():
            return {}
            
        try:
            citation_df = pd.read_parquet(citation_file)
            html_df = pd.read_parquet(html_file)
            
            citation_row = citation_df[citation_df['cid'] == cid].iloc[0] if not citation_df[citation_df['cid'] == cid].empty else None
            html_row = html_df[html_df['cid'] == cid].iloc[0] if not html_df[html_df['cid'] == cid].empty else None
            
            if citation_row is None or html_row is None:
                return {}
                
            return {
                'cid': cid,
                'title': citation_row['title'],
                'chapter': citation_row['chapter'],
                'place_name': citation_row['place_name'],
                'state_code': citation_row['state_code'],
                'state_name': citation_row['state_name'],
                'date': citation_row['date'],
                'bluebook_citation': citation_row['bluebook_citation'],
                'html_content': html_row['html']
            }
            
        except Exception as e:
            self.logger.error(f"Error getting document metadata for {cid} in {file_id}: {e}")
            return {}
    
    def search_across_files(
        self, 
        query_embedding: list[float], 
        max_files: int = 10, 
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Search for similar embeddings across multiple files.
        
        Args:
            query_embedding: Query embedding vector
            max_files: Maximum number of files to search
            top_k: Number of top results per file
            
        Returns:
            List of documents with similarity scores and metadata
        """
        if not self.data_path:
            return []
            
        embedding_files = self.list_embedding_files()
        
        # Limit number of files to search
        embedding_files = embedding_files[:max_files]
        
        all_results = []
        
        for file_path in embedding_files:
            file_id = os.path.basename(file_path).split('_')[0]
            
            # Search in this file
            file_results = self.search_embeddings_in_file(query_embedding, file_id, top_k)
            
            # Add metadata for each result
            for result in file_results:
                metadata = self.get_document_metadata(result['cid'], file_id)
                if metadata:
                    result.update(metadata)
                    all_results.append(result)
        
        # Sort all results by similarity score
        all_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top_k results across all files
        return all_results[:top_k]
    
    def get_db_connection(self):
        """
        Get a connection to the SQLite database.
        
        Returns:
            SQLite connection
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception as e:
            self.logger.error(f"Error connecting to database at {self.db_path}: {e}")
            raise

    def search_db_by_cid(self, cid: str) -> dict[str, Any]:
        """
        Search for a document in the database by content ID.
        
        Args:
            cid: Content ID
            
        Returns:
            Document data
        """
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT c.bluebook_cid, c.cid, c.title, c.chapter, c.place_name, c.state_name, c.date, 
                       c.bluebook_citation, h.html
                FROM citations c
                INNER JOIN c.html h ON c.cid = h.cid
                WHERE cid = ?
            ''', (cid,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return {}
                
            return {
                'id': row['id'],
                'cid': row['cid'],
                'title': row['title'],
                'chapter': row['chapter'],
                'place_name': row['place_name'],
                'state_name': row['state_name'],
                'date': row['date'],
                'bluebook_citation': row['bluebook_citation'],
                'html': row['html']
            }

        except Exception as e:
            self.logger.error(f"Error searching database for CID {cid}: {e}")
            return {}
