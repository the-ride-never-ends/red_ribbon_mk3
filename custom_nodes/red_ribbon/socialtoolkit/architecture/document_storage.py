
from datetime import datetime
from enum import Enum, StrEnum
import logging
from typing import Any, Callable, Optional
from uuid import uuid4


from pydantic import BaseModel


from custom_nodes.red_ribbon.utils_.database import DatabaseAPI, make_duckdb_database
from custom_nodes.red_ribbon.utils_.common import get_cid
from custom_nodes.red_ribbon.utils_.configs import configs as global_configs, Configs
from custom_nodes.red_ribbon.utils_.logger import logger as global_logger
from .dataclasses import Document, Vector


# from configs import Configs


class DocumentStatus(StrEnum):
    NEW = "new"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"

class VersionStatus(StrEnum):
    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"

class SourceType(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"

class StorageType(StrEnum):
    SQL = "sql"
    PARQUET = "parquet"
    CACHE = "cache"

class DocumentStorageConfigs(BaseModel):
    """Configuration for Document Storage"""
    database_connection_string: str = "sqlite:///documents.db" # TODO: Change to actual default
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    batch_size: int = 100
    vector_dim: int = 1536  # Common dimension for embeddings like OpenAI's
    storage_type: StorageType = StorageType.SQL




class DocumentStorage:
    """
    Document Storage system.

    Manages the storage and retrieval of documents, versions, metadata, and vectors
    """
    
    def __init__(self, *, resources: dict[str, Any], configs: Configs) -> None:
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including storage services
            configs: Configuration for Document Storage
        """
        self.resources = resources
        self.configs = configs

        # Extract needed services from resources
        self.db: DatabaseAPI = resources["db"]
        self.get_cid: Callable = resources["get_cid"]
        self.logger: logging.Logger = resources["logger"]

        self.logger.info("DocumentStorage initialized with services")

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()

    def execute(self, action: str, **kwargs) -> dict[str, Any]:
        """
        Execute document storage operations based on the action
        
        Args:
            action: Operation to perform (store, retrieve, update, delete)
            **kwargs: Operation-specific parameters
            
        Returns:
            Dictionary containing operation results
        """
        if not isinstance(action, str):
            raise TypeError(f"Action must be a string, got {type(action).__name__}")

        self.logger.info(f"Starting document storage operation: {action}")
        if self.db is None: # This route should be called if the node is from ComfyUI
            self.db = kwargs.get("db") # type: ignore
        if self.db is None:
            raise ValueError("self.db was not set at instantiation nor provided in kwargs")

        try:
            match action:
                case "store":
                    return self._store_documents(
                        documents=kwargs["documents"], 
                        metadata=kwargs["metadata"], 
                        vectors=kwargs["vectors"]
                    )
                case "retrieve":
                    return self.retrieve_documents(
                        query=kwargs["query"],
                    )
                case "update":
                    return self.update_documents(
                        kwargs["documents"]
                    )
                case "delete":
                    return self.delete_documents(
                        kwargs["cids"]
                    )
                case "_get_vectors":
                    return self._get_vectors(
                        kwargs["cids"]
                    )
                case _:
                    msg = f"Unknown action: {action}"
                    self.logger.error(msg)
                    raise ValueError(msg)
        except Exception as e:
            self.logger.error(f"Unexpected error executing action '{action}': {e}")
            raise e

    def store(self, 
              documents: list[dict[str, Any]], 
              metadata: list[dict[str, Any]], 
              vectors: list[dict[str, list[float]]]
              ) -> dict[str, Any]:
        """
        Store documents, metadata, and vectors

        Args:
            documents: Documents to store
            metadata: Metadata for the documents
            vectors: Vectors for the documents
            
        Returns:
            Dictionary with storage status
        """
        return self.execute("store", documents=documents, metadata=metadata, vectors=vectors)


    def _store_documents(self, 
                        *,
                        documents: list[dict[str, Any]], 
                        metadata: list[Any], 
                        vectors: list[Any]
                        ) -> dict[str, Any]:
        """
        Store documents, metadata, and vectors in the database
        
        Args:
            documents: List of documents to store.
        
        
        """
        try:
            # 1. Store source information if needed
            source_ids = self._store_sources(documents)

            # 2. Store documents
            cids = self._store_document_entries(documents, source_ids)

            # 3. Create versions for documents
            version_cids = self._create_versions(cids)

            # 4. Store metadata
            self._store_metadata(metadata, cids)

            # 5. Store content
            cids = self._store_content(documents, version_cids)

            # 6. Create version-content associations
            self._create_version_content_links(version_cids, cids)

            # 7. Store vectors
            self._store_vectors(vectors, cids)

            return {
                "success": True,
                "message": "Documents stored successfully",
                "cids": cids,
                "version_cids": version_cids,
                "cids": cids
            }
        except Exception as e:
            self.logger.error(f"Error storing documents: {e}")
            return {
                "success": False,
                "message": str(e),
                "cids": [],
                "version_cids": [],
                "cids": []
            }

    def retrieve_documents(self, *, query: str) -> dict[str, Any]:
        """
        Retrieve documents from the database
        
        Args:
            query: Plaintext query
        
        Returns:
            Dictionary containing the following:
            - success (bool): True if retrieval was successful, False otherwise.
            - message (str): Informational message about the retrieval operation, or error message if failed.
            - documents (list[Document]): List of retrieved documents.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty.
            DatabaseError: If there is an error executing the retrieval operation.

        Example:
            >>> query = "What is the local sales tax rate in Cheyenne, WY?"
            >>> result = document_storage.retrieve_documents(query=query)
            >>> print(result)
            >>> {
            ...     "success": True,
            ...     "message": "Retrieved 5 documents",
            ...     "documents": [Document(...), Document(...), ...]
            ... }
        """
        raise NotImplementedError("Retrieve documents not yet implemented")
        try:
            documents = []
            return {
                "success": True,
                "message": f"Retrieved {len(documents)} documents",
                "documents": documents
            }
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return {
                "success": False,
                "message": str(e),
                "documents": []
            }

    # TODO: Implement these methods
    def update_documents(self, documents: list[Any]) -> dict[str, Any]:
        """
        Update existing documents

        Args:
            documents: List of documents with updated information
        """
        raise NotImplementedError("Update documents not yet implemented")

    def delete_documents(self, cids: list[str]) -> dict[str, Any]:
        """
        Delete documents by ID from the database.

        Args:
            cids: List of CIDs of documents to delete

        Returns:
            A dictionary containing the following:
            - result (bool): True if deletion was successful, False otherwise.
            - message (str): Number of documents deleted, or an error message.

        Raises:
            TypeError: If cids is not a list of strings.
            ValueError: If cids list is empty, or if any ID is not found in the database.
            DatabaseError: If there is an error executing the delete operation.

        Example:
            >>> cids = ["bafkreig4k3pl6q...", "bafkreihdwdcef7...", ...]
            >>> result = document_storage.delete_documents(cids)
            >>> print(result)
            >>> {
            ... "result": True,
            ... "message": "Deleted 2 documents successfully."
            ... }
            >>> # Error case
            >>> cids = ["nonexistentid1", "nonexistentid2"]
            >>> result = document_storage.delete_documents(cids)
            >>> print(result)
            >>> {
            ... "result": False,
            ... "message": "Error deleting documents: Document IDs not found."
            ... }
        """
        raise NotImplementedError("Delete documents not yet implemented")

    def _get_vectors(self, cids: list[str]) -> dict[str, Any]:
        """Get vectors for the specified document IDs"""
        try:
            # Get content IDs for the documents
            content_ids_query = """
                SELECT c.cid FROM Contents c
                JOIN VersionsContents vc ON c.cid = vc.cid
                JOIN Versions v ON vc.version_cid = v.version_cid
                JOIN Documents d ON v.document_cid = d.document_cid
                WHERE d.document_cid IN ({}) AND v.current_version = 1
            """.format(','.join(['?']*len(cids)))
            
            content_ids_result = self.db.execute(content_ids_query, (cids))
            cids = [r["cid"] for r in content_ids_result]
            
            # Get vectors for the content
            vectors_query = f"""
                SELECT * FROM Vectors WHERE cid IN ({','.join(['?']*len(cids))})
            """
            vectors = self.db.execute(vectors_query, cids)
            msg = f"Retrieved {len(vectors)} vectors for documents"
            
            return {
                "success": True,
                "message": msg,
                "vectors": vectors
            }
        except Exception as e:
            self.logger.error(f"Error retrieving vectors: {e}")
            return {
                "success": False,
                "message": str(e),
                "vectors": []
            }
    
    # Helper methods for database operations
    # TODO: Something doesn't seem right here. Check carefully.
    def _store_sources(self, documents: list[Any]) -> dict[str, str]:
        """Store sources and return a mapping of URL to source_id"""
        source_map = {}
        for doc in documents:
            url = doc["url"]
            domain = self._extract_domain(url)
            
            if domain not in source_map:
                source_id = self.get_cid()
                
                # Check if source already exists
                query = "SELECT id FROM Sources WHERE id = ?"
                result = self.db.execute(query, (source_id))
                
                if not result:
                    # Insert new source
                    insert_query = "INSERT INTO Sources (id) VALUES (?)"
                    self.db.execute(insert_query, (source_id))
                
                source_map[domain] = domain  # Source ID is the domain
        
        return source_map
    
    def _store_document_entries(self, documents: list[Any], source_ids: dict[str, str]) -> list[str]:
        """Store document entries and return document IDs"""
        cids = []
        
        for doc in documents:
            url = doc["url"]
            domain = self._extract_domain(url)
            source_id = source_ids.get(domain)
            
            document_cid = self.get_cid()
            document_type = self._determine_document_type(url)
            
            # Insert document
            insert_query = """
                INSERT INTO Documents (
                    document_cid, source_id, url, document_type,
                    status, priority
                ) VALUES (?, ?, ?, ?, ?, ?)
            """
            
            params = (
                document_cid, 
                source_id, 
                url, 
                document_type,
                DocumentStatus.NEW.value,
                5  # Default priority
            )
            
            self.db.execute(insert_query, params)
            cids.append(document_cid)
        
        return cids
    
    def _create_versions(self, cids: list[str]) -> list[str]:
        """Create initial versions for documents"""
        version_cids = []
        
        for document_cid in cids:
            version_cid = self.get_cid()
            
            # Insert version
            insert_query = """
                INSERT INTO Versions (
                    version_cid, document_cid, current_version,
                    version_number, status, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
            """
            
            params = (
                version_cid,
                document_cid,
                True,  # Current version
                "1.0",  # Initial version
                VersionStatus.ACTIVE.value,
                datetime.now()
            )
            
            self.db.execute(insert_query, params)
            
            # Update document with current version ID
            update_query = """
                UPDATE Documents
                SET current_version_id = ?, status = ?
                WHERE document_cid = ?
            """
            
            update_params = (
                version_cid,
                DocumentStatus.COMPLETE.value,
                document_cid
            )
            
            self.db.execute(update_query, update_params)
            version_cids.append(version_cid)
        
        return version_cids
    
    def _store_metadata(self, metadata_list: list[Any], cids: list[str]) -> None:
        """Store metadata for documents"""
        for i, metadata in enumerate(metadata_list):
            if i >= len(cids):
                break
                
            document_cid = cids[i]
            metadata_id = self.get_cid()
            
            # Insert metadata
            insert_query = """
                INSERT INTO Metadatas (
                    metadata_id, document_cid, other_metadata,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
            """
            
            params = (
                metadata_id,
                document_cid,
                metadata["metadata"],
                datetime.now(),
                datetime.now()
            )
            
            self.db.execute(insert_query, params)
    
    def _store_content(self, documents: list[Any], version_cids: list[str]) -> list[str]:
        """Store content for document versions"""
        cids = []
        
        for i, doc in enumerate(documents):
            if i >= len(version_cids):
                break
                
            version_cid = version_cids[i]
            cid = self.get_cid()
            
            # Insert content
            insert_query = """
                INSERT INTO Contents (
                    cid, version_cid, raw_content,
                    processed_content, hash
                ) VALUES (?, ?, ?, ?, ?)
            """
            
            content = doc["content"]
            processed_content = content  # In reality, this might go through processing TODO: FUCKING IMPLEMENT THIS
            content_hash = self._generate_hash(content)
            
            params = (
                cid,
                version_cid,
                content,
                processed_content,
                content_hash
            )
            
            self.db.execute(insert_query, params)
            cids.append(cid)
        
        return cids
    
    def _create_version_content_links(self, version_cids: list[str], cids: list[str]) -> None:
        """Create links between versions and content"""
        for i, version_cid in enumerate(version_cids):
            if i >= len(cids):
                break
                
            cid = cids[i]
            
            # Insert version-content link
            insert_query = """
                INSERT INTO VersionsContents (
                    version_cid, cid, created_at, source_type
                ) VALUES (?, ?, ?, ?)
            """
            
            params = (
                version_cid,
                cid,
                datetime.now(),
                SourceType.PRIMARY.value
            )
            
            self.db.execute(insert_query, params)
    
    def _store_vectors(self, vectors: list[Any], cids: list[str]) -> None:
        """Store vectors for content"""
        for i, vector in enumerate(vectors):
            if i >= len(cids):
                break
                
            cid = cids[i]
            vector_id = self.get_cid()
            
            # Insert vector
            insert_query = """
                INSERT INTO Vectors (
                    vector_id, cid, vector_embedding, embedding_type
                ) VALUES (?, ?, ?, ?)
            """
            
            params = (
                vector_id,
                cid,
                vector["embedding"],
                vector["embedding_type"]
            )
            
            self.db.execute(insert_query, params)

    # Utility methods
    def _generate_uuid(self) -> str:
        """Generate a UUID string"""
        return str(uuid4())

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        import re
        match = re.search(r'https?://([^/]+)', url)
        return match.group(1) if match else url

    def _determine_document_type(self, url: str) -> str:
        """Determine document type from URL"""
        if url.endswith('.pdf'):
            return 'pdf'
        elif url.endswith('.doc') or url.endswith('.docx'):
            return 'word'
        else:
            return 'html'
    
    def _generate_hash(self, content: str) -> str:
        """Generate a hash for content"""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()

