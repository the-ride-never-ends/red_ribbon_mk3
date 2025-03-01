"""
Metadata manager for DocumentStorage
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MetadataManager:
    """
    Manager for document metadata operations
    """
    
    def __init__(self):
        logger.info("MetadataManager initialized")
        self.metadata = {}  # Dummy in-memory metadata store
    
    def create(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Create metadata for a document
        
        Args:
            document_id: ID of the document
            metadata: Metadata dictionary
            
        Returns:
            Boolean indicating success
        """
        if document_id in self.metadata:
            logger.warning(f"Metadata already exists for document ID: {document_id}")
            return False
            
        # Add system metadata
        enriched_metadata = metadata.copy()
        enriched_metadata.update({
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "version": 1
        })
        
        self.metadata[document_id] = enriched_metadata
        logger.info(f"Created metadata for document ID: {document_id}")
        return True
    
    def update(self, document_id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a document
        
        Args:
            document_id: ID of the document
            metadata: Metadata dictionary (partial update)
            
        Returns:
            Boolean indicating success
        """
        if document_id not in self.metadata:
            logger.warning(f"No metadata found for document ID: {document_id}")
            return False
            
        # Update existing metadata
        current_metadata = self.metadata[document_id]
        current_metadata.update(metadata)
        
        # Update system metadata
        current_metadata["updated_at"] = datetime.now().isoformat()
        current_metadata["version"] = current_metadata.get("version", 0) + 1
        
        logger.info(f"Updated metadata for document ID: {document_id}")
        return True
    
    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a document
        
        Args:
            document_id: ID of the document
            
        Returns:
            Metadata dictionary or None
        """
        return self.metadata.get(document_id)
    
    def delete(self, document_id: str) -> bool:
        """
        Delete metadata for a document
        
        Args:
            document_id: ID of the document
            
        Returns:
            Boolean indicating success
        """
        if document_id in self.metadata:
            del self.metadata[document_id]
            logger.info(f"Deleted metadata for document ID: {document_id}")
            return True
            
        logger.warning(f"No metadata found for document ID: {document_id}")
        return False
    
    def search(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search for documents by metadata
        
        Args:
            query: Query dictionary
            
        Returns:
            List of matching metadata dictionaries with document IDs
        """
        results = []
        
        for doc_id, meta in self.metadata.items():
            match = True
            for key, value in query.items():
                if key not in meta or meta[key] != value:
                    match = False
                    break
                    
            if match:
                result_item = meta.copy()
                result_item["document_id"] = doc_id
                results.append(result_item)
                
        return results