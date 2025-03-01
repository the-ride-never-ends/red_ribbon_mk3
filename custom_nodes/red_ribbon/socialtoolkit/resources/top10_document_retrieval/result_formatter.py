"""
Result formatter for Top10DocumentRetrieval
"""
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ResultFormatter:
    """
    Formats search results for display or further processing
    """
    
    def __init__(self):
        logger.info("ResultFormatter initialized")
    
    def format(self, documents: List[Dict[str, Any]], format_type: str = "summary") -> List[Dict[str, Any]]:
        """
        Format search results
        
        Args:
            documents: List of document objects
            format_type: Type of formatting to apply ("summary", "full", "highlight", etc.)
            
        Returns:
            List of formatted document dictionaries
        """
        logger.info(f"Formatting {len(documents)} documents as {format_type}")
        
        formatted_results = []
        
        for i, doc in enumerate(documents):
            rank = i + 1
            content = doc.get("content", "")
            url = doc.get("url", "")
            title = doc.get("title", f"Result {rank}")
            
            if format_type == "summary":
                # For summary format, truncate content to a brief excerpt
                excerpt = content[:150] + "..." if len(content) > 150 else content
                
                formatted_doc = {
                    "rank": rank,
                    "title": title,
                    "url": url,
                    "excerpt": excerpt,
                    "score": doc.get("rank_score", 0) or doc.get("similarity_score", 0)
                }
            
            elif format_type == "full":
                # For full format, include all content
                formatted_doc = {
                    "rank": rank,
                    "title": title,
                    "url": url,
                    "content": content,
                    "score": doc.get("rank_score", 0) or doc.get("similarity_score", 0)
                }
                
            elif format_type == "highlight":
                # For highlight format, include content with highlights (dummy implementation)
                formatted_doc = {
                    "rank": rank,
                    "title": title,
                    "url": url,
                    "content": content,
                    "highlights": [content[max(0, i-20):i+20] for i in range(0, len(content), 100) if i < len(content)],
                    "score": doc.get("rank_score", 0) or doc.get("similarity_score", 0)
                }
            
            else:
                # Default format
                formatted_doc = {
                    "rank": rank,
                    "title": title,
                    "url": url, 
                    "content": content,
                    "score": doc.get("rank_score", 0) or doc.get("similarity_score", 0)
                }
                
            formatted_results.append(formatted_doc)
            
        return formatted_results