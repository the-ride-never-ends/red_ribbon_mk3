"""
Result formatter for RelevanceAssessment
"""
import logging
from typing import Dict, List, Any, Tuple, Union

logger = logging.getLogger(__name__)

class ResultFormatter:
    """
    Formats search results for presentation
    """
    
    def __init__(self):
        logger.info("ResultFormatter initialized")
    
    def format(self, 
              ranked_results: List[Tuple[Dict[str, Any], float]], 
              format_type: str = "default",
              max_results: int = 10,
              include_scores: bool = False
             ) -> List[Dict[str, Any]]:
        """
        Format ranked results for presentation
        
        Args:
            ranked_results: List of (document, score) tuples
            format_type: Type of formatting to apply
            max_results: Maximum number of results to include
            include_scores: Whether to include relevance scores in output
            
        Returns:
            List of formatted result objects
        """
        logger.info(f"Formatting {len(ranked_results)} results as '{format_type}'")
        
        # Limit results to max_results
        results_to_format = ranked_results[:max_results]
        
        if format_type == "simple":
            return self._format_simple(results_to_format, include_scores)
        elif format_type == "detailed":
            return self._format_detailed(results_to_format, include_scores)
        elif format_type == "snippet":
            return self._format_snippet(results_to_format, include_scores)
        else:  # "default" or any other value
            return self._format_default(results_to_format, include_scores)
    
    def _format_default(self, 
                       results: List[Tuple[Dict[str, Any], float]], 
                       include_scores: bool
                      ) -> List[Dict[str, Any]]:
        """
        Apply default formatting to results
        """
        formatted_results = []
        
        for i, (doc, score) in enumerate(results):
            result = {
                "rank": i + 1,
                "id": doc.get("id", f"result_{i}"),
                "title": self._extract_title(doc),
                "url": doc.get("url", ""),
                "snippet": self._generate_snippet(doc)
            }
            
            if include_scores:
                result["relevance_score"] = score
                
            formatted_results.append(result)
            
        return formatted_results
    
    def _format_simple(self, 
                      results: List[Tuple[Dict[str, Any], float]], 
                      include_scores: bool
                     ) -> List[Dict[str, Any]]:
        """
        Apply simple formatting to results (just basic info)
        """
        formatted_results = []
        
        for i, (doc, score) in enumerate(results):
            result = {
                "id": doc.get("id", f"result_{i}"),
                "title": self._extract_title(doc),
                "url": doc.get("url", "")
            }
            
            if include_scores:
                result["relevance_score"] = score
                
            formatted_results.append(result)
            
        return formatted_results
    
    def _format_detailed(self, 
                        results: List[Tuple[Dict[str, Any], float]], 
                        include_scores: bool
                       ) -> List[Dict[str, Any]]:
        """
        Apply detailed formatting to results (include all metadata)
        """
        formatted_results = []
        
        for i, (doc, score) in enumerate(results):
            result = {
                "rank": i + 1,
                "id": doc.get("id", f"result_{i}"),
                "title": self._extract_title(doc),
                "url": doc.get("url", ""),
                "snippet": self._generate_snippet(doc),
                "timestamp": doc.get("timestamp", ""),
                "metadata": doc.get("metadata", {})
            }
            
            if include_scores:
                result["relevance_score"] = score
                
            formatted_results.append(result)
            
        return formatted_results
    
    def _format_snippet(self, 
                       results: List[Tuple[Dict[str, Any], float]], 
                       include_scores: bool
                      ) -> List[Dict[str, Any]]:
        """
        Format results with focus on content snippets
        """
        formatted_results = []
        
        for i, (doc, score) in enumerate(results):
            result = {
                "rank": i + 1,
                "id": doc.get("id", f"result_{i}"),
                "snippet": self._generate_snippet(doc, length=200)
            }
            
            if include_scores:
                result["relevance_score"] = score
                
            formatted_results.append(result)
            
        return formatted_results
    
    def _extract_title(self, document: Dict[str, Any]) -> str:
        """
        Extract a title from document
        """
        # Try to get title from various possible locations
        title = document.get("title", "")
        
        if not title:
            # Try to derive from content
            content = document.get("content", "")
            if content:
                # Take first line or first 50 chars as title
                title = content.split("\n")[0][:50]
                if len(title) == 50:
                    title += "..."
        
        if not title:
            # Last resort, use ID or URL
            title = document.get("id", document.get("url", "Untitled document"))
            
        return title
    
    def _generate_snippet(self, document: Dict[str, Any], length: int = 150) -> str:
        """
        Generate a content snippet from document
        """
        content = document.get("content", "")
        
        if not content:
            return ""
            
        # In a real implementation, would extract most relevant section
        # This is a simple implementation that just takes first N chars
        snippet = content[:length]
        if len(content) > length:
            snippet += "..."
            
        return snippet