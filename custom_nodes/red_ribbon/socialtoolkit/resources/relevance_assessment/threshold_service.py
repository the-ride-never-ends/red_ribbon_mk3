"""
Threshold service for RelevanceAssessment
"""
import logging
from typing import Callable, Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class ThresholdService:
    """
    Manages thresholds for relevance assessment decisions
    """
    
    def __init__(self):
        logger.info("ThresholdService initialized")
        
        # Default thresholds
        self.thresholds = {
            "high_relevance": 0.8,
            "medium_relevance": 0.5,
            "low_relevance": 0.3,
            "minimum_acceptable": 0.2
        }
    
    def set_threshold(self, threshold_name: str, value: float) -> None:
        """
        Set a named threshold value
        
        Args:
            threshold_name: Name of the threshold
            value: Threshold value
        """
        self.thresholds[threshold_name] = value
        logger.info(f"Set threshold '{threshold_name}' to {value}")
    
    def get_threshold(self, threshold_name: str, default: Optional[float] = None) -> float:
        """
        Get a threshold value by name
        
        Args:
            threshold_name: Name of the threshold
            default: Default value if threshold not found
            
        Returns:
            Threshold value
        """
        return self.thresholds.get(threshold_name, default)
    
    def classify_relevance(self, score: float) -> str:
        """
        Classify a relevance score into a category
        
        Args:
            score: Relevance score
            
        Returns:
            Relevance category
        """
        if score >= self.thresholds.get("high_relevance", 0.8):
            return "high"
        elif score >= self.thresholds.get("medium_relevance", 0.5):
            return "medium"
        elif score >= self.thresholds.get("low_relevance", 0.3):
            return "low"
        elif score >= self.thresholds.get("minimum_acceptable", 0.2):
            return "minimal"
        else:
            return "not_relevant"
    
    def filter_by_threshold(self, 
                           results: List[Union[float, Dict[str, Any], tuple]], 
                           threshold_name: str = "minimum_acceptable",
                           score_accessor: Optional[Callable[[Any], float]] = None
                          ) -> List[Union[float, Dict[str, Any], tuple]]:
        """
        Filter results by a threshold
        
        Args:
            results: List of results (can be scores, objects with scores, or tuples)
            threshold_name: Name of threshold to apply
            score_accessor: Function to extract score from complex result items
            
        Returns:
            Filtered list of results
        """
        threshold = self.thresholds.get(threshold_name, 0.2)
        
        filtered_results = []
        
        for item in results:
            # Extract score based on item type
            score: float
            if score_accessor:
                # Use provided accessor function
                accessor_output = score_accessor(item)
                assert isinstance(accessor_output, float), "Score accessor must return a float"
                score = float()
            elif isinstance(item, float):
                # Item is a score
                score = item
            elif isinstance(item, dict):
                # Item is a dict, look for score
                assert "score" in item, "Dictionary item must contain 'score' key"
                score = item["score"]
                assert isinstance(score, float), "'score' value must be a float"
            elif isinstance(item, tuple) and len(item) >= 2:
                # Item is a tuple with score as second element
                score = item[1]
                assert isinstance(score, float), "Second element of tuple must be a float score"
            else:
                # Can't determine score
                score = 0.0
            
            if score >= threshold:
                filtered_results.append(item)
                
        logger.info(f"Filtered {len(results)} results to {len(filtered_results)} using threshold '{threshold_name}' ({threshold})")
        return filtered_results