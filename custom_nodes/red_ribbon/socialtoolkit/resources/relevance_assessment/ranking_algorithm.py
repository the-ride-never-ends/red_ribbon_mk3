"""
Ranking algorithm for RelevanceAssessment
"""
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class RankingAlgorithm:
    """
    Ranks search results based on relevance scores and other factors
    """
    
    def __init__(self):
        logger.info("RankingAlgorithm initialized")
    
    def rank(self, 
             documents: List[Dict[str, Any]], 
             relevance_scores: List[float],
             query_analysis: Optional[Dict[str, Any]] = None,
             additional_factors: Optional[Dict[str, List[float]]] = None
            ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Rank documents based on relevance scores and additional factors
        
        Args:
            documents: List of documents to rank
            relevance_scores: Corresponding relevance scores
            query_analysis: Analysis of the search query
            additional_factors: Dictionary of additional ranking factors
            
        Returns:
            List of tuples containing (document, final_score)
        """
        logger.info(f"Ranking {len(documents)} documents")
        
        # Combine all factors to compute final scores
        final_scores = self._compute_final_scores(
            relevance_scores,
            additional_factors or {}
        )
        
        # Create document-score pairs
        doc_score_pairs = list(zip(documents, final_scores))
        
        # Sort by score in descending order
        ranked_results = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        
        return ranked_results
    
    def _compute_final_scores(self, 
                             relevance_scores: List[float], 
                             additional_factors: Dict[str, List[float]]
                            ) -> List[float]:
        """
        Compute final ranking scores using relevance and additional factors
        
        Args:
            relevance_scores: Base relevance scores
            additional_factors: Dictionary of additional factors with scores
            
        Returns:
            List of final ranking scores
        """
        # Weights for different factors
        weights = {
            "relevance": 0.6,
            "recency": 0.2,
            "popularity": 0.1,
            "authority": 0.1
        }
        
        # Initialize final scores with relevance scores (weighted)
        final_scores = [score * weights["relevance"] for score in relevance_scores]
        
        # Add weighted contributions from additional factors
        for factor, factor_scores in additional_factors.items():
            if factor in weights and len(factor_scores) == len(final_scores):
                for i, score in enumerate(factor_scores):
                    final_scores[i] += score * weights[factor]
        
        return final_scores