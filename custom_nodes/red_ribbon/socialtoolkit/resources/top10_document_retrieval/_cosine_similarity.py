

import numpy as np


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    # Convert to numpy arrays for efficient calculation
    vec1_np = np.array(vec1) if not isinstance(vec1, np.ndarray) else vec1
    vec2_np = np.array(vec2) if not isinstance(vec1, np.ndarray) else vec1
    
    # Calculate dot product
    dot = np.dot(vec1_np, vec2_np)
    
    # Calculate norms
    norm1 = np.linalg.norm(vec1_np)
    norm2 = np.linalg.norm(vec2_np, axis=1)
    
    # Calculate cosine similarity
    similarity = dot / (norm1 * norm2)
    return float(similarity)
