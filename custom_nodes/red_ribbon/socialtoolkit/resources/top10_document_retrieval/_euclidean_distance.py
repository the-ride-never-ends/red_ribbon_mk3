import numpy as np


def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
    """Calculate Euclidean distance between two vectors"""
    # Convert to numpy arrays for efficient calculation
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(vec1_np - vec2_np)
    return float(distance)