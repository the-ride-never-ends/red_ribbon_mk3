

import numpy as np


def dot_product(vec1, vec2):
    """
    Calculate dot product between two vectors
    """
    # Convert to numpy arrays for efficient calculation
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    # Calculate dot product
    dot = np.dot(vec1_np, vec2_np)
    return float(dot)