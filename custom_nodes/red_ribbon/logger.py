"""
Red Ribbon's Logger Module

Provides logging functionality for Red Ribbon modules.
"""
import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Get a logger instance for the specified name with consistent formatting.
    
    Args:
        name: The name for the logger (typically the module or class name)
        level: The logging level to use (default: INFO)
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Only add handler if it doesn't already have one
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger