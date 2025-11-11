"""Custom error definitions for the main Red Ribbon package."""

class InitializationError(RuntimeError):
    """
    Custom exception for initialization errors.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ConfigurationError(RuntimeError):
    """
    Custom exception for configuration errors.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)


class ResourceError(RuntimeError):
    """
    Custom exception for resource errors.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)


class LocalDependencyError(RuntimeError):
    """
    Custom exception for dependency errors.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)


class LibraryDependencyError(RuntimeError):
    """
    Custom exception for dependency errors.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)


class RedRibbonError(Exception):
    """
    Error that occurred in the main.py file.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"RedRibbonError: {self.message}"


class LLMError(Exception):
    """
    Custom exception for LLM-related errors.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"LLMError: {self.message}"
