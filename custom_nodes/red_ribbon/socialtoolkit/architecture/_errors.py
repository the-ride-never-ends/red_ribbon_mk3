"""
Error classes for the Social Toolkit architecture components.

Each step of the pipeline has its own custom error class to facilitate error handling and debugging.

"""


class UrlGenerationError(RuntimeError):
    """Custom exception for errors that occur during URL generation"""

    def __init__(self, *args):
        super().__init__(*args)


class WebsiteDocumentRetrievalError(RuntimeError):
    """Custom exception for errors that occur during website document retrieval"""

    def __init__(self, *args):
        super().__init__(*args)


class DecisionTreeError(RuntimeError):
    """Custom exception for errors that occur during decision tree execution"""

    def __init__(self, *args):
        super().__init__(*args)


class InitializationError(RuntimeError):
    """Custom exception for errors that occur during initialization"""

    def __init__(self, *args):
        super().__init__(*args)

class Top10DocumentRetrievalError(RuntimeError):
    """Custom exception for errors that occur during Top 10 document retrieval"""

    def __init__(self, *args):
        super().__init__(*args)

class LLMError(RuntimeError):
    """Custom exception for errors that occur during LLM interactions"""

    def __init__(self, *args):
        super().__init__(*args)

class RelevanceAssessmentError(RuntimeError):
    """Custom exception for errors that occur during relevance assessment"""

    def __init__(self, *args):
        super().__init__(*args)

class DocumentStorageError(RuntimeError):
    """Custom exception for errors that occur during document storage"""

    def __init__(self, *args):
        super().__init__(*args)

class CodebookError(RuntimeError):
    """Custom exception for errors that occur during variable codebook operations"""

    def __init__(self, *args):
        super().__init__(*args)

class DataclassError(RuntimeError):
    """Custom exception for errors that occur during dataclass operations"""

    def __init__(self, *args):
        super().__init__(*args)
