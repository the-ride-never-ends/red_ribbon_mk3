



class DecisionTreeError(RuntimeError):
    """Custom exception for errors that occur during decision tree execution"""

    def __init__(self, *args):
        super().__init__(*args)


class InitializationError(RuntimeError):
    """Custom exception for errors that occur during initialization"""

    def __init__(self, *args):
        super().__init__(*args)

class DocumentRetrievalError(RuntimeError):
    """Custom exception for errors that occur during document retrieval"""

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

class StorageError(RuntimeError):
    """Custom exception for errors that occur during document storage"""

    def __init__(self, *args):
        super().__init__(*args)

class CodebookError(RuntimeError):
    """Custom exception for errors that occur during variable codebook operations"""

    def __init__(self, *args):
        super().__init__(*args)

