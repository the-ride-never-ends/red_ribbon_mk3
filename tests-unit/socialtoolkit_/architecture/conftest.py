

class FixtureError(Exception):
    """Custom exception for fixture errors."""
    def __init__(self, msg: str):
        super().__init__(msg)
