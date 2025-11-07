
# Importing them should cause the database to be initialized and the tables to be created.
from .setup.setup_citation_db import setup_citation_db
from .setup.setup_embeddings_db import setup_embeddings_db
from .setup.setup_html_db import setup_html_db
from ._database import Database
from .factory import make_duckdb_database

DatabaseAPI = Database

__all__ = [
    "setup_citation_db",
    "setup_embeddings_db",
    "setup_html_db",
    "make_duckdb_database",
    "Database",
    "DatabaseAPI",
]
