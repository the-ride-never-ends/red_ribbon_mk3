"""
Socialtoolkit - Turn Law into Datasets
"""

from .socialtoolkit import make_socialtoolkit_api, SocialToolkitAPI
from ._demo_mode import (
    demo_document_storage,
    demo_top10_document_retrieval,
    demo_relevance_assessment,
    demo_database_enter,
    demo_prompt_decision_tree,
)

__all__ = [
    "make_socialtoolkit_api",
    "SocialToolkitAPI",
    "demo_document_storage",
    "demo_top10_document_retrieval",
    "demo_relevance_assessment",
    "demo_database_enter",
    "demo_prompt_decision_tree",
]
