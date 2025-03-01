

from typing import Any, Optional


from pydantic import BaseModel, Field




class SocialtoolkitConfigs(BaseModel):
    """Configuration for High Level Architecture workflow"""
    approved_document_sources: list[str]
    llm_api_config: dict[str, Any]
    document_retrieval_threshold: int = 10
    relevance_threshold: float = 0.7
    output_format: str = "json"

    codebook: Optional[dict[str, Any]] = None
    document_retrieval: Optional[dict[str, Any]] = None
    llm_service: Optional[dict[str, Any]] = None
    top10_retrieval: Optional[dict[str, Any]] = None
    relevance_assessment: Optional[dict[str, Any]] = None
    prompt_decision_tree: Optional[dict[str, Any]] = None














