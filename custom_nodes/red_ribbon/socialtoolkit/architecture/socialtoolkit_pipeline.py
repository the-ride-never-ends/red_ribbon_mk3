from pydantic import BaseModel
from typing import Any, Optional, Type
import logging

import openai





# from configs import Configs
OPEN_AI_API_KEY = "sk-1234567890abcdef1234567890abcdef"

class SocialtoolkitConfigs(BaseModel):
    """Configuration for High Level Architecture workflow"""
    approved_document_sources: list[str] = None
    llm_api_config: dict[str, Any] = None
    document_retrieval_threshold: int = 10
    relevance_threshold: float = 0.7
    output_format: str = "json"
    get_documents_from_web: bool = False


class SocialtoolkitPipeline:
    """
    High Level Architecture for document retrieval and data extraction system
    based on mermaid chart in README.md
    """

    def __init__(self, resources: dict[str, Any], configs: SocialtoolkitConfigs):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including services
            configs: Configuration for High Level Architecture
        """
        self.resources = resources
        self.configs: SocialtoolkitConfigs = configs
        self.logger = logging.getLogger(self.class_name)

        self.approved_document_sources: list[str] = self.configs.approved_document_sources

        self._web_scraper = None
        self._omni_converter = None
        self._db = None

        self.llm = resources["llm"]

        self.document_retrieval = resources["document_retrieval"]
        self.document_storage = resources["document_storage"]
        self.top10_document_retrieval = resources["top10_document_retrieval"]
        self.relevance_assessment = resources["relevance_assessment"]
        self.prompt_decision_tree = resources["prompt_decision_tree"]
        self.variable_codebook = resources["variable_codebook"]

        # Initialize services
        # TODO Get this openai shit out of here!
        if self.llm is None:
            self.llm_api = openai.OpenAI(api_key=OPEN_AI_API_KEY)
        else:# Default to OpenAI API
            self.llm_api = self.llm(resources, configs)

        self.logger.info("Socialtoolkit initialized with services")

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()

    def execute(self, input_data_point: str) -> dict[str, str] | list[dict[str, str]]:
        """
        Execute the control flow based on the mermaid chart
        
        Args:
            input_data_point: The question or information request. This can be a single request.

        Returns:
            Dictionary containing the output data point. 
            If the request was interpreted as having more than one response, a list of dictionaries is returned.
        """
        self.logger.info(f"Starting high level control flow with input: {input_data_point}")
        
        if self.configs.approved_document_sources:
            # Step 1: Get domain URLs from pre-approved sources
            domain_urls: list[str] = self.document_retrieval.execute(domain_urls)
            
            # Step 2: Retrieve documents from websites
            documents, metadata, vectors = self.document_retrieval.execute(domain_urls)
            documents: list[tuple[str, ...]]
            metadata: list[dict[str, Any]]
            vectors: list[dict[str, list[float]]]
            
            # Step 3: Store documents in document storage
            storage_successful: bool = self.document_storage.execute(documents, metadata, vectors)
            if storage_successful:
                self.logger.info("Documents stored successfully")
            else:
                self.logger.warning("Failed to store documents")
        
        # Step 4: Retrieve documents and document vectors
        stored_docs, stored_vectors = self.document_retrieval.execute(
            input_data_point,
            self.llm.execute("retrieve_documents")
        )
        stored_docs: list[tuple[str, ...]]
        stored_vectors: list[dict[str, list[float]]]
        
        # Step 5: Perform top-10 document retrieval
        potentially_relevant_docs = self.top10_document_retrieval.execute(
            input_data_point, 
            stored_docs, 
            stored_vectors
        )
        potentially_relevant_docs: list[tuple[str, ...]]
        
        # Step 6: Get variable definition from codebook
        prompt_sequence = self.variable_codebook.execute(self.llm, input_data_point)
        
        # Step 7: Perform relevance assessment
        relevant_documents = self.relevance_assessment.execute(
            potentially_relevant_docs,
            prompt_sequence,
            self.llm.execute("relevance_assessment")
        )
        
        # Step 8: Execute prompt decision tree
        output_data_point = self.prompt_decision_tree.execute(
            relevant_documents,
            prompt_sequence,
            self.llm.execute("prompt_decision_tree")
        )

        if output_data_point is None:
            self.logger.warning("Failed to execute prompt decision tree")
        else:
            self.logger.info(f"Completed high level control flow with output: {output_data_point}")
        
        return {"output_data_point": output_data_point}
        
    def _get_domain_urls(self) -> list[str]:
        """Extract domain URLs from pre-approved document sources"""
        # TODO
        pass
        