from pydantic import BaseModel
from typing import Any, Callable, Optional, Type
import logging

import openai


from .variable_codebook import VariableCodebook, Variable
from .document_storage import DocumentStorage
from .top10_document_retrieval import Top10DocumentRetrieval
from .relevance_assessment import RelevanceAssessment
from .prompt_decision_tree import PromptDecisionTree
from .document_retrieval_from_websites import DocumentRetrievalFromWebsites

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

    def __init__(self, *, resources: dict[str, Callable], configs: SocialtoolkitConfigs) -> None:
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

        self.document_retrieval_from_websites: DocumentRetrievalFromWebsites = resources["document_retrieval"]
        self.document_storage:                 DocumentStorage               = resources["document_storage"]
        self.top10_document_retrieval:         Top10DocumentRetrieval        = resources["top10_document_retrieval"]
        self.relevance_assessment:             RelevanceAssessment           = resources["relevance_assessment"]
        self.prompt_decision_tree:             PromptDecisionTree            = resources["prompt_decision_tree"]
        self.variable_codebook:                VariableCodebook              = resources["variable_codebook"]

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


    def execute(self, query: str) -> dict[str, str] | list[dict[str, str]]:
        """
        Execute the control flow based on the mermaid chart

        Args:
            query: The question or information request. 
                The query must contain a supported variable(s) and supported locality/localities

        Returns:
            Dictionary containing the following:

            If the request was interpreted as having more than one response, a list of dictionaries is returned.
        """
        self.logger.info(f"Starting high level control flow with input: {query}")

        if self.configs.approved_document_sources:
            # Step 1: Get domain URLs from pre-approved sources
            urls: list[str] = self.document_retrieval_from_websites.get_urls(query)

            # Step 2: Retrieve documents from websites
            documents, metadata, vectors = self.document_retrieval_from_websites.retrieve_documents(urls)
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
        stored_docs, stored_vectors = self.document_retrieval.execute(query)
        stored_docs: list[tuple[str, ...]]
        stored_vectors: list[dict[str, list[float]]]

        # Step 5: Perform top-10 document retrieval
        potentially_relevant_docs = self.top10_document_retrieval.execute(
            query, 
            stored_docs, 
            stored_vectors
        )
        potentially_relevant_docs: list[tuple[str, ...]]

        # Step 6: Get variable definition from codebook
        variable: Variable = self.variable_codebook.execute(self.llm, query)

        # Step 7: Perform relevance assessment
        relevant_documents = self.relevance_assessment.execute(
            potentially_relevant_docs,
            variable,
        )

        # Step 8: Execute prompt decision tree
        result: dict[str, str] | list[dict[str, str]] = self.prompt_decision_tree.execute(
            relevant_documents,
            variable,
        )

        if result is None:
            self.logger.warning("Failed to execute prompt decision tree")
            return {}
        if not result:
            self.logger.warning(f"Result for '{query}' is empty")
            return {}
        else:
            self.logger.info(f"Completed high level control flow for '{query}' with output: {result}")
            return result

    def _get_domain_urls(self) -> list[str]:
        """Extract domain URLs from pre-approved document sources"""
        # TODO
        pass


