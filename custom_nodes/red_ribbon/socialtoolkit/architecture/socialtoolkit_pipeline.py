import logging
from typing import Any, Callable, Optional, Type


import openai
from pydantic import BaseModel, NonNegativeInt, NonNegativeFloat


from .variable_codebook import VariableCodebook, Variable
from .document_storage import DocumentStorage
from .top10_document_retrieval import Top10DocumentRetrieval
from .relevance_assessment import RelevanceAssessment
from .prompt_decision_tree import PromptDecisionTree
from .document_retrieval_from_websites import DocumentRetrievalFromWebsites


from .dataclasses import Document, Vector
from custom_nodes.red_ribbon.utils_.llm_ import LLM
from ._errors import (
    UrlGenerationError,
    InitializationError,
    DecisionTreeError,
    WebsiteDocumentRetrievalError,
    Top10DocumentRetrievalError,
    LLMError,
    RelevanceAssessmentError,
    DocumentStorageError,
    CodebookError,
)

# from configs import Configs
OPEN_AI_API_KEY = "sk-1234567890abcdef1234567890abcdef"

class SocialtoolkitConfigs(BaseModel):
    """Configuration for High Level Architecture workflow"""
    approved_document_sources: Optional[list[str]] = None
    llm_api_config: Optional[dict[str, Any]] = None
    document_retrieval_threshold: NonNegativeInt = 10
    relevance_threshold: NonNegativeFloat = 0.7
    output_format: str = "json"
    get_documents_from_web: bool = False


class SocialtoolkitPipeline:
    """
    High Level Architecture for document retrieval and data extraction system
    based on mermaid chart in README.md
    """

    def __init__(self, *, resources: dict[str, Any], configs: SocialtoolkitConfigs) -> None:
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including services
            configs: Configuration for High Level Architecture
        """
        self.resources = resources
        self.configs = configs
        self.logger: logging.Logger = logging.getLogger(self.class_name)

        self.approved_document_sources: list[str] = self.configs.approved_document_sources

        self._web_scraper: Type = None
        self._omni_converter: Type = None
        self._db = None

        self.llm: LLM = resources["llm"]

        self.document_retrieval_from_websites: DocumentRetrievalFromWebsites = resources["document_retrieval"]
        self.document_storage:                 DocumentStorage               = resources["document_storage"]
        self.top10_document_retrieval:         Top10DocumentRetrieval        = resources["top10_document_retrieval"]
        self.relevance_assessment:             RelevanceAssessment           = resources["relevance_assessment"]
        self.prompt_decision_tree:             PromptDecisionTree            = resources["prompt_decision_tree"]
        self.variable_codebook:                VariableCodebook              = resources["variable_codebook"]

        # Initialize services
        # TODO Get this openai shit out of here!
        self.llm_api: LLM

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
        Execute the socialtoolkit pipeline based on an input query.

        Args:
            query: The question or information request. 
                The query must contain a supported variable(s) and supported locality/localities

        Returns:
            Dictionary containing the following:
                - success: bool indicating if execution was successful, or false if an error occurred
                - msg: (str) Message describing the result or error
                - document_text: list[Section] Concatenated document text used for decision tree
                - output_data_point: Extracted data point as a string
                - responses: List of LLM responses at each node
                - iterations: Number of iterations taken to reach final node
            
            If the request was interpreted as having more than one response, a list of dictionaries is returned.
        
        Example:
        >>> query = "What is the city sales tax rate in Cheyenne, WY?"
        >>> result = socialtoolkit_pipeline.execute(query)
        >>> print(result)
        >>> {
            "success": True,
            "msg": "Data point extracted successfully.",
            "document_text": [...],
            "output_data_point": "6.0%",
            "responses": [...],
            "iterations": 3
        }
        """
        self.logger.info(f"Starting high level control flow with input: {query}")

        # Step 1: Get variable definition from codebook
        get_var_action = "get_variable"
        try:
            var_result = self.variable_codebook.execute(get_var_action, llm=self.llm, query=query)
        except Exception as e:
            raise CodebookError(f"Unexpected exception while getting variable from codebook: {e}") from e

        if var_result['success'] == False:
            raise CodebookError(f"Failed to get variable from codebook: {var_result['message']}")

        variable: Variable = var_result['variable']

        self.logger.debug(f"Extracted variable from query: {variable.model_dump()}")

        if self.configs.approved_document_sources:
            # Step 2: Get domain URLs from pre-approved sources
            try:
                urls: list[str] = self.document_retrieval_from_websites.get_urls(query)
            except Exception as e:
                raise UrlGenerationError(f"Unexpected exception while generating URLs from approved sources: {e}") from e

            # Step 3: Retrieve documents from websites
            documents: list[dict[str, Any]]
            metadata: list[dict[str, Any]]
            vectors: list[dict[str, list[float]]]
            try:
                documents, metadata, vectors = self.document_retrieval_from_websites.retrieve_documents(urls)
            except Exception as e:
                raise WebsiteDocumentRetrievalError(f"Unexpected exception while retrieving documents from websites: {e}") from e

            self.logger.debug(
                f"Retrieved {len(documents)} documents, {len(metadata)} metadata entries, and {len(vectors)} vectors from websites"
            )

            if not documents or not metadata or not vectors:
                self.logger.warning("No documents, metadata, or vectors retrieved from websites. Aborting storage step.")
            else:
                # Step 4: Store documents in document storage
                store_action = "store"
                try:
                    store_result: dict[str, Any] = self.document_storage.execute(
                        store_action, documents=documents, metadata=metadata, vectors=vectors
                    )
                except Exception as e:
                    raise DocumentStorageError(f"Unexpected exception while storing documents: {e}") from e

                if store_result['success'] == True:
                    self.logger.info("Documents stored successfully")
                else:
                    self.logger.warning(f"Failed to store documents: {store_result['message']}")

        # Step 5: Retrieve documents and document vectors
        retrieve_action = "retrieve"
        stored_docs: list[Document]
        stored_vectors: list[Vector]
        try:
            retrieve_result = self.document_storage.execute(retrieve_action, query)
        except Exception as e:
            raise DocumentStorageError(f"Unexpected exception while retrieving documents: {e}") from e

        if retrieve_result['success'] == True:
            stored_docs = retrieve_result['documents']
            stored_vectors = retrieve_result['vectors']
            self.logger.info(f"Retrieved {len(stored_docs)} documents and {len(stored_vectors)} vectors from storage")
        else:
            self.logger.warning(f"Failed to retrieve documents: {retrieve_result['message']}")
            return {}

        # Step 6: Perform top-10 document retrieval
        try:
            potentially_relevant_docs: dict[str, Document] = self.top10_document_retrieval.execute(
                query, 
                stored_docs, 
                stored_vectors
            )
        except Exception as e:
            raise Top10DocumentRetrievalError(f"Unexpected exception during top-10 document retrieval: {e}") from e

        if not potentially_relevant_docs:
            self.logger.warning(f"No potentially relevant documents found for query '{query}'")
            return {}

        # Step 7: Perform relevance assessment
        try:
            relevant_documents: dict[str, Document] = self.relevance_assessment.execute(
                potentially_relevant_docs,
                variable,
            )
        except Exception as e:
            raise RelevanceAssessmentError(f"Unexpected exception during relevance assessment: {e}") from e

        if not relevant_documents:
            self.logger.warning(f"No relevant documents found after relevance assessment for query '{query}'")
            return {}

        # Step 8: Execute prompt decision tree
        try:
            result: dict[str, str] = self.prompt_decision_tree.execute(
                relevant_documents,
                variable,
            )
        except Exception as e:
            raise DecisionTreeError(f"Unexpected exception during prompt decision tree execution: {e}") from e

        if result['success'] == False:
            self.logger.warning(f"Prompt decision tree failed to produce a valid result for query '{query}'")
            return {}
        if not result: # Returned an empty dict
            self.logger.warning(f"Result for '{query}' is empty")
            return {}
        else:
            self.logger.info(f"Completed high level control flow for '{query}' with output: {result}")
            return result

