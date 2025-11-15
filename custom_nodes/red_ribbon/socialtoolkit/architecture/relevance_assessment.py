import logging
import traceback
import re
from typing import Any, Callable, Dict, Optional


from pydantic import BaseModel, Field, ValidationError


from custom_nodes.red_ribbon.utils_ import LLM
from .variable_codebook import Variable
from .dataclasses import Document, Section
from ._errors import LLMError, InitializationError, RelevanceAssessmentError


logger = logging.getLogger(__name__)


class RelevanceAssessmentConfigs(BaseModel):
    """Configuration for Relevance Assessment workflow"""
    criteria_threshold: float = 0.7  # Minimum relevance score threshold
    max_retries: int = 3  # Maximum number of retry attempts for LLM API calls
    max_citation_length: int = 500  # Maximum length of text citations
    use_hallucination_filter: bool = True  # Whether to filter for hallucinations


class RelevanceAssessment:
    """
    Relevance Assessment system based on mermaid flowchart in README.md
    Evaluates document relevance using LLM assessments
    """

    def __init__(self, *, 
                 resources: dict[str, Any],
                 configs: RelevanceAssessmentConfigs
                 ):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including services
            configs: Configuration for Relevance Assessment
        """
        self.resources = resources
        self.configs = configs
        self.criteria_threshold: float = configs.criteria_threshold

        self.logger: logging.Logger = resources['logger']
        self.llm: LLM = resources["llm"]
        self.cited_page_extractor: Callable = resources["cited_page_extractor"]

        self.logger.info("RelevanceAssessment initialized successfully.")

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()

    def execute(self,
               potentially_relevant_docs: list[Document],
               variable: Variable,
               llm: Optional[LLM] = None
              ) -> dict[str, Document]:
        """
        Wrapper to execute relevance assessment in ComfyUI node
        
        Args:
            potentially_relevant_docs: (list[Document]) list of potentially relevant documents
            variable: (Variable) Variable pydantic model containing definition and description of desired variable.
            llm: Optional LLM API instance

        Returns:
            Dictionary containing relevant documents and page numbers
        
        Raises:
            LLMError: If LLM API instance is not available as an argument or a class attribute
            ValueError: If llm is not provided and not available in resources
            RuntimeError: If relevance assessment fails unexpectedly
        """
        self.logger.info("Starting relevance assessment execution via execute() method")

        if not isinstance(potentially_relevant_docs, list):
            raise TypeError(f"potentially_relevant_docs must be a list, got {type(potentially_relevant_docs).__name__}")
        if not potentially_relevant_docs:
            raise ValueError("potentially_relevant_docs cannot be empty")
        if not isinstance(variable, Variable):
            raise TypeError(f"variable must be a Variable instance, got {type(variable).__name__}")

        self.logger.debug(f"Potentially relevant docs count: {len(potentially_relevant_docs)}")

        if llm is not None and not isinstance(llm, LLM):
            raise TypeError(f"llm must be an instance of LLM if provided, got {type(llm).__name__}")
        else:
            if self.llm is None:
                raise LLMError("LLM API instance was not provided as an argument of 'execute' or during initialization.")

        try:
            results = self.run(potentially_relevant_docs, variable, llm or self.llm)
        except Exception as e:
            self.logger.exception(f"Relevance assessment execution failed: {e}")
            raise RuntimeError(f"Relevance assessment execution failed: {e}.") from e
        return results


    def run(self, 
            potentially_relevant_docs: list[Document], 
            variable: Variable,
            llm: LLM
            ) -> dict[str, Document]:
        """
        Execute the relevance assessment flow.

        Args:
            potentially_relevant_docs: list of potentially relevant documents
            variable: Variable definition and description
            llm: LLM API instance

        Returns:
            Dictionary containing relevant documents and page numbers.
            Keys are the following:
                - "relevant_pages": list[str] list of relevant document contents
                - "relevant_doc_ids": (list[str]) list of document IDs deemed relevant
                - "page_numbers": (int) dict mapping document IDs to lists of relevant page numbers
                - "relevance_scores": (list[dict[str, Any]]) list of dicts with relevance scores and metadata

        Raises:
            TypeError: If input types are incorrect
            ValueError: If input values are invalid
        """
        input_docs: list[Document] = potentially_relevant_docs

        # Input validation
        if not isinstance(input_docs, list):
            raise TypeError(f"potentially_relevant_docs must be a list, got {type(input_docs).__name__}")
        if not isinstance(variable, Variable):
            raise TypeError(f"variable must be a Variable instance, got {type(variable).__name__}")
        if not isinstance(llm, LLM):
            raise TypeError(f"llm must be an LLM instance, got {type(llm).__name__}")

        if not input_docs:
            raise ValueError("potentially_relevant_docs cannot be empty")
        else:
            for idx, doc in enumerate(input_docs):
                if not isinstance(doc, Document):
                    raise TypeError(f"Each item {idx} must be a Document instance, got {type(doc).__name__}")

        self.logger.info(f"Starting relevance assessment for {len(potentially_relevant_docs)} documents")

        # Step 1: Assess document relevance
        assessment_results: list[Document] = self._assess_document_relevance(
            potentially_relevant_docs, 
            variable, 
            llm
        )

        # Step 2: Filter for hallucinations if configured
        if self.configs.use_hallucination_filter:
            assessment_results = self._filter_hallucinations(assessment_results, llm)
        
        # Step 3: Score relevance
        relevance_scores = self._score_relevance(assessment_results, potentially_relevant_docs)
        
        # Step 4: Apply threshold to separate relevant from irrelevant
        relevant_pages, discarded_pages = self._apply_threshold(relevance_scores)

        # Step 5: Extract page numbers
        page_numbers = self._extract_page_numbers(relevant_pages)

        # Step 6: Extract cited pages
        relevant_pages_content = self._extract_cited_pages(
            potentially_relevant_docs, page_numbers
        )

        self.logger.info(f"Completed relevance assessment: {len(relevant_pages_content)} relevant pages")

        sections: list[Section] = []
        for idx, page in enumerate(relevant_pages_content):
            try:
                section = Section(**page)
            except ValidationError as e:
                raise ValueError(f"Error creating Section object for page {idx}: {e}") from e
            sections.append(section)


        sections = [
            Section(**page) for page in relevant_pages
        ]

        return Document(section_list=sections)


    def assess(self, 
               potentially_relevant_docs: list[Document], 
              prompt_sequence: list[str], 
              llm: LLM
              ) -> list[Document]:
        """
        Public method to assess document relevance
        
        Args:
            potentially_relevant_docs: list of potentially relevant documents
            prompt_sequence: list of prompts to use for assessment
            llm: LLM API instance
            
        Returns:
            list of relevant documents
        """
        # Get variable definition from prompt sequence
        variable = {
            "prompt_sequence": prompt_sequence,
            "description": "Tax information for business operations"  # Default description if not available
        }
        
        result = self.run(potentially_relevant_docs, variable, llm)
        return result["relevant_pages"]
    
    def _assess_document_relevance(self, docs: list[Document], 
                                 variable: Variable, 
                                 llm: Any) -> list[dict[str, Any]]:
        """
        Assess document relevance using LLM
        
        Args:
            docs: list of documents to assess
            variable: Variable definition and description
            llm: LLM API instance
            
        Returns:
            list of assessment results
        """
        assessment_results = []
        
        for doc in docs:
            # Create assessment prompt
            assessment_prompt = self._create_assessment_prompt(doc, variable)
            
            # Get LLM assessment
            try:
                llm_response = llm.generate(assessment_prompt, max_tokens=1000)
                
                # Parse assessment results
                assessment = self._parse_assessment(llm_response, doc)
                assessment_results.append(assessment)
                
            except Exception as e:
                self.logger.error(f"Error assessing document {doc.get('id')}: {e}")
                # Add failed assessment
                assessment_results.append({
                    "doc_id": doc.get("id"),
                    "relevant": False,
                    "confidence": 0.0,
                    "citation": "",
                    "error": str(e)
                })
        
        return assessment_results
    
    def _filter_hallucinations(self, assessments: list[dict[str, Any]], 
                             llm: Any) -> list[dict[str, Any]]:
        """
        Filter for hallucinations in LLM assessments
        
        Args:
            assessments: list of assessment results
            llm: LLM API instance
            
        Returns:
            Filtered list of assessment results
        """
        filtered_assessments = []
        
        for assessment in assessments:
            # Skip already irrelevant assessments
            if not assessment.get("relevant", False):
                filtered_assessments.append(assessment)
                continue
                
            # Create hallucination check prompt
            hallucination_prompt = self._create_hallucination_prompt(assessment)
            
            try:
                # Get LLM hallucination check
                hallucination_response = llm.generate(hallucination_prompt, max_tokens=500)
                
                # Parse hallucination check
                is_hallucination = self._parse_hallucination_check(hallucination_response)
                
                if is_hallucination:
                    # Downgrade relevance for hallucinations
                    assessment["relevant"] = False
                    assessment["confidence"] = 0.0
                    assessment["hallucination"] = True
                
                filtered_assessments.append(assessment)
                
            except Exception as e:
                self.logger.error(f"Error checking hallucination for document {assessment.get('doc_id')}: {e}")
                # Keep original assessment in case of error
                filtered_assessments.append(assessment)
        
        return filtered_assessments
    
    def _score_relevance(self, 
                         assessments: list[dict[str, Any]], 
                       docs: list[Document]
                       ) -> list[dict[str, Document]]:
        """
        Score relevance based on LLM assessments
        
        Args:
            assessments: list of assessment results
            docs: list of original documents
            
        Returns:
            list of documents with relevance scores
        """
        # Create a dict mapping document ID to original document
        doc_map = {doc.get("id"): doc for doc in docs}
        
        relevance_scores = []
        
        for assessment in assessments:
            doc_id = assessment.get("doc_id")
            doc = doc_map.get(doc_id)
            
            if not doc:
                self.logger.warning(f"Document not found for ID: {doc_id}")
                continue
                
            # Calculate relevance score based on LLM confidence
            relevance_score = {
                "doc_id": doc_id,
                "page_number": assessment.get("page_number", 1),  # Default to page 1 if not specified
                "score": assessment.get("confidence", 0.0),
                "relevant": assessment.get("relevant", False),
                "citation": assessment.get("citation", ""),
                "content": doc.get("content", "")
            }
            
            relevance_scores.append(relevance_score)
        
        return relevance_scores
    
    def _apply_threshold(self, relevance_scores: list[dict[str, Any]]) -> tuple:
        """
        Apply threshold to relevance scores
        
        Args:
            relevance_scores: list of documents with relevance scores
            
        Returns:
            Tuple of (relevant_pages, discarded_pages)
        """
        relevant_pages = []
        discarded_pages = []
        
        for score in relevance_scores:
            if score.get("score", 0.0) >= self.criteria_threshold:
                relevant_pages.append(score)
            else:
                discarded_pages.append(score)
        
        return relevant_pages, discarded_pages
    
    def _extract_page_numbers(self, relevant_pages: list[dict[str, Any]]) -> dict[str, list[int]]:
        """
        Extract page numbers from relevant pages
        
        Args:
            relevant_pages: list of relevant pages
            
        Returns:
            Dictionary mapping document IDs to lists of page numbers
        """
        page_numbers: dict[str, list[int]] = {}
        
        for page in relevant_pages:
            doc_id = page.get("doc_id")
            page_number = page.get("page_number", 1)  # Default to page 1 if not specified
            
            if doc_id not in page_numbers:
                page_numbers[doc_id] = []
                
            page_numbers[doc_id].append(page_number)
        
        return page_numbers
    
    def _extract_cited_pages(self, 
                            docs: list[Document], 
                            page_numbers: dict[str, list[int]]
                           ) -> list[dict[str, Document]]:
        """
        Extract cited pages from documents
        
        Args:
            docs: list of original documents
            page_numbers: Dictionary mapping document IDs to lists of page numbers
            
        Returns:
            list of relevant page contents
        """
        # If cited page extractor service is available, use it
        if self.cited_page_extractor:
            return self.cited_page_extractor.extract(docs, page_numbers)

        # Fallback implementation
        cited_pages = []
        
        # Create a dict mapping document ID to original document
        doc_map = {doc.get("id"): doc for doc in docs}
        
        for doc_id, page_nums in page_numbers.items():
            doc = doc_map.get(doc_id)
            
            if not doc:
                self.logger.warning(f"Document not found for ID: {doc_id}")
                continue
                
            # Extract content for each page
            for page_num in page_nums:
                # In this simplified implementation, we assume the whole document is the content
                # In a real system, this would extract specific pages from multi-page documents
                cited_pages.append({
                    "doc_id": doc_id,
                    "page_number": page_num,
                    "content": doc.get("content", ""),
                    "title": doc.get("title", ""),
                    "url": doc.get("url", "")
                })
        
        return cited_pages
    
    def _create_assessment_prompt(self, doc: dict[str, Any], 
                                variable: dict[str, Any]) -> str:
        """
        Create assessment prompt for document relevance
        
        Args:
            doc: Document to assess
            variable: Variable definition and description
            
        Returns:
            Assessment prompt
        """
        # Get document content
        content = doc.get("content", "")[:5000]  # Limit content length
        
        # Get variable information
        description = variable.get("description", "")
        prompt_sequence = variable.get("prompt_sequence", [])
        
        # Create assessment prompt
        # TODO: We can get the confidence from the LLM response directly. No need to waste tokens asking for it.
        prompt = f"""
You are a document relevance assessor. Your task is to determine if the following document is relevant to the given information need.

Information Need: {description}

Key Questions:
{chr(10).join([f"- {p}" for p in prompt_sequence])}

Document Content:
{content}

Please assess the document's relevance to the information need based on the following criteria:
1. Does the document contain information directly related to the information need?
2. Does the document provide sufficient detail to answer at least one of the key questions?
3. Is the document from a credible source?

Provide your assessment in the following format:
RELEVANT: [Yes/No]
CONFIDENCE: [0.0-1.0]
CITATION: [Most relevant text snippet from the document that supports your assessment]
REASONING: [Brief explanation for your assessment]
"""
        return prompt

    def _parse_assessment(self, llm_response: str, doc: dict[str, Any]) -> dict[str, Any]:
        """
        Parse LLM assessment response
        
        Args:
            llm_response: LLM response text
            doc: Original document
            
        Returns:
            Parsed assessment
        """
        assessment = {
            "doc_id": doc.get("id"),
            "relevant": False,
            "confidence": 0.0,
            "citation": "",
            "reasoning": ""
        }
        
        try:
            # Parse relevant
            if "RELEVANT: Yes" in llm_response:
                assessment["relevant"] = True
                
            # Parse confidence
            confidence_match = re.search(r"CONFIDENCE: (0\.\d+|1\.0)", llm_response)
            if confidence_match:
                assessment["confidence"] = float(confidence_match.group(1))
                
            # Parse citation
            citation_match = re.search(r"CITATION: (.*?)(?=REASONING:|$)", llm_response, re.DOTALL)
            if citation_match:
                citation = citation_match.group(1).strip()
                # Truncate if necessary
                if len(citation) > self.configs.max_citation_length:
                    citation = citation[:self.configs.max_citation_length] + "..."
                assessment["citation"] = citation
                
            # Parse reasoning
            reasoning_match = re.search(r"REASONING: (.*?)$", llm_response, re.DOTALL)
            if reasoning_match:
                assessment["reasoning"] = reasoning_match.group(1).strip()
                
        except Exception as e:
            self.logger.error(f"Error parsing assessment: {e}")
        
        return assessment
    
    def _create_hallucination_prompt(self, assessment: dict[str, Any]) -> str:
        """
        Create prompt to check for hallucinations
        
        Args:
            assessment: Document assessment
            
        Returns:
            Hallucination check prompt
        """
        citation = assessment.get("citation", "")
        
        prompt = f"""
You are a fact-checking assistant. Your task is to analyze the following excerpt and determine if it directly addresses tax rates, specific tax information, or tax regulations.

Text excerpt:
{citation}

Please analyze this text and determine if it contains SPECIFIC information about tax rates, tax percentages, or tax regulations.
Answer with "HALLUCINATION: Yes" if the text does NOT contain specific tax information.
Answer with "HALLUCINATION: No" if the text DOES contain specific tax information.

Provide a brief explanation for your decision.
"""
        return prompt
    
    def _parse_hallucination_check(self, hallucination_response: str) -> bool:
        """
        Parse hallucination check response
        
        Args:
            hallucination_response: LLM response text
            
        Returns:
            True if hallucination detected, False otherwise
        """
        return "HALLUCINATION: Yes" in hallucination_response


def make_relevance_assessment(
        resources: dict[str, Any] = {}, 
        configs: RelevanceAssessmentConfigs = RelevanceAssessmentConfigs()
        ) -> RelevanceAssessment:
    """
    Factory function to create RelevanceAssessment instance
    
    Args:
        resources: Dictionary of resources including services
        configs: Configuration for Relevance Assessment
        
    Returns:
        RelevanceAssessment instance
    """
    _resources = {
        "llm": resources.get("llm"),
        "logger": resources.get("logger", logger),
        "cited_page_extractor": resources.get("cited_page_extractor"),
    }
    for key in resources:
        if key not in _resources:
            raise KeyError(f"Unexpected key in resources dictionary: {key}")
    return RelevanceAssessment(resources=_resources, configs=configs)
