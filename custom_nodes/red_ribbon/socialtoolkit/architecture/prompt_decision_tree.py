
from datetime import datetime
from enum import Enum
import logging
from typing import Dict, List, Any, Callable, Optional, Iterable, Literal
from pathlib import Path
import json


import networkx as nx
from pydantic import BaseModel, PositiveFloat, Field, PositiveInt, NonNegativeInt, NonNegativeFloat


from .dataclasses import Document, Section
from .variable_codebook import Variable, VariableCodebook
from ._errors import InitializationError
from ..configs_.socialtoolkit_configs import SocialtoolkitConfigs
from custom_nodes.red_ribbon.utils_.llm_ import LLM


class PromptDecisionTreeConfigs(BaseModel):
    """Configuration for Prompt Decision Tree workflow"""
    max_tokens_per_prompt: PositiveInt = 2000
    max_pages_to_concatenate: PositiveInt = 10
    max_iterations: NonNegativeInt = 5
    confidence_threshold: NonNegativeFloat = Field(default=0.7, ge=0.0, le=1.0)
    context_window_size: PositiveInt = 8192  # Maximum context window size for LLM
    enable_human_review: bool = True


class PromptDecisionTreeNodeType(str, Enum):
    """Types of nodes in the prompt decision tree"""
    INIT = "init"
    GET_TEXT = "get_text"
    QUESTION = "question"
    FINAL = "final"

class PromptDecisionTreeEdge(BaseModel):
    """Edge in the prompt decision tree"""
    condition: str
    next_node_id: str

class PromptDecisionTreeNode(BaseModel):
    """Node in the prompt decision tree"""
    id: str
    type: PromptDecisionTreeNodeType
    prompt: str
    edges: Optional[list[PromptDecisionTreeEdge]] = None
    is_final: bool = False



# TODO: Move these to yaml files.
_SECTION_PROMPT = """
'section' key is a string that represents the section's title (including numeric labels).
If it's not given, use `null`.
"""
_COMMENT_PROMPT = """
'comment' key is an optional one sentence explanation of how you determined the value.
If you think it's not necessary to explain, use `null`.
"""

def _init_base_graph(**kwargs) -> nx.DiGraph:
    """Initialize base directed graph for prompt decision tree"""
    return nx.DiGraph(
        SECTION_PROMPT=_SECTION_PROMPT,
        COMMENT_PROMPT=_COMMENT_PROMPT,
        **kwargs
    )

def _starts_with_yes(response: str) -> bool:
    """Determine if response indicates a 'yes' answer. Case insensitive."""
    return response.strip().lower().startswith("yes")

def _starts_with_no(response: str) -> bool:
    """Determine if response indicates a 'no' answer. Case insensitive."""
    return response.strip().lower().startswith("no")

def _does_not_start_with_no(response: str) -> bool:
    """Determine if response does not indicate a 'no' answer. Case insensitive."""
    return not _starts_with_no(response)







class PromptDecisionTree:
    """
    Prompt Decision Tree system.
    Executes a decision tree of prompts to extract information from documents
    """

    def __init__(self, *,
                 resources: dict[str, Any], 
                 configs: PromptDecisionTreeConfigs
                ):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including services
            configs: Configuration for Prompt Decision Tree
        """
        self.resources = resources
        self.configs = configs

        self.max_tokens_per_prompt: int  = configs.max_tokens_per_prompt
        self.max_iterations:        int  = configs.max_iterations
        self.human_review:          bool = configs.enable_human_review
        self.review_folder:         Path = configs.review_folder

        self.logger:                 logging.Logger        = resources['logger']
        self.llm:                    LLM                   = resources["llm"]
        self.variable_codebook:      VariableCodebook      = resources["variable_codebook"]
        self.extract_text_from_html: Callable              = resources["extract_text_from_html"]

        self.logger.info("PromptDecisionTree initialized.")

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()

    def run(self, relevant_sections: list[Section], variable: Variable) -> dict[str, Any]:
        """
        Execute the prompt decision tree and return detailed results.
        
        Args:
            relevant_sections: List of relevant document pages
            variable: Pydantic model containing a variable codebook entry 
            llm_api: LLM API instance
            
        Returns:
            Dictionary containing the output data point.

        Raises:
            TypeError: If relevant_sections is not a list of Sections or variable is not Variable
            ValueError: If relevant_sections is empty or contains invalid Section data, or if variable data is invalid
        """
        self.logger.info(f"Starting prompt decision tree with {len(relevant_sections)} pages")
        
        # Validate inputs
        if not isinstance(relevant_sections, list):
            raise TypeError(f"relevant_sections must be a list, got {type(relevant_sections).__name__}")
        if not isinstance(variable, Variable):
            raise TypeError(f"variable must be an instance of Variable, got {type(variable).__name__}")
        else:
            try:
                Variable.model_validate(variable)
            except Exception as e:
                raise ValueError(f"Invalid Variable data: {e}") from e

        if not relevant_sections:
            raise ValueError("relevant_sections cannot be empty")

        for idx, section in enumerate(relevant_sections):
            if not isinstance(section, Section):
                raise TypeError(f"section {idx} must be type Section, got {type(section).__name__}")
            else:
                try:
                    Section.model_validate(section)
                except Exception as e:
                    raise ValueError(f"Invalid Section data for section {idx}: {e}") from e

        # Step 1: Concatenate pages
        concatenated_pages = self._format_pages(relevant_sections)

        # Step 2: Execute prompt decision tree
        result = self._execute_decision_tree(concatenated_pages, variable)

        # Step 3: Flag errors for review if enabled
        if result['success'] == False:
            if self.configs.enable_human_review:
                self._save_for_review(result, concatenated_pages)

        self.logger.info(f"Completed prompt decision tree execution. Result Status: {result['success']}")
        return result

    def execute(self, relevant_sections: list[Any], variable: Variable) -> dict:
        """
        Public method to execute prompt decision tree
        
        Args:
            relevant_sections: List of relevant document pages
            variable: Pydantic model containing a variable codebook entry 
            llm_api: LLM API instance
            
        Returns:
            dict: Result dictionary. Contains the following keys:
                - success: bool indicating if execution was successful, or false if an error occurred
                - msg: (str) Message describing the result or error
                - document_text: list[Section] Concatenated document text used for decision tree
                - output_data_point: Extracted data point as a string
                - responses: List of LLM responses at each node
                - iterations: Number of iterations taken to reach final node

        Example:
            >>> relevant_sections = [{"content": "Property tax rate is 5%", "title": "Sample Title", "url": "http://example.com"}]
            >>> variable = = Variable(
            ...     label="Property Tax Rate",
            ...     item_name="property_tax_rate",
            ...     description="Annual property tax assessment rate",
            ...     units="Double (Percent)",
            ...     assumptions=Assumptions(business=BusinessAssumptions()),
            ...     prompt_decision_tree=PromptDecisionTree(nodes=[], edges=[])
            ... )
            >>> result = prompt_decision_tree.execute(relevant_sections, variable)
        """
        return self.run(relevant_sections, variable)
    
    def _format_pages(self, pages: list[Section]) -> str:
        """
        Format pages into a single document
        
        Args:
            pages: List of pages to concatenate
            
        Returns:
            Concatenated document text
        """
        concatenated_text = ""

        for idx, page in enumerate(pages, start=1):
            title = getattr(page, "title", f"Relevant Document {idx}")
            citation = page.bluebook_citation
            assert citation is not None, "Bluebook citation is required in each page."

            html = page.html
            assert html is not None, "HTML content is required in each page."
            content = self.extract_text_from_html(html)
            
            page_text = f"""
<document {idx}>
# {title}
## Citation: {citation}
## Content:
{content}
</doc>
"""
            concatenated_text += page_text

        # Tokenize the text to see if it's within context window
        tokens = self.llm.tokenizer.tokenize(concatenated_text)  # type: ignore[attr-defined]
        if tokens > self.configs.context_window_size:
            self.logger.warning(
                f"Concatenated document exceeds context window size "
                f"({len(tokens)} tokens > {self.configs.context_window_size} tokens). "
                f"Truncating to fit."
            )
            # Truncate to fit context window
            truncated_tokens = tokens[:self.configs.context_window_size]
            concatenated_text = self.llm.tokenizer.decode(truncated_tokens)  # type: ignore[attr-defined]

        return concatenated_text.strip()
    
    def _execute_decision_tree(self, 
                             document_text: str, 
                             variable: list[str], 
                             ) -> dict[str, Any]:  # type: ignore[return]
        """
        Execute the prompt decision tree

        Args:
            document_text: Concatenated document text
            variable: List of prompts in the decision tree
            llm_api: LLM API instance

        Returns:
            dict: Result dictionary. Contains the following keys:
                - success: bool indicating if execution was successful, or false if an error occurred
                - msg: (str) Message describing the result or error
                - document_text: list[Section] Concatenated document text used for decision tree
                - output_data_point: Extracted data point as a string
                - responses: List of LLM responses at each node
                - iterations: Number of iterations taken to reach final node
        """
        # Create a simplified decision tree from the prompt sequence
        decision_tree = self._create_decision_tree(variable)
        iteration = 0
        responses = []
        output_data_point = None
        output_dict: dict[str, Any] = {"success": None, "msg": None, "document_text": None}
        try:
            # Start with the first node
            current_node = decision_tree[0]

            # Follow the decision tree until a final node is reached or max iterations is exceeded
            while not current_node.is_final and iteration < self.max_iterations:
                # Generate prompt for the current node
                prompt = self._generate_node_prompt(current_node, document_text)

                # Get response from LLM#
                llm_response = self.llm.generate(prompt, max_tokens=self.max_tokens_per_prompt)  # type: ignore[attr-defined]
                responses.append({
                    "node_id": current_node.id,
                    "prompt": prompt,
                    "response": llm_response
                })
                
                # Determine next node based on response
                if current_node.edges:
                    next_node_id = self._determine_next_node(llm_response, current_node.edges)
                    current_node = next(
                        (node for node in decision_tree if node.id == next_node_id), 
                        decision_tree[-1]  # Default to the last node if not found
                    )
                else:
                    # No edges, move to the next node in sequence
                    node_index = decision_tree.index(current_node)
                    if node_index + 1 < len(decision_tree):
                        current_node = decision_tree[node_index + 1]
                    else:
                        # End of sequence, mark as final
                        current_node.is_final = True
                
                iteration += 1
            
            # Process the final response
            final_response = responses[-1]["response"] if responses else ""
            output_data_point = self._extract_output_data_point(final_response)
            
            output_dict = {
                "success": True,
                "msg": "Successfully executed decision tree",
            }
            
        except Exception as e:
            self.logger.exception(f"Error executing decision tree: {e}")
            output_dict = {
                "success": False,
                "msg": f"{type(e).__name__}: {str(e)}",
                "document_text": document_text,
            }
            return output_dict
        else:
            output_dict.update({
                "output_data_point": output_data_point,
                "responses": responses,
                "iterations": iteration
            })
            return output_dict

    def _create_decision_tree(self, variable: list[str]) -> list[PromptDecisionTreeNode]:
        """
        Create a decision tree from a prompt sequence
        
        This is a simplified implementation that creates a linear sequence of nodes.
        In a real system, this would create a proper tree structure with branches.
        
        Args:
            variable: List of prompts
            
        Returns:
            List of nodes in the decision tree
        """
        nodes = []
        
        for i, prompt in enumerate(variable):
            # Create a node for each prompt
            node = PromptDecisionTreeNode(
                id=f"node_{i}",
                type=PromptDecisionTreeNodeType.QUESTION,
                prompt=prompt,
                is_final=(i == len(variable) - 1)  # Last node is final
            )
            
            # Add edges if not the last node
            if i < len(variable) - 1:
                node.edges = [
                    PromptDecisionTreeEdge(
                        condition="default",
                        next_node_id=f"node_{i+1}"
                    )
                ]
                
            nodes.append(node)
        
        return nodes
    
    def _generate_node_prompt(self, node: PromptDecisionTreeNode, document_text: str) -> str:
        """
        Generate a prompt for a node in the decision tree
        
        Args:
            node: Current node in the decision tree
            document_text: Document text
            
        Returns:
            Prompt for the node
        """
        # Truncate document text if too long
        max_doc_length = self.configs.context_window_size - 500  # Reserve space for instructions
        if len(document_text) > max_doc_length:
            document_text = document_text[:max_doc_length] + "..."
            
        prompt = f"""
You are an expert tax researcher assisting with data extraction from official documents.
Please carefully analyze the following documents to answer this specific question:

QUESTION: {node.prompt}

DOCUMENTS:
{document_text}

Based solely on the information provided in these documents, please answer the question above.
If the answer is explicitly stated in the documents, provide the exact information along with its source.
If the answer requires interpretation, explain your reasoning clearly.
If the information is not available in the documents, respond with "Information not available in the provided documents."

Your answer should be concise, factual, and directly address the question.
"""
        return prompt
        
    def _determine_next_node(self, response: str, edges: list[PromptDecisionTreeEdge]) -> str:
        """
        Determine the next node based on the response
        
        This is a simplified implementation that just follows the default edge.
        In a real system, this would analyze the response to determine the path.
        
        Args:
            response: LLM response
            edges: List of edges from the current node
            
        Returns:
            ID of the next node
        """
        raise NotImplementedError("Edge condition evaluation not implemented.")

    def _extract_output_data_point(self, response: str) -> str:
        """
        Extract the output data point from the final response
        
        Args:
            response: Final LLM response
            
        Returns:
            Extracted output data point
        """
        # Look for patterns like "X%" or "X percent"
        import re
        
        # Try to find percentage patterns
        percentage_match = re.search(r'(\d+(?:\.\d+)?)\s*%', response)
        if percentage_match:
            return percentage_match.group(0)
            
        percentage_word_match = re.search(r'(\d+(?:\.\d+)?)\s+percent', response, re.IGNORECASE)
        if percentage_word_match:
            value = percentage_word_match.group(1)
            return f"{value}%"
            
        # Look for specific statements about rates
        rate_match = re.search(r'rate\s+is\s+(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        if rate_match:
            value = rate_match.group(1)
            return f"{value}%"
            
        # If no specific patterns are found, return a cleaned up version of the response
        # Limit to 100 characters for brevity
        cleaned_response = response.strip()
        if len(cleaned_response) > 100:
            cleaned_response = cleaned_response[:97] + "..."
            
        return cleaned_response
    
    def _save_for_review(self, result: dict[str, Any], document_text: str) -> None:
        """
        Save the result for human review if enabled.
        
        Args:
            result: Result from decision tree execution
            document_text: Document text
        """
        output_dict = {
            ""
            "error": result.get("msg", "Unknown Error"),
            "document_text": document_text,
            "responses": result.get("responses", [])
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        review_file = self.review_folder / f"review_request_{timestamp}.json"

        try:
            review_request_json = json.dumps(output_dict, indent=2)
            with open(review_file, 'w') as f:
                f.write(review_request_json)
        except json.JSONDecodeError as e:
            raise IOError(f"Failed to serialize review request to JSON: {e}") from e
        except Exception as e:
            raise IOError(f"Unexpected error while writing review request to JSON: {e}") from e
        else:
            self.logger.info(f"Saved review request to '{review_file}'")

