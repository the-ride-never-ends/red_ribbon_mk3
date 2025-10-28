
from enum import Enum
import logging
from typing import Dict, List, Any, Callable, Optional, Iterable, Literal


import networkx as nx
from pydantic import BaseModel, PositiveFloat, Field, PositiveInt, NonNegativeInt


from .variable_codebook import Variable
from ._errors import InitializationError
from ..configs_.socialtoolkit_configs import SocialtoolkitConfigs


class PromptDecisionTreeConfigs(BaseModel):
    """Configuration for Prompt Decision Tree workflow"""
    max_tokens_per_prompt: PositiveInt = 2000
    max_pages_to_concatenate: PositiveInt = 10
    max_iterations: NonNegativeInt = 5
    confidence_threshold: NonNegativeInt = Field(default=0.7, ge=0.0, le=1.0)
    context_window_size: PositiveInt = 8192  # Maximum context window size for LLM


class PromptDecisionTreeNodeType(str, Enum):
    """Types of nodes in the prompt decision tree"""
    INIT = "init"
    GET_TEXT = "get_text"
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
    edges: Optional[List[PromptDecisionTreeEdge]] = None
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

    def __init__(self, 
                 resources: dict[str, Callable], 
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

        # Extract needed services from resources
        self.logger: logging.Logger = resources['logger']
        self.llm = resources["llm"]
        self.variable_codebook = resources["variable_codebook"]

        self.logger.info("PromptDecisionTree initialized.")

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()

    def run(self, relevant_pages: List[Any], variable: Variable) -> Dict[str, Any]:
        """
        Execute the prompt decision tree and return detailed results.
        
        Args:
            relevant_pages: List of relevant document pages
            variable: Pydantic model containing a variable codebook entry 
            llm_api: LLM API instance
            
        Returns:
            Dictionary containing the output data point
        """
        self.logger.info(f"Starting prompt decision tree with {len(relevant_pages)} pages")

        # Step 1: Concatenate pages
        concatenated_pages = self._concatenate_pages(relevant_pages)

        # Step 2: Get desired data point codebook entry & prompt sequence
        # (Already provided as input parameter)

        # Step 3: Execute prompt decision tree
        result = self._execute_decision_tree(concatenated_pages, variable)

        # Step 4: Handle errors and unforeseen edge-cases if needed
        if result.get("error") and self.configs.enable_human_review:
            result = self._request_human_review(result, concatenated_pages)

        self.logger.info("Completed prompt decision tree execution")
        return result

    def execute(self, relevant_pages: List[Any], prompt_sequence: List[str]) -> str:
        """
        Public method to execute prompt decision tree
        
        Args:
            relevant_pages: List of relevant document pages
            prompt_sequence: List of prompts in the decision tree
            llm_api: LLM API instance
            
        Returns:
            str: Output data point as a string.

        Example:
            >>> relevant_pages = [{"content": "Tax rate is 5%", "title": "Sample Title", "url": "http://example.com"}]
            >>> prompt_sequence = ["What is the tax rate mentioned in the document?"]
            >>> result = prompt_decision_tree.execute(relevant_pages, prompt_sequence)
        """
        result = self.run(relevant_pages, prompt_sequence)
        return result.get("output_data_point", "")
    
    def _concatenate_pages(self, pages: List[Any]) -> str:
        """
        Concatenate pages into a single document
        
        Args:
            pages: List of pages to concatenate
            
        Returns:
            Concatenated document text
        """
        # Limit number of pages to avoid context window issues
        pages_to_use = pages[:self.configs.max_pages_to_concatenate]
        
        concatenated_text = ""
        
        for i, page in enumerate(pages_to_use):
            default_title = f"Document {i+1}"
            content = page.get("content", "")
            title = page.get("title", default_title)
            url = page.get("url", "")
            
            page_text = f"""
# {title}
## Source: {url}
## Content:
{content}
"""
            concatenated_text += page_text
            
        return concatenated_text.strip()
    
    def _execute_decision_tree(self, 
                             document_text: str, 
                             prompt_sequence: List[str], 
                             ) -> Dict[str, Any]:
        """
        Execute the prompt decision tree
        
        Args:
            document_text: Concatenated document text
            prompt_sequence: List of prompts in the decision tree
            llm_api: LLM API instance
            
        Returns:
            Dictionary containing the execution result
        """
        # Create a simplified decision tree from the prompt sequence
        decision_tree = self._create_decision_tree(prompt_sequence)
        iteration = 0
        responses = []
        output_data_point = None
        output_dict = {"success": None, "msg": None}
        try:
            # Start with the first node
            current_node = decision_tree[0]

            # Follow the decision tree until a final node is reached or max iterations is exceeded
            while not current_node.is_final and iteration < self.configs.max_iterations:
                # Generate prompt for the current node
                prompt = self._generate_node_prompt(current_node, document_text)
                
                # Get response from LLM#
                llm_response = self.llm.generate(prompt, max_tokens=self.configs.max_tokens_per_prompt)
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
            }
        else:
            output_dict.update({
                "output_data_point": output_data_point,
                "responses": responses,
                "iterations": iteration
            })
            return output_dict
    
    def _create_decision_tree(self, prompt_sequence: List[str]) -> List[PromptDecisionTreeNode]:
        """
        Create a decision tree from a prompt sequence
        
        This is a simplified implementation that creates a linear sequence of nodes.
        In a real system, this would create a proper tree structure with branches.
        
        Args:
            prompt_sequence: List of prompts
            
        Returns:
            List of nodes in the decision tree
        """
        nodes = []
        
        for i, prompt in enumerate(prompt_sequence):
            # Create a node for each prompt
            node = PromptDecisionTreeNode(
                id=f"node_{i}",
                type=PromptDecisionTreeNodeType.QUESTION,
                prompt=prompt,
                is_final=(i == len(prompt_sequence) - 1)  # Last node is final
            )
            
            # Add edges if not the last node
            if i < len(prompt_sequence) - 1:
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
        
    def _determine_next_node(self, response: str, edges: List[PromptDecisionTreeEdge]) -> str:
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
    
    def _request_human_review(self, result: Dict[str, Any], document_text: str) -> Dict[str, Any]:
        """
        Request human review for errors or low confidence results
        
        Args:
            result: Result from decision tree execution
            document_text: Document text
            
        Returns:
            Updated result after human review
        """
        if self.human_review:
            review_request = {
                "error": result.get("error"),
                "document_text": document_text,
                "responses": result.get("responses", [])
            }
            
            human_review_result = self.human_review.review(review_request)
            
            if human_review_result.get("success"):
                result["output_data_point"] = human_review_result.get("output_data_point", "")
                result["human_reviewed"] = True
                result["success"] = True
                result.pop("error", None)
        
        return result


logger = logging.getLogger(__name__)


def _make_prompt_decision_tree(
        resources: Optional[dict[str, Callable]] = None,
        configs: Optional[BaseModel] = None,
        ) -> PromptDecisionTree:
    """
    Factory function to create PromptDecisionTree instance.
    """


    resources = resources or {}
    configs = configs or SocialtoolkitConfigs()

    _resources_ = {
        "llm": resources.get("llm"),
    }

    try:
        return PromptDecisionTree(resources=_resources_, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize PromptDecisionTree: {e}") from e