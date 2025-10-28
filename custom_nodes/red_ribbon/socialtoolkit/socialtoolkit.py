#!/usr/bin/env python3
"""
Socialtoolkit - Turn Law into Datasets
"""
from __future__ import annotations
from .__version__ import __version__

from dataclasses import dataclass, field, InitVar
from functools import cached_property
import logging
import os
from pathlib import Path
import sys
from typing import Any, Optional, TypeVar


# Piece-wise Pipeline
from .architecture.document_retrieval_from_websites import DocumentRetrievalFromWebsites
from .architecture.document_storage import DocumentStorage
from .architecture.top10_document_retrieval import Top10DocumentRetrieval
from .architecture.relevance_assessment import RelevanceAssessment
from .architecture.variable_codebook import VariableCodebook
from .architecture.prompt_decision_tree import PromptDecisionTree


# Integrated Pipeline
from .architecture.socialtoolkit_pipeline import SocialtoolkitPipeline


from ..utils_.configs._configs import Configs
from ..utils_.llm._llm import LLM
from ..database import DatabaseAPI
from ..logger import make_logger
from ..utils_.nodes_.node_types import Node
from ..utils_.main_.instantiate import instantiate


Class = TypeVar('Class')
ClassInstance = TypeVar('ClassInstance')


class SocialToolkitAPI:
    """
    API for accessing Socialtoolkit from ComfyUI.
    
    Attributes:
        resources (dict[str, Any]): A dictionary of instantiated classes used to run Socialtoolkit.
        configs (Configs): The configuration settings for Socialtoolkit.

    Private Attributes:
        _document_retrieval_from_websites (ClassInstance): The class instance for document retrieval from websites.
        _document_storage (ClassInstance): The class instance for document storage.
        _top10_document_retrieval (ClassInstance): The class instance for top 10 document retrieval.
        _relevance_assessment (ClassInstance): The class instance for relevance assessment.
        _llm (ClassInstance): The class instance for the LLM service.
        _variable_codebook (ClassInstance): The class instance for the variable codebook.
        _prompt_decision_tree (ClassInstance): The class instance for the prompt decision tree.
        _socialtoolkit_pipeline (ClassInstance): The class instance for the Socialtoolkit pipeline.

    Properties:
        version (str): The version of the Socialtoolkit package.
    """
    def __init__(self, resources: dict[str, ClassInstance], configs: Configs):
        self.configs = configs
        self.resources = resources

        self._logger:                          logging.Logger  = make_logger(self.__class__.__name__)
        self._llm:                              ClassInstance = self.resources["llm"]
        self._db:                               ClassInstance = self.resources["db"]

        # Piece-wise Pipeline
        self._document_retrieval_from_websites: ClassInstance = self.resources["document_retrieval_from_websites"]
        self._document_storage:                 ClassInstance = self.resources["document_storage"]
        self._top10_document_retrieval:         ClassInstance = self.resources["top10_document_retrieval"]
        self._relevance_assessment:             ClassInstance = self.resources["relevance_assessment"]
        self._variable_codebook:                ClassInstance = self.resources["variable_codebook"]
        self._prompt_decision_tree:             ClassInstance = self.resources["prompt_decision_tree"]

        # Full Pipeline
        self._socialtoolkit_pipeline:           ClassInstance = self.resources["socialtoolkit_pipeline"]

    @property
    def version(self) -> str:
        return __version__

    def document_retrieval_from_websites(self, 
                                        action: str, 
                                        domain_urls: list[str]
                                        ) -> tuple[Any, Any, Any]:
        return self._document_retrieval_from_websites.execute(action, domain_urls)


    def document_storage(self, 
                         action: str, *args, **kwargs
                        ) -> Optional[tuple[Any, Any]]:
        return self._document_storage.execute(action, *args, **kwargs)


    def top10_document_retrieval(self,
                                 action: str, *args, **kwargs
                                ) -> Optional[Any]:
        return self._top10_document_retrieval.execute(action, *args, **kwargs)


    def llm(self, action: str, *args, **kwargs) -> Optional[Any]:
        return self._llm.execute(action, *args, **kwargs)


    def  relevance_assessment(self,
                              action: str, *args, **kwargs
                             ) -> Optional[Any]:
        return self._relevance_assessment.execute(action, *args, **kwargs)


    def variable_codebook(self, 
                          action: str, *args, **kwargs
                         ) -> Optional[Any]:
        return self._variable_codebook.execute(action, *args, **kwargs)


    def prompt_decision_tree(self,
                             action: str, *args, **kwargs
                            ) -> Optional[Any]:
        """"""
        return self._prompt_decision_tree.execute(action, *args, **kwargs)


    def socialtoolkit_pipeline(self, input_data_point: str, *args, **kwargs) -> dict[str, str] | list[dict[str, str]]:
        """
        Run an input data point through the Socialtoolkit pipeline.
        
        
        the whole Socialtoolkit pipeline on the input data point.
        
        Args:
            input_data_point: The input data point, in the form of a question or statement.

        Returns:
            A dictionary, with the input datapoint under the key "input", and the output data point under the key "output".
            If the input data point is interpreted as having more than one response, 
            a list of dictionaries in the same format is returned.

        Example:
            >>> configs = Configs()
            >>> resources = {
                   "document_retrieval_from_websites": DocumentRetrievalFromWebsites,
                   "document_storage": DocumentStorage,
                   "llm": LLM,
                   "socialtoolkit_pipeline": SocialtoolkitPipeline,
                   "codebook": VariableCodebook
                }
            >>> resources = instantiate(resources, configs)
            >>> api = SocialToolkitAPI(resources, configs)
            >>> response = api.execute("socialtoolkit_pipeline", configs.input_data_point)
            >>> print(response)
            >>> {
                   "input": "What is the local sales tax in Cheyenne, WY?",
                   "output": "6%"
                }
        """
        return self._socialtoolkit_pipeline.execute(input_data_point, *args, **kwargs)

    def get_domain_urls(self, *args, **kwargs) -> list[str]:
        """
        Get domain URLs from a database or text file
        # NOTE: This function is a placeholder and should be implemented in the final version
        """
        raise NotImplementedError("get_domain_urls function not implemented yet")

    def execute(self, func_name: str, *args, **kwargs) -> Optional[Any]:
        """
        Entry point for executing functions in the SocialtoolkitAPI.
        """
        match func_name:
            case "get_domain_urls":
                return self.get_domain_urls(*args, **kwargs)
            case "document_retrieval_from_websites":
                return self.document_retrieval_from_websites(*args, **kwargs)
            case "document_storage":
                return self.document_storage(*args, **kwargs)
            case "top10_document_retrieval":
                return self.top10_document_retrieval(*args, **kwargs)
            case "llm":
                return self.llm(*args, **kwargs)
            case "relevance_assessment":
                return self.relevance_assessment(*args, **kwargs)
            case "variable_codebook":
                return self.variable_codebook(*args, **kwargs)
            case "prompt_decision_tree":
                return self.prompt_decision_tree(*args, **kwargs)
            case "socialtoolkit_pipeline":
                return self.socialtoolkit_pipeline(*args, **kwargs)
            case _:
                raise ValueError(f"Function {func_name} not found in SocialToolkitAPI")



@dataclass
class SocialToolKitResources:
    """Container for classes used to run Socialtoolkit in ComfyUI"""

    _configs: InitVar[Configs]
    resources: dict[str, ClassInstance] = field(default_factory=dict)

    def __post_init__(self, _configs):
        self.resources = instantiate({
            "document_retrieval_from_websites": DocumentRetrievalFromWebsites,
            "document_storage": DocumentStorage,
            "top10_document_retrieval": Top10DocumentRetrieval,
            "llm": LLM,
            "db": DatabaseAPI,
            "relevance_assessment": RelevanceAssessment,
            "variable_codebook": VariableCodebook,
            "prompt_decision_tree": PromptDecisionTree,
            "socialtoolkit_pipeline": SocialtoolkitPipeline,
        }, _configs, "socialtoolkit")



# Main function that can be called when using this as a script
def main():
    """Main function for Socialtoolkit module"""
    try:
        configs = Configs()
        resources = SocialToolKitResources(configs).resources
        api = SocialToolkitAPI(resources, configs)

        print("Socialtoolkit loaded successfully")
        print(f"Version: {api.version}")
        print("Running pipeline based on settings in yaml files...")
        response = api.execute("socialtoolkit_pipeline", configs.input_data_point)
        print(response)
    except Exception as e:
        print(f"Error running socialtoolkit: {e}")
        return 1
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        return 1
    else:
        print("Program completed successfully.")
        return 0

if __name__ == "__main__":
    sys.exit(main())