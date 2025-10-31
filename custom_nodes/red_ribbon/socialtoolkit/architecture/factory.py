
from typing import Callable

from ._errors import InitializationError
from ..configs_.socialtoolkit_configs import SocialtoolkitConfigs
from .socialtoolkit_pipeline import SocialtoolkitPipeline
from .document_retrieval_from_websites import DocumentRetrievalFromWebsites
from .document_storage import DocumentStorage
from .top10_document_retrieval import Top10DocumentRetrieval
from .relevance_assessment import RelevanceAssessment
from .prompt_decision_tree import PromptDecisionTree
from .variable_codebook import VariableCodebook

from custom_nodes.red_ribbon.utils_ import make_logger
from custom_nodes.red_ribbon.utils_.database import DatabaseAPI


def make_variable_codebook(
    resources: dict[str, Callable] = {},
    configs: SocialtoolkitConfigs = lambda: SocialtoolkitConfigs(),
) -> VariableCodebook:
    """Factory function to create VariableCodebook instance"""
    try:
        return VariableCodebook(resources=resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize VariableCodebook: {e}") from e


def make_document_retrieval_from_websites(
    resources: dict[str, Callable] = {},
    configs: SocialtoolkitConfigs = lambda: SocialtoolkitConfigs(),
) -> DocumentRetrievalFromWebsites:
    """Factory function to create DocumentRetrievalFromWebsites instance"""
    try:
        return DocumentRetrievalFromWebsites(resources=resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize DocumentRetrievalFromWebsites: {e}") from e


def make_document_storage(
    resources: dict[str, Callable] = {},
    configs: SocialtoolkitConfigs = lambda: SocialtoolkitConfigs(),
) -> DocumentStorage:
    """Factory function to create DocumentStorage instance"""
    try:
        return DocumentStorage(resources=resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize DocumentStorage: {e}") from e


def make_top10_document_retrieval(
    resources: dict[str, Callable] = {},
    configs: SocialtoolkitConfigs = lambda: SocialtoolkitConfigs(),
) -> Top10DocumentRetrieval:
    """Factory function to create Top10DocumentRetrieval instance"""
    try:
        return Top10DocumentRetrieval(resources=resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize Top10DocumentRetrieval: {e}") from e


def make_relevance_assessment(
    resources: dict[str, Callable] = {},
    configs: SocialtoolkitConfigs = lambda: SocialtoolkitConfigs(),
) -> RelevanceAssessment:
    """Factory function to create RelevanceAssessment instance"""
    try:
        return RelevanceAssessment(resources=resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize RelevanceAssessment: {e}") from e


def make_prompt_decision_tree(
    resources: dict[str, Callable] = {},
    configs: SocialtoolkitConfigs = lambda: SocialtoolkitConfigs(),
) -> PromptDecisionTree:
    """Factory function to create PromptDecisionTree instance"""
    try:
        return PromptDecisionTree(resources=resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize PromptDecisionTree: {e}") from e


def make_socialtoolkit_pipeline(
    resources: dict[str, Callable] = {},
    configs: SocialtoolkitConfigs = lambda: SocialtoolkitConfigs(),
) -> SocialtoolkitPipeline:
    """Factory function to create SocialtoolkitPipeline instance"""

    common_resources = {
        "llm": resources.get("llm"),
        "db": resources.get("db"),
        "logger": resources.get("logger", make_logger("socialtoolkit_pipeline")),
    }

    try:
        _resources = {
            "document_retrieval": resources.get("document_retrieval", make_document_retrieval_from_websites(resources=common_resources, configs=configs)),
            "document_storage": resources.get("document_storage", make_document_storage(resources=common_resources, configs=configs)),
            "top10_document_retrieval": resources.get("top10_document_retrieval", make_top10_document_retrieval(resources=common_resources, configs=configs)),
            "relevance_assessment": resources.get("relevance_assessment", make_relevance_assessment(resources=common_resources, configs=configs)),
            "prompt_decision_tree": resources.get("prompt_decision_tree", make_prompt_decision_tree(resources=common_resources, configs=configs)),
            "variable_codebook": resources.get("variable_codebook", make_variable_codebook(resources=common_resources, configs=configs)),
        }
    except InitializationError as e:
        raise e
    except Exception as e:
        raise InitializationError(f"Unexpected error initializing dependencies for SocialtoolkitPipeline: {e}") from e

    try:
        return SocialtoolkitPipeline(_resources, configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize SocialtoolkitPipeline: {e}") from e
