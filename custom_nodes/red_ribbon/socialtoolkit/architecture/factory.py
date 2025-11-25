
from typing import Any, Callable


from pydantic import BaseModel, ValidationError


from ..configs_.socialtoolkit_configs import SocialtoolkitConfigs
from .socialtoolkit_pipeline import SocialtoolkitPipeline
from .document_retrieval_from_websites import DocumentRetrievalFromWebsites, DocumentRetrievalConfigs
from .document_storage import DocumentStorage, DocumentStorageConfigs
from .top10_document_retrieval import Top10DocumentRetrieval, Top10DocumentRetrievalConfigs
from .relevance_assessment import RelevanceAssessment, RelevanceAssessmentConfigs
from .prompt_decision_tree import PromptDecisionTree, PromptDecisionTreeConfigs
from .variable_codebook import VariableCodebook, VariableCodebookConfigs


from custom_nodes.red_ribbon.utils_ import (
   configs, Configs, make_logger, logger, make_duckdb_database, make_llm, get_cid
)
from custom_nodes.red_ribbon._custom_errors import (
    ResourceError,
    ConfigurationError,
    InitializationError,
)


def _validate_configs(configs: BaseModel, name: str) -> None:
    if not isinstance(name, str):
        raise TypeError(f"Name must be a string, got {type(name).__name__}")
    if not isinstance(configs, BaseModel):
        raise TypeError(f"Configs for {name} must be a Pydantic BaseModel instance, got {type(configs).__name__}")

    try:
        configs.model_validate(configs.model_dump())
    except ValidationError as e:
        raise ConfigurationError(f"Invalid configuration object for {name}: {e}") from e

def _validate_resources(resources: dict[str, Any], name: str) -> None:
    if not isinstance(resources, dict):
        raise TypeError(f"Resources for {name} must be a dictionary, got {type(resources).__name__}")


def _initialize(Class: Callable, configs: BaseModel, resources: dict[str, Any]) -> Any:
    """
    Generic initializer for classes with resources and configs
    
    Args:
        Class: The class to initialize
        configs: A Pydantic BaseModel instance of class-specific configurations
        resources: Dictionary of resources for the class

    Returns:
        An instance of the specified class

    Raises:
        TypeError: If configs is not a Pydantic BaseModel or resources is not a dictionary.
        ConfigurationError: If the provided configurations are missing or invalid.
        ResourceError: If there is an error creating common resources or dependencies.
        InitializationError: If there is an unexpected error initializing the class itself.
    """
    class_name = Class.__name__
    _validate_configs(configs, class_name)
    _validate_resources(resources, class_name)

    try:
        return Class(resources=resources, configs=configs)
    except KeyError as e:
        raise ResourceError(f"Missing required resource for {class_name}: {e}") from e
    except AttributeError as e:
        raise ConfigurationError(f"Missing required configuration for {class_name}: {e}") from e
    except Exception as e:
        raise InitializationError(f"Unexpected error initializing {class_name}: {e}") from e


def make_variable_codebook(
    resources: dict[str, Any] = {},
    configs: VariableCodebookConfigs = VariableCodebookConfigs(),
) -> VariableCodebook:
    """Factory function to create VariableCodebook instance"""
    return _initialize(VariableCodebook, configs, resources,) 


def make_document_retrieval_from_websites(
    resources: dict[str, Any] = {},
    configs: DocumentRetrievalConfigs = DocumentRetrievalConfigs(),
) -> DocumentRetrievalFromWebsites:
    """Factory function to create DocumentRetrievalFromWebsites instance"""
    _resources = {
        "get_cid": get_cid
    }
    for key in resources:
        if key not in _resources:
            _resources[key] = resources[key]
    return _initialize(DocumentRetrievalFromWebsites, configs, resources,) 



def make_document_storage(
    resources: dict[str, Any] = {},
    configs: Configs = configs,
) -> DocumentStorage:
    """Factory function to create DocumentStorage instance"""
    _resources = {
        "logger": resources.get("logger"),
        "db": resources.get("db"),
        "get_cid": resources.get("get_cid"),
    }
    _resources.update(resources)
    try:
        return DocumentStorage(resources=resources, configs=configs)
    except Exception as e:
        raise e


def make_top10_document_retrieval(
    resources: dict[str, Callable] = {}, 
    configs: Configs = configs
    ) -> Top10DocumentRetrieval:
    """
    Factory function to create Top10DocumentRetrieval instance
    
    Args:
        resources: Dictionary of resources
        configs: Configuration for Top-10 Document Retrieval
    
    Returns:
        Instance of Top10DocumentRetrieval
    """
    from ..resources.top10_document_retrieval._cosine_similarity import cosine_similarity
    from ..resources.top10_document_retrieval._dot_product import dot_product
    from ..resources.top10_document_retrieval._euclidean_distance import euclidean_distance

    _resources = {
        "logger": resources.get("logger", logger),
        "database": resources.get("database", make_duckdb_database()),
        "_encode_query": resources.get("_encode_query", None),
        "_similarity_search": resources.get("_similarity_search", None),
        "_retrieve_top_documents": resources.get("_retrieve_top_documents", None),
        "_cosine_similarity": resources.get("_cosine_similarity", cosine_similarity),
        "_dot_product": resources.get("_dot_product", dot_product),
        "_euclidean_distance": resources.get("_euclidean_distance", euclidean_distance),
    }
    return Top10DocumentRetrieval(resources=_resources, configs=configs)


def make_relevance_assessment(
    resources: dict[str, Any] = {},
    configs: RelevanceAssessmentConfigs = RelevanceAssessmentConfigs(),
) -> RelevanceAssessment:
    """Factory function to create RelevanceAssessment instance"""
    return _initialize(RelevanceAssessment, configs, resources,) 


def make_prompt_decision_tree(
    resources: dict[str, Any] = {},
    configs: PromptDecisionTreeConfigs = PromptDecisionTreeConfigs(),
) -> PromptDecisionTree:
    """Factory function to create PromptDecisionTree instance"""
    return _initialize(PromptDecisionTree, configs, resources,) 


def make_socialtoolkit_pipeline(
    resources: dict[str, Any] = {},
    configs: Configs = configs,
) -> SocialtoolkitPipeline:
    """
    Factory function to create SocialtoolkitPipeline instance

    Args:
        resources: (dict[str, Any]) Optional dictionary of resource overrides.
        configs: (SocialtoolkitConfigs) Configuration object for the SocialtoolkitPipeline.

    Returns:
        An instance of SocialtoolkitPipeline.

    Raises:
        ConfigurationError: If the provided configurations are invalid.
        ResourceError: If there is an error creating common resources or dependencies.
        InitializationError: If there is an unexpected error initializing the SocialtoolkitPipeline itself.
    """
    name = "SocialtoolkitPipeline"
    _validate_configs(configs, name)
    _validate_resources(resources, name)

    try:
        kwargs = {
            "resources": {
                "llm": resources.get("llm", make_llm()),
                "db": resources.get("db", make_duckdb_database()),
                "logger": resources.get("logger", make_logger("socialtoolkit_pipeline")),
            },
            "configs": configs,
        }
    except (ConfigurationError, ResourceError, InitializationError) as e:
        raise e
    except Exception as e:
        raise ResourceError(f"Unexpected error creating common resources for {name}: {e}") from e

    try:
        _resources = {
            "document_retrieval": resources.get("document_retrieval", make_document_retrieval_from_websites(**kwargs)),
            "document_storage": resources.get("document_storage", make_document_storage(**kwargs)),
            "top10_document_retrieval": resources.get("top10_document_retrieval", make_top10_document_retrieval(**kwargs)),
            "relevance_assessment": resources.get("relevance_assessment", make_relevance_assessment(**kwargs)),
            "prompt_decision_tree": resources.get("prompt_decision_tree", make_prompt_decision_tree(**kwargs)),
            "variable_codebook": resources.get("variable_codebook", make_variable_codebook(**kwargs)),
        }
    except (ConfigurationError, ResourceError, InitializationError) as e:
        raise e
    except Exception as e:
        raise ResourceError(f"Unexpected error initializing dependencies for {name}: {e}") from e

    try:
        return SocialtoolkitPipeline(configs=configs, resources=_resources)
    except KeyError as e:
        raise ResourceError(f"Missing required resource for {name}: {e}") from e
    except AttributeError as e:
        raise ConfigurationError(f"Missing required configuration for {name}: {e}") from e
    except Exception as e:
        raise InitializationError(f"Unexpected error initializing {name}: {e}") from e
