import asyncio
from collections import defaultdict
import logging
from pathlib import Path
from typing import Any, DefaultDict


from elm.ords.llm import StructuredLLMCaller
from .content import ValidationWithMemory
from .ordinances import OrdinanceValidator, OrdinanceExtractor


from pydantic import BaseModel, ValidationError
import yaml


class InitializationError(RuntimeError):
    def __init__(self, message: str):
        super().__init__(message)


def get_semaphore(limit: int = 5) -> asyncio.Semaphore:
    return asyncio.Semaphore(limit)


def _load_prompts() -> DefaultDict:
    """Load keywords from YAML files in the words directory."""
    paths = {
        file for file in (Path(__file__).parent / "prompts").glob("*.yaml")
    }
    prompts_from_yaml = defaultdict(dict)

    for path in paths:
        name = path.stem
        try:
            with open(path, "r") as f:
                prompts_from_yaml[name] = dict(yaml.safe_load(f)) 
        except FileNotFoundError as e:
            raise FileNotFoundError(f"YAML file for variable '{name}' not found: {e}")
        except yaml.YAMLError as e:
            raise IOError(f"Error parsing YAML file for variable '{name}': {e}")
        except ValidationError as e:
            raise ValueError(f"Validation error loading YAML file for variable '{name}': {e}")
        except Exception as e:
            raise IOError(f"Unexpected error loading YAML file for variable '{name}': {e}")
    return prompts_from_yaml

# NOTE Load outside the factory functions to avoid reloading on each call
_VALIDATION_PROMPTS = _load_prompts()

def _make_validation_with_memory(resources: dict[str, Any]) -> ValidationWithMemory:

    try:
        _resources = {
            "variable": resources['variable'],
            "structured_llm_caller": resources["structured_llm_caller"],
            "text_chunks": resources['text_chunks'],
            "logger": resources.get("logger", logging.getLogger(__name__)),
            "num_recall": resources.get("num_recall", 2)
        }
    except KeyError as e:
        raise InitializationError(f"Missing required resource for ValidationWithMemory: {e}") from e

    try:
        return ValidationWithMemory(**_resources)
    except Exception as e:
        raise InitializationError(f"Failed to initialize ValidationWithMemory: {e}") from e


def make_ordinance_validator(resources: dict[str, Any]) -> OrdinanceValidator:
    """Factory function to create an OrdinanceValidator instance.
    
    Args:
        resources: Dictionary containing required resources:
        'ordinance': str
        'llm_caller': StructuredLLMCaller 
        'chunks': list[str]

        Optional resources:
        'logger': Optional[logging.Logger], 
        'num_recall': Optional[int],
        'prompt_dict': Optional[DefaultDict[str, Any]].
    
    Returns:
        Initialized OrdinanceValidator instance.
    """
    try:
        ordinance = resources['ordinance']
        prompt_dict = resources.get('prompt_dict', _VALIDATION_PROMPTS)
        if ordinance not in prompt_dict:
            raise InitializationError(f"Prompts for ordinance '{ordinance}' not found in prompt dictionary")

        _resources = {
            "structured_llm_caller": resources['llm_caller'],
            "text_chunks": resources['chunks'],
            "variable": ordinance,
            "logger": resources.get("logger", logging.getLogger(__name__)),
            "num_recall": resources.get('ordinance', 2),
            "prompt_dict": prompt_dict[ordinance]['ORDINANCE_VALIDATOR']
        }
    except KeyError as e:
        raise InitializationError(f"Missing required resource for OrdinanceValidator: {e}") from e
    except InitializationError as e:
        raise e

    try:
        validation_with_memory = _make_validation_with_memory(resources)
        _resources['validation_with_memory'] = validation_with_memory
    except InitializationError as e:
        raise e

    try:
        return OrdinanceValidator(resources=_resources)
    except Exception as e:
        raise InitializationError(f"Failed to initialize StructuredLLMCaller: {e}") from e


def make_ordinance_extractor(*, 
    ordinance: str, 
    llm_caller: StructuredLLMCaller, 
    chunks: list[str], 
    logger: logging.Logger, 
    num_recall: int = 2, 
    prompt_dict: DefaultDict[str, Any] = _VALIDATION_PROMPTS
    ) -> OrdinanceExtractor:
    """Factory function to create an OrdinanceExtractor instance.

    Args:
        ordinance: Description
        llm_caller: Description
        chunks: Description
        logger: Description
        num_recall: Description
        prompt_dict: Description
    
    Returns:
        Initialized OrdinanceExtractor instance.
    """
    if ordinance not in prompt_dict:
        raise InitializationError(f"Prompts for ordinance '{ordinance}' not found in prompt dictionary")

    _resources = {
        "structured_llm_caller": llm_caller,
        "text_chunks": chunks,
        "variable": ordinance,
        "logger": logger,
        "num_recall": num_recall,
        "prompt_dict": prompt_dict[ordinance]['ORDINANCE_EXTRACTOR']
    }
    try:
        return OrdinanceExtractor(_resources)
    except Exception as e:
        raise InitializationError(f"Failed to initialize OrdinanceExtractor: {e}") from e

