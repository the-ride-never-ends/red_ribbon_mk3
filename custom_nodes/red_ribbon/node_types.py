from abc import ABC, abstractmethod
import importlib
import inspect
from typing import Any, Optional, Type


from pydantic import BaseModel


from easy_nodes import register_type
from easy_nodes.easy_nodes import AnythingVerifier


def registration_callback(register_these_classes: list[Type]) -> None:
    for this_class in register_these_classes:
        with_its_class_name_in_all_caps: str = this_class.__qualname__.upper()
        register_type(this_class, with_its_class_name_in_all_caps, verifier=AnythingVerifier())


def register_pydantic_models(
    module_names: list[str],
) -> None:
    """
    Loads Pydantic classes from specified modules and registers them.
    
    Args:
        module_names: list of module names to search for Pydantic models
        registration_callback: Optional function to call for each model (for registration)
            If None, a dummy registration function will be used
            
    Returns:
        Side-effect: registers Pydantic models with EasyNodes.
    """
    models = []
    for module_name in module_names:
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Find all Pydantic classes in the module
            for _, obj in inspect.getmembers(module):
                # Check if it's a class and a subclass of BaseModel but not BaseModel itself
                if (inspect.isclass(obj) and 
                    issubclass(obj, BaseModel) and 
                    obj is not BaseModel):
                    models.append(obj)
        except ImportError as e:
            print(f"Error importing module {module_name}: {e}")
        except Exception as e:
            print(f"Error processing module {module_name}: {e}")

    # Register the model using the provided callback
    try:
        registration_callback(models)
    except Exception as e:
        print(f"{type(e)} registering models: {e}")

    return models


class Node(ABC):

    # NOTE This is used for type hinting in the SocialToolkitNode class, 
    # as well as an example of how each class should be constructed.

    def __init__(self, resources: dict[str, Any], configs: 'BaseModel'):
        self.configs = configs
        self.resources = resources

    @abstractmethod
    def execute(self, action: str, *args, **kwargs) -> Optional[Any]:
        """
        Execute the action for the node.
        """

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()
