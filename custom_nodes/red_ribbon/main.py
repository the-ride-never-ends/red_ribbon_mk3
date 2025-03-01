"""
Red Ribbon - Main module for importing and registering all nodes
Author: Kyle Rose, Claude 3.7 Sonnet
Version: 0.1.0
"""
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Type, TypeVar, Optional


from easy_nodes import (
    NumberInput,
    ComfyNode,
    StringInput,
    Choice,
    show_text,
    register_type,
)
from easy_nodes.easy_nodes import AnythingVerifier


# Import components from subdirectories
from .socialtoolkit.socialtoolkit import SocialToolkitAPI, SocialToolKitResources
from .red_ribbon_core.red_ribbon import RedRibbonAPI
from .plug_in_play_transformer.plug_in_play_transformer import TransformerAPI
from .utils.utils import UtilsAPI
from .utils.instantiate import instantiate
# from .node_types import register_pydantic_models
# from custom_nodes.red_ribbon.socialtoolkit.types import Document, Metadata, Vectors

try: # TODO figure out what the hell is up with imports. It makes EasyNodes not so easy to debug!
    from .configs import Configs
except Exception as e:
    print(f"{type(e)} importing main module: {e}")
    raise e

Class = TypeVar('Class')
ClassInstance = TypeVar('ClassInstance')

class DatabaseAPI:
    """
    Generic class for database operations.
    NOTE: This is a composition class and should 
    
    """
    def __init__(self, 
                resources: dict[str, Callable] = None, 
                configs: Configs = None
                ) -> 'DatabaseAPI':
        self.configs = configs
        self.resources = resources
        self._enter = self.resources.get("enter")
        self._execute = self.resources.get("execute")
        self._exit = self.resources.get("exit")

    @classmethod
    def enter(cls, 
              resources: dict[str, Callable] = None, 
              configs: Configs = None
              ) -> 'DatabaseAPI':
        instance = cls(resources, configs)
        instance._enter()
        return instance

    def __enter__(self) -> 'DatabaseAPI':
        self._enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return self._exit()

    def exit(self) -> None:
        return self._exit()

    def execute(self, statement: str, *args, **kwargs):
        self._execute(statement, *args, **kwargs)








class RedRibbonPackage:
    """Main interface for the Red Ribbon package"""

    def __init__(self, resources: dict[str, Class] = None, configs: Configs = None):
        """Initialize the Red Ribbon package components"""
        self.configs = configs
        self.resources = resources or {}

        self.social: Type[SocialToolkitAPI] = self.resources.get("social")
        self.rr:     Type[RedRibbonAPI]     = self.resources.get("rr")
        self.trans:  Type[TransformerAPI]   = self.resources.get("trans")
        self.utils:  Type[UtilsAPI]         = self.resources.get("utils")

        self._missing_attributes: list[Optional[str]] = self.check_for_missing_attributes()
        self.startup_message()

    def check_for_missing_attributes(self):
        missing_attributes= [
            attr for attr in self.__dict__.keys() if not self.__dict__[attr]
        ]
        for attr in missing_attributes:
            match attr:
                case "configs":
                    raise ValueError("Critical attribute 'configs' is missing")
                case "resources":
                    raise ValueError("Critical attribute 'resources' is missing")
                case _:
                    print(f"WARNING: Node set '{attr}' is missing")
        return missing_attributes

    @property
    def version(self):
        """Get the version of the Red Ribbon package"""
        from .__version__ import __version__
        return __version__

    def startup_message(self):
        available_nodes = [
            node for node in self.__dict__.keys() 
            if node not in ["configs", "resources", self._missing_attributes]
        ]
        nodes = "- \n".join(available_nodes)
        print("Red Ribbon node package loaded successfully.")
        print(f"Version: {self.version}")
        print("Available Nodes:")
        print(nodes)


register_type(DatabaseAPI, "Database", AnythingVerifier)


@dataclass
class DatabaseResources:

    _resources: dict[str, ClassInstance] = field(default_factory=dict)
    
    def __post_init__(self): # TODO Implement instantiate function when not debugging
        self._resources = {
            "sqlite": "",
            "postgres": "",
            "oracle": "",
            "mysql": "",
            "duckdb": ""
        }

    @cached_property
    def resources(self) -> dict[str, ClassInstance]:
        """Instantiated classes use to run Socialtoolkit in ComfyUI"""
        return self._resources


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Database")
def database_enter(
    type: str = Choice(["SQLite", "PostgreSQL", "MySQL"]),
    connection_string: str = StringInput("sqlite:///database.db", multiline=False),
    username: str = StringInput("", multiline=False),
    password: str = StringInput("", multiline=False),
    timeout: int = NumberInput(default=30, min=1, max=300, step=1)
) -> DatabaseAPI:
    """
    Connect to a database.
    """
    resources = DatabaseResources.resources
    configs = Configs()
    configs.database.type = type
    configs.database.connection_string = connection_string
    configs.database.username = username
    configs.database.password = password
    configs.database.timeout = timeout
    return DatabaseAPI.enter(resources, configs)



configs = Configs()

resources = {
    "social": SocialToolkitAPI(SocialToolKitResources().resources, configs),
    # "rr": RedRibbonAPI(rr_resources, configs),
    # "trans": TransformerAPI(trans_resources, configs),
    # "utils": UtilsAPI(utils_resources, configs)
}
resources = None 
# Initialize the Red Ribbon package
red_ribbon = RedRibbonPackage(resources, configs)


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Rank and Sort Similar Search Results")
def rank_and_sort_similar_search_results(
    search_results: str,
    search_query: str,
    search_type: str,
    rank_by: str,
    sort_by: str
) -> list[str]:
    """
    Rank and sort similarity search results.
    """
    return red_ribbon.social.execute(
        "rank_and_sort_similar_search_results",
        search_results,
        search_query,
        search_type,
        rank_by,
        sort_by
    )


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Retrieve Documents from Websites",
    return_names=["Documents", "Metadata", "Vectors"],)
def document_retrieval_from_websites(
    domain_urls: list[str],
    database: DatabaseAPI
) -> tuple[str, str, str]:
    """
    Get documents from a list of domain URLs.
    """
    resources = DatabaseResources.resources
    with database() as db:
        return red_ribbon.social.execute(
            "document_retrieval_from_websites",
            domain_urls
        )


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Document Storage",
    return_names=["Documents", "Vectors"],)
def document_retrieval_from_websites(
    domain_urls: list[str]
) -> tuple[str, str]:
    """
    Get documents from a list of domain URLs.
    """
    return red_ribbon.social.execute(
        "document_retrieval_from_websites",
        domain_urls
    )


# Main function that can be called when using this as a script
def main():
    print("Red Ribbon package loaded successfully")
    print(f"Version: {red_ribbon.version()}")
    print("Available components:")
    print("- SocialToolkit")
    print("- RedRibbon Core")
    print("- Plug-in-Play Transformer")
    print("- Utils")

if __name__ == "__main__":
    main()