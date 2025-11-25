"""
Main module for importing and registering ComfyUI nodes in the Red Ribbon package\n
Author: Kyle Rose, Claude 3.7 Sonnet, Claude 4 Sonnet, Claude 4.5 Sonnet,
Version: 0.1.0
"""
import os
import time
from typing import Any, Optional, TypeVar, Optional
import logging


DEMO_MODE: bool = True


from ._custom_errors import (
    InitializationError,
    ConfigurationError,
    ResourceError,
    LocalDependencyError,
    LibraryDependencyError,
    RedRibbonError,
)


try:
    from rich.console import Console
    import torch
    import comfy # type: ignore

    from torch import Tensor, nn
    import comfy.utils # type: ignore
except ImportError as e:
    import traceback
    raise LibraryDependencyError(
        f"Critical import not found. Please install Comfy to use this package.\n{e}\n{traceback.format_exc()}"
    ) from e


try:
    from .custom_easy_nodes import (
        NumberInput,
        ComfyNode,
        StringInput,
        Choice,
        register_type,
        show_text
    )
    from .custom_easy_nodes.easy_nodes import AnythingVerifier
    from .socialtoolkit import SocialToolkitAPI
    from .red_ribbon_core import RedRibbonAPI
    from .plug_in_play_transformer import PlugInPlayTransformerAPI

    # Utility classes and functions
    # TODO figure out what the hell is up with imports. It makes EasyNodes not so easy to debug!
    from .utils_ import (
        Configs, LLM, make_logger, DatabaseAPI, 
        make_duckdb_database, get_red_ribbon_banner, instantiate
    )
    from ._custom_types import register_custom_types

    # Types
    from ._custom_types import (
        Prompts,
        Urls,
        Answers,
        Data,
        Laws,
        LlmApi,
    )
except ImportError as e:
    import traceback
    raise LocalDependencyError(
        f"Critical local import not found. Please ensure all Red Ribbon files are present.\n{e}\n{traceback.format_exc()}"
    ) from e


Class = TypeVar('Class')
ClassInstance = TypeVar('ClassInstance')



class RedRibbon:
    """Main interface for the Red Ribbon package"""
    _instance: 'RedRibbon' = None # type: ignore
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> 'RedRibbon':
        """Enforce singleton pattern"""
        if cls._instance is None:
            cls._instance = super(RedRibbon, cls).__new__(cls)
        return cls._instance

    def __init__(self, *, resources: dict[str, Any], configs: Configs) -> None:
        """Initialize the Red Ribbon package components"""
        if RedRibbon._initialized:
            return  # Prevent re-initialization
        RedRibbon._initialized = True
        self._call_count: int = 0
        print(f"_call_count == {self._call_count}")

        self.configs = configs
        self.resources = resources
        self.logger: logging.Logger = self.resources["logger"]
        self.console: Console = self.resources["console"]

        self.socialtoolkit: Optional[SocialToolkitAPI]         = self.resources.get("social")
        self.database:      Optional[DatabaseAPI]              = self.resources.get("database")
        self.llm:           Optional[LLM]                      = self.resources.get("llm")
        self.rr:            Optional[RedRibbonAPI]             = self.resources.get("rr")
        self.trans:         Optional[PlugInPlayTransformerAPI] = self.resources.get("trans")
        # TODO insert the class back when done debugging
        # self.utils: ModuleType                   = self.resources["utils"]

        register_custom_types()
        self._missing_attributes: list[str] = self._check_for_missing_attributes()
        self._print_startup_message()
        self._call_count += 1

    def _check_for_missing_attributes(self) -> list[str]:
        missing_attributes: list[str] = [
            attr for attr in self.__dict__.keys() if not self.__dict__[attr]
        ]
        for attr in missing_attributes:
            match attr:
                case "configs":
                    raise AttributeError("Critical attribute 'configs' is missing")
                case "resources":
                    raise AttributeError("Critical attribute 'resources' is missing")
                case _:
                    self.logger.warning(f"Node set '{attr}' is missing")
        return missing_attributes

    @property
    def VERSION(self) -> str:
        """Get the version of the Red Ribbon package"""
        from .__version__ import __version__
        return __version__

    @property
    def DEMO_NODE(self) -> str:
        """Get the demo mode status"""
        return '[green3]ON[/green3]' if DEMO_MODE else 'red3]OFF[/red3]'

    def _print_startup_message(self) -> None:
        dont_print_these_attributes: list[str] = [
            "configs", "resources", "logger", "_missing_attributes" , "client", "database", "llm",
            "console", "call_count"
        ]
        dont_print_these_attributes.extend(self._missing_attributes)
        attrs: list[str] = [
            attr for attr in self.__dict__.keys() if attr not in dont_print_these_attributes
        ]
        available_nodes = "\n".join(f"{i}. [bright_red]{name}[/bright_red]" for i, name in enumerate(attrs, start=1))
        red_ribbon_banner = get_red_ribbon_banner(without_logo=True)
        for line in red_ribbon_banner:
            self.console.print(line)
            time.sleep(0.025)
        self.console.print(f"""
                                    Red Ribbon loaded successfully.
                                    Version: {self.VERSION}
                                    DEMO MODE is {self.DEMO_NODE}
                                    *****************
                                    Available Modules:
                                    {available_nodes}
        """)


def _make_red_ribbon() -> RedRibbon:
    """Factory function to create and initialize the Red Ribbon package"""
    try:
        configs = Configs()
    except Exception as e:
        raise ConfigurationError(f"Failed to initialize global configs for RedRibbon: {e}") from e

    try:
        _resources = {
            #"social": SocialToolkitAPI(SocialToolKitResources(configs).resources, configs),
            # "rr": RedRibbonAPI(rr_resources, configs),
            # "utils": UtilsAPI(utils_resources, configs)
            "logger": make_logger("RedRibbon"),
            "trans": PlugInPlayTransformerAPI(configs=configs),
            "database": make_duckdb_database(configs=configs),
            "console": Console(),
        }
    except Exception as e:
        raise ResourceError(f"Failed to initialize global resources for RedRibbon: {e}") from e

    # Initialize the Red Ribbon package
    try:
        red_ribbon = RedRibbon(resources=_resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize RedRibbon package: {e}") from e
    return red_ribbon


red_ribbon = _make_red_ribbon()


#####################
#### COMFY NODES ####
#####################


def _raise_if_no_socialtoolkit() -> None:
    if red_ribbon.socialtoolkit is None:
        raise RedRibbonError("SocialToolkitAPI is not initialized in RedRibbon.")

@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    always_run=True,
    display_name="Database")
def database_enter(
    type: str = Choice(["DuckDB","SQLite", "PostgreSQL", "MySQL"]), # type: ignore[assignment]
    username: str = StringInput("", multiline=False), # type: ignore[assignment]
    password: str = StringInput("", multiline=False), # type: ignore[assignment]
    port: int = NumberInput(default=3306), # type: ignore[assignment]
    timeout: int = NumberInput(default=30, min=1, max=300, step=1) # type: ignore[assignment]
) -> Data:
    """
    Select a database type to use in the pipeline.
    """
    if DEMO_MODE:
        from .socialtoolkit import demo_database_enter
        database_api = demo_database_enter()
    else:
        print(f"Initializing {type} database...")
        configs = Configs() # TODO Make it auto-construct the connection string based on the type
        # Create appropriate connection string based on type
        user_path = os.path.expanduser("~")
        match type:
            case "DuckDB":
                configs.connection_string = f"duckdb:///{user_path}/red_ribbon_data/database.duckdb"
            case "SQLite":
                configs.connection_string = f"sqlite:///{user_path}/red_ribbon_data/database.db"
            case "PostgreSQL":
                host = "localhost"
                db_name = "red_ribbon"
                configs.connection_string = f"postgresql://{username}:{password}@{host}:{port}/{db_name}"
            case "MySQL":
                host = "localhost" 
                db_name = "red_ribbon"
                configs.connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}"
            case _:
                raise ValueError(f"Unknown database type: {type}")
        configs.timeout = timeout

        # Select the database type and initialize is as a DatabaseAPI
        # TODO add the rest of the supported database types
        match type:
            case "duckdb": 
                database_api: DatabaseAPI = make_duckdb_database(configs=configs)
            case _:
                raise NotImplementedError(f"Database type '{type}' is not supported yet. Sorry!")
        red_ribbon.database = database_api
    print("Database initialized successfully.")
    return database_api


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    always_run=True,
    display_name="Get links from the file",
    return_names=["links"],)
def load_data(
    the_file: str = StringInput("links.xlsx", multiline=False), # type: ignore[assignment]
) -> Urls:
    if DEMO_MODE:
        from .socialtoolkit import demo_load_data
        urls = demo_load_data()
    else:
        _raise_if_no_socialtoolkit()
        print("Loading URLs from input file...")
        urls = red_ribbon.socialtoolkit.execute(
            "load_data",
            file_path=the_file
        )
        print("URLs loaded successfully.")
    return urls


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Get laws from the Web.",)
def document_retrieval_from_websites(
    links: Urls,
) -> Laws: # dict[str, Any]
    """
    Get documents from a list of domain URLs.
    """
    if DEMO_MODE:
        from .socialtoolkit import demo_document_retrieval_from_websites
        return demo_document_retrieval_from_websites(links)
    else:
        _raise_if_no_socialtoolkit()
        #with DatabaseAPI() as db: # TODO
        return red_ribbon.socialtoolkit.execute(
            "document_retrieval_from_websites",
            links,
            db_service=None,
        )


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    is_output_node=True,
    display_name="Save the laws we find.")
def document_storage(
    laws: Laws,
) -> Laws:
    """
    Get documents and their vectors from a database.
    """
    #documents, metadata, vectors = laws['documents'], laws["metadata"], laws["vectors"]
    if DEMO_MODE:
        from .socialtoolkit import demo_document_storage
        return demo_document_storage()
    else:
        _raise_if_no_socialtoolkit()
        return red_ribbon.socialtoolkit.execute(
            "document_storage",
            db_service=None,
            documents=laws.documents, 
            metadata=laws.metadata, 
            vectors=laws.vectors
        )


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="The Question",
    return_names=["Question"],)
def input_text(
    question: str = StringInput("Local Sales Tax in Cheyenne, WY", multiline=True), # type: ignore[assignment]
) -> str:
    """
    Input text for the pipeline.
    """
    question = question.strip()
    return question


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Find the laws we want.",
    return_names=["Laws"],)
def top10_document_retrieval(
    question: str,
    laws: Laws,
    return_how_many: int = NumberInput(default=10, min=1, max=100, step=1, display="slider") # type: ignore[assignment]
) -> Laws:
    """
    Get a list of the top X documents based on a data point.
    """
    # documents, vectors = laws['documents'], laws["vectors"]
    if DEMO_MODE:
        from .socialtoolkit import demo_top10_document_retrieval
        return demo_top10_document_retrieval(laws, return_how_many)
    else:
        _raise_if_no_socialtoolkit()
        return red_ribbon.socialtoolkit.execute(
            "top10_document_retrieval",
            question,
            num_documents=return_how_many,
            documents=laws.documents, 
            vectors=laws.vectors
        )


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Use AI to find the laws we want.",)
def relevance_assessment(
        laws: Laws, 
        ai: LlmApi, # NOTE return_how_many is a dummy slider to keep the UI consistent
        return_how_many: int = NumberInput(default=3, min=1, max=100, step=1, display="slider") # type: ignore[assignment]
    ) -> Laws:
    """
    Determine how relevant a list of documents are to a query.
    """
    _ = return_how_many
    if DEMO_MODE:
        from .socialtoolkit import demo_relevance_assessment
        return demo_relevance_assessment()
    else:
        print("Running relevance assessment...")
        _raise_if_no_socialtoolkit()
        try:
            with open("question.txt", "r") as f:
                question = f.read()
        except Exception as e:
            raise IOError(f"Failed to read question from 'question.txt': {e}") from e

        documents = red_ribbon.socialtoolkit.execute(
            "relevance_assessment",
            question,
            documents=laws, 
            llm=ai
        )
        return documents



@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Instructions for the AI",
    return_names=["Instructions"],)
def variable_codebook(
    question: str,
) -> Prompts:
    """
    Load a variable codebook entry from a database.
    This entry consists of a directed graph with attached metadata.
    """
    answer = None
    if DEMO_MODE:
        from .socialtoolkit._demo_mode import demo_variable_codebook
        answer = demo_variable_codebook(question)
    else:
        print("Loading variable codebook...")
        _raise_if_no_socialtoolkit()
        try:
            with DatabaseAPI() as db:
                prompts = red_ribbon.socialtoolkit.execute(
                    "variable_codebook",
                    question,
                    database=db, 
                )
        except Exception as e:
            raise NotImplementedError(f"Variable codebook still needs to be debugged: {e}") from e
        print("Variable codebook loaded successfully.")
    return answer


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Use AI to answer the question.",
    return_names=["Answers"],)
def prompt_decision_tree(
    laws: Laws,
    ai: LlmApi,
) -> Answers:
    """
    Run an AI-powered decision tree to extract answers from a list of documents.
    """
    if DEMO_MODE:
        from .socialtoolkit._demo_mode import demo_prompt_decision_tree
        answer = demo_prompt_decision_tree()
    else:
        documents = laws
        _raise_if_no_socialtoolkit()
        try:
            with open("question.txt", "r") as f:
                question = f.read()
        except Exception as e:
            raise IOError(f"Failed to read question from 'question.txt': {e}") from e

        print("Loading codebook entry...")
        prompts = variable_codebook(question)
        print("Running prompt decision tree...")

        answer = red_ribbon.socialtoolkit.execute(
            "prompt_decision_tree",
            question,
            documents=documents, 
            prompts=prompts,
            llm=ai
        )
        print("Prompt decision tree complete.")
    return answer


@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="AI",
    always_run=True,
    return_names=["AI"],)
def llm_api(
    instructions: Prompts | None = None,
    name: str = Choice(["gpt-4o"]), # type: ignore[assignment]
) -> LlmApi:
    """
    Load a large language model (LLM) from a local file or API.
    """
    # Hardcode for the demo
    temperature = 0.7
    max_tokens = 4096
    top_p = 1.0
    red_ribbon
    if DEMO_MODE:
        from .socialtoolkit._demo_mode import demo_llm_api
        llm = demo_llm_api(name, instructions, red_ribbon)
    else:
        print("Loading LLM...")

        _raise_if_no_socialtoolkit()

        llm = red_ribbon.socialtoolkit.execute(
            "llm",
            model_name=name,
            temperature=temperature, 
            max_tokens=max_tokens,
            top_p=top_p
        )
        print("LLM loaded successfully.")
    return llm


@ComfyNode(
    category="Socialtoolkit",
    is_output_node=True,
    color="#1f1f1f",
    bg_color="#454545",
    display_name="The Answer")
def answer(
    answer: Answers
):
    """
    Display the LLM's answer.
    """
    _ = answer
    if DEMO_MODE:
        from .socialtoolkit._demo_mode import demo_answer
        demo_answer()
    else:
        match answer:
            case str():
                show_text(answer)
            case list():
                for line in answer:
                    show_text(line)
            case _:
                raise TypeError(f"Answer must be a string or a list of strings, got {type(answer).__name__}.")
    return 


# Main function that can be called when using this as a script
def main() -> None:
    print("Red Ribbon package loaded successfully")
    print(f"Version: {red_ribbon.version()}")
    print("Available components:")
    print("- SocialToolkit")
    print("- RedRibbon Core")
    print("- Plug-in-Play Transformer")
    print("- Utils")

if __name__ == "__main__":
    main()
