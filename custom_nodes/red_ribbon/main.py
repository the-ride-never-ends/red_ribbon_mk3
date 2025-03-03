"""
Red Ribbon - Main module for importing and registering all nodes
Author: Kyle Rose, Claude 3.7 Sonnet
Version: 0.1.0
"""
DEMO_MODE = True
IN_COMFY = True

import contextvars
from dataclasses import dataclass, field
from functools import cached_property
import os
from pathlib import Path
import random
import sys
import time
from typing import Callable, Type, TypeVar, Optional

import comfy.utils

try:
    import comfy
except ImportError:
    print("Comfy not found. Please install Comfy to use this package.")
    sys.exit(1)


from easy_nodes import (
    NumberInput,
    ComfyNode,
    StringInput,
    Choice,
    show_text,
    register_type,
)
from easy_nodes.easy_nodes import AnythingVerifier, _curr_preview as easy_nodes_curr_preview
from networkx import DiGraph # NOTE We do this so that we can register the nx.DiGraph type in ComfyUI
import openai
from tqdm import tqdm

# Import components from subdirectories
# Modules
from .socialtoolkit.socialtoolkit import SocialToolkitAPI, SocialToolKitResources
from .red_ribbon_core.red_ribbon import RedRibbonAPI
from .plug_in_play_transformer.plug_in_play_transformer import TransformerAPI


# Utility functions
from .configs import Configs # TODO figure out what the hell is up with imports. It makes EasyNodes not so easy to debug!
from .database import DatabaseAPI
from .llm import Llm
from .logger import get_logger
from .utils.instantiate import instantiate


Class = TypeVar('Class')
ClassInstance = TypeVar('ClassInstance')


class RedRibbon:
    """Main interface for the Red Ribbon package"""

    def __init__(self, resources: dict[str, Class] = None, configs: Configs = None):
        """Initialize the Red Ribbon package components"""
        self.configs = configs
        self.resources = resources or {}
        self.logger = get_logger(self.__class__.__name__)

        self.socialtoolkit: Type[SocialToolkitAPI] = self.resources["social"]
        #self.rr:     Type[RedRibbonAPI]     = self.resources["rr"]
        #self.trans:  Type[TransformerAPI]   = self.resources["trans"]
        # TODO insert the class back when done debugging
        #self.utils:  Type                   = self.resources["utils"]

        self._missing_attributes: list[Optional[str]] = self.check_for_missing_attributes()
        self._print_startup_message()

    def check_for_missing_attributes(self) -> list[str]:
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
                    self.logger.warning(f"Node set '{attr}' is missing")
        return missing_attributes

    @property
    def version(self):
        """Get the version of the Red Ribbon package"""
        from .__version__ import __version__
        return __version__

    def _print_startup_message(self):
        dont_print_these_attributes = ["configs", "resources", "logger", "_missing_attributes"]
        dont_print_these_attributes.extend(self._missing_attributes)
        attrs = [
            attr for attr in self.__dict__.keys() if attr not in dont_print_these_attributes
        ]
        available_nodes = "\n".join(f"{i}. {name}" for i, name in enumerate(attrs, start=1))
        print(f"""
            Red Ribbon loaded successfully.
            Version: {self.version}
            DEMO MODE is {'ON' if DEMO_MODE else 'OFF'}
            Available Modules:
            {available_nodes}
        """)
        # print(f"Version: {self.version}")
        # print(f"DEMO MODE is {'ON' if DEMO_MODE else 'OFF'}")
        # print("Available Modules:")
        # print(available_nodes)


from contextlib import contextmanager
import shutil
import tempfile
@contextmanager
def tempdir():
    """
    Source: pytorch
    """
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)




AnyType = TypeVar("*") #  Note: Any from Typing is just an alias for object anyways.
# Force the AnyType to always be equal in not equal comparisons.
# Source: https://github.com/rgthree/rgthree-comfy/blob/main/py/display_any.py#L15
AnyType.__ne__ = lambda x: False 


Excel = TypeVar("Excel", str, list[str], dict[str, str])

Vectors = TypeVar("Vectors", list[float], list[list[float]])
Metadata = TypeVar("Metadata", str, dict, DiGraph, tuple, list[str], list[dict], list[tuple], list[DiGraph])
Prompts = TypeVar("Prompts", str, dict, DiGraph)

Urls = TypeVar("Urls", str, list[str])
Answers = TypeVar("Answers", str, list[str])


# Compound type
Data = TypeVar("Data", DatabaseAPI, Excel)
Documents = TypeVar("Documents", str, list[str])
Laws = TypeVar("Laws", list[str], str, dict, tuple[Documents, Metadata, Vectors])

# Register the types with ComfyUI
types = {
    "Database": DatabaseAPI, "Llm": Llm, "Configs": Configs, 
    "Prompts": Prompts, "DiGraph": DiGraph, "dict": dict,
    "Vectors": Vectors, "Documents": Documents, "Urls": Urls,
    "Metadata": Metadata, "AnyType": AnyType, "Answers": Answers,
    "Excel": Excel, "Data": Data, "Laws": Laws
}
for type_name, type_class in types.items():
    try:
        type_class.__qualname__ 
    except AttributeError:  # If the class doesn't have a __qualname__ attribute, monkeypatch one in.
         # This came up when testing TypeVar aliases.
         type_class.__qualname__ = type_name
         #print(f"Added __qualname__ to type {type_class.__name__}")

    register_type(type_class, type_name, verifier=AnythingVerifier())



configs = Configs()
resources = {
    "social": SocialToolkitAPI(SocialToolKitResources(configs).resources, configs),
    # "rr": RedRibbonAPI(rr_resources, configs),
    # "trans": TransformerAPI(trans_resources, configs),
    # "utils": UtilsAPI(utils_resources, configs)
}
# Initialize the Red Ribbon package
red_ribbon = RedRibbon(resources, configs)


from .utils.database.resources.duckdb import DuckDB


#####################
#### COMFY NODES ####
#####################

def random_sleep(start: int = None, stop: int = None) -> None:
    if DEMO_MODE:
        if start is None or stop is None:
            start, stop = 1, 3
        rand_int = random.randint(start, stop)
        time.sleep(rand_int)

@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    always_run=True,
    display_name="Database")
def database_enter(
    type: str = Choice(["DuckDB","SQLite", "PostgreSQL", "MySQL"]),
    username: str = StringInput("", multiline=False),
    password: str = StringInput("", multiline=False),
    port: int = NumberInput(default=3306),
    timeout: int = NumberInput(default=30, min=1, max=300, step=1)
) -> Data:
    """
    Select a database type to use in the pipeline.
    """
    if DEMO_MODE:
        print("Logging into database...")
        random_sleep()
        database_api = "mock_database"
        print("Login successful.")
    else:
        print(f"Initializing {type} database...")
        configs = Configs() # TODO Make it auto-construct the connection string based on the type
        # Create appropriate connection string based on type
        match type:
            case "DuckDB":
                configs.connection_string = f"duckdb:///{{os.path.expanduser('~')}}/red_ribbon_data/database.duckdb"
            case "SQLite":
                configs.connection_string = f"sqlite:///{{os.path.expanduser('~')}}/red_ribbon_data/database.db"
            case "PostgreSQL":
                host = "localhost"
                db_name = "red_ribbon"
                configs.connection_string = f"postgresql://{username}:{password}@{host}:{port}/{db_name}"
            case "MySQL":
                host = "localhost" 
                db_name = "red_ribbon"
                configs.connection_string = f"mysql+pymysql://{username}:{password}@{host}:{port}/{db_name}"
        configs.timeout = timeout

        # Select the database type and intialize is as a DatabaseAPI
        match type:
            case "duckdb": # TODO add the rest of the cases
                database_api: DatabaseAPI = DuckDB(configs=configs)
            case _:
                raise NotImplementedError(f"Database type '{type}' is not supported yet. Sorry!")
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
    the_file: str = StringInput("links.xlsx", multiline=False),
) -> Urls:
    if DEMO_MODE:
        print("Loading links...")
        random_sleep(1,2)
        urls = "mock_urls"
        print("Links loaded successfully.")
    else:
        print("Loading URLs from input file...")
        urls = red_ribbon.socialtoolkit.execute(
            "load_data",
            file_path=the_file
        )
        print("URLs loaded successfully.")
    return urls


# 37
_DEMO_INPUT_URLS = [
    # Springhill, LA = Query "Local Sales Tax in Springhill, LA"
    "https://library.municode.com/la/springhill/codes/code_of_ordinances",
    # Cheyenne, WY = Query "Local Sales Tax in Cheyenne, WY"
    "https://library.municode.com/wy/cheyenne/codes/code_of_ordinances",
    # San Jose, CA = Query "Vending Machine Laws in San Jose, CA",
    "https://library.municode.com/ca/san_jose/codes/code_of_ordinances",
]

_DEMO_RELEVANT_URLS = [
    # Springhill, LA
    "https://library.municode.com/la/springhill/codes/code_of_ordinances?nodeId=COOR_CH98TA_ARTIINGE_S98-1SAUSTA",
    "https://taxfoundation.org/location/louisiana/",
    "https://webstersalestax.org/current-rates/"

    # Cheyenne, WY
    "https://library.municode.com/wy/cheyenne/codes/code_of_ordinances?nodeId=TIT3REFI_CH3.08TA_3.08.010REEXINRE",
    "https://taxfoundation.org/location/wyoming/",
    "https://www.avalara.com/taxrates/en/state-rates/wyoming/counties/laramie-county.html",

    # San Jose, CA
    "https://library.municode.com/ca/san_jose/codes/code_of_ordinances?nodeId=TIT20ZO_CH20.80SPUSRE_PT10OUVEFA_20.80.820EXDMPE"
    "https://library.municode.com/ca/san_jose/codes/code_of_ordinances?nodeId=TIT6BULIRE_CH6.54PEPEOR",
]

# @ComfyNode(
#     category="Socialtoolkit",
#     color="#1f1f1f",
#     bg_color="#454545",
#     display_name="Get links from the file",
#     return_names=["links"],)
# def get_domain_urls(
#     data: Data,
# ) -> Urls:
#     """
#     Get domain URLs from a database or text file
#     """
#     print("Loading URLs...")

#     if DEMO_MODE:
#         mock_range = len(_DEMO_INPUT_URLS) * 2

#         for i in range(len(mock_range)):
#             print(f"Got URL {i+1}.")

#         return _DEMO_INPUT_URLS
#     else:
#         if isinstance(data, DatabaseAPI):
#             database = data # Call a spade a spade
#             with database.enter() as db:
#                 return red_ribbon.socialtoolkit.execute(
#                     "get_domain_urls",
#                     db_service=db
#                 )


_VECTOR_LENGTH = 3072 # See: https://platform.openai.com/docs/guides/embeddings
_SELECTED_DOMAIN_URLS = []


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
        print("Gettings laws from the web...")
        mock_vectors = []
        random_sleep()
        len_domain_urls = len(links)
        print(f"Found {len_domain_urls} laws. Downloading...")

        pbar = comfy.utils.ProgressBar(total=len_domain_urls)
        for i in range(len_domain_urls):
            mock_vectors.append([random.uniform(-5.0, 5.0) for _ in range(_VECTOR_LENGTH)])
            random_sleep(1,2)
            #print(f"Downloaded Law {i+1} of {len_domain_urls}")
            pbar.update(i+1)

        return {
            "documents": _SELECTED_DOMAIN_URLS,
            "metadata": "mock_metadata",
            "vectors": mock_vectors
        }
    else:
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
    documents, metadata, vectors = laws['documents'], laws["metadata"], laws["vectors"]

    if DEMO_MODE:
        print("Saving laws...")
        random_sleep(2,5)
        print("Laws saved.")
        return {
            "documents": "mock_documents",
            "metadata": "mock_metadata",
            "vectors": [[random.uniform(-5.0, 5.0) for _ in range(_VECTOR_LENGTH)] for _ in range(10)]
        }
    else:
        #with DatabaseAPI() as db:
            return red_ribbon.socialtoolkit.execute(
                "document_storage",
                db_service=None,
                documents=documents, 
                metadata=metadata, 
                vectors=vectors
            )




@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="The Question",
    return_names=["Question"],)
def input_text(
    question: str = StringInput("Local Sales Tax in Cheyenne, WY", multiline=True),
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
    return_how_many: int = NumberInput(default=10, min=1, max=100, step=1, display="slider")
) -> Laws:
    """
    Get a list of the top X documents based on a data point.
    """
    documents, vectors = laws['documents'], laws["vectors"]
    if DEMO_MODE:
        print(f"Filtering down to {return_how_many} documents...")
        time.sleep(5)
        return ["mock_document_1", "mock_document_2", "mock_document_3"]
    else:
        return red_ribbon.socialtoolkit.execute(
            "top10_document_retrieval",
            question,
            num_documents=return_how_many,
            documents=documents, 
            vectors=vectors
        )

@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Use AI to find the laws we want.",
    return_names=["Laws"],)
def relevance_assessment(
        laws: Laws, 
        ai: Llm, 
        return_how_many: int = NumberInput(default=10, min=1, max=100, step=1, display="slider") # Dummy slider.
    ) -> Laws:
    """
    Determine how relevant a list of documents are to a query.
    """
    if DEMO_MODE:
        print("Filtering laws with AI...")
        random_sleep(1,2)
        documents = ["mock_document_1", "mock_document_2", "mock_document_3"]
        print("Laws filtered successfully.")
    else:
        print("Running relevance assessment...")
        with open("question.txt", "r") as f:
            question = f.read()
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
    if DEMO_MODE:
        print("Loading AI instructions...")
        random_sleep(2,5)
        match question:
            case question if "Cheyenne" in question:
                prompts = "6.0%"
            case question if "Springhill" in question:
                prompts = "10.5%"
            case question if "San Jose" in question:
                prompts = "Exceptions for Holiday sales, city-approved locations, newsracks, on-site businesses, and farmers' markets."
            case _:
                prompts = "mock_prompts"
        print("AI instructions loaded.")
    else:
        print("Loading variable codebook...")
        # with DatabaseAPI() as db:
        #     prompts = red_ribbon.socialtoolkit.execute(
        #         "variable_codebook",
        #         question,
        #         database=db, 
        #     )
        print("Variable codebook loaded successfully.")
    return prompts

@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Use AI to answer the question.",
    return_names=["Answers"],)
def prompt_decision_tree(
    laws: Laws,
    ai: Llm,
) -> str:
    """
    Run an AI-powered decision tree to extract answers from a list of documents.
    """
    if DEMO_MODE:
        print("AI is reviewing the laws...")
        random_sleep(5,7)
        answer = "mock_answer"
        print("Law review complete. Answering question...")
        random_sleep(5,7)
        print("Question answered.")
    else:
        documents = laws
        with open("question.txt", "r") as f:
            question = f.read()
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

_LLM_MODELS = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet", "llama-2-70b", "gpt-4o"]

# temperature: float = NumberInput(default=0.7, min=0.1, max=1.0, step=0.1),
# max_tokens: int = NumberInput(default=4096, min=1, max=10000, step=1),
# top_p: float = NumberInput(default=1.0, min=0.1, max=1.0, step=0.1),

@ComfyNode(
    category="Socialtoolkit",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="AI",
    always_run=True,
    return_names=["AI"],)
def llm_api(
    instructions: Prompts = None, # Dummy input
    name: str = Choice(["gpt-4o"]),
) -> Llm:
    """
    Load a large language model (LLM) from a local file or API.
    """
    # Hardcode for the demo
    temperature = 0.7
    max_tokens = 4096
    top_p = 1.0

    if DEMO_MODE:
        print("Loading AI...")
        random_sleep(2,3)
        llm = "mock_llm"
        print("AI loaded.")
    else:
        print("Loading LLM...")
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
    answer: str
):
    """
    Display the LLM's answer.
    """
    print(f"The answer is: {answer}")
    show_text(answer)
    return 


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