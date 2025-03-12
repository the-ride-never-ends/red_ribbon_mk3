"""
Main module for importing and registering ComfyUI nodes in the Red Ribbon package\n
Author: Kyle Rose, Claude 3.7 Sonnet\n
Version: 0.1.0
"""
DEMO_MODE = True
IN_COMFY = True


import contextvars
from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
import importlib
from importlib import resources
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Callable, Generator, Type, TypeVar, Optional


class RedRibbonError(Exception):
    """
    Error that occurred in the main.py file.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"RedRibbonError: {self.message}"

try:
    import torch
    import comfy
except ImportError as e:
    msg = "Critical import not found. Please install Comfy to use this package."
    raise RedRibbonError(msg) from e

# OPEN_AI_API_KEY = "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"
# try:
#     subprocess.run([f'export OPENAI_API_KEY="{OPEN_AI_API_KEY}"'], shell=True, check=True)
# except subprocess.CalledProcessError as e:
#     print(f"Failed to set OpenAI API key. Please set it manually: {e}")
#     sys.exit(1)

from easy_nodes import (
    NumberInput,
    ComfyNode,
    StringInput,
    Choice,
    show_text,
    register_type,
)
from torch import nn
from easy_nodes.easy_nodes import AnythingVerifier, _curr_preview as easy_nodes_curr_preview
from networkx import DiGraph # NOTE We do this so that we can register the nx.DiGraph type in ComfyUI
from pydantic import BaseModel, Field
import openai
from tqdm import tqdm


# Import components from subdirectories
# Modules
import comfy.utils
from .socialtoolkit.socialtoolkit import SocialToolkitAPI, SocialToolKitResources
from .red_ribbon_core.red_ribbon import RedRibbonAPI, RedRibbonResources
from .plug_in_play_transformer.plug_in_play_transformer import TransformerAPI, TransformerResources


# Utility functions
from .configs import Configs # TODO figure out what the hell is up with imports. It makes EasyNodes not so easy to debug!
from .database import DatabaseAPI
from .llm import Llm
from .logger import get_logger
from .utils.main_.red_ribbon_banner import get_red_ribbon_banner
from .utils.common.safe_format import safe_format


ModuleType = TypeVar('ModuleType')
Class = TypeVar('Class')
ClassInstance = TypeVar('ClassInstance')


class Implementations(str, Enum):
    """
    The available client implementations for each part of the the Red Ribbon package.
    
    """
    # Database Implementations
    DUCKDB = "DuckDB"
    SQLITE = "SQLite"
    POSTGRESQL = "PostgreSQL"
    MYSQL = "MySQL"

    # LLM Implementations
    OPENAI = "OpenAI"

    # Transformer Implementations
    HUGGINGFACE = "HuggingFace"
    TORCH = "Torch"

    # Logger implementations
    PYTHON = "Python"
    PYDANTIC = "Pydantic"


def get_implementations(impl_class: Implementations) -> Generator[ModuleType, None, None]:
    """Get the implementations for the Red Ribbon package"""
    implementations = [imp.lower() for imp in impl_class]
    # Check to see if the module exists.
    this_dir = Path(__file__).parent
    for file, _, _, in this_dir.walk():
        if file.stem in implementations:
            module = importlib.import_module(f"{__package__}.{file.stem}")
            return module


def get_resources(module_name: str) -> 'Resources' | dict[str, Class]:
    pass


class Resources(BaseModel):
    resources: list[ModuleType] = Field(default_factory=get_resources)

    @cached_property
    def resources(self):
        return {
            "document_retrieval_from_websites": self.document_retrieval_from_websites,
            "document_storage": self.document_storage,
            "top10_document_retrieval": self.top10_document_retrieval,
            "relevance_assessment": None
        }

    @cached_property
    def database(self):
        return DatabaseAPI(self.resources, configs)

    @cached_property
    def socialtoolkit(self):
        return SocialToolkitAPI(self.resources, configs)

    @cached_property
    def comfy(self):
        return ComfyType(socialtoolkit=self.socialtoolkit)

    @cached_property
    def rr(self):
        return RedRibbonAPI(self.resources, configs)

    def __getitem__(self, key: str) -> Optional[Any]:
        try:
            return dict(self)[key]
        except KeyError:
            return None

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        return self.__getitem__(key) or default

def _get_value_from_base_model(self: BaseModel, key: str) -> Any:
    try:
        return dict(self)[key]
    except KeyError as e:
        raise KeyError(f"Key '{key}' not found in {self.__qualname__}") from e




class ComfyType(BaseModel):
    pass



class RedRibbon:
    """Main interface for the Red Ribbon package"""

    def __init__(self, resources: dict[str, Class] = None, configs: Configs = None):
        """Initialize the Red Ribbon package components"""
        self.configs = configs
        self.resources = resources or {}
        self.logger = get_logger(self.__class__.__name__)
        self.llm = self.resources["llm"] or openai.OpenAI()

        self.socialtoolkit: Type[SocialToolkitAPI] = self.resources["social"]
        self.database: Type[DatabaseAPI] = self.resources["database"]
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
    
    @property
    def demo_mode(self):
        """Get the demo mode status"""
        

    def _print_startup_message(self):
        dont_print_these_attributes = ["configs", "resources", "logger", "_missing_attributes" , "client"]
        dont_print_these_attributes.extend(self._missing_attributes)
        attrs = [
            attr for attr in self.__dict__.keys() if attr not in dont_print_these_attributes
        ]
        available_nodes = "\n".join(f"{i}. {name}" for i, name in enumerate(attrs, start=1))
        red_ribbon_banner = get_red_ribbon_banner(without_logo=True)
        for line in red_ribbon_banner:
            print(line)
        print(f"""
                                    Red Ribbon loaded successfully.
                                    Version: {self.version}
                                    DEMO MODE is {'ON' if DEMO_MODE else 'OFF'}
                                    *****************
                                    Available Modules:
                                    {available_nodes}
        """)
        # print(f"Version: {self.version}")
        # print(f"DEMO MODE is {'ON' if DEMO_MODE else 'OFF'}")
        # print("Available Modules:")
        # print(available_nodes)






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

# Laws = TypeVar("Laws", list[str], str, dict, tuple[Documents, Metadata, Vectors, Optional[Prompts], Optional[Answers]])
LlmApi = TypeVar("LlmApi", str, Llm, dict)

@dataclass
class Laws:
    documents: Optional[Documents] = None
    metadata: Optional[Metadata] = None
    vectors: Optional[Vectors] = None
    prompts: Optional[Prompts] = None
    answers: Optional[Answers] = None




# Register the types with ComfyUI
types = {
    "Database": DatabaseAPI, "Llm": Llm, "Configs": Configs, 
    "Prompts": Prompts, "DiGraph": DiGraph, "dict": dict,
    "Vectors": Vectors, "Documents": Documents, "Urls": Urls,
    "Metadata": Metadata, "AnyType": AnyType, "Answers": Answers,
    "Excel": Excel, "Data": Data, "Laws": Laws, "LlmApi": LlmApi,
}
for type_name, type_class in types.items():
    try:
        type_class.__qualname__ 
    except AttributeError:  # If the class doesn't have a __qualname__ attribute, monkeypatch one in.
         # This came up when testing TypeVar aliases.
         type_class.__qualname__ = type_name
         #print(f"Added __qualname__ to type {type_class.__name__}")

    register_type(type_class, type_name, verifier=AnythingVerifier())


class ModuleType(str, Enum):
    pass





configs = Configs()
resources = Resources()
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


def stream_llm_response(llm: dict, func_name: str, **kwargs) -> None:
    print(f"llm_dict:\n{llm}")
    messages = llm['ai']['messages'][func_name][1]["content"]
    print(f"messages: {messages}")
    if kwargs:
        llm['ai']['messages'][func_name][1]["content"] = safe_format(messages, **kwargs)
    collected_chunks = []
    collected_messages = []
    response = llm['ai']['client'].chat.completions.create(
        model=llm["model"],
        messages=llm['ai']['messages'][func_name][1]["content"],
        temperature=0.3,
        stream=True
    )
    timer_wait = 0.01
    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk.choices[0].delta.content  # extract the message
        collected_messages.append(chunk_message)  # save the message
        print_chunk = []
        for message in collected_messages:
            if message is None:
                break
            print_chunk.append(message)
            flattened_chunk = "".join(print_chunk)
            sys.stdout.write('\r')
            sys.stdout.write(flattened_chunk)
            sys.stdout.flush()
            time.sleep(timer_wait)


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

        # Select the database type and initialize is as a DatabaseAPI
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

_DEMO_RELEVANT_URLS = {
    # Springhill, LA
    "springhill": [
        "https://library.municode.com/la/springhill/codes/code_of_ordinances?nodeId=COOR_CH98TA_ARTIINGE_S98-1SAUSTA",
        "https://taxfoundation.org/location/louisiana/",
        "https://webstersalestax.org/current-rates/"
    ],
    # Cheyenne, WY
    "cheyenne": [
        "https://library.municode.com/wy/cheyenne/codes/code_of_ordinances?nodeId=TIT3REFI_CH3.08TA_3.08.010REEXINRE",
        "https://taxfoundation.org/location/wyoming/",
        "https://www.avalara.com/taxrates/en/state-rates/wyoming/counties/laramie-county.html",
        "https://www.cityoflaramie.org/FAQ.aspx?QID=104",
    ],
    # San Jose, CA
    "san_jose": [
        "https://library.municode.com/ca/san_jose/codes/code_of_ordinances?nodeId=TIT20ZO_CH20.80SPUSRE_PT10OUVEFA_20.80.820EXDMPE"
        "https://library.municode.com/ca/san_jose/codes/code_of_ordinances?nodeId=TIT6BULIRE_CH6.54PEPEOR",
    ],
}

def flatten_nested_dictionary_into_list(input_dict: dict) -> list:
    output = []
    for key, value in input_dict.items():
        if isinstance(value, list):
            output.extend(value)
        else:
            if isinstance(value, dict):
                flatten_nested_dictionary_into_list(value)
            else:
                output.append(value)
    return output

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

def mock_internet_check(links: Urls) -> None:
    print("Gettings laws from the web...")
    #random_sleep()
    print("Checking government websites...")
    #random_sleep()
    print("Checking Google...")
    #random_sleep()
    print("Checking Bing...")
    random_sleep()
    
    len_domain_urls = len(links)
    random_number_of_urls = len_domain_urls * round(random.uniform(0, 5))
    print(f"Found {random_number_of_urls} laws. Downloading...")
    return random_number_of_urls



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
        mock_vectors = []
        random_number_of_urls = mock_internet_check(links)
        pbar = comfy.utils.ProgressBar(total=random_number_of_urls)
        for i,_ in enumerate([random_number_of_urls], start=1):
            mock_vectors.append([random.uniform(-5.0, 5.0) for _ in range(_VECTOR_LENGTH)])
            time.sleep(0.1)
            #random_sleep(1,2)
            print(f"Downloading law {i}.")
            if i == 1:
                pbar.update(1)
            else:
                pbar.update(i+1)
        print("Laws downloaded. Printing...")
        link_dict = flatten_nested_dictionary_into_list(_DEMO_RELEVANT_URLS)
        for link in link_dict:
            print(link)

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
    #documents, metadata, vectors = laws['documents'], laws["metadata"], laws["vectors"]

    if DEMO_MODE:
        print("Saving laws to disk...")
        random_sleep(2,5)
        print("Laws saved.")
        return {
            "documents": [f"mock_document {i}" for i in range(10)],
            "metadata": "mock_metadata",
            "vectors": [[random.uniform(-5.0, 5.0) for _ in range(_VECTOR_LENGTH)] for _ in range(10)]
        }
    else:
        #with DatabaseAPI() as db:
            return red_ribbon.socialtoolkit.execute(
                "document_storage",
                db_service=None,
                documents=laws.documents, 
                metadata=laws.metadata, 
                vectors=laws.vectors
            )


INPUT_TEXT_OPTIONS = [
    "Local Sales Tax in Cheyenne, WY",
    "What is the local sales tax in Springhill, Lousiana?",
    "List exceptions to local vending machine laws in San Jose, CA",
]

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
    # documents, vectors = laws['documents'], laws["vectors"]
    if DEMO_MODE:
        print(f"Filtering down to {return_how_many} documents...")
        random_sleep(3,5)
        print("Laws filtered successfully.")
        return laws
    else:
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
        ai: LlmApi, 
        return_how_many: int = NumberInput(default=3, min=1, max=100, step=1, display="slider") # Dummy slider.
    ) -> Laws:
    """
    Determine how relevant a list of documents are to a query.
    """
    return_how_many
    if DEMO_MODE:
        print("Filtering laws with AI...")
        random_sleep(1,2)
        print("Laws filtered successfully. Printing results:")
        question = ai['question']
        documents = []
        match question:
            case question if "Cheyenne" in question:
                documents = _DEMO_RELEVANT_URLS['cheyenne']
            case question if "Springhill" in question:
                documents = _DEMO_RELEVANT_URLS['springhill']
            case question if "San Jose" in question:
                documents = _DEMO_RELEVANT_URLS['san_jose']
            case _: # Default case is to return all the documents
                sub_dicts = _DEMO_RELEVANT_URLS.values()
                for doc in sub_dicts:
                    documents.extend(doc)
        #stream_llm_response(ai, "relevance_assessment", relevant_docs=documents)
        print(f"#########\n{documents}")
        laws['documents'] = documents
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
    return question


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
        global ANSWER
        print("Loading AI instructions...")
        random_sleep(2,3)
        match question:
            case question if "Cheyenne" in question:
                print("Found applicable instructions: Local Sales Tax.")
                ANSWER = answer = "6.0%"
            case question if "Springhill" in question:
                print("Found applicable instructions: Local Sales Tax.")
                ANSWER = answer = "10.5%"
            case question if "San Jose" in question:
                print("Found applicable instructions: Vending Machine Laws.")
                ANSWER = answer = "Exceptions for Holiday sales, city-approved locations, newsracks, on-site businesses, and farmers' markets."
            case _:
                ANSWER = answer = "mock_prompts"
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
    #print(f"prompt decision tree laws: {laws}")
    if DEMO_MODE:
        show_text("AI is reviewing the laws...")
        random_sleep(3,4)
        show_text("Law review complete. Answering question...")
        random_sleep(3,4)
        print("Question answered.")
        answer = "blank" #laws['answer']
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

_LLM_MODELS = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet", "gpt-4o"]

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
    instructions: Prompts = None,
    name: str = Choice(["gpt-4o"]),
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
        print(f"AI model '{name}' selected.")
        print("Loading AI...")
        random_sleep(2,3)
        print(f"{name} loaded.")
        
        llm = {
            "model": name,
            "answer": instructions[0],
            "instructions": instructions[1],
            "question": instructions[1],
            "prompts": instructions[1],
            "ai": {
                "client": red_ribbon.client,
                "messages": {
                    f'llm_api': [],
                    f"relevance_assessment": []
        },}}
        system_prompt: dict = {"role": "system", "content": "You are a helpful assistant.\nYou are currently assisting in a demo presentation for software product.\nSpeak in a stereotypically robotic tone."}
        llm['ai']['messages']["llm_api"] = [
            system_prompt,
            {"role": "user", "content": "You've been initialized. Say hello to the audience!"},
        ]
        llm['ai']['messages']["relevance_assessment"] = [
            system_prompt,
            {"role": "user", "content": "You've found these URLs to be relevant. Present them to the audience.\n###### {relevant_docs}"},
        ]
        # Let's see what happens!
        #stream_llm_response(llm, "llm_api")
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
    answer: Answers
):
    """
    Display the LLM's answer.
    """
    print(f"The answer is: {ANSWER}")
    show_text(ANSWER)
    return 

######## Plug-in-play Transformer Nodes ########

from torch import Tensor

torch_types = {
    "Tensor": Tensor,
}


for type_name, type_class in types.items():
    try:
        type_class.__qualname__ 
    except AttributeError:  # If the class doesn't have a __qualname__ attribute, monkeypatch one in.
         # This came up when testing TypeVar aliases.
         type_class.__qualname__ = type_name
         #print(f"Added __qualname__ to type {type_class.__name__}")

    register_type(type_class, type_name, verifier=AnythingVerifier())



@ComfyNode(
    category="Plug-in-Play Transformer/residual",
    color="#1f1f1f",
    bg_color="#454545",
    display_name="Add Residual")
def add_residual(
    x: Tensor,
    residual: Tensor,
) -> Tensor:
    """
    Add a residual connection to the input tensor.

    Inputs:
      - x: Original input tensor
      - residual: Tensor to be added as residual
    
    Outputs:
      - output: Result after adding residual
    """
    return x + residual





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