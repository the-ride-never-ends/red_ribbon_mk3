#!/usr/bin/env python3
"""Example on parsing an existing PDF file on-disk for ordinances."""
# Standard library imports
import json
import logging
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, TypedDict
import traceback

import time




# Third-party imports
from bs4 import BeautifulSoup
import duckdb
import openai
import requests
import yaml
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from pydantic import (
    AfterValidator as AV,
    BaseModel,
    BeforeValidator as BV,
    DirectoryPath,
    FilePath,
    Field,
    PrivateAttr,
    SecretStr
)


# ELM imports
from elm import ApiBase
from elm.base import ApiBase
from elm.ords.extraction.apply import (
    # check_for_ordinance_info,
    extract_ordinance_text_with_llm,
    extract_ordinance_values
)
from elm.ords.extraction.ordinance import OrdinanceExtractor
from elm.ords.llm import LLMCaller
from elm.ords.services.openai import OpenAIService
from elm.ords.services.provider import RunningAsyncServices as ARun
from elm.ords.utilities import RTS_SEPARATORS
# from elm.utilities import validate_azure_api_params
from elm.web.document import HTMLDocument, PDFDocument
from elm.ords.llm import LLMCaller, StructuredLLMCaller
from elm.ords.extraction.date import DateExtractor


# Other imports
from rex import init_logger



class InitializationError(RuntimeError):
    """Custom exception for errors initializing classes."""
    def __init__(self, message: str):
        super().__init__(message)


SUPPORTED_FILE_TYPES = [
    "html",
    "json",
    "parquet",
]


class HTMLDocumentWithTokens(HTMLDocument):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tokens: int = 0
        self._model = 'gpt-5'
        self.input_price: float | None = 1.25 # https://platform.openai.com/docs/pricing?latest-pricing=standard

    @property
    def token_count(self) -> int:
        if not self._tokens:
            self._tokens = ApiBase.count_tokens(self.text, model=self._model)
        return self._tokens

    @property
    def doc_cost(self) -> float:
        if self.input_price is None:
            return 0.0
        return round(( self.token_count / 1_000_000 ) * self.input_price, 2)


class OUTPUT_COLUMNS(TypedDict):
    county: str
    state: str
    FIPS: str
    feature: str
    fixed_value: str
    mult_value: str
    mult_type: str
    adder: str
    min_dist: str
    max_dist: str
    value: str
    units: str
    ord_year: str
    last_updated: str
    section: str
    source: str
    comment: str


class CHECK_COLS(TypedDict):
    fixed_value: str
    mult_value: str
    adder: str
    min_dist: str
    max_dist: str
    value: str


from pathlib import Path

def _get_project_root(marker='pyproject.toml'):
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Project root with {marker} not found")

_ROOT_DIR = _get_project_root()

class Paths(BaseModel):
    _ROOT_DIR: DirectoryPath = _ROOT_DIR
    _HOME_DIR: DirectoryPath = _ROOT_DIR.parent
    _OUTPUT_TO_HUGGING_FACE_DIR: DirectoryPath = _ROOT_DIR / "output_to_hugging_face"
    _INPUT_FROM_SQL: DirectoryPath = _ROOT_DIR / "input_from_sql"
    _CONFIG_YAML_PATH: DirectoryPath = _ROOT_DIR / "configs.yaml"
    _SQL_CONFIG_YAML_PATH: DirectoryPath = _ROOT_DIR / "sql_configs.yaml"

    def __iter__(self):
        for (attr, value) in self.__dict__.items():
            value: Path
            yield value

    @property
    def ROOT_DIR(self):
        return self._ROOT_DIR
    @property
    def HOME_DIR(self):
        return self._HOME_DIR
    @property
    def OUTPUT_TO_HUGGING_FACE_DIR(self):
        return self._OUTPUT_TO_HUGGING_FACE_DIR
    @property
    def CONFIG_YAML_PATH(self):
        return self._CONFIG_YAML_PATH
    @property
    def SQL_CONFIG_YAML_PATH(self):
        return self._SQL_CONFIG_YAML_PATH
    @property
    def INPUT_FROM_SQL(self):
        return self._INPUT_FROM_SQL

# Iterate over Paths Enum
# paths = Paths()

# for path in paths:
#     if path not in {paths.ROOT_DIR, paths.HOME_DIR}:  # Use set for efficiency
#         if path.suffix in ('.csv', '.yaml', '.txt', '.json'):  # Use .suffix instead of endswith()
#             if not path.parent.exists():
#                 path.parent.mkdir(parents=True, exist_ok=True)
#             if not path.exists():
#                 path.touch()
#                 print(f"Created file {path.name} at {path}")
#         else:
#             if not path.exists():
#                 path.mkdir(parents=True, exist_ok=True)
#                 print(f"Created directory {path.name} at {path}")



def make_log_level_an_int(value: str|int) -> int:
    if isinstance(value, int):
        return value
    else:
        match value.upper():
            case "DEBUG":
                return logging.DEBUG
            case "INFO":
                return logging.INFO
            case "WARNING":
                return logging.WARNING
            case "ERROR":
                return logging.ERROR
            case "CRITICAL":
                return logging.CRITICAL
            case _:
                raise ValueError(f"Invalid log level: {value}")

def check_if_this_type_is_supported(value: str) -> str:
    if value not in SUPPORTED_FILE_TYPES:
        raise NotImplementedError(
            f"File type '{value}' is not currently supported. Supported types are: {', '.join(SUPPORTED_FILE_TYPES)}"
        )
    return value


class Tables(BaseModel):
    """Tables in the database"""
    DATA_TABLE_NAMES: list[str]
    METADATA_TABLE_NAMES: list[str]
    

class Sql(BaseModel):
    """SQL database connection details"""
    HOST: str
    USER: str
    PORT: int
    PASSWORD: str
    DATABASE_NAME: str
    OUTPUT_FOLDER: str = "input_from_sql"
    LIMIT: int = 100000
    BATCH_SIZE: int = 5000
    PARTITION_COLUMN: str = "id"
    COMPRESSION_TYPE: str = "gzip"
    SQL_TYPE: str = "mysql"

    _tables: Tables | None = None

    def __init__(self, **data):
        tables_data = data.pop("TABLES")
        super().__init__(**data)
        self._tables = Tables(**tables_data)

    @property
    def tables(self) -> Tables:
        if self._tables is None:
            raise ValueError("Tables configuration has not been loaded.")
        return self._tables


class Configs(BaseModel):
    """General configuration for the program"""
    HUGGING_FACE_USER_ACCESS_TOKEN: str
    REPO_ID: str
    TARGET_DIR_NAME: str

    BATCH_SIZE: int = Field(default=100, ge=1, le=5000)
    CLEAR_HASHES_CSV: bool = Field(default=True)
    FILE_PATH_ENDING: Annotated[
        str, AV(check_if_this_type_is_supported)
    ] = Field(default="html")

    HUGGING_FACE_UPLOAD_CONCURRENCY_LIMIT: int = Field(default=4, ge=1, le=10)
    LOG_LEVEL: Annotated[
        int, BV(make_log_level_an_int)
    ] = Field(default=logging.INFO, ge=logging.DEBUG, le=logging.CRITICAL)
    GET_FROM_SQL: bool = Field(default=False)
    OPENAI_API_KEY: SecretStr = Field(default=SecretStr(""))

    _paths: Paths = PrivateAttr(default_factory=Paths)
    _sql: Sql | None = None

    def __init__(self, **data):
        paths = Paths()
        print("Loading general configs...")
        with open(paths.CONFIG_YAML_PATH, "r") as f:
            data = dict(yaml.safe_load(f))
        super().__init__(**data)

        print("General configs loaded. Loading Sql configs...")
        with open(paths.SQL_CONFIG_YAML_PATH, "r") as f:
            sql_data = dict(yaml.safe_load(f))

        self._sql = Sql(**sql_data)

        print("Sql Configs loaded.")
        print("All configs loaded successfully.")

    @property
    def paths(self) -> Paths:
        return self._paths

    @property
    def sql(self) -> Sql:
        if self._sql is None:
            raise ValueError("SQL configuration has not been loaded.")
        return self._sql


def _get_model_pricing():
    url = "https://platform.openai.com/docs/pricing?latest-pricing=standard"

    response = requests.get(url)
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch model pricing from {url}. Status code: {response.status_code}")

    soup = BeautifulSoup(response.text, "html.parser")
    pricing_table = soup.find("tbody")
    if not pricing_table:
        raise ValueError("Failed to find pricing table on the OpenAI pricing page.")

    model_pricing = {}
    for row in pricing_table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 2:
            model_name = cols[0].get_text(strip=True)
            model_price_input = cols[1].get_text(strip=True)
            model_price_cached_input = cols[2].get_text(strip=True)
            model_price_output = cols[3].get_text(strip=True)

            model_pricing[model_name] = {
                "input": model_price_input,
                "cached_input": model_price_cached_input,
                "output": model_price_output
            }

    return model_pricing


class DuckDbSetup:

    def __init__(self, *, resources, configs):
        self.resources = resources
        self.configs = configs

        self.db_name: str = configs.sql.DATABASE_NAME
        self.host: str = configs.sql.HOST
        self.user: str = configs.sql.USER
        self.password: str = configs.sql.PASSWORD
        self.compression_type: str = configs.sql.COMPRESSION_TYPE
        self.data_table_names: list[str] = configs.sql.tables.DATA_TABLE_NAMES
        self.metadata_table_names: list[str] = configs.sql.tables.METADATA_TABLE_NAMES
        self.limit: int = configs.sql.LIMIT
        self.batch_size: int = configs.sql.BATCH_SIZE
        self.partition_column: str = configs.sql.PARTITION_COLUMN

        self.sql_type: str = resources['sql_type']

        self.connection_string: str = f"{self.sql_type}://{self.user}:{self.password}@{self.host}/{self.db_name}"
        self.db_typed = f"{self.sql_type}_db"
        self.data_table: str = f"{self.db_typed}.{self.db_name}.{self.data_table_names[0]}"
        self.raw_api_output_table: str = f"{self.db_typed}.{self.db_name}.{self.data_table_names[1]}"
        self.html_metadata_table: str = f"{self.db_typed}.{self.db_name}.{self.metadata_table_names[0]}"
        self.place_metadata_table: str = f"{self.db_typed}.{self.db_name}.{self.metadata_table_names[1]}"

        self.install_db: str | None = f'INSTALL {self.sql_type};'
        self.load_db: str | None = f'LOAD {self.sql_type};'
        self.attach_db: str | None = f"ATTACH '{self.connection_string}' AS {self.db_typed} (TYPE {self.sql_type.upper()}, READ_ONLY);"

    def setup_duckdb(self, duckdb_module: ModuleType) -> ModuleType:
        """Setup DuckDB with SQL extension and establish connection."""
        duckdb_module.sql(self.install_db)
        duckdb_module.sql(self.load_db)
        duckdb_module.sql(self.attach_db)
        return duckdb_module


def _setup_duckdb(
        duckdb_module: ModuleType = duckdb, 
        resources: dict[str, Any] = {}, 
        configs: Configs | None = None
        ) -> tuple[ModuleType, DuckDbSetup]:

    _resources = {
        "sql_type": resources.get("sql_type", "mysql")
    }

    if configs is None:
        raise ValueError("Configs must be provided to setup DuckDB.")

    try:
        setup = DuckDbSetup(resources=_resources, configs=configs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize DuckDbSetup class: {e}") from e
    try:
        module = setup.setup_duckdb(duckdb_module)
    except Exception as e:
        raise InitializationError(f"Failed to setup DuckDB module with MySQL: {e}") from e
    return module, setup


def _make_llm_service(configs):
    try:
        client = openai.AsyncOpenAI(
            api_key=configs.OPENAI_API_KEY.get_secret_value(),
        )
    except Exception as e:
        raise InitializationError(f"Failed to initialize AsyncOpenAI client: {e}") from e
    try:
        llm_service = OpenAIService(client, rate_limit=1e9)
    except Exception as e:
        raise InitializationError(f"Failed to initialize LLM service: {e}") from e
    else:
        print("OpenAI LLM service initialized successfully.")
    return [llm_service]


def _make_ordinance_extractor(logger, kwargs):
    try:
        extractor = OrdinanceExtractor(LLMCaller(**kwargs), logger)
    except Exception as e:
        raise InitializationError(f"Failed to initialize OrdinanceExtractor: {e}") from e
    return extractor


def _make_llm_caller(kwargs):
    try:
        return StructuredLLMCaller(**kwargs)
    except Exception as e:
        raise InitializationError(f"Failed to initialize StructuredLLMCaller: {e}") from e




def _make_text_splitter(*, model: str) -> RecursiveCharacterTextSplitter:
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            RTS_SEPARATORS,
            chunk_size=3000,
            chunk_overlap=300,
            length_function=partial(ApiBase.count_tokens, model=model),
        )
    except Exception as e:
        raise InitializationError(f"Failed to initialize text splitter: {e}") from e
    return text_splitter


from .architecture.validation import make_ordinance_extractor, make_ordinance_validator


async def check_for_ordinance_info(doc, text_splitter, ordinance="wind", **kwargs):
    """Parse a single document for ordinance information.

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document potentially containing ordinance information. Note
        that if the document's metadata contains the
        ``"contains_ord_info"`` key, it will not be processed. To force
        a document to be processed by this function, remove that key
        from the documents metadata.
    text_splitter : obj
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    **kwargs
        Keyword-value pairs used to initialize an
        `elm.ords.llm.LLMCaller` instance.

    Returns
    -------
    elm.web.document.BaseDocument
        Document that has been parsed for ordinance text. The results of
        the parsing are stored in the documents metadata. In particular,
        the metadata will contain a ``"contains_ord_info"`` key that
        will be set to ``True`` if ordinance info was found in the text,
        and ``False`` otherwise. If ``True``, the metadata will also
        contain a ``"date"`` key containing the most recent date that
        the ordinance was enacted (or a tuple of `None` if not found),
        and an ``"ordinance_text"`` key containing the ordinance text
        snippet. Note that the snippet may contain other info as well,
        but should encapsulate all of the ordinance text.
    """
    info_key, date_key, text_key = "contains_ord_info", "date", "ordinance_text"
    if info_key in doc.attrs:
        return doc

    llm_caller = _make_llm_caller(kwargs)
    chunks = text_splitter.split_text(doc.text)
    resources = {
        "ordinance": ordinance,
        "llm_caller": llm_caller,
        "chunks": chunks
    }
    validator = make_ordinance_validator(resources)
    doc.attrs[info_key] = await validator.parse()
    if doc.attrs[info_key]:
        doc.attrs[date_key] = await DateExtractor(llm_caller).parse(doc)
        doc.attrs[text_key] = validator.ordinance_text

    return doc



def main():
    try:
        print("Starting logging...")
        init_logger('elm', log_level='DEBUG')

        print("Loading configurations...")
        configs = Configs()
        print("Configurations loaded successfully.")
        logger = logging.getLogger(__name__)

        test_gnis = "2411080" # City of Merced, CA
        ordinance = "wind"
        model = "gpt-5"

        fp_txt_all = test_gnis + '_all.txt'
        fp_txt_clean = test_gnis + '_clean.txt'
        fp_ords = test_gnis + '_ords.csv'

        conn: duckdb.DuckDBPyConnection | None = None

        print("Setting up duckdb connection to MySQL server...")
        conn, setup = _setup_duckdb(configs=configs)
        print("DuckDB connection established successfully.")

        if conn is not None:
            df = conn.sql(f"SELECT * FROM {setup.raw_api_output_table} WHERE gnis={test_gnis}").to_df()
            print(f"{len(df)} rows retrieved from DuckDB.")
            # print(f"DEBUG df: {df.head()}")
            # print(f"DEBUG df columns: {df.columns.tolist()}")
        else:
            raise RuntimeError("Failed to establish DuckDB connection: conn object was None.")

        debug_stop = 0
        html_list = []

        for row in df.itertuples():
            content_json = json.loads(row.content_json)
            html = f"{content_json["Content"]}\n{content_json["TitleHtml"]}"
            html_list.append(html)
            debug_stop += 1

        doc = HTMLDocumentWithTokens(html_list)
        print("Document parsed and loaded successfully.")
        print(f"Total document tokens: {doc.token_count}")
        print(f"Total document cost (${doc.input_price} USD per million tokens): ${doc.doc_cost}")

        text_splitter = _make_text_splitter(model=model)

        # # # setup LLM and Ordinance service/utility classes
        # # azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()

        print("Initializing LLM services...")
        services = _make_llm_service(configs=configs)
        print("LLM services initialized successfully.")
        print("Initializing OrdinanceExtractor...")
        kwargs = {
            "llm_service": services,
            "model": model,
            "temperature": 0
        }
        extractor = _make_ordinance_extractor(logger, kwargs)
        print("OrdinanceExtractor initialized successfully.")

        # """The following three function calls present three (equivalent) ways to
        # call ELM async ordinance functions. The three functions 1) check ordinance
        # documents for relevant ordinance info, 2) extract the relevant text, and 3)
        # run the decision tree to get structured ordinance data from the
        # unstructured legal text."""

        # # 1) call async func using a partial function (`run_async`)
        # checker = check_for_ordinance_info(
        #     doc, text_splitter, ordinance=ordinance, **kwargs
        # )
        # doc = ARun.run(services[0], checker)

        run_async = partial(ARun.run, services)
        doc = run_async(check_for_ordinance_info(doc, text_splitter, ordinance=ordinance, **kwargs)) # NOTE Relevance Assessment

        # # 2) Build coroutine first the use it to call async func
        # # (extract_ordinance_text_with_llm is an async function)
        # extract = extract_ordinance_text_with_llm(doc, text_splitter, extractor) # NOTE Extract Ordinance Text
        # doc = ARun.run(services, extract)

        # # 3) Build coroutine and use it to call async func in one go
        # doc = ARun.run(services, extract_ordinance_values(doc, **kwargs)) # NOTE Decision Tree

        # # save outputs
        doc.attrs['ordinance_values'].to_csv(fp_ords)
        with open(fp_txt_all, 'w') as f:
            f.write(doc.attrs["ordinance_text"])
        with open(fp_txt_clean, 'w') as f:
            f.write(doc.attrs["cleaned_ordinance_text"])
    except KeyboardInterrupt:
        print("Pipeline interrupted by user.")
        return 1
    except Exception as e:
        print(f"An error occurred: {e}:\n{traceback.format_exc()}")
        return 1
    else:
        print("Pipeline completed successfully.")
        return 0 

if __name__ == "__main__":
    sys.exit(main())
