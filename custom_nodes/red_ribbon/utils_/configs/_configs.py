"""
Configs module - Configuration constants loaded from YAML files.
"""
import logging
import os
from pathlib import Path
from typing import Any, Optional, Literal


from openai.types import EmbeddingModel, ChatModel, ModerationModel
from pydantic import (
    BaseModel, 
    DirectoryPath, 
    Field, 
    FilePath, 
    PositiveInt, 
    PositiveFloat, 
    NonNegativeInt, 
    SecretStr, 
    ValidationError
)
import yaml


from custom_nodes.red_ribbon._custom_errors import ConfigurationError
from ..common import get_value_from_base_model, get_value_with_default_from_base_model


_VERSION_DIR = Path(__file__).parent.parent.parent


assert (_VERSION_DIR / "__version__.py").exists(), f"_VERSION_DIR is incorrectly specified: {_VERSION_DIR}"


class DatabaseConfigs(BaseModel):
    AMERICAN_LAW_DATA_DIR:            DirectoryPath = _VERSION_DIR / "data"
    DATABASE_CONNECTION_POOL_SIZE:    PositiveInt = 10
    DATABASE_CONNECTION_TIMEOUT:      PositiveInt = 30
    DATABASE_CONNECTION_MAX_OVERFLOW: PositiveInt = 20
    DATABASE_CONNECTION_MAX_AGE:      PositiveInt = 300


class Paths(BaseModel):
    THIS_FILE:             DirectoryPath = Path(__file__).resolve()
    THIS_DIR:              DirectoryPath = THIS_FILE.parent
    VERSION_DIR:           DirectoryPath = _VERSION_DIR
    CUSTOM_NODES_DIR:      DirectoryPath = THIS_DIR.parent
    COMFYUI_DIR:           DirectoryPath = CUSTOM_NODES_DIR.parent
    LLM_OUTPUTS_DIR:       DirectoryPath = COMFYUI_DIR / "output" / "red_ribbon_outputs"
    LLM_MODELS_DIR:        DirectoryPath = COMFYUI_DIR / "models" / "llm_models"
    SOCIALTOOLKIT_DIR:     DirectoryPath = VERSION_DIR / "socialtoolkit"
    DATABASE_DIR:          DirectoryPath = VERSION_DIR / "database"
    DB_PATH:               FilePath      = VERSION_DIR / "red_ribbon.db"
    AMERICAN_LAW_DATA_DIR: DirectoryPath = VERSION_DIR / "data"
    AMERICAN_LAW_DB_PATH:  FilePath      = VERSION_DIR / "data" / "american_law.db"
    PROMPTS_DIR:           DirectoryPath = VERSION_DIR / "utils_" / "llm_" / "prompts"

    def __getitem__(self, key: str) -> Optional[Any]:
        return get_value_from_base_model(self, key)

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        return get_value_with_default_from_base_model(self, key, default)

    class ConfigDict:
        frozen = True  # Make the model immutable (read-only)

try:
    _PATHS = Paths()
except ValidationError as e:
    raise ConfigurationError(f"_Paths failed to validate in _configs.py: {e}") from e
except Exception as e:
    raise ConfigurationError(f"Unexpected error while initializing global paths in _configs.py: {e}") from e


class VariableCodebookConfigs(BaseModel):
    variable_file_paths: dict[str, str] = Field(default_factory=dict)
    load_from_file: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    default_assumptions_enabled: bool = True

    def __getitem__(self, key: str) -> Optional[Any]:
        return get_value_from_base_model(self, key)

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        return get_value_with_default_from_base_model(self, key, default)



# NOTE: All lower-case field names are considered mutable.
class Configs(BaseModel):
    """
    Configuration constants for the red ribbon package.
    All UPPER_CASE fields are read-only, lower_case fields are mutable.

    Attributes:
        database (DatabaseConfigs): Pydantic model of Database related configurations.
        paths (Paths): Pydantic model of various important paths.
        variable_codebook (VariableCodebookConfigs): Pydantic model of Variable Codebook configurations
    
        RETRIEVAL_COUNT (PositiveInt): Number of documents to retrieve.
        similarity_threshold (float): Minimum similarity score.
        RANKING_METHOD (str): Method for ranking documents. Options: cosine_similarity, dot_product, euclidean.
        USE_FILTER (bool): Whether to filter results.
        FILTER_CRITERIA (dict[str, Any]): Criteria for filtering results.
        USE_RERANKING (bool): Whether to use re-ranking.
        OPENAI_API_KEY (SecretStr): OpenAI API key.
        OPENAI_MODEL (str): OpenAI model to use.
        OPENAI_SMALL_MODEL (str): Smaller OpenAI model to use.
        OPENAI_EMBEDDING_MODEL (str): OpenAI embedding model to use.
        EMBEDDING_DIMENSIONS (PositiveInt): Dimensions of the embeddings.
        TEMPERATURE (PositiveFloat): Temperature setting for the LLM.
        MAX_TOKENS (PositiveInt): Maximum tokens for the LLM.
        LOG_LEVEL (Literal[10, 20, 30, 40, 50]): Logging level.
        SIMILARITY_SCORE_THRESHOLD (float): Similarity score threshold.
        SEARCH_EMBEDDING_BATCH_SIZE (NonNegativeInt): Batch size for searching embeddings.
        connection_string (Optional[str]): Database connection string.
        timeout (Optional[PositiveInt]): Database timeout in seconds.
    """
    database:          DatabaseConfigs = Field(default_factory=DatabaseConfigs)
    paths:             Paths = Field(default=_PATHS)
    variable_codebook: VariableCodebookConfigs = Field(default_factory=VariableCodebookConfigs)


    # Pipeline Configurations
    INPUT_DATAPOINT: str = "sales tax info"  # Example input data point
    get_from_internet: bool = True  # Whether to retrieve documents from the internet

    # Top-10 Document Retrieval
    RETRIEVAL_COUNT: PositiveInt = 10  # Number of documents to retrieve
    similarity_threshold: float = 0.6  # Minimum similarity score
    RANKING_METHOD: str = "cosine_similarity"  # Options: cosine_similarity, dot_product, euclidean
    USE_FILTER: bool = False  # Whether to filter results
    FILTER_CRITERIA: dict[str, Any] = Field(default_factory=dict)
    USE_RERANKING: bool = False  # Whether to use re-ranking
    OUTPUT_DIR: DirectoryPath = _PATHS.LLM_OUTPUTS_DIR  # Directory to save outputs

    # LLM
    OPENAI_API_KEY:                   SecretStr = Field(default_factory=lambda: SecretStr(os.environ.get("OPENAI_API_KEY", "")), min_length=1)
    OPENAI_MODEL:                     ChatModel = Field(default="gpt-4o-mini", min_length=1)
    OPENAI_SMALL_MODEL:               ChatModel = Field(default="gpt-5-nano", min_length=1)
    OPENAI_EMBEDDING_MODEL:           EmbeddingModel = Field(default="text-embedding-3-small", min_length=1)
    OPENAI_MODERATION_MODEL:          ModerationModel = Field(default="omni-moderation-latest", min_length=1)
    DEFAULT_SYSTEM_PROMPT:            str = Field(default="You are a helpful assistant.")
    EMBEDDING_DIMENSIONS:             PositiveInt = 1536
    TEMPERATURE:                      PositiveFloat = 0.0
    MAX_TOKENS:                       PositiveInt = 4096
    LOG_LEVEL:                        Literal[10, 20, 30, 40, 50] = logging.DEBUG
    SIMILARITY_SCORE_THRESHOLD:       float = 0.4
    SEARCH_EMBEDDING_BATCH_SIZE:      NonNegativeInt = 10000

    connection_string:                Optional[str] = None  # Database connection string
    timeout:                          Optional[PositiveInt] = None  # Database timeout in seconds

    def __setattr__(self, name: str, value: Any) -> None:
        if not isinstance(name, str):
            raise TypeError(f"Attribute name must be a string, got {type(name.__name__)}")
        if name.isupper():
            raise ConfigurationError(f"Cannot modify read-only configuration: {name}")
        return super().__setattr__(name, value) 

    def __getitem__(self, key: str) -> Optional[Any]:
        return get_value_from_base_model(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(f"Key must be a string, got {type(key.__name__)}")
        if key.isupper():
            raise ConfigurationError(f"Cannot modify read-only configuration key: {key}")
        try:
            setattr(self, key, value)
        except AttributeError:
            raise KeyError(f"Key '{key}' not found in Configs")

    def get(self, key: str, default: Any | None = None) -> Optional[Any]:
        return get_value_with_default_from_base_model(self, key, default)

try:
    configs = Configs()
except ValidationError as e:
    raise ConfigurationError(f"Default global configs failed to validate in _configs.py: {e}") from e
except Exception as e:
    raise ConfigurationError(f"Unexpected error while initializing global configs in _configs.py: {e}") from e

# class Paths(BaseModel):
#     THIS_FILE = Path(__file__).resolve()
#     THIS_DIR = THIS_FILE.parent
#     CUSTOM_NODES_DIR = THIS_DIR.parent
#     COMFYUI_DIR = CUSTOM_NODES_DIR.parent
#     LLM_OUTPUTS_DIR = COMFYUI_DIR / "output" / "red_ribbon_outputs"
#     LLM_MODELS_DIR = COMFYUI_DIR / "models" / "llm_models"

#     class Config:
#         frozen = True  # Make the model immutable (read-only)

# # class RedRibbonConfigs(BaseModel):
# #     pass


# # class SocialToolkitConfigs(BaseModel):
# #     """Configuration for High Level Architecture workflow"""
# #     approved_document_sources: list[str]
# #     llm_api_config: dict[str, Any]
# #     document_retrieval_threshold: int = 10
# #     relevance_threshold: float = 0.7
# #     output_format: str = "json"

# #     codebook: Optional[dict[str, Any]] = None
# #     document_retrieval: Optional[dict[str, Any]] = None
# #     llm: Optional[dict[str, Any]] = None
# #     top10_retrieval: Optional[dict[str, Any]] = None
# #     relevance_assessment: Optional[dict[str, Any]] = None
# #     prompt_decision_tree: Optional[dict[str, Any]] = None


# # class ConfigsBase(BaseModel):
# #     """Base model for configuration with read-only fields."""
    
# #     class Config:
# #         frozen = True  # Make the model immutable (read-only)


# # @lru_cache()
# # def get_config():
# #     """
# #     Load configuration from YAML files and cache the result.
# #     Returns a read-only Configs object.
# #     """
# #     base_dir = os.path.dirname(os.path.abspath(__file__))
    
# #     # Load main configs
# #     config_path = os.path.join(base_dir, "configs.yaml")
# #     config_data = {}
# #     if os.path.exists(config_path):
# #         with open(config_path, 'r') as f:
# #             config_data = yaml.safe_load(f) or {}
    
# #     # Load private configs (overrides main configs)
# #     private_config_path = os.path.join(base_dir, "private_configs.yaml")
# #     private_config_data = {}
# #     if os.path.exists(private_config_path):
# #         with open(private_config_path, 'r') as f:
# #             private_config_data = yaml.safe_load(f) or {}
    
# #     # Merge configs, with private taking precedence
# #     merged_config = {**config_data, **private_config_data}
    
# #     return Configs(**merged_config)


# class Configs(BaseModel):
#     """
#     Configuration constants loaded from YAML files.
#     All fields are read-only. 
    
#     Loads from:
#     - configs.yaml (base configuration)
#     - private_configs.yaml (overrides base configuration)
#     """
#     # Add your configuration fields here with defaults
#     # Example:
#     API_URL: str = Field(default="http://localhost:8000", description="API URL")
#     DEBUG_MODE: bool = Field(default=False, description="Enable debug mode")
#     MAX_BATCH_SIZE: int = Field(default=4, description="Maximum batch size")
#     MODEL_PATHS: Dict[str, str] = Field(default_factory=dict, description="Paths to models")
#     CUSTOM_SETTINGS: Dict[str, Any] = Field(default_factory=dict, description="Custom configuration settings")

#     # _paths: Paths = Field(default_factory=Paths)
#     # _socialtoolkit: SocialToolkitConfigs = Field(default_factory=SocialToolkitConfigs)
#     # _red_ribbon: RedRibbonConfigs = Field(default_factory=RedRibbonConfigs)
    
#     # # Access the singleton instance through this class method
#     # @classmethod
#     # def get(cls) -> 'Configs':
#     #     """Get the singleton instance of Configs."""
#     #     return get_config()
    
#     # @property
#     # def paths(self) -> Paths:
#     #     return self._paths
    
#     # @property
#     # def socialtoolkit(self) -> SocialToolkitConfigs:
#     #     return self._socialtoolkit
    
#     # @property
#     # def red_ribbon(self) -> RedRibbonConfigs:
#     #     return self._red_ribbon