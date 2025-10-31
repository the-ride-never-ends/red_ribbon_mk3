"""
Configs module - Configuration constants loaded from YAML files.
"""
from pathlib import Path
from typing import Any, Optional



from pydantic import BaseModel, DirectoryPath, Field, FilePath
import yaml


from ..common.get_value_from_base_model import get_value_from_base_model
from ..common.get_value_with_default_from_base_model import get_value_with_default_from_base_model
import custom_nodes.red_ribbon.main as main_module


class DatabaseConfigs(BaseModel):
    pass


class Paths(BaseModel):
    THIS_FILE:         DirectoryPath = Path(__file__).resolve()
    THIS_DIR:          DirectoryPath = THIS_FILE.parent
    VERSION_DIR:       DirectoryPath = Path(main_module.__file__).parent
    CUSTOM_NODES_DIR:  DirectoryPath = THIS_DIR.parent
    COMFYUI_DIR:       DirectoryPath = CUSTOM_NODES_DIR.parent
    LLM_OUTPUTS_DIR:   DirectoryPath = COMFYUI_DIR / "output" / "red_ribbon_outputs"
    LLM_MODELS_DIR:    DirectoryPath = COMFYUI_DIR / "models" / "llm_models"
    SOCIALTOOLKIT_DIR: DirectoryPath = VERSION_DIR / "socialtoolkit"
    DATABASE_DIR:      DirectoryPath = VERSION_DIR / "database"
    DB_PATH:           FilePath = VERSION_DIR / "red_ribbon.db"

    def __getitem__(self, key: str) -> Optional[Any]:
        return get_value_from_base_model(self, key)

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        return get_value_with_default_from_base_model(self, key, default)

    class ConfigDict:
        frozen = True  # Make the model immutable (read-only)


class VariableCodebookConfigs(BaseModel):
    variables_path: str = "variables.json"
    load_from_file: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    default_assumptions_enabled: bool = True

    def __getitem__(self, key: str) -> Optional[Any]:
        return get_value_from_base_model(self, key)

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        return get_value_with_default_from_base_model(self, key, default)


class Configs(BaseModel):
    database:          DatabaseConfigs = None
    paths:             Paths = Field(default_factory=Paths)
    variable_codebook: VariableCodebookConfigs = Field(default_factory=VariableCodebookConfigs)

    # Top-10 Document Retrieval
    retrieval_count: int = 10  # Number of documents to retrieve
    similarity_threshold: float = 0.6  # Minimum similarity score
    ranking_method: str = "cosine_similarity"  # Options: cosine_similarity, dot_product, euclidean
    use_filter: bool = False  # Whether to filter results
    filter_criteria: dict[str, Any] = {}
    use_reranking: bool = False  # Whether to use re-ranking

    def __getitem__(self, key: str) -> Optional[Any]:
        return get_value_from_base_model(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        try:
            setattr(self, key, value)
        except AttributeError:
            raise KeyError(f"Key '{key}' not found in Configs")

    def get(self, key: str, default: Any = None) -> Optional[Any]:
        return get_value_with_default_from_base_model(self, key, default)

configs = Configs()

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