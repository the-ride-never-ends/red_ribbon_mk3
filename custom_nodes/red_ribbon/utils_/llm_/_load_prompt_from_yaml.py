from pathlib import Path
from typing import Literal, Never


from pydantic import (
    BaseModel, 
    Field,
    ValidationError,
)
import yaml


from custom_nodes.red_ribbon.utils_.logger import logger
from custom_nodes.red_ribbon.utils_.configs import configs, Configs
from custom_nodes.red_ribbon.utils_.common._safe_format import safe_format



def validate_prompt(prompt: str) -> None:
    if "role" not in prompt:
        raise ValidationError("Prompt must contain 'role' key.")
    if "content" not in prompt:
        raise ValidationError("Prompt must contain 'content' key.")
    return None

class Settings(BaseModel):
    """
    Settings for the LLM client.
    """
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

class PromptFields(BaseModel):
    role: str
    content: str

class Prompt(BaseModel):
    client: Literal['openai', 'anthropic']
    settings: Settings = Field(default_factory=Settings)
    system_prompt: PromptFields
    user_prompt: PromptFields

    def safe_format(self, **kwargs) -> dict:
        """
        Safely insert kwargs into the system and user prompts.
        Only keys that exist in the respective prompt templates will be used for formatting.
        """
        if not kwargs:
            return self.model_dump()

        sys_kwargs = {k: v for k, v in kwargs.items() if k in self.system_prompt.content}
        user_kwargs = {k: v for k, v in kwargs.items() if k in self.user_prompt.content}

        self.system_prompt.content = safe_format(self.system_prompt.content, **sys_kwargs)
        self.user_prompt.content = safe_format(self.user_prompt.content, **user_kwargs)
        return self.model_dump()


def load_prompt_from_yaml(name: str, configs: Configs, **kwargs) -> Prompt:
    assert hasattr(configs, 'PROMPTS_DIR'), "Configs must have PROMPTS_DIR attribute"
    prompt_dir = configs.PROMPTS_DIR
    assert isinstance(prompt_dir, Path), "PROMPTS_DIR must be a Path object"
    prompt_path = prompt_dir / f"{name}.yaml"
    assert prompt_path.is_file(), f"Prompt YAML file '{prompt_path}' does not exist."

    try:
        with open(prompt_path, 'r') as file:
            prompt_dict = dict(yaml.safe_load(file))
            prompt_obj = Prompt.model_validate(prompt_dict)
            prompt_obj.safe_format(**kwargs)
            return prompt_obj
    except Exception as e:
        msg = f"Unexpected Error loading prompt from YAML file '{prompt_path}': {e}"
        logger.error(msg)
        raise IOError(msg) from e