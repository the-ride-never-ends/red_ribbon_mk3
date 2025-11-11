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


def load_prompt_from_yaml(name: str, configs: Configs, **kwargs) -> Prompt:
    prompt_dir = configs.PROMPTS_DIR
    prompt_path = prompt_dir / f"{name}.yaml"

    with open(prompt_path, 'r') as file:
        prompt = dict(yaml.safe_load(file))
        return Prompt.model_validate(prompt).safe_format(**kwargs)
