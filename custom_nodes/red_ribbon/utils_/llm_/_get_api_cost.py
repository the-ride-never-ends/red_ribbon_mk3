import logging
from typing import Any, Optional

import tiktoken


from ._constants import MODEL_USAGE_COSTS_USD_PER_MILLION_TOKENS


def _check_if_string(value: Any, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"'{name}' must be a string, got {type(value).__name__}")

def get_api_cost(prompt: str, data: str, out: str, model: str, logger: logging.Logger) -> Optional[int]:
    """Calculate the cost of an LLM API call based on token usage."""

    _check_if_string(prompt, "prompt")
    _check_if_string(data, "data")
    _check_if_string(out, "out")
    _check_if_string(model, "model")

    model_costs: dict | None = MODEL_USAGE_COSTS_USD_PER_MILLION_TOKENS.get(model)
    model_keys: set[str] = set(tiktoken.model.MODEL_PREFIX_TO_ENCODING.keys()).union(set(tiktoken.model.MODEL_TO_ENCODING.keys()))

    # Initialize the tokenizer for the GPT model
    if model_costs is None:
        logger.error(f"Model {model} not found in usage costs.")
        return None

    if model not in model_keys:
        logger.error(f"Model {model} not found in tiktoken.")
        return None

    tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model(model)

    # request and response
    request = prompt + data
    response = out

    # Tokenize 
    request_tokens, response_tokens = tokenizer.encode(request), tokenizer.encode(response)

    # Counting the total tokens for request and response separately
    input_tokens, output_tokens = len(request_tokens), len(response_tokens)

    model_key: dict = model_costs[model]

    # Actual costs per 1 million tokens
    cost_per_1M_input_tokens, cost_per_1M_output_tokens = model_key["input"], model_key["output"]

    if cost_per_1M_output_tokens is None:
        output_cost = 0
    else:
        output_cost = (output_tokens / 10**6) * cost_per_1M_output_tokens

    input_cost = (input_tokens / 10**6) * cost_per_1M_input_tokens
    total_cost = input_cost + output_cost
    return total_cost
