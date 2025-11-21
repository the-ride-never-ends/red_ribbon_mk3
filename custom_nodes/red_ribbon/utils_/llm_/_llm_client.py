"""
LLM Client implementation for American Law database.
Provides integration with OpenAI APIs and RAG components for legal research.
"""
import asyncio
import os
import logging
from pathlib import Path
import sqlite3
from typing import Annotated as Ann, Any, Callable, Optional, Type, Union
from functools import cached_property


import duckdb
import numpy as np
import pandas as pd
from openai import AsyncOpenAI, OpenAI, OpenAIError
from openai.types import Completion, CreateEmbeddingResponse, ChatModel, EmbeddingModel, ModerationModel
from pydantic import (
    AfterValidator as AV, 
    BaseModel, 
    BeforeValidator as BV, 
    computed_field, 
    Field,
    PrivateAttr, 
    TypeAdapter, 
    ValidationError,
    PositiveInt,
    PositiveFloat,
    NonNegativeFloat,
    NonNegativeInt,
)

from custom_nodes.red_ribbon.utils_ import logger, configs, Configs
from ._load_prompt_from_yaml import load_prompt_from_yaml, Prompt
from ._get_api_cost import get_api_cost
from ._embeddings_manager import EmbeddingsManager


def __validate(_Model: Type, value: Any, name: str):
    try:
        _Model(value=value)
    except ValidationError as e:
        raise ValueError(f"'{name}' must be a { _Model.value.annotation}: {e.errors()}") from e

def _validate_strings(list_of_strings: list[str] | str) -> None:
    """Type checker for list of texts"""
    if isinstance(list_of_strings, str):
        list_of_strings = [list_of_strings]
    class _Model(BaseModel):
        value: list[str] = Field(..., ge=1)
    __validate(_Model, list_of_strings, "list_of_strings")

def _strip(text: str) -> str:
    return text.strip()

class ParserError(RuntimeError):
    """Error when parsing an LLM response fails."""

    def __init__(self, message: str):
        super().__init__(message)


class OpenAiLLMOutput(BaseModel):
    """
    The output from a Large Language Model (LLM) return.

    Attributes:
        response (str): The stripped response text from the LLM
        system_prompt (str): The stripped system prompt used for the LLM interaction.
        user_message (str): The stripped user message sent to the LLM.
        total_tokens (NonNegativeInt): The amount of context tokens used in the interaction.
        model (str): The name/identifier of the LLM model used.
        response_parser (Callable): A function to parse the raw response into a structured format.

    Computed Properties:
        cost (float): The calculated API cost in USD for the interaction based on token usage and model pricing.
        parsed_response (Any): The asynchronously parsed response using the provided parser function.

    Raises:
        RuntimeError: If the response parsing fails using the provided parser function.

    Examples:
        >>> def simple_parser(text):
        ...     return text.upper()
        >>> output = OpenAiLLMOutput(
        ...     response="Hello, world!",
        ...     system_prompt="You are a helpful assistant.",
        ...     user_message="Say hello.",
        ...     total_tokens=50,
        ...     model="gpt-3.5-turbo",
        ...     response_parser=simple_parser
        ... )
        >>> print(output.response)
        'Hello, world!'
        >>> print(output.cost)
        0.00012
    """
    completion: Completion = Field(..., exclude=True)
    system_prompt: Ann[str, AV(_strip)]
    user_message: Ann[str, AV(_strip)]
    total_tokens: NonNegativeInt
    model: ChatModel
    response_parser: Callable = Field(default_factory=lambda: lambda x: x, exclude=True)

    @property
    def raw_response(self) -> Optional[str]:
        """The raw response text from the LLM"""
        return self.completion.choices[0].text

    @computed_field # type: ignore[prop-decorator]
    @cached_property
    def response(self) -> Optional[Any]:
        """Returns the response using the response_parser function."""
        if self.raw_response is None:
            return None
        try:
            return self.response_parser(self.response)
        except Exception as e:
            raise ParserError(f"Error parsing LLM response: {e}") from e

    @computed_field # type: ignore[prop-decorator]
    @property
    def cost(self) -> float:
        if self.raw_response is None:
            return 0.0
        result = get_api_cost(self.system_prompt, self.user_message, out=self.response, model=self.model)
        return float(result) if result is not None else 0.0


class OpenAiEmbeddingGeneration(BaseModel):
    """
    Pydantic model for generating embeddings from text using OpenAI's embedding models.

    Attributes:
        client (OpenAI): Synchronous OpenAI client for API calls
        async_client (AsyncOpenAI): Asynchronous OpenAI client for API calls
        embedding_model (EmbeddingModel): The OpenAI embedding model to use
        texts (Union[list[str], str]): Text(s) to generate embeddings for

    Properties:
        raw_response: The raw response object from the embedding API

    Methods:
        embed: Synchronously generate embeddings
        async_embed: Asynchronously generate embeddings

    Examples:
        >>> client = OpenAI(api_key="your-key")
        >>> async_client = AsyncOpenAI(api_key="your-key")
        >>> embedding_gen = OpenAiEmbeddingGeneration(
        ...     client=client,
        ...     async_client=async_client,
        ...     texts=["Hello world", "AI is amazing"]
        ... )
        >>> embeddings = embedding_gen.embed()
        >>> len(embeddings)
        2
    """
    client: OpenAI = Field(..., exclude=True)
    async_client: AsyncOpenAI = Field(..., exclude=True)
    embedding_model: EmbeddingModel = Field(default="text-embedding-3-small", ge=1)
    texts: Ann[Union[list[str], str], AV(_validate_strings)]

    _raw_response: Optional[CreateEmbeddingResponse] = PrivateAttr(default=None)

    @property
    def raw_response(self) -> Optional[CreateEmbeddingResponse]:
        """The raw response object from the embedding API"""
        return self._raw_response

    def _parse_response(self, response: CreateEmbeddingResponse) -> list[list[float]]:
        """Extract embeddings from the API response"""
        self._raw_response = response
        return [data.embedding for data in response.data]

    async def async_embed(self) -> list[list[float]]:
        """Asynchronously generate embeddings for the provided texts using an OpenAI embedding model.

        Returns:
            list[list[float]]: A list of embedding vectors

        Raises:
            TypeError: If the client attribute is not an AsyncOpenAI instance.
            OpenAIError: If there's an error with the OpenAI API request (e.g., invalid
                        API key, rate limiting, model unavailable).
            RuntimeError: If an unexpected error occurs during the embedding generation process.

        Example:
            >>> embedding_generation = OpenAiEmbeddingGeneration(texts=["Hello world", "AI is amazing"])
            >>> embeddings = embedding_generation.embed()
            >>> len(embeddings)
            2
            >>> isinstance(embeddings[0], list)
            True
        """
        try:
            response = await self.async_client.embeddings.create(
                input=self.texts,
                model=self.embedding_model
            )
        except OpenAIError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Unexpected error generating embeddings with async embed: {e}") from e
        return self._parse_response(response)

    def embed(self) -> list[list[float]]:
        """Generate embeddings for the provided texts using an OpenAI embedding model.

        Returns:
            list[list[float]]: A list of embedding vectors

        Raises:
            TypeError: If the client attribute is not an OpenAI instance.
            OpenAIError: If there's an error with the OpenAI API request (e.g., invalid
                        API key, rate limiting, model unavailable).
            RuntimeError: If an unexpected error occurs during the embedding generation process.

        Example:
            >>> embedding_generation = OpenAiEmbeddingGeneration(texts=["Hello world", "AI is amazing"])
            >>> embeddings = embedding_generation.embed()
            >>> len(embeddings)
            2
            >>> isinstance(embeddings[0], list)
            True
        """
        try:
            response = self.client.embeddings.create(
                input=self.texts,
                model=self.embedding_model
            )
        except OpenAIError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Unexpected error generating embeddings with async embed: {e}") from e
        return self._parse_response(response)




class OpenAiLLMGeneration(BaseModel):
    """
    LLM Generation model for creating chat completions using OpenAI API.

    Attributes:
        client: OpenAI client for API calls
        async_client: AsyncOpenAI client for async API calls
        configs: Configuration object containing API settings
        user_message: The user's input message to the LLM
        system_prompt: System-level instructions for the LLM
        model: The specific chat model to use for generation
        max_tokens: Maximum number of tokens in the response
        temperature: Controls randomness in the response (0.0 = deterministic)
        response_parser: Function to parse the raw response into structured format

    Properties:
        raw_response: The raw response object from the LLM API
        messages: List of message dictionaries for the chat completion
        response_kwargs: Keyword arguments for the chat completion API call

    Methods:
        generate: Synchronously generate a response from the LLM
        async_generate: Asynchronously generate a response from the LLM

    Examples:
        >>> llm_generation = OpenAiLLMGeneration(
        ...     client=openai_client,
        ...     configs=config_obj,
        ...     user_message="What is the capital of France?",
        ...     system_prompt="You are a geography expert.",
        ...     model="gpt-4"
        ... )
        >>> result = llm_generation.generate()
        >>> print(result.response)
        'The capital of France is Paris.'
    """
    client: OpenAI = Field(..., exclude=True)
    async_client: AsyncOpenAI = Field(..., exclude=True)
    configs: Configs = Field(...,exclude=True)
    user_message: Ann[str, AV(_strip)] = Field(
        min_length=1, default="Write a test response telling the user that the LLM client is working."
    )
    system_prompt: Ann[str, AV(_strip)] = Field(default="You are a helpful assistant.")
    model: ChatModel = Field(default="gpt-5",min_length=1)
    max_tokens: PositiveInt = 4096
    temperature: NonNegativeFloat = 0.0 # Deterministic output
    response_parser: Callable = Field(default_factory=lambda: lambda x: x, exclude=True)

    _raw_response: Optional[Completion] = PrivateAttr(default=None)

    @property
    def raw_response(self) -> Optional[Completion]:
        return self._raw_response

    @property
    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_message}
        ]

    @property
    def response_kwargs(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _parse_output(self, response: Completion) -> OpenAiLLMOutput:
        self._raw_response: Completion | None = response
        if self._raw_response is None:
            raise ValueError("Response from chat completion was None.")
        try:
            return OpenAiLLMOutput(
                completion=self._raw_response,
                system_prompt=self.system_prompt,
                user_message=self.user_message,
                model=self.model,
                response_parser=self.response_parser,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )
        except ValidationError as e:
            raise ValueError(f"LLM output failed to validate: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error parsing LLM output: {e}") from e

    def generate(self) -> OpenAiLLMOutput:
        try:
            response = self.client.chat.completions.create(**self.response_kwargs)
        except OpenAIError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Unexpected error generating response: {e}") from e
        return self._parse_output(response)

    async def async_generate(self) -> OpenAiLLMOutput:
        try:
            response = await self.async_client.chat.completions.create(**self.response_kwargs)
        except OpenAIError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Unexpected error generating async response: {e}") from e
        return self._parse_output(response)






class OpenAiClient:

    def __init__(self, *, resources: dict[str, Any], configs: Configs):
        """
        Initialize the OpenAI client for American Law dataset RAG.
        
        Args:
            resources: Dictionary of resources for embeddings and database queries
            configs: Configuration object
        """
        self.resources = resources
        self.configs = configs

        self.model: ChatModel = configs.OPENAI_MODEL
        self.embedding_model: EmbeddingModel = configs.OPENAI_EMBEDDING_MODEL
        self.moderation_model: ModerationModel = configs.OPENAI_MODERATION_MODEL
        self.embedding_dimensions: int = configs.EMBEDDING_DIMENSIONS
        self.temperature: float = configs.TEMPERATURE
        self.max_tokens: int = configs.MAX_TOKENS
        self.data_path: Path = configs.paths.AMERICAN_LAW_DATA_DIR
        self.db_path: Path = configs.paths.AMERICAN_LAW_DB_PATH

        self.logger: logging.Logger = resources['logger']
        self.clean_html: Callable = resources['clean_html']
        self.client: OpenAI = resources['client']
        self.async_client: AsyncOpenAI = resources['async_client']

        # Set data paths
        self.logger.info(f"Initialized OpenAiClient: LLM model: {self.model}, embedding model: {self.embedding_model}")
        self.logger.debug(f"OpenAiClient attributes\n '{dir(self)}'")

    def _was_flagged(self, texts: str | list[str], response: Any) -> bool:
        if response.results[0].flagged:
            self.logger.warning(f"Messages flagged by moderation: {texts}")
            return True
        return False

    def moderate(self, texts: str | list[str]) -> bool:
        """
        Check if text content violates OpenAI's usage policies using the moderation API.
        
        Args:
            texts: Single string or list of strings to check for policy violations
            
        Returns:
            bool: True if content is flagged as violating policies, False otherwise
            
        Raises:
            OpenAIError: If the moderation API call fails
            RuntimeError: If an unexpected error occurs during moderation
        """
        match texts:
            case list():
                for idx, text in enumerate(texts):
                    if not isinstance(text, str):
                        raise TypeError(f"Item at index {idx} in texts must be a string, got {type(text).__name__}")
            case str():
                texts = [texts]
            case _:
                raise TypeError(f"'texts' must be a string or list of strings, got {type(texts).__name__}")

        try:
            response = self.client.moderations.create(
                model=self.moderation_model,
                input=texts,
            )
        except OpenAIError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during moderation: {e}") from e
        return self._was_flagged(texts, response)

    async def async_moderate(self, texts: str | list[str]) -> bool:
        """
        Check if text content violates OpenAI's usage policies using the moderation API.
        
        Args:
            texts: Single string or list of strings to check for policy violations
            
        Returns:
            bool: True if content is flagged as violating policies, False otherwise
            
        Raises:
            OpenAIError: If the moderation API call fails
            RuntimeError: If an unexpected error occurs during moderation
        """
        if not isinstance(texts, list):
            texts = [texts]
        try:
            response = await self.async_client.moderations.create(
                model=self.moderation_model,
                input=texts,
            )
        except OpenAIError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during moderation: {e}") from e
        return self._was_flagged(texts, response)

    def _make_embedding_generation(self, texts: str | list[str]) -> OpenAiEmbeddingGeneration:
        """Make an OpenAiEmbeddingGeneration instance."""
        try:
            return OpenAiEmbeddingGeneration(
                client=self.client,
                async_client=self.async_client,
                embedding_model=self.embedding_model,
                texts=texts
            )
        except ValidationError as e:
            raise ValueError(f"Invalid input for embedding generation: {e}") from e


    async def async_get_embeddings(self, texts: str | list[str]) -> list[list[float]]:
        """
        Asynchronously generate embeddings for a list of text inputs using OpenAI's embedding model.
        
        Args:
            texts: String or list of strings to generate embeddings for.
            
        Returns:
            List of embedding vectors, where each vector is a list of floats.
            
        Raises:
            TypeError: 
            ValueError: If the input list is empty or contains empty strings.
            OpenAIError: If the API call fails.
            RuntimeError: If an unexpected error occurs during embedding generation.
        """
        _validate_strings(texts)
        embedding_generation = self._make_embedding_generation(texts)
        return await embedding_generation.async_embed()


    def get_embeddings(self, texts: str | list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text inputs using OpenAI's embedding model.

        Args:
            texts: string or list of strings to generate embeddings for.

        Returns:
            List of embedding vectors, where each vector is a list of floats.

        Raises:
            TypeError: If the input is not a list of strings
            ValueError: If input parameters fail validation
            OpenAIError: If the API call fails.
            RuntimeError: If an unexpected error occurs during embedding generation.
        """
        _validate_strings(texts)
        embedding_generation = self._make_embedding_generation(texts)
        return embedding_generation.embed()

    def _make_llm_generation(self, 
                             user_message: str, 
                             system_prompt: str, 
                             response_parser: Callable,
                             model: Optional[ChatModel], 
                             temperature: Optional[float], 
                             max_tokens: Optional[int], 
                             ) -> OpenAiLLMGeneration:
        """Make an OpenAiLLMGeneration instance."""
        try:
            return OpenAiLLMGeneration(
                client=self.client,
                async_client=self.async_client,
                configs=self.configs,
                user_message=user_message,
                system_prompt=system_prompt,
                model=model or self.model,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                response_parser=response_parser,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid input for LLM generation: {e}") from e

    async def async_get_response(self,
                            user_message: str = "Write a test response telling the user that the LLM client is working.",
                            system_prompt: str = "You are a helpful assistant.",
                            response_parser: Callable = lambda x: x,
                            model: Optional[ChatModel] = None,
                            temperature: Optional[float] = None,
                            max_tokens: Optional[int] = None,
                            ) -> Optional[Any]:
        """
        Asynchronously generate a response using the OpenAI API.

        Args:
            user_message: The user's message
            system_prompt: The system prompt
            response_parser: Function to parse the response content. Defaults to identity function.
            model: Optional model to override the configuration default
            temperature: Optional temperature to override the configuration default
            max_tokens: Optional max tokens to override the configuration default

        Returns:
            Optional[Any]: The parsed response content or the full API response if response_parser is not provided.

        Raises:
            ValueError: If input parameters fail validation
            OpenAIError: If the API call fails
            RuntimeError: If an unexpected error occurs during response generation
            ParsingError: If an error occurs while parsing the response
        """
        llm_generation = self._make_llm_generation(
            user_message=user_message,
            system_prompt=system_prompt,
            response_parser=response_parser,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm_output = await llm_generation.async_generate()
        return llm_output.response


    def get_response(self,
                    user_message: str = "Write a test response telling the user that the LLM client is working.",
                    system_prompt: str = "You are a helpful assistant.",
                    response_parser: Callable = lambda x: x,
                    model: Optional[ChatModel] = None,
                    temperature: Optional[float] = None,
                    max_tokens: Optional[int] = None,
                    ) -> Optional[str]:
        """
        Generate a response using the OpenAI API.

        Args:
            user_message: The user's message
            system_prompt: The system prompt
            response_parser: Function to parse the response content. Defaults to identity function.
            model: Optional model to override the configuration default
            temperature: Optional temperature to override the configuration default
            max_tokens: Optional max tokens to override the configuration default

        Returns:
            The parsed response content or the full API response if response_parser is not provided.

        Raises:
            ValueError: If input parameters fail validation
            OpenAIError: If the API call fails
            RuntimeError: If an unexpected error occurs during response generation
            ParsingError: If an error occurs while parsing the response
        """
        llm_generation = self._make_llm_generation(
            user_message=user_message,
            system_prompt=system_prompt,
            response_parser=response_parser,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        llm_output = llm_generation.generate()
        return llm_output.response


class AsyncOpenAiClient:
    """
    Asynchronous client for LLM API integration with RAG capabilities.
    Handles embeddings integration and semantic search against the law database.
    """
    
    def __init__(self, *, resources: dict[str, Any], configs: Configs):
        """
        Initialize the OpenAI client for American Law dataset RAG.
        
        Args:
            resources: Dictionary of resources for embeddings and database queries
            configs: Configuration object
        """
        self.resources = resources
        self.configs = configs

        self.api_key: str = configs.OPENAI_API_KEY.get_secret_value()
        if not self.api_key:
            raise ValueError("API key must be provided either as an argument or as an environment variable (e.g. EXPORT OPENAI_API_KEY=...)")

        self.logger: logging.Logger = resources['logger']
        self._get_embeddings: Callable = resources['async_get_embeddings']
        self._search_embeddings: Callable = resources['search_embeddings']
        self._execute_sql_query: Callable = resources['execute_sql_query']

        self.client: AsyncOpenAI = AsyncOpenAI(api_key=self.api_key)
        self.model: str = configs.OPENAI_MODEL
        self.embedding_model: str = configs.OPENAI_EMBEDDING_MODEL
        self.embedding_dimensions: int = configs.EMBEDDING_DIMENSIONS
        self.temperature: float = configs.TEMPERATURE
        self.max_tokens: int = configs.MAX_TOKENS

        # Set data paths
        self.data_path: Path = configs.paths.AMERICAN_LAW_DATA_DIR
        self.db_path: Path = configs.paths.AMERICAN_LAW_DB_PATH

        self.clean_html: Callable = resources['clean_html']

        logger.info(f"Initialized AsyncLLMClient client")


    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text inputs using OpenAI's embedding model.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        try:
            # Prepare texts by stripping whitespace and handling empty strings
            processed_texts = [text.strip() for text in texts]
            processed_texts = [text if text else " " for text in processed_texts]

            return await self._get_embeddings(processed_texts)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise e


    async def get_single_embedding(self, text: str) -> list[float]:
        """
        Generate an embedding for a single text input.
        
        Args:
            text: Text string to generate an embedding for
            
        Returns:
            Embedding vector
        """
        embeddings = await self.get_embeddings([text])
        if embeddings:
            return embeddings[0]
        return []


    async def search_embeddings(
        self, 
        query: str, 
        gnis: Optional[str] = None, 
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Search for relevant documents using embeddings.
        
        Args:
            query: Search query
            gnis: Optional file ID to limit search to
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with similarity scores
        """
        # Generate embedding for the query
        query_embedding: list[float] = await self.get_single_embedding(query)
        
        results: list[dict] = []
        # Create the SQL query
        sql_query = f"""
        WITH query_embedding AS (
            SELECT UNNEST($1) as vec
        ), 
        similarity_scores AS (
            SELECT 
            c.id, 
            c.cid, 
            c.title, 
            c.chapter, 
            c.place_name, 
            c.state_name, 
            c.date, 
            c.bluebook_citation, 
            c.content, 
            DOT_PRODUCT(e.embedding, ARRAY(SELECT vec FROM {query_embedding})) / 
            (SQRT(DOT_PRODUCT(e.embedding, e.embedding)) * 
                SQRT(DOT_PRODUCT(ARRAY(SELECT vec FROM {query_embedding}), 
                        ARRAY(SELECT vec FROM {query_embedding})))) as similarity_score 
            FROM 
            citations c 
            JOIN 
            embeddings e ON c.id = e.citation_id 
            WHERE 
            1=1
        """
        
        # Add GNIS filter if provided
        if gnis:
            sql_query += f" AND c.gnis = '{gnis}'"
            
        sql_query += f"""
        )
        SELECT * FROM similarity_scores
        ORDER BY similarity_score DESC
        LIMIT {top_k}
        """
        try:
            # Execute the query with the embedding as parameter
            results_list: Optional[list[dict[str, Any]]] = self._execute_sql_query(sql_query)
            if results_list is None:
                return []
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []

        # UNpack into a list of dictionaries
        for row in results_list:
            results.append({
                'id': row['id'],
                'cid': row['cid'],
                'title': row['title'],
                'chapter': row['chapter'],
                'place_name': row['place_name'],
                'state_name': row['state_name'],
                'date': row['date'],
                'bluebook_citation': row['bluebook_citation'],
                'content': row['content'],
                'similarity_score': float(row['similarity_score'])
            })

        return results

    def execute_sql_query(self, query: str, raise_on_e: bool = False) -> list[dict[str, Any]]:
        try:
            # Execute the query with the embedding as parameter
            result = self._execute_sql_query(query)
            return result if result is not None else []
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            if raise_on_e:
                raise e
            return []


    async def query_database(self, query: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """
        Query the database for relevant laws.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching law records
        """
        # Simple text search of citations
        sql_query = f"""
            SELECT id, cid, title, chapter, place_name, state_name, date, 
                    bluebook_citation, content
            FROM citations
            WHERE lower(search_text) LIKE '%{query.lower()}%'
            ORDER BY place_name, title
        """
        if limit is not None:
            if limit > 0:
                # Add limit to the SQL query
                sql_query += f" LIMIT {limit};"
        # Execute query and fetch results
        result = self.execute_sql_query(sql_query)
        return result if result is not None else []


    async def generate_rag_response(
        self, 
        query: str, 
        use_embeddings: bool = True,
        top_k: int = 5,
        system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Generate a response using RAG (Retrieval Augmented Generation).
        
        Args:
            query: User query
            use_embeddings: Whether to use embeddings for search
            top_k: Number of context documents to include
            system_prompt: Custom system prompt
            
        Returns:
            Dictionary with the generated response and context used
        """
        # Retrieve relevant context
        context_docs = []
        
        if use_embeddings:
            # Use embedding-based semantic search
            context_docs = await self.search_embeddings(query, top_k=top_k)
        else:
            # Use database text search as fallback
            context_docs = await self.query_database(query, limit=top_k)

        # Build context for the prompt
        context_text = "Relevant legal information:\n\n"
        references = "Citation(s):\n\n"

        for i, doc in enumerate(context_docs):
            context_text += f"[{i+1}] {doc.get('title', 'Untitled')} - {doc.get('place_name', 'Unknown location')}, {doc.get('state_name', 'Unknown state')}\n"
            references += f"{i+1}. {doc.get('bluebook_citation', 'No citation available')}\n"
            context_text += f"Citation: {doc.get('bluebook_citation', 'No citation available')}\n"
            # Limit html to avoid excessively long prompts
            html = doc.get('html', '')
            if html:
                # Turn the HTML into a string.
                content = self.clean_html(html)
                content = content[:1000] + "..." if len(content) > 1000 else content
                context_text += f"Content: {content}\n\n"
        
        # Default system prompt if none provided
        if system_prompt is None:
            system_prompt = """
    You are a legal research assistant specializing in American municipal and county laws. 
    Answer questions based on the provided legal context information. 
    If the provided context doesn't contain enough information to answer the question confidently, 
    acknowledge the limitations of the available information and suggest what additional 
    information might be helpful.
    For legal citations, use Bluebook format when available. Be concise but thorough.
            """
        # Generate response using OpenAI
        prompt_dir = self.configs.paths.PROMPTS_DIR
        prompt: Prompt = load_prompt_from_yaml("generate_rag_response", prompt_dir=prompt_dir, query=query, context=context_text)
        if system_prompt is not None:
            prompt.system_prompt.content = system_prompt.strip()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": f"Question: {query}\n\n{context_text}"}
                ]
            )
            
            generated_response = response.choices[0].message.content
            if generated_response:
                generated_response = generated_response.strip()
                # Append the citations
                generated_response += f"\n\n{references.strip()}"
            else:
                generated_response = "No response generated."
            
            return {
                "query": query,
                "response": generated_response,
                "context_used": [doc.get('bluebook_citation', 'No citation') for doc in context_docs],
                "model_used": self.model,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
        except Exception as e:
            self.logger.error(f"Error generating RAG response: {e}")
            return {
                "query": query,
                "response": f"Error generating response: {str(e)}",
                "context_used": [],
                "model_used": self.model,
                "error": str(e)
            }
