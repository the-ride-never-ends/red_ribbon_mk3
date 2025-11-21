"""
OpenAI Client implementation for American Law database.
Provides integration with OpenAI APIs and RAG components for legal research.
"""
import logging
from pathlib import Path
from typing import Any, Callable, Optional


import duckdb
from openai import AsyncOpenAI, APIError, OpenAIError, RateLimitError, Timeout,  APIConnectionError
from openai.types.chat import ChatCompletionMessage
import pandas as pd
from pydantic import (
    BaseModel, 
    computed_field, 
    FilePath,
    PrivateAttr, 
)

from custom_nodes.red_ribbon.utils_ import logger, configs, Configs
from .._load_prompt_from_yaml import load_prompt_from_yaml, Prompt


from .openai_client import calculate_cost


class DuckDbSqlDatabase(BaseModel):
    """
    Class to handle SQL database operations.
    """
    db_path: FilePath

    _conn: duckdb.DuckDBPyConnection = PrivateAttr(default=None)
    _query: str = PrivateAttr(default="")

    def __enter__(self) -> 'DuckDbSqlDatabase':
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn:
            self._conn.close()
        self._conn = None
        return

    def connect(self) -> None:
        if self._conn is None:
            self._conn = duckdb.connect(self.db_path, read_only=True)

    def execute_query(self, 
                      query: str, 
                      return_as: Optional[str] = None
                      ) -> Optional[pd.DataFrame | list[dict[str, Any]] | list[tuple[Any, ...]]]:
        if self._conn:
            with self._conn.cursor() as cursor:
                try:
                    cursor.execute(query)
                    if return_as is None:
                        self._conn.commit()
                except Exception as e:
                    logger.error(f"Error executing query: {e}")
                    if return_as is None:
                        self._conn.rollback()

                match return_as:
                    case "df":
                        return cursor.fetchdf()
                    case "dict":
                        return cursor.fetchdf().to_dict('records')
                    case "tuple":
                        return cursor.fetchall()
                    case _: # This route is for database alterations like CREATE TABLE
                        return None

    def close(self):
        if self._conn:
            self._conn.close()


class AsyncLLMInput(BaseModel):
    client: Any
    user_message: str # NOTE For RAG, the found documents should be passed here.
    system_prompt: str = "You are a helpful assistant."
    use_rag: bool = False
    max_tokens: int = 4096
    temperature: float = 0 # Deterministic output
    response_parser: Callable = lambda x: x # This should be a partial function.
    formatting: Optional[str] = None
    db_path: Optional[FilePath] = None


    _configs: Configs = PrivateAttr(default=configs)


    async def get_response(self) -> Optional[Any]:
        if self.use_rag:
            # Get an embedding of user_message
            user_message_embedding = await self._get_embedding(self.user_message)
            with DuckDbSqlDatabase(db_path=self.db_path) as db:
                # Use RAG to find relevant documents
                context = await self._use_rag(user_message_embedding, self.system_prompt)
                if context:
                    self.user_message = context[0]
                    self.system_prompt = context[1]
                else:
                    return "No relevant documents found."

        try:
            _response = await self.client.chat.completions.create(
                model=self._configs.OPENAI_MODEL,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[
                    {"role": "system", "content": self.system_prompt.strip()},
                    {"role": "user", "content": self.user_message}
                ]
            )
        except Exception as e:
            logger.error(f"{type(e)} generating response: {e}")
            return "Error generating response. Please try again."

        if _response.choices[0].message.content:
            return AsyncLLMOutput(
                response=_response.choices[0].message.content.strip(),
                system_prompt=self.system_prompt.strip(),
                user_message=self.user_message,
                context_used=_response.usage.total_tokens,
                model=self._configs.OPENAI_MODEL,
                response_parser=self.response_parser,
            )
        else:
            return "No response generated. Please try again."

    async def _get_embedding(self, message: str) -> list[float]:
        """
        Generate an embedding of the input message.

        Args:
            message: Message to generate an embedding for
        
        Returns:
            Embedding vector
        """
        # Strip the user message to remove newlines and leading/trailing whitespace
        message = message.strip('\n').strip()

        try:
            response = await self.client.embeddings.create(
                input=message,
                model=self._configs.OPENAI_EMBEDDING_MODEL
            )
        except Exception as e:
            logger.error(f"{type(e)} generating embedding: {e}")
            return []

        if response.data[0].embedding:
            return response.data[0].embedding
        else:
            logger.error("No embedding generated. Please try again.")
            return []

    async def _use_rag(self, user_message: str, system_message: str) -> Optional[tuple[str, str]]:
        """
        Use RAG to find relevant documents and generate a response.
        
        Args:
            user_message: The user's message
            system_message: The system prompt
        """
        


class AsyncLLMOutput(BaseModel):
    response: str
    system_prompt: str
    user_message: str
    context_used: int
    model: str
    response_parser: Callable

    _configs: Configs = PrivateAttr(default=configs)

    @computed_field # type: ignore[prop-decorator]
    @property
    def cost(self) -> float:
        if self.response is not None:
            return calculate_cost(self.system_prompt, self.user_message, self.response, self.model)
        else: 
            return 0

    async def get_parsed_response(self) -> Any:
        """Asynchronously parse the response using the provided parser"""
        return self.response_parser(self.response)


class LLMOutput(BaseModel):
    response: str
    system_prompt: str
    user_message: str
    context_used: int
    response_parser: Callable

    _configs: Configs = PrivateAttr(default=configs)

    @computed_field # type: ignore[prop-decorator]
    @property
    def cost(self) -> float:
        if self.response is not None:
            return calculate_cost(self.system_prompt, self.user_message, self.response, self._configs.OPENAI_MODEL)
        else: 
            return 0

    def response(self) -> Any:
        return self.response_parser(self.response)


class AsyncOpenAIClient:
    """
    Asynchronous client for OpenAI API integration with RAG capabilities for the American Law dataset.
    Handles embeddings integration and semantic search against the law database.
    """
    
    def __init__(self, *, resources: dict[str, Any], configs: Configs) -> None:
        """
        Initialize the OpenAI client for American Law dataset RAG.
        
        Args:
            resources: Dictionary containing shared resources like logger and utilities
            configs: Configuration object containing API keys, model settings, and paths
        """
        self.configs = configs
        self.resources = resources

        self.logger: logging.Logger = self.resources["logger"]
        self.clean_html: Callable = self.resources["clean_html"]

        self.api_key: str = configs.OPENAI_API_KEY.get_secret_value()
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as an argument or in the OPENAI_API_KEY environment variable")
        
        # Define model parameters
        self.model:                str = configs.OPENAI_MODEL
        self.small_model:          str = configs.OPENAI_SMALL_MODEL
        self.embedding_model:      str = configs.OPENAI_EMBEDDING_MODEL
        self.embedding_dimensions: int = configs.EMBEDDING_DIMENSIONS
        self.temperature:          float = configs.TEMPERATURE
        self.max_tokens:           int = configs.MAX_TOKENS

        # Start the async client
        self.client:               AsyncOpenAI = AsyncOpenAI(api_key=self.api_key)

        # Set data paths
        self.data_path: Path = configs.paths.AMERICAN_LAW_DATA_DIR
        self.db_path:   Path = configs.paths.AMERICAN_LAW_DB_PATH

        self._latest_response: Optional[str] = None

        self.logger.info(f"Initialized AsyncOpenAI client: ")
        self.logger.debug(f"LLM model: {self.model}, embedding model: {self.embedding_model}")

    @property
    def total_tokens(self) -> int:
        """
        Get the total tokens used in the last request.
        
        Returns:
            Total tokens used
        """
        return self._latest_response.usage.total_tokens if self._latest_response else 0

    async def chat_completion(self,
        messages: list[dict[str, str]] = [],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
        ) -> Optional[str]:
        """
        Generate a chat completion using the OpenAI API.
        
        Args:
            messages: List of message dictionaries for the chat completion
            model: Optional LLM to use (defaults to configs model)
            temperature: Optional temperature setting (defaults to configs temperature)
            max_tokens: Optional max tokens setting (defaults to configs max tokens)

        Returns:
            Chat completion response

        Raises:
            RuntimeError: If there is an error generating the chat completion
        """
        # Reset the latest response if it's not None
        if self._latest_response is not None:
            self._latest_response = None
        try:
            response = await self.client.chat.completions.create(
                model=model or self.model,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                messages=messages
            )
            self._latest_response = response
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.exception(f"{type(e).__name__}  generating chat completion: {e}")
            raise RuntimeError(f"Error generating chat completion: {e}") from e 

    async def get_embeddings(self, texts: list[str] | str) -> list[list[float]] | None:
        """
        Generate embeddings for a list of text inputs using OpenAI's embedding model.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            An embedding vector, list of embedding vectors, or None if no embeddings are generated
        """
        if not texts:
            return []
        if isinstance(texts, str):
            texts = [texts]

        try:
            # Prepare texts by stripping whitespace and handling empty strings
            processed_texts = [text.strip() for text in texts]
            processed_texts = [text if text else " " for text in processed_texts]

            response = await self.client.embeddings.create(
                input=processed_texts,
                model=self.embedding_model
            )

            # Extract the embedding vectors from the response
            embeddings: list[list[float]] = [data.embedding for data in response.data]
            return embeddings if embeddings else []

        except Exception as e:
            self.logger.exception(f"{type(e).__name__}  generating chat completion: {e}")
            raise RuntimeError(f"Error generating embeddings: {e}") from e

    async def search_embeddings(
        self, 
        query: str, 
        top_k: int = 5,
        gnis: Optional[str] = None, 
    ) -> list[dict[str, Any]]:
        """
        Search for relevant documents using embeddings.
        
        Args:
            query: Search query
            gnis: Optional file ID to limit search to
            top_k: Number of top results to return
            
        Returns:
            List of relevant documents with similarity scores

        Raises:
            TypeError: If input types are incorrect
            ValueError: If input values are invalid
        """
        # Input validation
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__} instead.")
        else:
            if not query.strip():
                raise ValueError("Query must be a non-empty string.")

        if gnis is not None and not isinstance(gnis, str):
            raise TypeError(f"GNIS must be a string if provided, got {type(gnis).__name__} instead.")
        else:
            if not gnis.strip():
                raise ValueError("GNIS must be a non-empty string if provided.")

        if not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__} instead.")
        else:
            if top_k <= 0:
                raise ValueError(f"top_k must be a positive integer, got {top_k} instead.")


        # Generate embedding for the query
        try:
            query_embedding = await self.get_embeddings(query)
        except Exception as e:
            self.logger.exception(f"Error generating embedding for query: {e}")
            return []

        query_embedding = query_embedding[0] if query_embedding else None

        if not query_embedding:
            self.logger.error("No embedding generated for the query.")
            return []

        try:
            # Connect to the database
            conn = duckdb.connect(self.db_path, read_only=True)
            
            # Create the SQL query
            sql_query = """
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
                DOT_PRODUCT(e.embedding, ARRAY(SELECT vec FROM query_embedding)) / 
                (SQRT(DOT_PRODUCT(e.embedding, e.embedding)) * 
                 SQRT(DOT_PRODUCT(ARRAY(SELECT vec FROM query_embedding), 
                          ARRAY(SELECT vec FROM query_embedding)))) as similarity_score 
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
                
            # Execute the query with the embedding as parameter
            result_df = conn.execute(sql_query, [query_embedding]).fetchdf()
            
            # Convert to list of dictionaries
            results = []
            for _, row in result_df.iterrows():
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

            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching embeddings with DuckDB: {e}")
            return []
        
    async def query_database(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Query the database for relevant laws using DuckDB.
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of matching law records
        """
        try:
            # Connect to the database
            # NOTE: DuckDB can also connect to SQLite files
            conn = duckdb.connect(self.db_path, read_only=True)

            # Simple text search
            sql_query = f"""
                SELECT id, cid, title, chapter, place_name, state_name, date, 
                       bluebook_citation, content
                FROM citations
                WHERE lower(search_text) LIKE '%{query.lower()}%'
                ORDER BY place_name, title
                LIMIT {limit}
            """
            # Execute query and fetch results
            result_df = conn.execute(sql_query).fetchdf()
            
            # Convert DataFrame to list of dictionaries
            results = result_df.to_dict('records')
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying database with DuckDB: {e}")
            raise RuntimeError(f"Error querying database: {e}") from e


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
        prompt: Prompt = load_prompt_from_yaml("generate_rag_response", self.configs, query=query, context=context_text)
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
            
            generated_response = response.choices[0].message.content.strip()
            
            # Append the citations
            generated_response += f"\n\n{references.strip()}"
            
            return {
                "query": query,
                "response": generated_response,
                "context_used": [doc.get('bluebook_citation', 'No citation') for doc in context_docs],
                "model_used": self.model,
                "total_tokens": response.usage.total_tokens
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