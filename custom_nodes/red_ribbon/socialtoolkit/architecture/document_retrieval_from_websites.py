
from datetime import datetime
from enum import Enum
import logging
import re
from typing import Any, Annotated as Ann, Callable, Optional, Union
import queue

try:
    from requests import Timeout
except ImportError:
    Timeout = TimeoutError
from pydantic import AfterValidator as AV, BaseModel, Field, HttpUrl, ValidationError

from custom_nodes.red_ribbon.utils_ import (
    logger, DatabaseAPI, LLM
)
from custom_nodes.red_ribbon._custom_errors import DatabaseError
from .document_storage import DocumentStorage


class WebpageType(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"


class DocumentRetrievalConfigs(BaseModel):
    """Configuration for Document Retrieval from Websites workflow"""
    timeout_seconds: int = 30
    max_retries: int = 3
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    dynamic_rendering_wait_time: int = 5
    selenium_enabled: bool = False
    headers: dict[str, str] = Field(default_factory=dict)
    batch_size: int = 10
    follow_links: bool = False
    max_depth: int = 1


def _validate_url_completeness(urls: list[HttpUrl]) -> list[HttpUrl]:
    # Complete URL regex pattern
    complete_url_pattern = r'^(http|https)://[^\s/$.?#].[^\s]*$'
    for url in urls:
        _url = str(url)
        if not re.match(complete_url_pattern, _url):
            raise ValueError(f"URL '{url}' is not a complete URL with scheme")
    return urls


class ValidUrls(BaseModel):
    urls: Ann[list[HttpUrl], AV(_validate_url_completeness)]


class DocumentRetrievalFromWebsites:
    """
    Document Retrieval from Websites for data extraction system
    based on mermaid chart in README.md
    """

    def __init__(self, *, 
                 resources: dict[str, Any],
                configs: DocumentRetrievalConfigs
        ) -> None:
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including services
            configs: Configuration for Document Retrieval
        """
        self.resources = resources
        self.configs = configs
        self.logger: logging.Logger = resources['logger']
        self.db: DatabaseAPI = resources["db"]
        self.llm: LLM = resources["llm"]

        # Extract needed services from resources
        self.static_parser = resources["static_parser"]
        self.dynamic_parser = resources["dynamic_parser"]
        self.data_extractor = resources["data_extractor"]
        self.vector_generator = resources["vector_generator"]
        self.metadata_generator = resources["metadata_generator"]
        self.document_storage: DocumentStorage = resources["document_storage"]
        self.url_path_generator = resources["url_path_generator"]
        self._get_cid: Callable = resources["get_cid"]

        assert hasattr(self.static_parser, 'parse'), "static_parser must have parse method"
        assert hasattr(self.dynamic_parser, 'parse'), "dynamic_parser must have parse method"
        assert hasattr(self.data_extractor, 'extract'), "data_extractor must have extract method"
        assert hasattr(self.vector_generator, 'generate'), "vector_generator must have generate method"
        assert hasattr(self.metadata_generator, 'generate'), "metadata_generator must have generate method"
        assert hasattr(self.document_storage, 'store'), "document_storage must have store method"
        assert hasattr(self.url_path_generator, 'generate'), "url_path_generator must have generate method"

        self.logger.info("DocumentRetrievalFromWebsites initialized with services")


    def execute(self, 
        domain_urls: list[str]
        ) -> dict[str, Union[list[Any], list[dict[str, Any]], list[dict[str, list[float]]]]]:
        """
        Execute the document retrieval flow based on the mermaid chart
        
        Args:
            domain_urls: List of domain URLs to retrieve documents from
            
        Returns:
            Dictionary containing retrieved documents, metadata, and vectors

        Raises:
            ValueError: If domain_urls are invalid
        """
        if not isinstance(domain_urls, list):
            raise TypeError(f"domain_urls must be a list, got {type(domain_urls).__name__}")
        for url in domain_urls:
            if not isinstance(url, str):
                raise TypeError(f"Each domain URL must be a string, got {type(url).__name__}")

        try:
            _ = ValidUrls.model_validate({"urls": domain_urls})
        except ValidationError as e:
            raise ValueError(f"Invalid domain URLs provided: {e.errors()}") from e
        except Exception as e:
            msg = f"Unexpected error validating domain URLS: {e}"
            self.logger.exception(msg)
            raise RuntimeError(msg) from e

        self.logger.info(f"Starting document retrieval from {len(domain_urls)} domains")

        all_documents: list[dict[str, Any]] = []
        all_metadata = []
        all_vectors = []

        raw_datum = []

        for domain_url in domain_urls:
            # Step 1: Generate URLs from domain URL
            urls: list[str] = self._generate_urls(domain_url)

            self.logger.info(f"Generated {len(urls)} URLs from domain {domain_url}")
            self.logger.debug(f"Generated URLs: {urls}")
            url_queue = queue.Queue()

            # Initialize queue with URLs and retry counter
            for url in urls:
                url_queue.put_nowait((url, 0))  # (url, retry_count)

            while url_queue.qsize() > 0:
                # Step 2: Determine webpage type and parse accordingly
                url, retry_count = url_queue.get_nowait()
                webpage_type: WebpageType = self._determine_webpage_type(url)
                parser: Callable = lambda url: None

                match webpage_type:
                    case WebpageType.STATIC:
                        parser = self.static_parser.parse
                    case WebpageType.DYNAMIC:
                        parser = self.dynamic_parser.parse
                    case _:
                        self.logger.warning(f"Unknown webpage type for URL: {url}")
                        raise ValueError(f"Unknown webpage type for URL: {url}")

                try:
                    raw_data = parser(url)
                    raw_datum.append((url, raw_data))
                except (Timeout, TimeoutError) as e:
                    if retry_count < self.configs.max_retries:
                        url_queue.put_nowait((url, retry_count + 1))
                        self.logger.warning(f"Timeout parsing URL '{url}': {e}. Retry {retry_count + 1}/{self.configs.max_retries}")
                    else:
                        msg = f"Max retries ({self.configs.max_retries}) exceeded for URL '{url}' due to timeout. Skipping...\n{e}"
                        self.logger.error(msg)
                        continue
                except Exception as e:
                    msg = f"Unexpected Error parsing URL '{url}': {e}"
                    self.logger.exception(msg)
                    raise e

        for (url, raw_data) in raw_datum:
            # Step 3: Extract structured data from raw data
            raw_strings = self.data_extractor.extract(raw_data)

            # Step 4: Generate documents, vectors, and metadata
            documents = self._create_documents(raw_strings, url)
            document_vectors = self.vector_generator.generate(documents)
            document_metadata = self.metadata_generator.generate(documents, url)

            all_documents.extend(documents)
            all_vectors.extend(document_vectors)
            all_metadata.extend(document_metadata)

        # Step 5: Store documents, vectors, and metadata
        try:
            self.document_storage.store(all_documents, all_metadata, all_vectors)
        except Exception as e:
            raise DatabaseError(f"Failed to store documents: {e}") from e

        self.logger.info(f"Retrieved and stored {len(all_documents)} documents")
        return {
            "documents": all_documents,
            "metadata": all_metadata,
            "vectors": all_vectors
        }

    def get_urls(self, query: str) -> list[str]:
        """
        Get URLs from an input data point

        Args:
            query: Natural language input to extract domain URLs.

        Returns:
            List of generated URLs

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty.
            RuntimeError: If URL generation fails.

        Examples:
            >>> query = "What is the local sales tax in Cheyenne, WY?"
            >>> document_retrieval = DocumentRetrievalFromWebsites(resources, configs)
            >>> urls = document_retrieval.get_urls(query)
            >>> print(urls)
            ['https://www.cheyennecity.org/tax-info', 'https://www.wyoming.gov/tax-rates']
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")
        query = query.strip()
        if not query:
            raise ValueError("query cannot be empty")

        # Get unique domain URLs from input data point
        domain_urls: set[str] = self._query_to_domain_urls(query)
        self.logger.info(f"Extracted {len(domain_urls)} unique domain URLs from query")

        url_list = []
        for url in domain_urls:
            try:
                generated_urls = self._generate_urls(url)
                url_list.extend(generated_urls)
            except Exception as e:
                msg = f"Failed to generate URLs from domain '{url}': {e}"
                self.logger.exception(msg)
                raise RuntimeError(msg) from e

        return url_list

    def _query_to_domain_urls(self, query: str) -> set[str]:
        """Turn a plain English query into a set of domain URLs."""
        raise NotImplementedError("_query_to_domain_urls method not implemented")

    def retrieve_documents(self, domain_urls: list[str]) -> tuple[list[Any], list[Any], list[Any]]:
        """
        Public method to retrieve documents from websites

        Args:
            domain_urls: List of domain URLs to retrieve documents from

        Returns:
            Tuple of (documents, metadata, vectors)
        """
        result = self.execute(domain_urls)
        return (
            result["documents"],
            result["metadata"],
            result["vectors"]
        )

    def _generate_urls(self, domain_url: str) -> list[str]:
        """Generate URLs from domain URL using URL path generator"""
        return self.url_path_generator.generate(domain_url)
    
    def _get_timestamp(self) -> Any:
        """Get current timestamp from timestamp service"""
        return datetime.now()

    def _determine_webpage_type(self, url: str) -> WebpageType:
        """
        Determine whether a webpage is static or dynamic
        
        This is a simple implementation that could be enhanced with
        more sophisticated detection mechanisms
        TODO: Improve detection logic.
        """
        # Check URL patterns that typically indicate dynamic content
        dynamic_indicators = {
            "#!", "?", "api", "ajax", "load", "spa", "react", 
            "angular", "vue", "dynamic", "js-rendered"
        }

        for indicator in dynamic_indicators:
            if indicator in url.lower():
                return WebpageType.DYNAMIC

        return WebpageType.STATIC

    def _create_documents(self, raw_strings: list[str], url: str) -> list[Any]:
        """Create documents from raw strings"""
        # Implementation would create document objects from raw text content
        # TODO: This is a placeholder implementation
        documents = []
        for idx, content in enumerate(raw_strings):
            try:
                id: str = self._get_cid(content + url)
            except Exception as e:
                msg = f"Failed to generate CID for document {idx} from URL '{url}': {e}"
                self.logger.exception(msg)
                raise RuntimeError(msg) from e

            documents.append({
                "id": id,
                "content": content,
                "url": url,
                "timestamp": datetime.now()
            })
        return documents
