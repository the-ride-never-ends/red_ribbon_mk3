from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import logging


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
    headers: Dict[str, str] = {}
    batch_size: int = 10
    follow_links: bool = False
    max_depth: int = 1


class DocumentRetrievalFromWebsites:
    """
    Document Retrieval from Websites for data extraction system
    based on mermaid chart in README.md
    """
    
    def __init__(self, resources: Dict[str, Any] = None, configs: DocumentRetrievalConfigs = None):
        """
        Initialize with injected dependencies and configuration
        
        Args:
            resources: Dictionary of resources including services
            configs: Configuration for Document Retrieval
        """
        self.resources = resources
        self.configs = configs
        self.logger = logging.getLogger(self.class_name)
        
        # Extract needed services from resources
        self.static_webpage_parser = resources["static_webpage_parser"]
        self.dynamic_webpage_parser = resources["dynamic_webpage_parser"]
        self.data_extractor = resources["data_extractor"]
        self.vector_generator = resources["vector_generator"]
        self.metadata_generator = resources["metadata_generator"]
        self.document_storage = resources["document_storage_service"]
        self.url_path_generator = resources["url_path_generator"]
        
        self.logger.info("DocumentRetrievalFromWebsites initialized with services")

    @property
    def class_name(self) -> str:
        """Get class name for this service"""
        return self.__class__.__name__.lower()

    def execute(self, 
        domain_urls: List[str]
        ) -> Dict[str, Union[List[Any], List[dict[str, Any]], List[dict[str, list[float]]]]]:
        """
        Execute the document retrieval flow based on the mermaid chart
        
        Args:
            domain_urls: List of domain URLs to retrieve documents from
            
        Returns:
            Dictionary containing retrieved documents, metadata, and vectors
        """
        self.logger.info(f"Starting document retrieval from {len(domain_urls)} domains")
        
        all_documents = []
        all_metadata = []
        all_vectors = []
        
        for domain_url in domain_urls:
            # Step 1: Generate URLs from domain URL
            urls: list[str] = self._generate_urls(domain_url)

            for url in urls:
                # Step 2: Determine webpage type and parse accordingly
                webpage_type: WebpageType = self._determine_webpage_type(url)

                match webpage_type:
                    case WebpageType.STATIC:
                        raw_data = self.static_webpage_parser.parse(url)
                    case WebpageType.DYNAMIC:
                        raw_data = self.dynamic_webpage_parser.parse(url)
                    case _:
                        self.logger.warning(f"Unknown webpage type for URL: {url}")
                        raise ValueError(f"Unknown webpage type for URL: {url}")

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
        self.document_storage.store(all_documents, all_metadata, all_vectors)

        self.logger.info(f"Retrieved and stored {len(all_documents)} documents")
        return {
            "documents": all_documents,
            "metadata": all_metadata,
            "vectors": all_vectors
        }

    def get_urls(self, query: str) -> List[str]:
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
        # Get domain URLs from input data point


        domain_urls = []

        urls = {
            self._generate_urls(domain_url) for domain_url in domain_urls
        }
        return list(urls)

    def retrieve_documents(self, domain_urls: List[str]) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Public method to retrieve documents from websites

        Args:
            domain_urls: List of domain URLs to retrieve documents from

        Returns:
            Tuple of (documents, metadata, vectors)
        """
        result = self.run(domain_urls)
        return (
            result["documents"],
            result["metadata"],
            result["vectors"]
        )

    def _generate_urls(self, domain_url: str) -> List[str]:
        """Generate URLs from domain URL using URL path generator"""
        return self.url_path_generator.generate(domain_url)
        
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

    def _create_documents(self, raw_strings: List[str], url: str) -> List[Any]:
        """Create documents from raw strings"""
        # Implementation would create document objects from raw text content
        # TODO: This is a placeholder implementation
        documents = []
        for i, content in enumerate(raw_strings):
            documents.append({
                "id": f"{url}_{i}",
                "hash": hash(content + url),
                "content": content,
                "url": url,
                "timestamp": self.resources["timestamp_service"].now()
            })
        return documents