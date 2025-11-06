# SocialToolkit Gherkin Documentation Summary

This directory contains comprehensive Gherkin feature files documenting the externally verifiable behavior of the SocialToolkit system.

## Architecture Layer (6 feature files)

### 1. prompt_decision_tree.feature (292 lines)
Documents the PromptDecisionTree system for executing decision trees of prompts to extract information from documents.

**Key behaviors documented:**
- Execute method returns extracted data points
- Control flow returns dictionary with success, output_data_point, responses, iterations
- Pages are concatenated up to max_pages_to_concatenate
- Decision tree execution with node traversal
- LLM prompt generation with document context
- Output data point extraction from final response
- NgramValidator text analysis utilities
- Decision tree node creation from prompt sequences
- Human review integration for error handling

### 2. top10_document_retrieval.feature (141 lines)
Documents the Top-10 Document Retrieval system for finding the most relevant documents using vector search.

**Key behaviors documented:**
- Execute returns dictionary with relevant_documents, scores, top_doc_ids
- Returns at most N documents where N is retrieval_count
- Documents are ranked by similarity score in descending order
- Similarity scores respect configured threshold
- Input type validation (rejects non-string queries)
- Handles document and vector parameters (provided vs. fetched from storage)
- Ranking method configuration (cosine similarity, dot product, euclidean)
- Result documents contain required metadata (id, content, url, title)

### 3. document_retrieval_from_websites.feature (195 lines)
Documents the Document Retrieval from Websites system for web scraping and data extraction.

**Key behaviors documented:**
- Execute accepts list of domain URLs
- Returns dictionary with documents, metadata, vectors
- Static and dynamic webpage parsing
- URL generation expands domain URLs to page URLs
- Data extraction converts raw data to structured strings
- Documents are created with URL context
- Vectors are generated for all documents
- Metadata is generated for all documents
- Documents, vectors, and metadata are stored
- Batch processing configuration
- HTTP request failure handling
- User agent configuration

### 4. relevance_assessment.feature (197 lines)
Documents the Relevance Assessment system for evaluating document relevance using LLM assessments.

**Key behaviors documented:**
- Control flow returns dictionary with relevant_pages, relevant_doc_ids, page_numbers, relevance_scores
- Documents filtered by criteria_threshold
- Hallucination filter when enabled
- Relevance scores calculated for each document (0.0 to 1.0)
- Page numbers extracted from relevant documents
- Cited pages extracted by page numbers
- LLM API usage with max_retries configuration
- Citations truncated to maximum length
- Assess method provides public interface
- Input type validation
- Relevant document IDs tracked

### 5. document_storage.feature (290 lines)
Documents the Document Storage system for managing documents with metadata and vectors.

**Key behaviors documented:**
- Execute accepts action parameter (store, retrieve, update, delete)
- Action parameter validation
- Returns dictionary with operation results
- Store operation persists documents with metadata and vectors
- Store generates IDs for new documents
- Store validates document-metadata-vector alignment
- Retrieve operation fetches documents by ID
- Retrieve returns empty for non-existent IDs
- Update operation modifies existing documents
- Delete operation removes documents completely
- Cache service usage when enabled
- Batch processing respects batch_size configuration
- Vector dimensions validation
- Document status tracking (new, processing, complete, error)
- Storage type configuration (SQL, parquet, cache)

### 6. variable_codebook.feature (302 lines)
Documents the Variable Codebook system for managing variable definitions with assumptions and prompt sequences.

**Key behaviors documented:**
- Control flow accepts action parameter (get_variable, get_prompt_sequence, get_assumptions, add_variable, update_variable)
- Action validation
- Returns dictionary with operation results
- Variable structure contains required fields (label, item_name, description, units)
- Optional fields (assumptions, prompt_decision_tree)
- Assumptions object structure (general, specific, business_owner, business, taxes)
- Prompt sequence extraction from decision tree
- Get prompt sequence for input extracts variable name
- Add variable persists new variable
- Update variable modifies existing variable
- Variables loaded from file when configured
- Cache service usage when enabled
- Default assumptions application when enabled
- Prompt decision tree as DiGraph
- Keyword matching maps input to variables

## Resource Layer (4 feature files)

### 7. vector_search_engine.feature (108 lines)
Documents the VectorSearchEngine for document similarity search.

**Key behaviors documented:**
- Add vectors stores documents and embeddings
- Search returns top-K similar documents
- Search results include required document fields
- Results ordered by similarity score
- Query vector parameter required
- Search operations logged

### 8. ranking_algorithm.feature (125 lines)
Documents the RankingAlgorithm for ranking documents by relevance.

**Key behaviors documented:**
- Rank accepts documents and query
- Ranking score based on keyword frequency
- Title matches receive bonus score
- Keywords at document start receive bonus
- Rank score stored in document
- Documents sorted in descending order by rank score
- Query keywords extracted and used for scoring
- Content matching is case-insensitive
- Ranking operations logged

### 9. cache_manager.feature (181 lines)
Documents the CacheManager for time-based caching with expiration.

**Key behaviors documented:**
- Get method retrieves cached values
- Get returns None for non-existent or expired keys
- Set method stores values with timestamp
- Set overwrites existing keys
- Cache TTL configuration affects expiration
- Clear method removes all cache entries
- Get stats returns cache information (size, keys, ttl_seconds)
- Expired entries removed on get
- Cache operations logged

### 10. query_processor.feature (174 lines)
Documents the QueryProcessor for query normalization and tokenization.

**Key behaviors documented:**
- Process accepts string query
- Returns dictionary with original_query, normalized_query, tokens, keywords
- Normalization converts to lowercase
- Normalization strips whitespace
- Tokenization splits query into words
- Keywords filter extracts significant words (length > 3)
- Original query preserved unchanged
- Handles special characters
- Query processing logged
- Empty or whitespace-only queries handled
- Unicode and non-ASCII characters handled

## Documentation Statistics

- **Total feature files:** 10
- **Total lines of documentation:** 2,005 lines
- **Architecture components:** 6 feature files (1,417 lines)
- **Resource components:** 4 feature files (588 lines)

## Documentation Principles

All feature files follow these principles:

1. **Externally Verifiable:** Focus on observable inputs, outputs, and behaviors
2. **Specific:** Concrete examples with actual values, not abstract descriptions
3. **Given-When-Then:** Standard Gherkin pattern for all scenarios
4. **Rules-Based:** Organized by behavioral rules for clarity
5. **Comprehensive:** Cover success cases, error cases, edge cases, and configuration
6. **Type Safety:** Document input type validation and error handling
7. **Configuration-Aware:** Document how configuration affects behavior
8. **Logging:** Document logging behavior for observability

## Usage

These feature files serve as:
- **Specification:** Define expected behavior for implementation
- **Test Basis:** Foundation for pytest-bdd test implementations
- **Documentation:** Readable documentation of system behavior
- **Contract:** API contract for component interactions
