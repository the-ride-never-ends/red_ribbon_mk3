Feature: Document Storage
  As a document management system
  I want to store, retrieve, update, and delete documents with metadata and vectors
  So that documents can be efficiently managed and searched

  Background:
    Given a DocumentStorage instance is initialized
    And a database connection is available
    And a cache service is available
    And a vector store service is available
    And an ID generator service is available

  Rule: Execute Method Accepts Action Parameter

    Scenario: Execute with "store" action
      Given action parameter is "store"
      And documents, metadata, and vectors are provided
      When I call execute with the action
      Then the store_documents method is called
      And documents are persisted to storage

    Scenario: Execute with "retrieve" action
      Given action parameter is "retrieve"
      And document IDs are provided
      When I call execute with the action
      Then documents are fetched from storage
      And matching documents are returned

    Scenario: Execute with "update" action
      Given action parameter is "update"
      And document updates are provided
      When I call execute with the action
      Then documents are updated in storage
      And updated documents are returned

    Scenario: Execute with "delete" action
      Given action parameter is "delete"
      And document IDs to delete are provided
      When I call execute with the action
      Then documents are removed from storage
      And deletion confirmation is returned

  Rule: Execute Validates Action Parameter Type

    Scenario: Execute rejects non-string action
      Given an integer action value 123
      When I call execute with the invalid action
      Then a TypeError is raised
      And the error message indicates "Action must be a string"

    Scenario: Execute rejects None as action
      Given a None value as action
      When I call execute with the invalid action
      Then a TypeError is raised

    Scenario: Execute rejects unknown action string
      Given action parameter is "invalid_action"
      When I call execute with the unknown action
      Then an error is raised indicating unknown action

  Rule: Execute Returns Dictionary with Operation Results

    Scenario: Store operation returns success status
      Given a store operation with 3 documents
      When execute completes successfully
      Then the result contains key "status"
      And the status value is "success"
      And the result contains key "stored_count"
      And stored_count is 3

    Scenario: Retrieve operation returns documents
      Given a retrieve operation for 2 document IDs
      When execute completes successfully
      Then the result contains key "documents"
      And the documents list contains 2 items
      And each document is a complete document object

    Scenario: Update operation returns updated documents
      Given an update operation for 1 document
      When execute completes successfully
      Then the result contains key "updated_documents"
      And updated_documents list contains the modified document

    Scenario: Delete operation returns deletion count
      Given a delete operation for 5 document IDs
      When execute completes successfully
      Then the result contains key "deleted_count"
      And deleted_count is 5

  Rule: Store Operation Persists Documents with Metadata and Vectors

    Scenario: Store saves all three components
      Given 2 documents with corresponding metadata and vectors
      When store operation is executed
      Then 2 documents are saved to database
      And 2 metadata records are saved
      And 2 vectors are saved to vector store
      And all components share the same document IDs

    Scenario: Store generates IDs for new documents
      Given documents without IDs
      When store operation is executed
      Then the ID generator creates unique IDs
      And IDs are assigned to documents
      And IDs are used across documents, metadata, and vectors

    Scenario: Store validates document-metadata-vector alignment
      Given 3 documents but only 2 metadata records
      When store operation is executed
      Then a validation error is raised
      And the error indicates mismatched counts

  Rule: Retrieve Operation Fetches Documents by ID

    Scenario: Retrieve returns documents for valid IDs
      Given document IDs ["doc1", "doc2"] exist in storage
      When retrieve operation is executed with these IDs
      Then exactly 2 documents are returned
      And document IDs match the requested IDs
      And documents include all stored fields

    Scenario: Retrieve returns empty for non-existent IDs
      Given document IDs ["nonexistent1", "nonexistent2"]
      And these IDs do not exist in storage
      When retrieve operation is executed
      Then an empty documents list is returned
      And no error is raised

    Scenario: Retrieve returns partial results for mixed IDs
      Given document IDs ["existing", "nonexistent"]
      And only "existing" is in storage
      When retrieve operation is executed
      Then exactly 1 document is returned
      And the document ID is "existing"

  Rule: Update Operation Modifies Existing Documents

    Scenario: Update modifies document content
      Given a document with ID "doc1" exists
      And an update with new content for "doc1"
      When update operation is executed
      When the document is retrieved after update
      Then the document has the new content
      And other fields remain unchanged

    Scenario: Update modifies document metadata
      Given a document with ID "doc2" exists
      And updated metadata for "doc2"
      When update operation is executed
      Then the metadata is updated in storage
      And the document content remains unchanged

    Scenario: Update fails for non-existent document
      Given a document ID "nonexistent"
      And an update for this ID
      When update operation is executed
      Then an error is raised or update count is 0
      And the error indicates document not found

  Rule: Delete Operation Removes Documents Completely

    Scenario: Delete removes document and associated data
      Given a document with ID "doc1" exists with metadata and vector
      When delete operation is executed for "doc1"
      Then the document is removed from database
      And the associated metadata is removed
      And the associated vector is removed
      And retrieve for "doc1" returns empty

    Scenario: Delete handles non-existent IDs gracefully
      Given document IDs ["nonexistent1", "nonexistent2"]
      When delete operation is executed
      Then no error is raised
      And deleted_count is 0

  Rule: Cache Service Is Used When Enabled

    Scenario: Cache is checked before database on retrieve
      Given cache_enabled is configured as True
      And document "doc1" is in cache
      When retrieve operation is executed for "doc1"
      Then the document is returned from cache
      And the database is not queried

    Scenario: Cache is populated after database retrieval
      Given cache_enabled is configured as True
      And document "doc2" is not in cache but in database
      When retrieve operation is executed for "doc2"
      Then the document is fetched from database
      And the document is added to cache
      And cache TTL is set to cache_ttl_seconds

    Scenario: Cache is bypassed when disabled
      Given cache_enabled is configured as False
      When retrieve operation is executed
      Then the cache is not checked
      And all retrieval is from database

  Rule: Batch Processing Respects Batch Size Configuration

    Scenario: Large store operation is batched
      Given 250 documents to store
      And batch_size is configured as 100
      When store operation is executed
      Then documents are stored in 3 batches
      And first two batches contain 100 documents
      And last batch contains 50 documents

    Scenario: Small store operation uses single batch
      Given 30 documents to store
      And batch_size is configured as 100
      When store operation is executed
      Then documents are stored in 1 batch
      And the batch contains all 30 documents

  Rule: Vector Dimensions Are Validated

    Scenario: Store validates vector dimensions match configuration
      Given vector_dim is configured as 1536
      And vectors with dimension 1536
      When store operation is executed
      Then vectors are stored successfully

    Scenario: Store rejects mismatched vector dimensions
      Given vector_dim is configured as 1536
      And vectors with dimension 768
      When store operation is executed
      Then a validation error is raised
      And the error indicates dimension mismatch

  Rule: Document Status Tracking

    Scenario: New documents have status "new"
      Given new documents without status field
      When store operation is executed
      Then documents are assigned status "new"
      And the status is persisted

    Scenario: Document status can be updated
      Given a document with status "new"
      And an update changing status to "complete"
      When update operation is executed
      Then the document status is "complete"

  Rule: Storage Type Configuration Determines Backend

    Scenario: SQL storage type uses SQL database
      Given storage_type is configured as "sql"
      When store operation is executed
      Then documents are stored in SQL database
      And SQL queries are used for operations

    Scenario: Parquet storage type uses parquet files
      Given storage_type is configured as "parquet"
      When store operation is executed
      Then documents are stored as parquet files
      And parquet operations are used

    Scenario: Cache storage type uses cache only
      Given storage_type is configured as "cache"
      When store operation is executed
      Then documents are stored in cache only
      And no database persistence occurs

  Rule: Database Connection Handling

    Scenario: Execute uses provided database connection
      Given a database connection is provided in resources
      When execute is called
      Then the provided database connection is used
      And operations execute within database context

    Scenario: Execute accepts database in kwargs when resources db is None
      Given resources database is None
      And a database connection is in kwargs
      When execute is called
      Then the kwargs database is used
      And operations complete successfully

  Rule: Execute Logs Operations

    Scenario: Execute logs operation start
      Given a store operation
      When execute is called
      Then a log message indicates "Starting document storage operation: store"

    Scenario: Execute logs operation completion
      Given a retrieve operation completes successfully
      Then a log message indicates operation completion
      And the log includes result counts
