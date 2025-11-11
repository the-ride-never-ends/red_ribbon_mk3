# Function and Class stubs from '/home/kylerose1946/red_ribbon_mk3/custom_nodes/red_ribbon/socialtoolkit/architecture/document_storage.py'

Files last updated: 1762819347.7065544

Stub file last updated: 2025-11-10 16:03:22

## DocumentStatus

```python
class DocumentStatus(str, Enum):
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## DocumentStorage

```python
class DocumentStorage:
    """
    Document Storage system.

Manages the storage and retrieval of documents, versions, metadata, and vectors
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## DocumentStorageConfigs

```python
class DocumentStorageConfigs(BaseModel):
    """
    Configuration for Document Storage
    """
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## SourceType

```python
class SourceType(str, Enum):
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## StorageType

```python
class StorageType(str, Enum):
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## VersionStatus

```python
class VersionStatus(str, Enum):
```
* **Async:** False
* **Method:** False
* **Class:** N/A

## __init__

```python
def __init__(self, resources: Dict[str, Any], configs: DocumentStorageConfigs):
    """
    Initialize with injected dependencies and configuration

Args:
    resources: Dictionary of resources including storage services
    configs: Configuration for Document Storage
    """
```
* **Async:** False
* **Method:** True
* **Class:** DocumentStorage

## class_name

```python
@property
def class_name(self) -> str:
    """
    Get class name for this service
    """
```
* **Async:** False
* **Method:** True
* **Class:** DocumentStorage

## delete_documents

```python
def delete_documents(self, doc_ids: list[str]) -> Dict[str, Any]:
    """
    Delete documents by ID from the database.

Args:
    doc_ids: List of IDs of documents to delete.

Returns:
    A dictionary containing the following:
    - result (bool): True if deletion was successful, False otherwise.
    - deleted_count (int): Number of documents deleted.
    - message (str): Informational message about the deletion operation.
     If result is True, message is "Documents deleted successfully."
     If result is False, message contains the error details.

Raises:
    TypeError: If doc_ids is not a list of strings.
    ValueError: If doc_ids list is empty, or if any ID is not found in the database.
    DatabaseError: If there is an error executing the delete operation.
    """
```
* **Async:** False
* **Method:** True
* **Class:** DocumentStorage

## execute

```python
def execute(self, action: str, **kwargs) -> Dict[str, Any]:
    """
    Execute document storage operations based on the action

Args:
    action: Operation to perform (store, retrieve, update, delete)
    **kwargs: Operation-specific parameters
    
Returns:
    Dictionary containing operation results
    """
```
* **Async:** False
* **Method:** True
* **Class:** DocumentStorage

## get_documents_and_vectors

```python
def get_documents_and_vectors(self, doc_ids: list[str] = None, filters: Dict[str, Any] = None) -> Tuple[list[Any], list[Any]]:
    """
    Retrieve documents and their vectors

Args:
    doc_ids: Optional list of document IDs to retrieve
    filters: Optional filters to apply
    
Returns:
    Tuple of (documents, vectors)
    """
```
* **Async:** False
* **Method:** True
* **Class:** DocumentStorage

## get_vectors

```python
def get_vectors(self, doc_ids: list[str]) -> Dict[str, Any]:
    """
    Get vectors for the specified document IDs
    """
```
* **Async:** False
* **Method:** True
* **Class:** DocumentStorage

## retrieve_documents

```python
def retrieve_documents(self, doc_ids: list[str] = None, filters: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Retrieve documents from the database
    """
```
* **Async:** False
* **Method:** True
* **Class:** DocumentStorage

## store

```python
def store(self, documents: list[Any], metadata: list[Any], vectors: list[Any]) -> Dict[str, Any]:
    """
    Store documents, metadata, and vectors

Args:
    documents: Documents to store
    metadata: Metadata for the documents
    vectors: Vectors for the documents
    
Returns:
    Dictionary with storage status
    """
```
* **Async:** False
* **Method:** True
* **Class:** DocumentStorage

## store_documents

```python
def store_documents(self, documents: list[Any], metadata: list[Any], vectors: list[Any]) -> Dict[str, Any]:
    """
    Store documents, metadata, and vectors in the database
    """
```
* **Async:** False
* **Method:** True
* **Class:** DocumentStorage

## update_documents

```python
def update_documents(self, documents: list[Any]) -> Dict[str, Any]:
    """
    Update existing documents
    """
```
* **Async:** False
* **Method:** True
* **Class:** DocumentStorage
