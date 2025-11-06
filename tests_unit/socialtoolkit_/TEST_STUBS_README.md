# SocialToolkit Test Stubs

This directory contains pytest test stub files generated from Gherkin feature documentation.

## Overview

Each `.feature` file has a corresponding `test_*.py` file containing pytest test stubs. These stubs include:

- **Module-level docstrings** containing the complete Feature description
- **Fixtures** for Background steps (where applicable)
- **Test classes** organized by Rules
- **Test methods** for each Scenario assertion with complete step descriptions in docstrings
- **One Given-When-Then per test** - Each test method contains exactly one Then assertion

## Structure

```
tests-unit/socialtoolkit_/
├── architecture/
│   ├── document_retrieval_from_websites.feature
│   ├── test_document_retrieval_from_websites.py (641 lines, 54 tests)
│   ├── document_storage.feature
│   ├── test_document_storage.py (896 lines, 82 tests)
│   ├── prompt_decision_tree.feature
│   ├── test_prompt_decision_tree.py (848 lines, 78 tests)
│   ├── relevance_assessment.feature
│   ├── test_relevance_assessment.py (615 lines, 51 tests)
│   ├── top10_document_retrieval.feature
│   ├── test_top10_document_retrieval.py (441 lines, 38 tests)
│   ├── variable_codebook.feature
│   └── test_variable_codebook.py (851 lines, 76 tests)
└── resources/
    ├── cache_manager.feature
    ├── test_cache_manager.py (463 lines, 42 tests)
    ├── query_processor.feature
    ├── test_query_processor.py (442 lines, 41 tests)
    ├── ranking_algorithm.feature
    ├── test_ranking_algorithm.py (314 lines, 26 tests)
    ├── vector_search_engine.feature
    └── test_vector_search_engine.py (271 lines, 24 tests)
```

## Statistics

- **10 test stub files** created (5,782 lines total)
- **512 test methods** total (split from 264 original scenarios)
- **1:1 correspondence** with Gherkin feature files
- **All Gherkin lines preserved** in docstrings
- **Valid Python syntax** in all files
- **One Given-When-Then per test** - Each test has exactly one Then assertion

## Test Stub Format

### Module-Level Docstring

Each test file starts with the complete Feature description from the Gherkin file:

```python
"""
Feature: Cache Manager
  As a caching system
  I want to store and retrieve values with time-based expiration
  So that frequently accessed data can be served quickly

  Background:
    Given a CacheManager instance is initialized with default TTL
"""
```

### Background Fixtures

Background steps are converted to pytest fixtures:

```python
@pytest.fixture
def a_cachemanager_instance_is_initialized_with_default_ttl():
    """
    Given a CacheManager instance is initialized with default TTL
    """
    pass
```

### Test Classes by Rule

Each Rule from the Gherkin file becomes a test class:

```python
class TestGetMethodRetrievesCachedValues:
    """
    Rule: Get Method Retrieves Cached Values
    """
```

### Test Methods - One Given-When-Then Per Test

Each assertion from the original Scenario becomes a separate test method. This ensures each test validates exactly one behavior:

```python
def test_get_returns_cached_value_when_key_exists_and_not_expired(self):
    """
    Scenario: Get returns cached value when key exists and not expired
      Given a cache entry with key "query1" and value {"results": []}
      And the entry was created 60 seconds ago
      And cache_ttl_seconds is 3600
      When I call get with key "query1"
      Then the cached value is returned
    """
    pass

def test_get_returns_cached_value_when_key_exists_and_not_expired_1(self):
    """
    Scenario: Get returns cached value when key exists and not expired
      Given a cache entry with key "query1" and value {"results": []}
      And the entry was created 60 seconds ago
      And cache_ttl_seconds is 3600
      When I call get with key "query1"
      Then the value matches {"results": []}
    """
    pass
```

**Key Points:**
- Each test has only one `Then` assertion
- Multiple `And` clauses after `Then` in the original Gherkin are split into separate tests
- `And` clauses are converted to `Then` when they become the assertion
- Each test shares the same Given/When setup but validates a different assertion

## Next Steps

To implement these test stubs:

1. **Replace `pass` statements** with actual test implementation
2. **Implement fixtures** to set up test dependencies
3. **Add assertions** based on the Then/And steps in docstrings
4. **Add setup/teardown** as needed for test isolation
5. **Run tests** with `pytest tests-unit/socialtoolkit_/`

## Example Implementation Pattern

```python
@pytest.fixture
def cache_manager():
    """
    Given a CacheManager instance is initialized with default TTL
    """
    return CacheManager(cache_ttl_seconds=3600)

class TestGetMethodRetrievesCachedValues:
    """
    Rule: Get Method Retrieves Cached Values
    """

    def test_get_returns_cached_value_when_key_exists_and_not_expired(self, cache_manager):
        """
        Scenario: Get returns cached value when key exists and not expired
          Given a cache entry with key "query1" and value {"results": []}
          And the entry was created 60 seconds ago
          And cache_ttl_seconds is 3600
          When I call get with key "query1"
          Then the cached value is returned
          And the value matches {"results": []}
        """
        # Given
        test_value = {"results": []}
        cache_manager.set("query1", test_value)
        
        # When
        result = cache_manager.get("query1")
        
        # Then
        assert result is not None
        assert result == test_value
```

## Validation

All test stub files have been validated for:

✅ Valid Python syntax (using `ast.parse()`)
✅ Proper pytest structure (fixtures, classes, methods)
✅ Complete Gherkin preservation in docstrings
✅ One-to-one correspondence with feature files

## Usage

Run all tests (will currently pass as they only contain `pass`):

```bash
pytest tests-unit/socialtoolkit_/
```

Run specific architecture tests:

```bash
pytest tests-unit/socialtoolkit_/architecture/
```

Run specific resource tests:

```bash
pytest tests-unit/socialtoolkit_/resources/
```

Run a specific test file:

```bash
pytest tests-unit/socialtoolkit_/architecture/test_cache_manager.py
```

Run with verbose output to see all test names:

```bash
pytest tests-unit/socialtoolkit_/ -v
```
