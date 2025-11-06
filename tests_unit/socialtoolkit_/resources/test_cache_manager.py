"""
Feature: Cache Manager
  As a caching system
  I want to store and retrieve values with time-based expiration
  So that frequently accessed data can be served quickly

  Background:
    Given a CacheManager instance is initialized with default TTL
"""

import pytest
import time
from unittest.mock import Mock, patch
from custom_nodes.red_ribbon.socialtoolkit.resources.top10_document_retrieval.cache_manager import CacheManager


# Fixtures for Background

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
        """
        # Arrange
        test_value = {"results": []}
        cache_manager.set("query1", test_value)
        
        # Act
        result = cache_manager.get("query1")
        
        # Assert
        assert result is not None

    def test_get_returns_cached_value_when_key_exists_and_not_expired_1(self, cache_manager):
        """
        Scenario: Get returns cached value when key exists and not expired
          Given a cache entry with key "query1" and value {"results": []}
          And the entry was created 60 seconds ago
          And cache_ttl_seconds is 3600
          When I call get with key "query1"
          Then the value matches {"results": []}
        """
        # Arrange
        test_value = {"results": []}
        cache_manager.set("query1", test_value)
        
        # Act
        result = cache_manager.get("query1")
        
        # Assert
        assert result == {"results": []}

    def test_get_returns_none_for_non_existent_key(self, cache_manager):
        """
        Scenario: Get returns None for non-existent key
          Given no cache entry exists for key "missing_key"
          When I call get with key "missing_key"
          Then None is returned
        """
        # Arrange - no setup needed, cache is empty
        
        # Act
        result = cache_manager.get("missing_key")
        
        # Assert
        assert result is None

    def test_get_returns_none_for_expired_entry(self, cache_manager):
        """
        Scenario: Get returns None for expired entry
          Given a cache entry with key "old_query"
          And the entry was created 7200 seconds ago
          And cache_ttl_seconds is 3600
          When I call get with key "old_query"
          Then None is returned
        """
        # Arrange
        cache_manager.set("old_query", {"data": "old"})
        # Simulate entry created 7200 seconds ago
        cache_manager.cache_timestamps["old_query"] = time.time() - 7200
        
        # Act
        result = cache_manager.get("old_query")
        
        # Assert
        assert result is None

    def test_get_returns_none_for_expired_entry_1(self, cache_manager):
        """
        Scenario: Get returns None for expired entry
          Given a cache entry with key "old_query"
          And the entry was created 7200 seconds ago
          And cache_ttl_seconds is 3600
          When I call get with key "old_query"
          Then the expired entry is removed from cache
        """
        # Arrange
        cache_manager.set("old_query", {"data": "old"})
        cache_manager.cache_timestamps["old_query"] = time.time() - 7200
        
        # Act
        cache_manager.get("old_query")
        
        # Assert
        assert "old_query" not in cache_manager.cache


class TestSetMethodStoresValueswithTimestamp:
    """
    Rule: Set Method Stores Values with Timestamp
    """

    def test_set_stores_value_in_cache(self, cache_manager):
        """
        Scenario: Set stores value in cache
          Given a key "new_query" and value {"data": "test"}
          When I call set with the key and value
          Then the value is stored in cache
        """
        # Arrange
        key = "new_query"
        value = {"data": "test"}
        
        # Act
        cache_manager.set(key, value)
        
        # Assert
        assert key in cache_manager.cache

    def test_set_stores_value_in_cache_1(self, cache_manager):
        """
        Scenario: Set stores value in cache
          Given a key "new_query" and value {"data": "test"}
          When I call set with the key and value
          Then the current timestamp is recorded
        """
        # Arrange
        key = "new_query"
        value = {"data": "test"}
        
        # Act
        before_time = time.time()
        cache_manager.set(key, value)
        after_time = time.time()
        
        # Assert
        assert key in cache_manager.cache_timestamps
        assert before_time <= cache_manager.cache_timestamps[key] <= after_time

    def test_set_overwrites_existing_key(self, cache_manager):
        """
        Scenario: Set overwrites existing key
          Given a cache entry with key "query1" and old value
          And a new value for key "query1"
          When I call set with key and new value
          Then the old value is replaced
        """
        # Arrange
        key = "query1"
        old_value = {"data": "old"}
        new_value = {"data": "new"}
        cache_manager.set(key, old_value)
        
        # Act
        cache_manager.set(key, new_value)
        
        # Assert
        assert cache_manager.cache[key] == new_value

    def test_set_overwrites_existing_key_1(self, cache_manager):
        """
        Scenario: Set overwrites existing key
          Given a cache entry with key "query1" and old value
          And a new value for key "query1"
          When I call set with key and new value
          Then the timestamp is updated to current time
        """
        # Arrange
        key = "query1"
        old_value = {"data": "old"}
        new_value = {"data": "new"}
        cache_manager.set(key, old_value)
        old_timestamp = cache_manager.cache_timestamps[key]
        time.sleep(0.01)  # Ensure time difference
        
        # Act
        cache_manager.set(key, new_value)
        
        # Assert
        assert cache_manager.cache_timestamps[key] > old_timestamp


class TestCacheTTLConfigurationAffectsExpiration:
    """
    Rule: Cache TTL Configuration Affects Expiration
    """

    def test_short_ttl_causes_faster_expiration(self):
        """
        Scenario: Short TTL causes faster expiration
          Given cache_ttl_seconds is configured as 60
          And a cache entry was created 61 seconds ago
          When I call get for that entry
          Then None is returned
        """
        # Arrange
        cache_manager = CacheManager(cache_ttl_seconds=60)
        cache_manager.set("test_key", {"data": "test"})
        cache_manager.cache_timestamps["test_key"] = time.time() - 61
        
        # Act
        result = cache_manager.get("test_key")
        
        # Assert
        assert result is None

    def test_short_ttl_causes_faster_expiration_1(self):
        """
        Scenario: Short TTL causes faster expiration
          Given cache_ttl_seconds is configured as 60
          And a cache entry was created 61 seconds ago
          When I call get for that entry
          Then the entry is considered expired
        """
        # Arrange
        cache_manager = CacheManager(cache_ttl_seconds=60)
        cache_manager.set("test_key", {"data": "test"})
        cache_manager.cache_timestamps["test_key"] = time.time() - 61
        
        # Act
        result = cache_manager.get("test_key")
        
        # Assert
        assert result is None  # Expired entries return None

    def test_long_ttl_keeps_entries_longer(self):
        """
        Scenario: Long TTL keeps entries longer
          Given cache_ttl_seconds is configured as 86400 (24 hours)
          And a cache entry was created 3600 seconds ago (1 hour)
          When I call get for that entry
          Then the cached value is returned
        """
        # Arrange
        cache_manager = CacheManager(cache_ttl_seconds=86400)
        test_value = {"data": "test"}
        cache_manager.set("test_key", test_value)
        cache_manager.cache_timestamps["test_key"] = time.time() - 3600
        
        # Act
        result = cache_manager.get("test_key")
        
        # Assert
        assert result is not None

    def test_long_ttl_keeps_entries_longer_1(self):
        """
        Scenario: Long TTL keeps entries longer
          Given cache_ttl_seconds is configured as 86400 (24 hours)
          And a cache entry was created 3600 seconds ago (1 hour)
          When I call get for that entry
          Then the entry is not expired
        """
        # Arrange
        cache_manager = CacheManager(cache_ttl_seconds=86400)
        test_value = {"data": "test"}
        cache_manager.set("test_key", test_value)
        cache_manager.cache_timestamps["test_key"] = time.time() - 3600
        
        # Act
        result = cache_manager.get("test_key")
        
        # Assert
        assert result == test_value

    def test_ttl_is_set_during_initialization(self):
        """
        Scenario: TTL is set during initialization
          Given CacheManager is initialized with cache_ttl_seconds=7200
          When get_stats is called
          Then ttl_seconds in stats is 7200
        """
        # Arrange
        cache_manager = CacheManager(cache_ttl_seconds=7200)
        
        # Act
        stats = cache_manager.get_stats()
        
        # Assert
        assert stats["ttl_seconds"] == 7200


class TestClearMethodRemovesAllCacheEntries:
    """
    Rule: Clear Method Removes All Cache Entries
    """

    def test_clear_removes_all_cached_values(self, cache_manager):
        """
        Scenario: Clear removes all cached values
          Given 5 cache entries exist
          When I call clear
          Then all cache entries are removed
        """
        # Arrange
        for i in range(5):
            cache_manager.set(f"key{i}", {"data": f"value{i}"})
        
        # Act
        cache_manager.clear()
        
        # Assert
        assert len(cache_manager.cache) == 0

    def test_clear_removes_all_cached_values_1(self, cache_manager):
        """
        Scenario: Clear removes all cached values
          Given 5 cache entries exist
          When I call clear
          Then all timestamps are removed
        """
        # Arrange
        for i in range(5):
            cache_manager.set(f"key{i}", {"data": f"value{i}"})
        
        # Act
        cache_manager.clear()
        
        # Assert
        assert len(cache_manager.cache_timestamps) == 0

    def test_clear_removes_all_cached_values_2(self, cache_manager):
        """
        Scenario: Clear removes all cached values
          Given 5 cache entries exist
          When I call clear
          Then the cache is empty
        """
        # Arrange
        for i in range(5):
            cache_manager.set(f"key{i}", {"data": f"value{i}"})
        
        # Act
        cache_manager.clear()
        
        # Assert
        assert cache_manager.get_stats()["size"] == 0

    def test_clear_on_empty_cache_completes_without_error(self, cache_manager):
        """
        Scenario: Clear on empty cache completes without error
          Given the cache is empty
          When I call clear
          Then the operation completes successfully
        """
        # Arrange - cache is already empty
        
        # Act & Assert - should not raise exception
        cache_manager.clear()
        assert True  # If we reach here, no exception was raised

    def test_clear_on_empty_cache_completes_without_error_1(self, cache_manager):
        """
        Scenario: Clear on empty cache completes without error
          Given the cache is empty
          When I call clear
          Then the cache remains empty
        """
        # Arrange - cache is already empty
        
        # Act
        cache_manager.clear()
        
        # Assert
        assert len(cache_manager.cache) == 0


class TestGetStatsReturnsCacheInformation:
    """
    Rule: Get Stats Returns Cache Information
    """

    def test_stats_include_cache_size(self, cache_manager):
        """
        Scenario: Stats include cache size
          Given 10 entries are cached
          When I call get_stats
          Then the result contains key "size" with value 10
        """
        # Arrange
        for i in range(10):
            cache_manager.set(f"key{i}", {"data": f"value{i}"})
        
        # Act
        stats = cache_manager.get_stats()
        
        # Assert
        assert "size" in stats
        assert stats["size"] == 10

    def test_stats_include_cached_keys(self, cache_manager):
        """
        Scenario: Stats include cached keys
          Given cache contains keys ["query1", "query2", "query3"]
          When I call get_stats
          Then the result contains key "keys"
        """
        # Arrange
        for key in ["query1", "query2", "query3"]:
            cache_manager.set(key, {"data": "test"})
        
        # Act
        stats = cache_manager.get_stats()
        
        # Assert
        assert "keys" in stats

    def test_stats_include_cached_keys_1(self, cache_manager):
        """
        Scenario: Stats include cached keys
          Given cache contains keys ["query1", "query2", "query3"]
          When I call get_stats
          Then keys list includes ["query1", "query2", "query3"]
        """
        # Arrange
        expected_keys = ["query1", "query2", "query3"]
        for key in expected_keys:
            cache_manager.set(key, {"data": "test"})
        
        # Act
        stats = cache_manager.get_stats()
        
        # Assert
        assert set(stats["keys"]) == set(expected_keys)

    def test_stats_include_ttl_configuration(self, cache_manager):
        """
        Scenario: Stats include TTL configuration
          Given cache_ttl_seconds is 3600
          When I call get_stats
          Then the result contains key "ttl_seconds" with value 3600
        """
        # Arrange - using fixture with TTL 3600
        
        # Act
        stats = cache_manager.get_stats()
        
        # Assert
        assert stats["ttl_seconds"] == 3600

    def test_stats_reflect_empty_cache(self, cache_manager):
        """
        Scenario: Stats reflect empty cache
          Given the cache is empty
          When I call get_stats
          Then size is 0
        """
        # Arrange - cache is empty by default
        
        # Act
        stats = cache_manager.get_stats()
        
        # Assert
        assert stats["size"] == 0

    def test_stats_reflect_empty_cache_1(self, cache_manager):
        """
        Scenario: Stats reflect empty cache
          Given the cache is empty
          When I call get_stats
          Then keys list is empty
        """
        # Arrange - cache is empty by default
        
        # Act
        stats = cache_manager.get_stats()
        
        # Assert
        assert stats["keys"] == []


class TestExpiredEntriesAreRemovedOnGet:
    """
    Rule: Expired Entries Are Removed on Get
    """

    def test_get_removes_expired_entry_from_cache(self, cache_manager):
        """
        Scenario: Get removes expired entry from cache
          Given a cache entry that expired 100 seconds ago
          When I call get for that entry
          Then None is returned
        """
        # Arrange
        cache_manager.set("expired_key", {"data": "test"})
        cache_manager.cache_timestamps["expired_key"] = time.time() - (cache_manager.cache_ttl_seconds + 100)
        
        # Act
        result = cache_manager.get("expired_key")
        
        # Assert
        assert result is None

    def test_get_removes_expired_entry_from_cache_1(self, cache_manager):
        """
        Scenario: Get removes expired entry from cache
          Given a cache entry that expired 100 seconds ago
          When I call get for that entry
          Then the entry is deleted from self.cache
        """
        # Arrange
        cache_manager.set("expired_key", {"data": "test"})
        cache_manager.cache_timestamps["expired_key"] = time.time() - (cache_manager.cache_ttl_seconds + 100)
        
        # Act
        cache_manager.get("expired_key")
        
        # Assert
        assert "expired_key" not in cache_manager.cache

    def test_get_removes_expired_entry_from_cache_2(self, cache_manager):
        """
        Scenario: Get removes expired entry from cache
          Given a cache entry that expired 100 seconds ago
          When I call get for that entry
          Then the timestamp is deleted from self.cache_timestamps
        """
        # Arrange
        cache_manager.set("expired_key", {"data": "test"})
        cache_manager.cache_timestamps["expired_key"] = time.time() - (cache_manager.cache_ttl_seconds + 100)
        
        # Act
        cache_manager.get("expired_key")
        
        # Assert
        assert "expired_key" not in cache_manager.cache_timestamps

    def test_cache_size_decreases_when_expired_entry_removed(self, cache_manager):
        """
        Scenario: Cache size decreases when expired entry removed
          Given 5 cache entries with 1 expired
          When I call get for the expired entry
          Then the cache size is reduced to 4
        """
        # Arrange
        for i in range(5):
            cache_manager.set(f"key{i}", {"data": f"value{i}"})
        # Expire one entry
        cache_manager.cache_timestamps["key2"] = time.time() - (cache_manager.cache_ttl_seconds + 100)
        
        # Act
        cache_manager.get("key2")
        
        # Assert
        assert len(cache_manager.cache) == 4

    def test_cache_size_decreases_when_expired_entry_removed_1(self, cache_manager):
        """
        Scenario: Cache size decreases when expired entry removed
          Given 5 cache entries with 1 expired
          When I call get for the expired entry
          Then only 4 timestamps remain
        """
        # Arrange
        for i in range(5):
            cache_manager.set(f"key{i}", {"data": f"value{i}"})
        cache_manager.cache_timestamps["key2"] = time.time() - (cache_manager.cache_ttl_seconds + 100)
        
        # Act
        cache_manager.get("key2")
        
        # Assert
        assert len(cache_manager.cache_timestamps) == 4
