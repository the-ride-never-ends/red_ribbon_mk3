Feature: Cache Manager
  As a caching system
  I want to store and retrieve values with time-based expiration
  So that frequently accessed data can be served quickly

  Background:
    Given a CacheManager instance is initialized with default TTL

  Rule: Get Method Retrieves Cached Values

    Scenario: Get returns cached value when key exists and not expired
      Given a cache entry with key "query1" and value {"results": []}
      And the entry was created 60 seconds ago
      And cache_ttl_seconds is 3600
      When I call get with key "query1"
      Then the cached value is returned
      And the value matches {"results": []}

    Scenario: Get returns None for non-existent key
      Given no cache entry exists for key "missing_key"
      When I call get with key "missing_key"
      Then None is returned

    Scenario: Get returns None for expired entry
      Given a cache entry with key "old_query"
      And the entry was created 7200 seconds ago
      And cache_ttl_seconds is 3600
      When I call get with key "old_query"
      Then None is returned
      And the expired entry is removed from cache

  Rule: Set Method Stores Values with Timestamp

    Scenario: Set stores value in cache
      Given a key "new_query" and value {"data": "test"}
      When I call set with the key and value
      Then the value is stored in cache
      And the current timestamp is recorded

    Scenario: Set overwrites existing key
      Given a cache entry with key "query1" and old value
      And a new value for key "query1"
      When I call set with key and new value
      Then the old value is replaced
      And the timestamp is updated to current time

  Rule: Cache TTL Configuration Affects Expiration

    Scenario: Short TTL causes faster expiration
      Given cache_ttl_seconds is configured as 60
      And a cache entry was created 61 seconds ago
      When I call get for that entry
      Then None is returned
      And the entry is considered expired

    Scenario: Long TTL keeps entries longer
      Given cache_ttl_seconds is configured as 86400 (24 hours)
      And a cache entry was created 3600 seconds ago (1 hour)
      When I call get for that entry
      Then the cached value is returned
      And the entry is not expired

    Scenario: TTL is set during initialization
      Given CacheManager is initialized with cache_ttl_seconds=7200
      When get_stats is called
      Then ttl_seconds in stats is 7200

  Rule: Clear Method Removes All Cache Entries

    Scenario: Clear removes all cached values
      Given 5 cache entries exist
      When I call clear
      Then all cache entries are removed
      And all timestamps are removed
      And the cache is empty

    Scenario: Clear on empty cache completes without error
      Given the cache is empty
      When I call clear
      Then the operation completes successfully
      And the cache remains empty

  Rule: Get Stats Returns Cache Information

    Scenario: Stats include cache size
      Given 10 entries are cached
      When I call get_stats
      Then the result contains key "size" with value 10

    Scenario: Stats include cached keys
      Given cache contains keys ["query1", "query2", "query3"]
      When I call get_stats
      Then the result contains key "keys"
      And keys list includes ["query1", "query2", "query3"]

    Scenario: Stats include TTL configuration
      Given cache_ttl_seconds is 3600
      When I call get_stats
      Then the result contains key "ttl_seconds" with value 3600

    Scenario: Stats reflect empty cache
      Given the cache is empty
      When I call get_stats
      Then size is 0
      And keys list is empty

  Rule: Expired Entries Are Removed on Get

    Scenario: Get removes expired entry from cache
      Given a cache entry that expired 100 seconds ago
      When I call get for that entry
      Then None is returned
      And the entry is deleted from self.cache
      And the timestamp is deleted from self.cache_timestamps

    Scenario: Cache size decreases when expired entry removed
      Given 5 cache entries with 1 expired
      When I call get for the expired entry
      Then the cache size is reduced to 4
      And only 4 timestamps remain

  Rule: Cache Operations Are Logged

    Scenario: Initialization logs TTL value
      Given cache_ttl_seconds is 1800
      When CacheManager is initialized
      Then a log message indicates "CacheManager initialized with TTL: 1800 seconds"

    Scenario: Cache hit is logged
      Given a valid cache entry for key "test_query"
      When get returns the cached value
      Then a log message indicates "Cache hit for key: test_query"

    Scenario: Cache miss does not log hit
      Given no entry exists for key "missing"
      When get returns None
      Then no cache hit is logged

    Scenario: Set operation is logged
      Given key "new_query" and value to cache
      When set is called
      Then a log message indicates "Caching results for key: new_query"

    Scenario: Clear operation is logged
      When clear is called
      Then a log message indicates "Clearing cache"

  Rule: Key Parameter Must Be String

    Scenario: Get accepts string keys
      Given a string key "valid_key"
      When get is called with the key
      Then the operation completes successfully

    Scenario: Set accepts string keys
      Given a string key "valid_key" and a value
      When set is called
      Then the operation completes successfully

  Rule: Value Parameter Must Be Dictionary

    Scenario: Set stores dictionary values
      Given a dictionary value {"results": [], "count": 0}
      When set is called with the value
      Then the dictionary is stored correctly

  Rule: Cache Uses Current Time for Expiration Checks

    Scenario: Time comparison determines expiration
      Given a cache entry with timestamp T
      And current time is T + TTL + 1
      When get is called
      Then the entry is expired
      And None is returned

    Scenario: Entry within TTL is not expired
      Given a cache entry with timestamp T
      And current time is T + (TTL / 2)
      When get is called
      Then the entry is not expired
      And the cached value is returned
