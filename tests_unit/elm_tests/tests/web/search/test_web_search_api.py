# -*- coding: utf-8 -*-
"""ELM Web searches using search engines tests"""
import os
from pathlib import Path

import pytest

import elm.web.search.duckduckgo
import elm.web.search.google
from elm.web.search.base import APISearchEngineLinkSearch


SE_API_TO_TEST = [
    (
        elm.web.search.duckduckgo.APIDuckDuckGoSearch,
        {"verify": False}
    )
]

if os.getenv(elm.web.search.google.APIGoogleCSESearch.API_KEY_VAR):
    SE_API_TO_TEST.append((elm.web.search.google.APIGoogleCSESearch, {}))


def test_api_key_read_from_env(monkeypatch):
    """Test that API search engine reads environ"""
    test_key = "TEST-KEY"
    monkeypatch.setenv("TEST_API_KEY_VAR", test_key)

    class MockAPISearchEngine(APISearchEngineLinkSearch):
        """MockAPISearchEngine"""

        API_KEY_VAR = "TEST_API_KEY_VAR"

        async def _search(self, *__, **___):
            return []

    mock_api_key = MockAPISearchEngine().api_key

    assert mock_api_key == test_key, f"expected MockAPISearchEngine API key to be {test_key}, got {mock_api_key}"


def test_no_api_key_var():
    """Test that API search engine does not break if var name is None"""

    class MockAPISearchEngine(APISearchEngineLinkSearch):
        """MockAPISearchEngine"""

        async def _search(self, *__, **___):
            return []
        
    mock_api_key = MockAPISearchEngine().api_key

    assert mock_api_key is None, f"expected MockAPISearchEngine API key to be None, got {mock_api_key}"


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == "true",
                    reason="Fails in GHA due to rate limiting")
@pytest.mark.parametrize("queries", 
    [
        ['1. "NREL elm"'],
        ['1. "NREL elm"', "NREL reV"],
    ]
)
@pytest.mark.parametrize("se", SE_API_TO_TEST)
class TestBasicSearchQuery:

    @pytest.mark.asyncio
    async def test_len_output_equals_len_queries(self, queries, se):
        """
        GIVEN a set of search queries and search engine
        WHEN the results of the search engine are retrieved for each query
        THEN the number of results returned should equal the number of queries
        """
        num_results = 7
        se_class, kwargs = se
        search_engine = se_class(**kwargs)
        out = await search_engine.results(*queries, num_results=num_results)
        len_out, len_queries = len(out), len(queries)

        assert len_out == len_queries, f"expected {len_queries} results from these queries, got {len_out}"


    @pytest.mark.asyncio
    async def test_query_returns_expected_num_results(self, queries, se):
        """
        GIVEN a set of search queries and search engine
        WHEN the results of the search engine are retrieved for each query
        THEN each query should return between 1 and the specified number of results
        """
        num_results = 7
        se_class, kwargs = se
        search_engine = se_class(**kwargs)
        out = await search_engine.results(*queries, num_results=num_results)

        for idx, results in enumerate(out, start=1):
            assert 0 < len(results) <= num_results, \
                f"expected between 1 and {num_results} results for query {idx}, got {len(results)}\nqueries: {queries}"


    @pytest.mark.asyncio
    async def test_all_urls_start_with_http(self, queries, se):
        """
        GIVEN a set of search queries and search engine
        WHEN the results of the search engine are retrieved for each query
        THEN all URLs in the results should start with 'http'
        """
        num_results = 7
        se_class, kwargs = se
        search_engine = se_class(**kwargs)
        out = await search_engine.results(*queries, num_results=num_results)

        for idx, results in enumerate(out, start=1):
            assert all(link.startswith("http") for link in results), \
                f"expected all links to start with 'http' for query {idx}, got {results}"


    @pytest.mark.asyncio
    async def test_urls_do_not_have_plus_signs(self, queries, se):
        """
        GIVEN a set of search queries and search engine
        WHEN the results of the search engine are retrieved for each query
        THEN all URLs in the results should not contain the '+' character
        """
        num_results = 7
        se_class, kwargs = se
        search_engine = se_class(**kwargs)
        out = await search_engine.results(*queries, num_results=num_results)

        for idx, results in enumerate(out, start=1):
            assert all("+" not in link for link in results), \
                f"expected no '+' in links for query {idx}, got {results}"



if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
