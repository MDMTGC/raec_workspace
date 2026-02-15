"""
Tests for web module
"""

import pytest
from pathlib import Path
import tempfile
import os

from .activity_log import ActivityLog, ActivityType, WebActivity
from .fetcher import WebFetcher, FetchResult
from .search import WebSearch, SearchResults


class TestActivityLog:
    """Test activity logging"""

    def test_log_fetch(self, tmp_path):
        """Test logging a fetch"""
        log = ActivityLog(tmp_path / "test_activity.db")

        activity = log.log_fetch(
            url="https://example.com",
            reason="Testing fetch logging",
            success=True,
            result_summary="Found example content",
            triggered_by="user"
        )

        assert activity.id is not None
        assert activity.url == "https://example.com"
        assert activity.reason == "Testing fetch logging"
        assert activity.success is True
        assert activity.triggered_by == "user"

    def test_log_search(self, tmp_path):
        """Test logging a search"""
        log = ActivityLog(tmp_path / "test_activity.db")

        activity = log.log_search(
            query="test query",
            reason="Testing search logging",
            success=True,
            result_summary="Found 5 results",
            triggered_by="autonomous"
        )

        assert activity.id is not None
        assert activity.query == "test query"
        assert activity.triggered_by == "autonomous"

    def test_get_recent(self, tmp_path):
        """Test retrieving recent activity"""
        log = ActivityLog(tmp_path / "test_activity.db")

        # Add several activities
        for i in range(5):
            log.log_fetch(
                url=f"https://example{i}.com",
                reason=f"Test {i}",
                success=True,
                result_summary=f"Result {i}",
                triggered_by="user"
            )

        recent = log.get_recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert "example4" in recent[0].url

    def test_get_autonomous(self, tmp_path):
        """Test filtering autonomous activity"""
        log = ActivityLog(tmp_path / "test_activity.db")

        log.log_fetch("https://user.com", "User action", True, "ok", "user")
        log.log_fetch("https://auto.com", "Auto action", True, "ok", "autonomous")

        autonomous = log.get_autonomous()
        assert len(autonomous) == 1
        assert autonomous[0].url == "https://auto.com"

    def test_get_stats(self, tmp_path):
        """Test activity statistics"""
        log = ActivityLog(tmp_path / "test_activity.db")

        log.log_fetch("https://a.com", "r", True, "ok", "user")
        log.log_fetch("https://b.com", "r", False, "err", "autonomous")
        log.log_search("query", "r", True, "ok", "user")

        stats = log.get_stats()
        assert stats["total_activities"] == 3
        assert stats["fetches"] == 2
        assert stats["searches"] == 1
        assert stats["autonomous_actions"] == 1


class TestWebFetcher:
    """Test web fetcher"""

    def test_fetch_result_summary(self):
        """Test FetchResult summary"""
        result = FetchResult(
            url="https://example.com",
            success=True,
            title="Example",
            content="This is the content of the page",
            links=[],
            error=None,
            status_code=200
        )

        summary = result.summary(max_length=20)
        assert "This is" in summary
        assert "..." in summary

    def test_fetch_result_error_summary(self):
        """Test FetchResult error summary"""
        result = FetchResult(
            url="https://example.com",
            success=False,
            title=None,
            content=None,
            links=[],
            error="Connection refused",
            status_code=None
        )

        summary = result.summary()
        assert "Failed" in summary
        assert "Connection refused" in summary


class TestSearchResults:
    """Test search results"""

    def test_search_results_format(self):
        """Test formatting search results"""
        from .search import SearchResult

        results = SearchResults(
            query="test query",
            results=[
                SearchResult(
                    title="Result 1",
                    url="https://example1.com",
                    snippet="First result snippet"
                ),
                SearchResult(
                    title="Result 2",
                    url="https://example2.com",
                    snippet="Second result snippet"
                )
            ],
            success=True,
            error=None
        )

        formatted = results.format()
        assert "test query" in formatted
        assert "Result 1" in formatted
        assert "example1.com" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
