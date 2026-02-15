"""
Web Search - Query-based web search with transparency

Uses DuckDuckGo for privacy-respecting search.
Every search is logged with reason and results.
"""

import requests
from dataclasses import dataclass
from typing import List, Optional
import re
import urllib.parse

from .activity_log import ActivityLog


@dataclass
class SearchResult:
    """A single search result"""
    title: str
    url: str
    snippet: str

    def __str__(self) -> str:
        return f"{self.title}\n{self.url}\n{self.snippet}"


@dataclass
class SearchResults:
    """Collection of search results"""
    query: str
    results: List[SearchResult]
    success: bool
    error: Optional[str]

    def summary(self) -> str:
        """Brief summary of results"""
        if not self.success:
            return f"Search failed: {self.error}"
        return f"Found {len(self.results)} results for '{self.query}'"

    def format(self, max_results: int = 5) -> str:
        """Format results for display"""
        if not self.success:
            return f"Search failed: {self.error}"
        if not self.results:
            return f"No results found for '{self.query}'"

        lines = [f"Search: {self.query}", "=" * 40]
        for i, r in enumerate(self.results[:max_results], 1):
            lines.append(f"\n{i}. {r.title}")
            lines.append(f"   {r.url}")
            lines.append(f"   {r.snippet[:150]}...")

        return "\n".join(lines)


class WebSearch:
    """
    Web search with full transparency.

    Uses DuckDuckGo HTML search (no API key needed).
    All searches are logged with reason and results.
    """

    def __init__(self, activity_log: Optional[ActivityLog] = None):
        self.activity_log = activity_log or ActivityLog()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAEC/1.0 (Autonomous AI Assistant; Transparent Web Access)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        })
        self.timeout = 10

    def search(
        self,
        query: str,
        reason: str,
        triggered_by: str = "user",
        max_results: int = 10
    ) -> SearchResults:
        """
        Search the web.

        Args:
            query: Search query
            reason: Why this search is happening (logged for transparency)
            triggered_by: "user" or "autonomous"
            max_results: Maximum results to return

        Returns:
            SearchResults with findings or error
        """
        try:
            # Use DuckDuckGo HTML search
            results = self._duckduckgo_search(query, max_results)

            search_results = SearchResults(
                query=query,
                results=results,
                success=True,
                error=None
            )

            # Log the search
            self.activity_log.log_search(
                query=query,
                reason=reason,
                success=True,
                result_summary=f"Found {len(results)} results",
                triggered_by=triggered_by
            )

            return search_results

        except Exception as e:
            self.activity_log.log_search(
                query=query,
                reason=reason,
                success=False,
                result_summary=f"Error: {str(e)}",
                triggered_by=triggered_by
            )

            return SearchResults(
                query=query,
                results=[],
                success=False,
                error=str(e)
            )

    def _duckduckgo_search(self, query: str, max_results: int) -> List[SearchResult]:
        """
        Search using DuckDuckGo HTML interface.

        This is a simple scraper approach. For production, consider:
        - DuckDuckGo Instant Answer API
        - SearXNG self-hosted instance
        - Brave Search API
        """
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()

        # Parse results
        results = []
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # DuckDuckGo HTML results have class 'result'
        for result_div in soup.select('.result')[:max_results]:
            try:
                # Title and URL
                title_elem = result_div.select_one('.result__title a')
                if not title_elem:
                    continue

                title = title_elem.get_text(strip=True)

                # DuckDuckGo wraps URLs in a redirect
                href = title_elem.get('href', '')
                # Extract actual URL from uddg parameter
                if 'uddg=' in href:
                    url_match = re.search(r'uddg=([^&]+)', href)
                    if url_match:
                        actual_url = urllib.parse.unquote(url_match.group(1))
                    else:
                        actual_url = href
                else:
                    actual_url = href

                # Snippet
                snippet_elem = result_div.select_one('.result__snippet')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

                results.append(SearchResult(
                    title=title,
                    url=actual_url,
                    snippet=snippet
                ))

            except Exception:
                continue  # Skip malformed results

        return results

    def search_and_summarize(
        self,
        query: str,
        reason: str,
        triggered_by: str = "user"
    ) -> str:
        """Search and return a formatted summary"""
        results = self.search(query, reason, triggered_by)
        return results.format()
