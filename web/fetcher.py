"""
Web Fetcher - URL content retrieval with transparency

Fetches web pages and extracts readable content.
Every fetch is logged with reason and result.
"""

import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional, List
from urllib.parse import urlparse
import re

from .activity_log import ActivityLog


@dataclass
class FetchResult:
    """Result of a web fetch"""
    url: str
    success: bool
    title: Optional[str]
    content: Optional[str]  # Cleaned text content
    links: List[str]  # Extracted links
    error: Optional[str]
    status_code: Optional[int]

    def summary(self, max_length: int = 200) -> str:
        """Get a brief summary of the result"""
        if not self.success:
            return f"Failed: {self.error}"
        if not self.content:
            return "No content extracted"
        text = self.content[:max_length]
        if len(self.content) > max_length:
            text += "..."
        return text


class WebFetcher:
    """
    Fetches and parses web content.

    All fetches are logged with full transparency:
    - What URL was accessed
    - Why it was accessed
    - What was found
    - Whether it was user-triggered or autonomous
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

    def fetch(
        self,
        url: str,
        reason: str,
        triggered_by: str = "user"
    ) -> FetchResult:
        """
        Fetch a URL and extract content.

        Args:
            url: The URL to fetch
            reason: Why this fetch is happening (logged for transparency)
            triggered_by: "user" or "autonomous"

        Returns:
            FetchResult with content or error
        """
        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme:
                url = "https://" + url
            elif parsed.scheme not in ('http', 'https'):
                return self._log_failure(url, reason, triggered_by, "Invalid URL scheme")
        except Exception as e:
            return self._log_failure(url, reason, triggered_by, f"Invalid URL: {e}")

        # Fetch
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # Parse content
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract title
            title = None
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Remove script, style, nav elements
            for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                tag.decompose()

            # Extract main content
            content = self._extract_content(soup)

            # Extract links
            links = self._extract_links(soup, url)

            result = FetchResult(
                url=url,
                success=True,
                title=title,
                content=content,
                links=links,
                error=None,
                status_code=response.status_code
            )

            # Log success
            self.activity_log.log_fetch(
                url=url,
                reason=reason,
                success=True,
                result_summary=f"Title: {title or 'None'}. {len(content)} chars extracted.",
                triggered_by=triggered_by
            )

            return result

        except requests.Timeout:
            return self._log_failure(url, reason, triggered_by, "Request timed out")
        except requests.HTTPError as e:
            return self._log_failure(url, reason, triggered_by, f"HTTP {e.response.status_code}")
        except requests.RequestException as e:
            return self._log_failure(url, reason, triggered_by, str(e))
        except Exception as e:
            return self._log_failure(url, reason, triggered_by, f"Parse error: {e}")

    def _log_failure(
        self,
        url: str,
        reason: str,
        triggered_by: str,
        error: str
    ) -> FetchResult:
        """Log a failed fetch and return error result"""
        self.activity_log.log_fetch(
            url=url,
            reason=reason,
            success=False,
            result_summary=f"Error: {error}",
            triggered_by=triggered_by
        )
        return FetchResult(
            url=url,
            success=False,
            title=None,
            content=None,
            links=[],
            error=error,
            status_code=None
        )

    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract readable text content from parsed HTML"""
        # Try to find main content area
        main_content = None
        for selector in ['main', 'article', '[role="main"]', '.content', '#content']:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            # Fall back to body
            body = soup.find('body')
            text = body.get_text(separator='\n', strip=True) if body else ""

        # Clean up whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract relevant links from the page"""
        links = []
        parsed_base = urlparse(base_url)

        for a in soup.find_all('a', href=True):
            href = a['href']

            # Skip anchors, javascript, mailto
            if href.startswith(('#', 'javascript:', 'mailto:')):
                continue

            # Make absolute
            if href.startswith('/'):
                href = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
            elif not href.startswith(('http://', 'https://')):
                continue

            links.append(href)

        # Deduplicate while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)

        return unique_links[:20]  # Limit to top 20

    def fetch_multiple(
        self,
        urls: List[str],
        reason: str,
        triggered_by: str = "user"
    ) -> List[FetchResult]:
        """Fetch multiple URLs"""
        return [self.fetch(url, reason, triggered_by) for url in urls]
