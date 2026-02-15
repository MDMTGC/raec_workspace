"""
RAEC Web Module - Transparent internet access with full logging

Every web action is logged and disclosed. RAEC sees the world,
but you always know what it's looking at and why.
"""

from .fetcher import WebFetcher, FetchResult
from .search import WebSearch, SearchResult
from .activity_log import ActivityLog, WebActivity

__all__ = [
    'WebFetcher',
    'FetchResult',
    'WebSearch',
    'SearchResult',
    'ActivityLog',
    'WebActivity',
]
