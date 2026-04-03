import requests
import xml.etree.ElementTree as ET
import json
import time
import os
from typing import Dict, List

# ─── Disk-persisted cache (24-hour TTL) ───────────────────
_ARXIV_CACHE_PATH = "data/arxiv_cache.json"
_ARXIV_CACHE_TTL  = 86400  # 24 hours in seconds

# In-memory mirror so we only hit disk once per process
_arxiv_cache: Dict[str, dict] = {}   # key → {"count": int, "ts": float}


def _load_disk_cache():
    """Load the persisted arXiv cache from disk into the in-memory mirror."""
    global _arxiv_cache
    if os.path.exists(_ARXIV_CACHE_PATH):
        try:
            with open(_ARXIV_CACHE_PATH, "r") as f:
                _arxiv_cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            _arxiv_cache = {}


def _save_disk_cache():
    """Flush the in-memory cache to disk."""
    try:
        os.makedirs(os.path.dirname(_ARXIV_CACHE_PATH), exist_ok=True)
        with open(_ARXIV_CACHE_PATH, "w") as f:
            json.dump(_arxiv_cache, f)
    except OSError as e:
        print(f"⚠️  [arXiv cache] Could not write cache to disk: {e}")


# Load on module import so the cache is ready on the very first call
_load_disk_cache()
# ──────────────────────────────────────────────────────────


class ArxivService:
    BASE_URL = "https://export.arxiv.org/api/query?"

    def __init__(self, delay_seconds: float = 3.0):
        self.delay_seconds    = delay_seconds
        self.last_query_time  = 0.0

    def _wait_for_rate_limit(self):
        elapsed = time.time() - self.last_query_time
        if elapsed < self.delay_seconds:
            time.sleep(self.delay_seconds - elapsed)
        self.last_query_time = time.time()

    def get_total_results(self, search_query: str) -> int:
        self._wait_for_rate_limit()
        params = {"search_query": search_query, "max_results": 1}
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            namespaces = {
                'atom':       'http://www.w3.org/2005/Atom',
                'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
            }
            total_elem = root.find('opensearch:totalResults', namespaces)
            return int(total_elem.text) if total_elem is not None else 0
        except Exception as e:
            print(f"Error fetching from arXiv: {e}")
            return 0

    def get_mention_count_for_year(self, keyword: str, year: int) -> int:
        cache_key = f"{keyword}|{year}"

        # ── Check cache (in-memory mirror, populated from disk on startup) ──
        if cache_key in _arxiv_cache:
            entry = _arxiv_cache[cache_key]
            if time.time() - entry["ts"] < _ARXIV_CACHE_TTL:
                print(f"[arXiv Cache HIT] {keyword} ({year})")
                return entry["count"]

        # ── Cache miss: call API ────────────────────────────────────────────
        start_date = f"{year}01010000"
        end_date   = f"{year}12312359"
        query      = f'all:"{keyword}" AND submittedDate:[{start_date} TO {end_date}]'
        count      = self.get_total_results(query)

        # ── Write to in-memory mirror and flush to disk ─────────────────────
        _arxiv_cache[cache_key] = {"count": count, "ts": time.time()}
        _save_disk_cache()

        return count

    def fetch_research_signals(self, keywords: List[str], years: List[int]) -> Dict[str, Dict[int, int]]:
        results = {}
        for kw in keywords:
            results[kw] = {}
            for year in years:
                results[kw][year] = self.get_mention_count_for_year(kw, year)
        return results