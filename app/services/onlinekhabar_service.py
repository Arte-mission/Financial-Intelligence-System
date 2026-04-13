"""
OnlineKhabar scraper for Datus AI — Nepal-native financial news source.

Targets:
  - https://english.onlinekhabar.com/category/economy  (macro / economy)
  - https://english.onlinekhabar.com/?s=nepal+rastra+bank  (NRB / banking)
  - https://english.onlinekhabar.com/?s=nepse  (market)
  - https://english.onlinekhabar.com/?s=nepal+economy  (macro fallback)
  - https://english.onlinekhabar.com/?s=banking+nepal  (banking fallback)

HTML structure (confirmed via live inspection):
  - Article content blocks: div.ok-post-contents
  - Title + URL: first <a> tag inside the content block
  - Date: text in span/div children containing "ago" or a date pattern

Returns normalized article dicts: {title, url, source, published_at}
"""

import requests
import logging
import re
import time
from typing import List, Dict, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Ordered by priority. Each entry: (url, per_page_limit)
SCRAPE_TARGETS = [
    ("https://english.onlinekhabar.com/category/economy",   12),
    ("https://english.onlinekhabar.com/?s=nepal+rastra+bank", 8),
    ("https://english.onlinekhabar.com/?s=nepse",            8),
    ("https://english.onlinekhabar.com/?s=nepal+economy",    6),
    ("https://english.onlinekhabar.com/?s=banking+nepal",    6),
]

# ── Per-URL HTTP response cache ──────────────────────────────────────────────
# Each entry: { url: {"html": str, "ts": float} }
# TTL is deliberately shorter than the market-mood TTL (600s) so fresh content
# is always available on the first genuine refresh after new articles publish.
_PAGE_CACHE: Dict[str, Dict] = {}
PAGE_CACHE_TTL = 600   # 10 minutes per URL — aligned with MACRO_CACHE_TTL


def _parse_articles_from_page(html: str, source_url: str, limit: int) -> List[Dict[str, Any]]:
    """
    Parse articles from an OnlineKhabar page HTML.
    Returns up to `limit` normalized article dicts.
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # Primary selector: div.ok-post-contents contains title + link
    content_blocks = soup.find_all("div", class_="ok-post-contents")

    if not content_blocks:
        # Fallback: scan all article tags or h2/h3 with anchors
        content_blocks = soup.find_all("article")

    for block in content_blocks:
        if len(results) >= limit:
            break

        # Find the first anchor with non-empty text — that is the title link
        anchor = block.find("a", href=True)
        if not anchor:
            continue

        title = anchor.get_text(strip=True)
        url = anchor.get("href", "").strip()

        if not title or not url:
            continue

        # Skip non-article-looking hrefs
        if not url.startswith("http"):
            url = "https://english.onlinekhabar.com" + url

        # Extract date — look for any text that looks like a relative or absolute date
        published_at = ""
        date_candidates = block.find_all(["span", "div", "time"])
        for elem in date_candidates:
            text = elem.get_text(strip=True)
            # Relative: "5 hours ago", "2 days ago"
            if re.search(r"\d+\s+(second|minute|hour|day|week|month)s?\s+ago", text, re.I):
                published_at = text
                break
            # Absolute: "April 11, 2026"
            if re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d+,?\s+\d{4}", text, re.I):
                published_at = text
                break

        results.append({
            "title": title,
            "url": url,
            "source": "OnlineKhabar",
            "published_at": published_at,
        })

    return results


def fetch_onlinekhabar_news() -> List[Dict[str, Any]]:
    """
    Scrapes multiple OnlineKhabar pages for Nepal financial news.
    Each page response is cached for PAGE_CACHE_TTL seconds (8 min).
    Deduplicates results by normalized title.
    Returns a list of article dicts: {title, url, source, published_at}.
    """
    seen_titles: set = set()
    all_articles: List[Dict[str, Any]] = []
    now = time.time()

    for page_url, limit in SCRAPE_TARGETS:
        try:
            # ── Check per-URL page cache first ───────────────────────────
            page_cached = _PAGE_CACHE.get(page_url)
            if page_cached and (now - page_cached["ts"] < PAGE_CACHE_TTL):
                html = page_cached["html"]
                age  = int(now - page_cached["ts"])
                logger.info(f"Page cache hit for {page_url} (age {age}s)")
            else:
                # ── Live fetch ──────────────────────────────────────────
                logger.info(f"OnlineKhabar live fetch: {page_url}")
                response = requests.get(page_url, headers=HEADERS, timeout=12)
                response.raise_for_status()
                html = response.text
                _PAGE_CACHE[page_url] = {"html": html, "ts": now}

            articles = _parse_articles_from_page(html, page_url, limit)
            added = 0

            for article in articles:
                title_key = re.sub(r"[^\w\s]", "", article["title"].lower()).strip()
                if not title_key or title_key in seen_titles:
                    continue
                seen_titles.add(title_key)
                all_articles.append(article)
                added += 1

            logger.info(f"  → {added} unique articles from {page_url}")

        except requests.exceptions.RequestException as e:
            logger.warning(f"OnlineKhabar fetch failed for {page_url}: {e}")
        except Exception as e:
            logger.warning(f"OnlineKhabar parse error for {page_url}: {e}")

    logger.info(f"OnlineKhabar total: {len(all_articles)} deduplicated articles")
    return all_articles
