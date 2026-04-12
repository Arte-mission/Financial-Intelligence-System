import requests
import logging
from typing import List, Dict, Any, Tuple
from app.utils.config import settings

logger = logging.getLogger(__name__)

# Separate query buckets — each targets a distinct category of Nepal finance news.
# Splitting avoids OR-query dilution where one dominant keyword (NEPSE) crowds out banking/macro results.
QUERY_BUCKETS = [
    {"query": "NEPSE",                  "category_hint": "market",  "limit": 8},
    {"query": "Nepal Rastra Bank",      "category_hint": "banking", "limit": 8},
    {"query": "Nepal banks financial",  "category_hint": "banking", "limit": 6},
    {"query": "Nepal economy",          "category_hint": "macro",   "limit": 6},
    {"query": "financial policy Nepal", "category_hint": "macro",   "limit": 5},
]


def fetch_nepal_business_news() -> Tuple[List[Dict[str, Any]], str]:
    """
    Fetches Nepal financial news across multiple targeted query buckets.
    Deduplicates by title, applies per-bucket limits, and returns a balanced
    cross-category article pool aimed at 15–30 inputs for the sentiment engine.
    Returns (articles, queries_used_summary).
    """
    api_key = settings.NEWSDATA_API_KEY
    if not api_key:
        logger.info("NewsData not configured — running in OnlineKhabar-only mode.")
        return [], ""

    url = "https://newsdata.io/api/1/news"
    seen_titles: set = set()
    all_articles: List[Dict[str, Any]] = []
    queries_used: List[str] = []

    for bucket in QUERY_BUCKETS:
        query = bucket["query"]
        per_bucket_limit = bucket["limit"]

        params = {
            "apikey": api_key,
            "q": query,
            "language": "en",
        }

        try:
            logger.info(f"NewsData bucket fetch: '{query}'")
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            bucket_added = 0

            for item in results:
                if bucket_added >= per_bucket_limit:
                    break

                title = item.get("title", "").strip()
                if not title:
                    continue

                # Deduplicate by normalized title
                title_key = title.lower()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

                all_articles.append({
                    "title": title,
                    "source": item.get("source_id", "Unknown"),
                    "published_at": item.get("pubDate", ""),
                })
                bucket_added += 1

            if results:
                queries_used.append(query)

        except requests.exceptions.RequestException as e:
            logger.error(f"NewsData request failed for bucket '{query}': {e}")
        except ValueError as e:
            logger.error(f"JSON parse error for bucket '{query}': {e}")

    summary = " | ".join(queries_used) if queries_used else ""
    logger.info(f"Total deduplicated articles fetched: {len(all_articles)}")
    return all_articles, summary
