from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timezone
from app.schemas import MarketMoodResponse, MarketMoodSnapshotResponse, Drift, MacroLayerStatus
from app.services.newsdata_service import fetch_nepal_business_news
from app.services.onlinekhabar_service import fetch_onlinekhabar_news
from app.services.sentiment_service import (
    calculate_sentiment, get_mood_label, get_tiered_mood_label,
    categorize_article, score_materiality,
    extract_drivers, compute_macro_confidence, get_category_coverage_note,
    score_relevance
)
from app.database import get_db
from app.models import NewsArticle, MarketMoodSnapshot
from app.utils.config import settings
import time
import logging

logger = logging.getLogger(__name__)

# NewsData is optional. The app runs normally without it.
# Only attempt the fallback when a key is actually configured.
NEWSDATA_ENABLED: bool = bool(settings.NEWSDATA_API_KEY)

router = APIRouter(prefix="/market-mood", tags=["Market Sentiment"])

# Cache holds last computed response + state used for change detection
_mood_cache = {
    "data":             None,
    "timestamp":        0,
    "fingerprint":      None,   # str hash of last headline set + labels
    "pulse_label":      None,   # last Market Pulse label written to DB
    "climate_label":    None,   # last Macro Climate label written to DB
}

CAME_TTL_SECONDS = 600  # 10 minutes: realistic cadence for Nepal financial news sources


def _make_fingerprint(headlines: list, score: float, pulse: str, climate: str) -> str:
    """Deterministic fingerprint of the current article set + signal labels.
    Two refreshes with identical inputs will produce the same string.
    """
    import hashlib, json
    payload = {
        "headlines": sorted(h.lower().strip() for h in headlines),
        "score":     round(score, 2),
        "pulse":     pulse,
        "climate":   climate,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()

@router.get("", response_model=MarketMoodResponse)
def get_market_sentiment(db: Session = Depends(get_db)):
    """
    Primary source: OnlineKhabar (Nepal-native financial news).
    Fallback: NewsData.io — only used when NEWSDATA_API_KEY is configured
    and OnlineKhabar yields fewer than MIN_ANALYZED_BEFORE_FALLBACK relevant articles.
    App runs fully in OnlineKhabar-only mode when the key is absent.
    Includes a 600-second memory cache (10 min — matches realistic Nepal news cadence).
    """
    global _mood_cache
    current_time = time.time()
    if _mood_cache["data"] and (current_time - _mood_cache["timestamp"] < CAME_TTL_SECONDS):
        cached = _mood_cache["data"]
        return cached.model_copy(update={"cache_hit": True})
        
    # --- Source pipeline: OnlineKhabar primary, NewsData fallback ---
    MIN_ANALYZED_BEFORE_FALLBACK = 4  # if OKH yields fewer than this, add NewsData
    MIN_MACRO_ARTICLES = 3

    seen_titles: set = set()
    raw_articles: list = []          # each dict has keys: title, source, published_at, url
    source_breakdown: dict = {"OnlineKhabar": 0, "NewsData": 0}

    def _dedup_add(articles_in, source_tag):
        """Add articles that pass the relevance gate and aren't duplicates."""
        added = 0
        for art in articles_in:
            title = art.get("title", "").strip()
            if not title:
                continue
            tkey = title.lower()
            if tkey in seen_titles:
                continue
            # Relevance gate: must score >= 2 to be financially material
            if score_relevance(title) < 2:
                continue
            seen_titles.add(tkey)
            art["_source_tag"] = source_tag
            raw_articles.append(art)
            added += 1
        return added

    # 1. Primary: OnlineKhabar
    try:
        ok_articles = fetch_onlinekhabar_news()
        ok_added = _dedup_add(ok_articles, "OnlineKhabar")
        source_breakdown["OnlineKhabar"] = ok_added
        logger.info(f"OnlineKhabar contributed {ok_added} relevant articles")
    except Exception as e:
        logger.warning(f"OnlineKhabar pipeline failed, proceeding to fallback: {e}")
        ok_added = 0

    # 2. Optional fallback: NewsData (only when key is configured)
    if ok_added < MIN_ANALYZED_BEFORE_FALLBACK and NEWSDATA_ENABLED:
        try:
            nd_articles, _ = fetch_nepal_business_news()
            nd_added = _dedup_add(nd_articles, "NewsData")
            source_breakdown["NewsData"] = nd_added
            logger.info(f"NewsData fallback contributed {nd_added} additional articles")
        except Exception as e:
            logger.warning(f"NewsData fallback failed (continuing with OnlineKhabar only): {e}")
    elif ok_added < MIN_ANALYZED_BEFORE_FALLBACK and not NEWSDATA_ENABLED:
        logger.info(
            f"OnlineKhabar yielded {ok_added} articles (below threshold of {MIN_ANALYZED_BEFORE_FALLBACK}). "
            "NewsData not configured — proceeding with OnlineKhabar-only mode."
        )

    query_used = "OnlineKhabar" + (" + NewsData" if source_breakdown["NewsData"] > 0 else "")

    if not raw_articles:
        return MarketMoodResponse(
            average_score=0.0,
            mood_label="Neutral",
            headline_count=0,
            latest_headlines=[],
            macro_sentiment="Neutral",
            score=0.0,
            drivers=[],
            cache_hit=False,
            raw_score=0.0,
            smoothed_score=0.0,
            previous_score=0.0,
            filtered_article_count=0,
            analyzed_article_count=0,
            query_used=query_used,
            categories_breakdown={},
            analyzed_headlines=[],
            source_breakdown=source_breakdown
        )

    fetched_count = len(raw_articles)  # after relevance gate, before category noise filter

    total_weighted_score = 0.0
    total_weight = 0.0
    # Separate accumulators for two-layer split
    market_score_sum = 0.0
    market_weight_sum = 0.0
    macro_score_sum = 0.0      # banking + macro
    macro_weight_sum = 0.0
    headlines = []
    drivers_list = []
    categories_breakdown: dict = {"market": 0, "banking": 0, "macro": 0}
    high_materiality_count = 0  # articles with materiality == 1.0

    weights = {"market": 1.0, "banking": 0.9, "macro": 0.7}

    for article_data in raw_articles:
        title = article_data.get("title", "")
        if title:
            category = categorize_article(title)
            if category == "noise":
                continue
                
            score = calculate_sentiment(title)
            category_weight = weights.get(category, 0.0)
            materiality = score_materiality(title)

            # Combined weight = category importance × article materiality
            effective_weight = category_weight * materiality

            total_weighted_score += score * effective_weight
            total_weight += effective_weight

            if materiality == 1.0:
                high_materiality_count += 1

            # Accumulate into layer-specific buckets
            if category == "market":
                market_score_sum += score * effective_weight
                market_weight_sum += effective_weight
            else:  # banking or macro
                macro_score_sum += score * effective_weight
                macro_weight_sum += effective_weight

            headlines.append(title)
            categories_breakdown[category] = categories_breakdown.get(category, 0) + 1
            
            # Save fetched headlines to SQLite
            exists = db.query(NewsArticle).filter(NewsArticle.title == title).first()
            if not exists:
                new_article = NewsArticle(
                    title=title,
                    source=article_data.get("source", ""),
                    published_at=article_data.get("published_at", ""),
                    sentiment_score=score,
                    query_used=query_used
                )
                db.add(new_article)
            
            # Extract investor-grade semantic drivers (pass materiality for fallback gate)
            article_drivers = extract_drivers(title, score, category, materiality)
            drivers_list.extend(article_drivers)


    db.commit()
            
    count = len(headlines)
    new_score_raw = total_weighted_score / total_weight if total_weight > 0 else 0.0
    
    previous_score = 0.0
    has_previous = False
    if _mood_cache["data"] is not None:
        previous_score = _mood_cache["data"].score
        has_previous = True
        
    if has_previous:
        smoothed_score = (new_score_raw * 0.6) + (previous_score * 0.4)
    else:
        smoothed_score = new_score_raw
        
    change = smoothed_score - previous_score if has_previous else 0.0
    abs_change = abs(change)
    
    if abs_change < 0.05:
        drift_status = "Stable"
    elif abs_change <= 0.15:
        drift_status = "Moderate"
    else:
        drift_status = "Significant"
        
    if change > 0:
        dir_text = "improved"
    elif change < 0:
        dir_text = "weakened"
    else:
        dir_text = "held steady"
        
    unique_drivers = list(dict.fromkeys(drivers_list))[:4]
    if not unique_drivers:
        unique_drivers = ["mixed domestic signals"] if count > 0 else []

    # --- Two-layer signal computation ---
    market_pulse_score = market_score_sum / market_weight_sum if market_weight_sum > 0 else 0.0
    market_pulse_label = get_mood_label(market_pulse_score)

    macro_article_count = categories_breakdown.get("banking", 0) + categories_breakdown.get("macro", 0)
    market_article_count = categories_breakdown.get("market", 0)
    macro_coverage_sufficient = macro_article_count >= MIN_MACRO_ARTICLES

    if macro_coverage_sufficient:
        raw_macro_score = macro_score_sum / macro_weight_sum if macro_weight_sum > 0 else 0.0
        macro_climate_label = get_tiered_mood_label(raw_macro_score)
    else:
        macro_climate_label = "Insufficient Coverage"

    macro_layer_status = MacroLayerStatus(
        market_pulse=market_pulse_label,
        macro_climate=macro_climate_label,
        market_article_count=market_article_count,
        macro_article_count=macro_article_count,
        coverage_sufficient=macro_coverage_sufficient
    )

    # Macro confidence + category coverage
    macro_confidence = compute_macro_confidence(count, categories_breakdown, high_materiality_count)
    coverage_note = get_category_coverage_note(categories_breakdown)

    # Coverage-aware delta summary
    if not macro_coverage_sufficient:
        # Be explicit that this is a market-only signal
        base_summary = (
            f"Current signal is based on {market_article_count} NEPSE market headline{'s' if market_article_count != 1 else ''}, "
            f"with limited banking or macroeconomic confirmation."
        )
        delta_summary = base_summary
    elif not has_previous or abs_change < 0.03:
        if unique_drivers:
            base_summary = f"Macro sentiment is broadly unchanged, with {unique_drivers[0]} establishing a stable baseline."
        else:
            base_summary = "Market sentiment remains balanced as sector gains continue to offset broader index pressure."
        delta_summary = f"{base_summary} {coverage_note}".strip() if coverage_note else base_summary
    else:
        if change > 0:
            base_summary = f"Market sentiment improved as constructive drivers like {unique_drivers[0] if unique_drivers else 'positive sector rotation'} supported the broader backdrop."
        else:
            base_summary = f"Market mood remains cautious as {unique_drivers[0] if unique_drivers else 'banking and policy concerns'} continue to pressure investor sentiment."
        delta_summary = f"{base_summary} {coverage_note}".strip() if coverage_note else base_summary
    
    average_score_rounded = round(smoothed_score, 2)
    change_rounded = round(change, 2)
    mood_label   = get_mood_label(smoothed_score)        # 3-tier: legacy
    tiered_label = get_tiered_mood_label(smoothed_score) # 5-tier: investor-facing display

    # -----------------------------------------------------------------
    # Fingerprint-gated snapshot write
    # Only write a new DB row when something materially changed:
    #   - score moved by >= 0.03, OR
    #   - Market Pulse or Macro Climate label changed
    # This prevents timeline from filling with identical Neutral rows.
    # -----------------------------------------------------------------
    new_fingerprint = _make_fingerprint(
        headlines, average_score_rounded,
        macro_layer_status.market_pulse,
        macro_layer_status.macro_climate
    )
    previous_fingerprint = _mood_cache.get("fingerprint")
    previous_pulse       = _mood_cache.get("pulse_label")
    previous_climate     = _mood_cache.get("climate_label")

    pulse_changed   = (macro_layer_status.market_pulse != previous_pulse)
    climate_changed = (macro_layer_status.macro_climate != previous_climate)
    score_moved     = abs_change >= 0.03
    articles_changed = (new_fingerprint != previous_fingerprint)

    should_write_snapshot = (
        previous_fingerprint is None          # always write first snapshot
        or pulse_changed
        or climate_changed
        or score_moved
    )

    if should_write_snapshot:
        snapshot = MarketMoodSnapshot(
            average_score=average_score_rounded,
            mood_label=tiered_label,
            headline_count=count
        )
        db.add(snapshot)
        db.commit()
        logger.info(
            f"Snapshot written: score={average_score_rounded}, pulse={macro_layer_status.market_pulse}, "
            f"climate={macro_layer_status.macro_climate}, reason=score_moved:{score_moved} "
            f"pulse_changed:{pulse_changed} climate_changed:{climate_changed}"
        )
    else:
        logger.info(
            f"Snapshot skipped (no material change): fingerprint={new_fingerprint[:8]}…, score={average_score_rounded}"
        )

    # Source-change status for UI transparency
    if previous_fingerprint is None or articles_changed:
        source_change_status = "New relevant coverage detected"
    else:
        source_change_status = "No new relevant articles since last refresh"
    
    response_data = MarketMoodResponse(
        average_score=average_score_rounded,
        mood_label=tiered_label,
        headline_count=count,
        latest_headlines=headlines[:3],
        macro_sentiment=tiered_label,
        score=average_score_rounded,
        drivers=unique_drivers,
        last_updated=datetime.now(timezone.utc).isoformat(),
        refresh_interval_seconds=600,
        drift=Drift(status=drift_status, change=change_rounded),
        delta_summary=delta_summary,
        macro_confidence=macro_confidence,
        macro_layer_status=macro_layer_status,
        # Diagnostics
        cache_hit=False,
        raw_score=round(new_score_raw, 4),
        smoothed_score=round(smoothed_score, 4),
        previous_score=round(previous_score, 4),
        filtered_article_count=fetched_count,
        analyzed_article_count=count,
        query_used=query_used,
        categories_breakdown=categories_breakdown,
        analyzed_headlines=headlines,
        source_breakdown=source_breakdown,
        source_change_status=source_change_status
    )

    _mood_cache["data"]          = response_data
    _mood_cache["timestamp"]     = time.time()
    _mood_cache["fingerprint"]   = new_fingerprint
    _mood_cache["pulse_label"]   = macro_layer_status.market_pulse
    _mood_cache["climate_label"] = macro_layer_status.macro_climate
    
    return response_data

@router.get("/history", response_model=List[MarketMoodSnapshotResponse])
def get_market_history(db: Session = Depends(get_db), limit: int = 50):
    """
    Returns the latest mood snapshots from SQLite, ordered newest first.
    """
    snapshots = db.query(MarketMoodSnapshot).order_by(MarketMoodSnapshot.created_at.desc()).limit(limit).all()
    return snapshots
