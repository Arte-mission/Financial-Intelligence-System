from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List, Dict
from datetime import datetime, timezone, timedelta
import threading

from app.schemas import (
    MarketMoodResponse, MarketMoodSnapshotResponse,
    Drift, MacroLayerStatus, SignalTimelineEntry,
)
from app.services.newsdata_service import fetch_nepal_business_news
from app.services.onlinekhabar_service import fetch_onlinekhabar_news
from app.services.sentiment_service import (
    calculate_sentiment, get_mood_label, get_tiered_mood_label,
    categorize_article, score_materiality,
    extract_drivers, compute_macro_confidence, get_category_coverage_note,
    score_relevance, score_article_priority,
    build_macro_reasoning, build_delta_intelligence,
    apply_recency_filter_and_weight,
)
from app.database import get_db, SessionLocal
from app.models import NewsArticle, MarketMoodSnapshot
from app.utils.config import settings
import time
import logging

logger = logging.getLogger(__name__)

# NewsData is optional. The app runs normally without it.
NEWSDATA_ENABLED: bool = bool(settings.NEWSDATA_API_KEY)

router = APIRouter(prefix="/market-mood", tags=["Market Sentiment"])

# ── Macro response cache ────────────────────────────────────────────────────
# Stale-while-revalidate tiers (Sprint 1):
#   [0 … MACRO_CACHE_TTL]       → Tier 1: fresh — return instantly
#   [MACRO_CACHE_TTL … STALE]   → Tier 2: stale — return instantly + bg refresh
#   [> STALE]                   → Tier 3: expired — blocking recompute
MACRO_CACHE_TTL       = 600    # 10 min — primary TTL
MACRO_STALE_SERVE_TTL = 1200   # 20 min — stale serve window
MACRO_CACHE: dict = {}

# ── Background revalidation lock ─────────────────────────────────────────
_MACRO_REVALIDATING: bool = False
_MACRO_REVALIDATE_LOCK = threading.Lock()

# ── Sprint 2: In-memory Signal Timeline ──────────────────────────────────
# Stores the last N DISTINCT macro state transitions (not every refresh).
# Only updated when score changes ≥ 0.03 OR label changes OR drivers change.
# Prevents "always Neutral" spam in the timeline display.
_SIGNAL_TIMELINE: List[Dict] = []
_SIGNAL_TIMELINE_MAX = 5
_TIMELINE_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_source_fingerprint(raw_articles: list) -> str:
    """Fingerprint of the raw article list fetched from all sources.
    Built BEFORE the relevance gate — captures the true page state.
    Normalisation: lowercase title, strip punctuation, trim whitespace.
    Incorporates URL so a title reuse on a different story is detected.
    Incorporates published_at so a timestamp change (e.g. article correction)
    triggers recomputation even when title+URL are identical.
    """
    import hashlib, json, re
    entries = []
    for a in raw_articles:
        title     = re.sub(r"[^\w\s]", "", a.get("title", "").lower()).strip()
        url       = (a.get("url") or "").strip().rstrip("/")
        published = (a.get("published_at") or "").strip()
        entries.append(f"{title}|{url}|{published}")
    payload = sorted(entries)
    return hashlib.md5(json.dumps(payload).encode()).hexdigest()


def _make_fingerprint(headlines: list, score: float, pulse: str, climate: str) -> str:
    """Deterministic fingerprint of the scored headline set + signal labels.
    Used for DB snapshot gating (write only when signal materially changes).
    """
    import hashlib, json
    payload = {
        "headlines": sorted(h.lower().strip() for h in headlines),
        "score":     round(score, 2),
        "pulse":     pulse,
        "climate":   climate,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _update_signal_timeline(
    score: float,
    label: str,
    drivers: List[str],
    article_count: int,
    change_reason: str = "",
) -> None:
    """
    Append a state entry to the in-memory signal timeline ONLY when the
    transition is meaningful:
      - First entry (cold start)
      - Label changed from previous entry
      - Score moved ≥ 0.03 from previous entry
      - Drivers set changed AND score moved ≥ 0.01

    Duplicate Neutral states (score < 0.03 change) are suppressed so the
    timeline reads as real state changes, not background refresh artifacts.
    """
    global _SIGNAL_TIMELINE
    with _TIMELINE_LOCK:
        if _SIGNAL_TIMELINE:
            last = _SIGNAL_TIMELINE[-1]
            score_delta = abs(score - last["score"])
            # Suppress consecutive Neutral — prevents "always Neutral" spam
            if label == "Neutral" and last["label"] == "Neutral" and score_delta < 0.03:
                return
            # Suppress any transition with too-small a score change and same label
            if label == last["label"] and score_delta < 0.03:
                return

        entry = {
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "score":         round(score, 3),
            "label":         label,
            "drivers":       drivers[:3],
            "article_count": article_count,
            "change_reason": change_reason or "",
        }
        _SIGNAL_TIMELINE.append(entry)
        if len(_SIGNAL_TIMELINE) > _SIGNAL_TIMELINE_MAX:
            _SIGNAL_TIMELINE.pop(0)   # FIFO — oldest evicted first


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@router.get("", response_model=MarketMoodResponse)
@router.get("", response_model=MarketMoodResponse)
def get_market_sentiment(db: Session = Depends(get_db)):
    """
    Returns instantaneous macro market sentiment strictly from memory cache.
    The real heavyweight data-fetching and analysis happens permanently in the
    background via `start_macro_refresh_loop()`.
    """
    _entry = MACRO_CACHE.get("market_mood")
    
    if not _entry or not _entry.get("data"):
        # Rare ultra-cold start fallback if endpoint is hit before the very first background cycle completes
        logger.info("Macro cache missing — computing synchronously for initial request.")
        return _process_market_data(db)

    # Return pure in-memory cache instantly
    return _entry["data"].model_copy(update={
        "cache_hit": True, 
        "data_freshness": "fresh"
    })

def start_macro_refresh_loop():
    """
    Permanent background thread that executes market data scraping and processing every 10 minutes.
    This effectively replaces the passive stale-while-revalidate model with a backend-driven cycle.
    """
    def _loop():
        # Infinite cycle
        while True:
            try:
                db = SessionLocal()
                try:
                    logger.info("Background macro loop executing interval cycle...")
                    _process_market_data(db)
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"Background macro loop iteration failed: {e}")
            
            # Sleep until the next cycle seamlessly
            time.sleep(MACRO_CACHE_TTL)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    logger.info("🟢 Macro background refresh loop started.")


def _process_market_data(db: Session) -> MarketMoodResponse:
    """
    Full macro sentiment pipeline.
    Called synchronously on cold start / TTL expiry, or from background thread.

    Sprint 2 additions vs Sprint 1:
      - Dynamic article selection (recency × relevance priority, cap 30)
      - Article delta tracking (new / removed from pool)
      - Lighter EMA (0.85 new / 0.15 prev) — score responds to real changes
      - Noise gate: if |Δscore| < 0.02 AND label unchanged → return cached
      - Signal timeline updated only on meaningful state transitions
      - macro_reasoning + delta_intelligence generated on every full run
    """
    current_time = time.time()
    _entry = MACRO_CACHE.get("market_mood")

    MIN_ANALYZED_BEFORE_FALLBACK = 4
    MIN_MACRO_ARTICLES = 3

    # ── Step 1: Fetch raw articles ────────────────────────────────────────────
    fetched_raw: list = []
    source_breakdown: dict = {"OnlineKhabar": 0, "NewsData": 0}

    try:
        ok_articles = fetch_onlinekhabar_news()
        fetched_raw.extend(ok_articles)
        logger.info(f"OnlineKhabar fetched {len(ok_articles)} raw articles")
    except Exception as e:
        logger.warning(f"OnlineKhabar pipeline failed: {e}")
        ok_articles = []

    ok_raw_count = len(ok_articles)

    nd_articles_raw: list = []
    if ok_raw_count < MIN_ANALYZED_BEFORE_FALLBACK and NEWSDATA_ENABLED:
        try:
            nd_articles_raw, _ = fetch_nepal_business_news()
            fetched_raw.extend(nd_articles_raw)
            logger.info(f"NewsData raw fetch: {len(nd_articles_raw)} articles")
        except Exception as e:
            logger.warning(f"NewsData fallback failed: {e}")
    elif ok_raw_count < MIN_ANALYZED_BEFORE_FALLBACK and not NEWSDATA_ENABLED:
        logger.info(
            f"OnlineKhabar: {ok_raw_count} raw articles (below threshold {MIN_ANALYZED_BEFORE_FALLBACK}). "
            "NewsData not configured — OnlineKhabar-only mode."
        )

    # ── Source fingerprint — skip ALL compute if articles unchanged ───────────
    new_source_fp  = _make_source_fingerprint(fetched_raw)
    prev_source_fp = _entry["source_fp"] if _entry else None

    if prev_source_fp is not None and new_source_fp == prev_source_fp and _entry and _entry["data"]:
        logger.info(
            f"Source fingerprint unchanged ({new_source_fp[:8]}…) — "
            "resetting TTL, skipping recomputation."
        )
        MACRO_CACHE["market_mood"]["ts"] = time.time()
        
        last_checked_iso = datetime.now(timezone.utc).isoformat()
        updated_data = _entry["data"].model_copy(update={
            "cache_hit": True,
            "source_change": False,
            "data_freshness": "fresh",
            "source_change_status": "No new relevant articles",
            "last_checked": last_checked_iso,
            "next_check_at": (datetime.now(timezone.utc) + timedelta(seconds=MACRO_CACHE_TTL)).isoformat(),
        })
        MACRO_CACHE["market_mood"]["data"] = updated_data
        return updated_data

    # ── Step 2: Relevance gate + deduplication ────────────────────────────────
    seen_titles: set = set()
    raw_articles: list = []

    def _dedup_add(articles_in, source_tag):
        added = 0
        for art in articles_in:
            title = art.get("title", "").strip()
            if not title:
                continue
            tkey = title.lower()
            if tkey in seen_titles:
                continue
            if score_relevance(title) < 2:
                continue
            seen_titles.add(tkey)
            art["_source_tag"] = source_tag
            raw_articles.append(art)
            added += 1
        return added

    ok_added = _dedup_add(ok_articles, "OnlineKhabar")
    source_breakdown["OnlineKhabar"] = ok_added
    logger.info(f"OnlineKhabar contributed {ok_added} relevant articles")

    if nd_articles_raw:
        nd_added = _dedup_add(nd_articles_raw, "NewsData")
        source_breakdown["NewsData"] = nd_added
        logger.info(f"NewsData contributed {nd_added} relevant articles")

    query_used = "OnlineKhabar" + (" + NewsData" if source_breakdown["NewsData"] > 0 else "")

    # ── Recency filter + window-aware weight stamping ─────────────────────────
    # Filters to the 72-hour core window; falls back to 7 days if coverage is thin.
    # Each surviving article gains `_recency_weight` used during scoring below.
    raw_articles, _recency_window = apply_recency_filter_and_weight(raw_articles)
    logger.info(
        f"Recency window: {_recency_window} — {len(raw_articles)} articles retained "
        f"(OK={source_breakdown['OnlineKhabar']}, ND={source_breakdown.get('NewsData', 0)})"
    )

    # ── Sprint 2: Dynamic article selection — recency × relevance priority ─────
    # Sort by priority score (freshest + most relevant articles scored first),
    # then cap at 30 to prevent low-quality article overflow from diluting scores.
    raw_articles.sort(key=lambda a: score_article_priority(a), reverse=True)
    raw_articles = raw_articles[:30]

    # ── Sprint 2: Article delta — track new vs removed articles ───────────────
    prev_titles = (_entry.get("article_titles") or set()) if _entry else set()
    curr_titles  = {a.get("title", "") for a in raw_articles}
    article_delta_dict = {
        "new":     len(curr_titles - prev_titles),
        "removed": len(prev_titles - curr_titles),
        "total":   len(curr_titles),
    }

    if not raw_articles:
        return MarketMoodResponse(
            average_score=0.0,
            mood_label="Neutral",
            headline_count=0,
            latest_headlines=[],
            macro_sentiment="Neutral",
            score=0.0,
            drivers=[],
            # ── Performance metadata ──────────────────────────────────────────
            cache_hit=False,
            last_checked=datetime.now(timezone.utc).isoformat(),
            last_updated=datetime.now(timezone.utc).isoformat(),
            source_change=False,
            data_freshness="fresh",
            # ── Diagnostics ───────────────────────────────────────────────────
            raw_score=0.0,
            smoothed_score=0.0,
            previous_score=0.0,
            filtered_article_count=0,
            analyzed_article_count=0,
            query_used=query_used,
            categories_breakdown={},
            analyzed_headlines=[],
            source_breakdown=source_breakdown,
            article_delta=article_delta_dict,
            signal_timeline=[
                SignalTimelineEntry(**e) for e in _SIGNAL_TIMELINE
            ],
        )

    fetched_count = len(raw_articles)

    # ── Step 3: Score each article ────────────────────────────────────────────
    total_weighted_score = 0.0
    total_weight = 0.0
    market_score_sum = 0.0;  market_weight_sum = 0.0
    macro_score_sum  = 0.0;  macro_weight_sum  = 0.0
    headlines = []
    drivers_list = []
    categories_breakdown: dict = {"market": 0, "banking": 0, "macro": 0}
    high_materiality_count = 0
    weights = {"market": 1.0, "banking": 0.9, "macro": 0.7}

    for article_data in raw_articles:
        title = article_data.get("title", "")
        if not title:
            continue
        category = categorize_article(title)
        if category == "noise":
            continue

        score           = calculate_sentiment(title)
        category_weight  = weights.get(category, 0.0)
        materiality      = score_materiality(title)
        recency_weight   = article_data.get("_recency_weight", 0.30)  # stamped by apply_recency_filter_and_weight
        effective_weight = category_weight * materiality * recency_weight

        total_weighted_score += score * effective_weight
        total_weight += effective_weight

        if materiality == 1.0:
            high_materiality_count += 1

        if category == "market":
            market_score_sum += score * effective_weight
            market_weight_sum += effective_weight
        else:
            macro_score_sum += score * effective_weight
            macro_weight_sum += effective_weight

        headlines.append(title)
        categories_breakdown[category] = categories_breakdown.get(category, 0) + 1

        # Persist article to SQLite (idempotent)
        exists = db.query(NewsArticle).filter(NewsArticle.title == title).first()
        if not exists:
            db.add(NewsArticle(
                title=title,
                source=article_data.get("source", ""),
                published_at=article_data.get("published_at", ""),
                sentiment_score=score,
                query_used=query_used,
            ))

        article_drivers = extract_drivers(title, score, category, materiality)
        drivers_list.extend(article_drivers)

    db.commit()

    count = len(headlines)
    new_score_raw = total_weighted_score / total_weight if total_weight > 0 else 0.0

    # Previous state
    previous_score = 0.0
    has_previous   = False
    if _entry and _entry["data"] is not None:
        previous_score = _entry["data"].score
        has_previous   = True

    # ── Sprint 2: Lighter EMA (0.85 / 0.15) ──────────────────────────────────
    # Was 0.60 / 0.40 — the heavy prior weight made scores sticky even when
    # articles changed substantially. 0.85 new data tracks real changes while
    # still smoothing single-article noise spikes.
    if has_previous:
        smoothed_score = (new_score_raw * 0.85) + (previous_score * 0.15)
    else:
        smoothed_score = new_score_raw

    change     = smoothed_score - previous_score if has_previous else 0.0
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

    # Two-layer signal computation
    market_pulse_score = market_score_sum / market_weight_sum if market_weight_sum > 0 else 0.0
    market_pulse_label = get_mood_label(market_pulse_score)

    macro_article_count  = categories_breakdown.get("banking", 0) + categories_breakdown.get("macro", 0)
    market_article_count = categories_breakdown.get("market", 0)
    macro_coverage_sufficient = macro_article_count >= MIN_MACRO_ARTICLES

    if macro_coverage_sufficient:
        raw_macro_score    = macro_score_sum / macro_weight_sum if macro_weight_sum > 0 else 0.0
        macro_climate_label = get_tiered_mood_label(raw_macro_score)
    else:
        macro_climate_label = "Insufficient Coverage"

    macro_layer_status = MacroLayerStatus(
        market_pulse=market_pulse_label,
        macro_climate=macro_climate_label,
        market_article_count=market_article_count,
        macro_article_count=macro_article_count,
        coverage_sufficient=macro_coverage_sufficient,
    )

    macro_confidence = compute_macro_confidence(count, categories_breakdown, high_materiality_count)
    coverage_note    = get_category_coverage_note(categories_breakdown)

    average_score_rounded = round(smoothed_score, 2)
    change_rounded        = round(change, 2)
    mood_label            = get_mood_label(smoothed_score)
    tiered_label          = get_tiered_mood_label(smoothed_score)

    # ── Sprint 2: Noise gate ──────────────────────────────────────────────────
    # If scores barely moved AND label is unchanged, broadcast as "refreshed but
    # unchanged" without writing a new snapshot or updating the timeline.
    # db.commit() above already saved new articles; this only gates the signal layer.
    _prev_tiered = get_tiered_mood_label(previous_score) if has_previous else None
    if has_previous and abs_change < 0.02 and tiered_label == _prev_tiered and _entry and _entry["data"]:
        logger.info(
            f"Noise gate: Δscore={abs_change:.4f} < 0.02, label={tiered_label} unchanged "
            f"— skipping snapshot + timeline update"
        )
        MACRO_CACHE["market_mood"]["ts"] = time.time()
        last_checked_iso = datetime.now(timezone.utc).isoformat()
        
        updated_data = _entry["data"].model_copy(update={
            "cache_hit":             False,
            "source_change":         True,
            "data_freshness":        "fresh",
            "last_checked":          last_checked_iso,
            "next_check_at":         (datetime.now(timezone.utc) + timedelta(seconds=MACRO_CACHE_TTL)).isoformat(),
            # We explicitly do NOT touch last_updated here, keeping the previous value.
            "source_change_status":  "New articles detected, but macro signals unchanged",
            "article_delta":         article_delta_dict,
            "analyzed_headlines":    headlines,
        })
        MACRO_CACHE["market_mood"]["data"] = updated_data
        return updated_data

    # ── Fingerprint-gated snapshot write ─────────────────────────────────────
    new_fingerprint      = _make_fingerprint(
        headlines, average_score_rounded,
        macro_layer_status.market_pulse,
        macro_layer_status.macro_climate,
    )
    previous_fingerprint = _entry["fingerprint"]   if _entry else None
    previous_pulse       = _entry["pulse_label"]    if _entry else None
    previous_climate     = _entry["climate_label"]  if _entry else None

    pulse_changed    = (macro_layer_status.market_pulse   != previous_pulse)
    climate_changed  = (macro_layer_status.macro_climate  != previous_climate)
    score_moved      = abs_change >= 0.03
    articles_changed = (new_fingerprint != previous_fingerprint)

    should_write_snapshot = (
        previous_fingerprint is None
        or pulse_changed
        or climate_changed
        or score_moved
    )

    if should_write_snapshot:
        snapshot = MarketMoodSnapshot(
            average_score=average_score_rounded,
            mood_label=tiered_label,
            headline_count=count,
        )
        db.add(snapshot)
        db.commit()
        logger.info(
            f"Snapshot written: score={average_score_rounded}, "
            f"pulse={macro_layer_status.market_pulse}, "
            f"climate={macro_layer_status.macro_climate}"
        )
    else:
        logger.info(
            f"Snapshot skipped (no material change): fp={new_fingerprint[:8]}…, "
            f"score={average_score_rounded}"
        )

    # Source-change metadata
    _articles_changed_bool = (previous_fingerprint is None or articles_changed)
    source_change_status = (
        "New relevant coverage detected"
        if _articles_changed_bool
        else "No new relevant articles since last refresh"
    )

    # ── Sprint 2: Previous state metadata for delta intelligence ─────────────
    prev_drivers   = list(_entry["data"].drivers) if _entry and _entry["data"] else []
    prev_label_str = _entry["data"].mood_label    if _entry and _entry["data"] else tiered_label

    # ── Sprint 2: Delta intelligence — natural-language change description ────
    delta_text = build_delta_intelligence(
        new_label=tiered_label,
        prev_label=prev_label_str,
        new_score=average_score_rounded,
        prev_score=round(previous_score, 2),
        new_drivers=unique_drivers,
        prev_drivers=prev_drivers,
        article_delta=article_delta_dict,
    )

    # ── Sprint 2: Coverage-aware delta summary ────────────────────────────────
    if not macro_coverage_sufficient:
        base_summary = (
            f"Current signal is based on {market_article_count} NEPSE market "
            f"headline{'s' if market_article_count != 1 else ''}, with limited "
            "banking or macroeconomic confirmation."
        )
        delta_summary = base_summary
    elif not has_previous or abs_change < 0.03:
        if unique_drivers:
            base_summary = (
                f"Macro sentiment broadly unchanged — "
                f"{unique_drivers[0]} establishing a stable baseline."
            )
        else:
            base_summary = "Market sentiment remains balanced as sector gains continue to offset broader index pressure."
        delta_summary = f"{base_summary} {coverage_note}".strip() if coverage_note else base_summary
    else:
        if change > 0:
            base_summary = (
                f"Market sentiment improved as {unique_drivers[0] if unique_drivers else 'positive sector rotation'} "
                "supported the broader backdrop."
            )
        else:
            base_summary = (
                f"Market mood remains cautious as "
                f"{unique_drivers[0] if unique_drivers else 'banking and policy concerns'} "
                "continue to pressure investor sentiment."
            )
        delta_summary = f"{base_summary} {coverage_note}".strip() if coverage_note else base_summary

    # ── Sprint 2: Macro reasoning — rich investor-grade explanation ───────────
    macro_reasoning_text = build_macro_reasoning(
        label=tiered_label,
        score=average_score_rounded,
        drivers=unique_drivers,
        confidence=macro_confidence,
        article_count=count,
        change=change_rounded,
        has_previous=has_previous,
        prev_label=prev_label_str,
        categories_breakdown=categories_breakdown,
    )

    # ── Sprint 2: Signal timeline update ─────────────────────────────────────
    _update_signal_timeline(
        score=average_score_rounded,
        label=tiered_label,
        drivers=unique_drivers,
        article_count=count,
        change_reason=delta_text,
    )

    # ── Build response ────────────────────────────────────────────────────────
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
        delta_summary=delta_text,              # Sprint 2: richer delta intelligence text
        macro_confidence=macro_confidence,
        macro_layer_status=macro_layer_status,
        # ── Performance metadata ───────────────────────────────────────────────
        cache_hit=False,
        source_change=_articles_changed_bool,
        last_checked=datetime.now(timezone.utc).isoformat(),
        data_freshness="fresh",
        # ── Diagnostics ────────────────────────────────────────────────────────
        raw_score=round(new_score_raw, 4),
        smoothed_score=round(smoothed_score, 4),
        previous_score=round(previous_score, 4),
        filtered_article_count=fetched_count,
        analyzed_article_count=count,
        query_used=query_used,
        categories_breakdown=categories_breakdown,
        analyzed_headlines=headlines,
        source_breakdown=source_breakdown,
        source_change_status=source_change_status,
        # ── Sprint 2: new fields ───────────────────────────────────────────────
        macro_reasoning=macro_reasoning_text,
        article_delta=article_delta_dict,
        raw_article_count=ok_raw_count,
        relevant_article_count=count,
        freshest_article_at=fetched_raw[0].get("published_at", "Just now") if fetched_raw else "Just now",
        next_check_at=(datetime.now(timezone.utc) + timedelta(seconds=MACRO_CACHE_TTL)).isoformat(),
        signal_timeline=[
            SignalTimelineEntry(
                timestamp=e["timestamp"],
                score=e["score"],
                label=e["label"],
                drivers=e.get("drivers", []),
                article_count=e.get("article_count", 0),
                change_reason=e.get("change_reason", ""),
            )
            for e in _SIGNAL_TIMELINE
        ],
    )

    # ── Update cache ──────────────────────────────────────────────────────────
    MACRO_CACHE["market_mood"] = {
        "data":           response_data,
        "ts":             time.time(),
        "source_fp":      new_source_fp,
        "fingerprint":    new_fingerprint,
        "pulse_label":    macro_layer_status.market_pulse,
        "climate_label":  macro_layer_status.macro_climate,
        "article_titles": curr_titles,     # Sprint 2: for article delta tracking
    }
    logger.info(
        f"MACRO_CACHE populated: score={average_score_rounded}, "
        f"label={tiered_label}, articles={count}, "
        f"delta=new:{article_delta_dict['new']}/rem:{article_delta_dict['removed']}"
    )

    return response_data


@router.get("/history", response_model=List[MarketMoodSnapshotResponse])
def get_market_history(db: Session = Depends(get_db), limit: int = 50):
    """
    Returns the latest mood snapshots from SQLite, ordered newest first.
    """
    snapshots = (
        db.query(MarketMoodSnapshot)
        .order_by(MarketMoodSnapshot.created_at.desc())
        .limit(limit)
        .all()
    )
    return snapshots
