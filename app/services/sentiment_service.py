"""
sentiment_service.py — Nepal financial sentiment analysis layer.

Key functions:
  - calculate_sentiment(): Hybrid TextBlob + Nepal keyword scorer
  - get_mood_label(): 3-tier label (Bullish/Neutral/Bearish)
  - get_tiered_mood_label(): 5-tier calibrated label for macro signals
  - score_relevance(): relevance gate (0–10 score, gate at >= 2)
  - score_article_priority(): recency × relevance priority for dynamic selection
  - categorize_article(): market / banking / macro / noise
  - score_materiality(): article importance weight (High/Medium/Low)
  - extract_drivers(): semantic investor-grade driver phrases
  - compute_macro_confidence(): confidence with materiality awareness
  - get_category_coverage_note(): investor note on single-category dominance
  - build_macro_reasoning(): full investor-grade macro explanation
  - build_delta_intelligence(): natural-language delta between two states
"""

from textblob import TextBlob
from typing import Tuple, List, Dict, Any, Optional
import re
from datetime import datetime, timezone, timedelta


# ── Nepal financial keyword sentiment tables ──────────────────────────────────
#
# TextBlob polarity on factual business text ("NEPSE index rises 15 points")
# returns near-zero because those sentences contain no subjective words.
# These domain-specific tables give each keyword a signed weight so the
# scorer produces meaningful directional output for Nepal financial headlines.
#
# Weights are calibrated so a single strong signal (~0.25) pushes the final
# blended score clearly above the 0.10 / -0.10 Bullish/Bearish threshold.
#
_FIN_POSITIVE: List[Tuple[str, float]] = [
    # Market momentum
    ("rises",           0.18), ("rise",           0.15), ("risen",         0.15),
    ("gains",           0.18), ("gain",            0.15),
    ("rally",           0.20), ("rallies",         0.20),
    ("recover",         0.15), ("recovery",        0.15),
    ("bullish",         0.22), ("positive outlook", 0.18),
    # Liquidity / rates — positive direction
    ("liquidity surplus",   0.22), ("liquidity ease",    0.20),
    ("liquidity improve",   0.20), ("liquidity easing",  0.20),
    ("rate cut",            0.25), ("policy rate cut",   0.25),
    ("rate reduce",         0.20), ("rate lower",        0.20),
    ("rate falls",          0.18), ("rates fall",        0.18),
    ("interest rate cut",   0.25), ("rate decline",      0.18),
    # Corporate actions
    ("dividend",        0.14), ("bonus share",     0.20),
    ("ipo oversubscribed",  0.22),
    ("profit",          0.13), ("profitable",      0.15),
    # Macro / institutional positive — compound-only (no bare 'surge' to avoid inflation surge)
    ("remittance surge",    0.22), ("remittance record",  0.20),
    ("remittance rise",     0.18), ("remittance increase", 0.16),
    ("nepse surge",         0.25), ("market surge",       0.22),
    ("share surge",         0.20), ("index surge",        0.20),
    ("foreign reserve rise",0.18), ("reserve increase",  0.15),
    ("growth forecast",     0.15), ("gdp grow",          0.15),
    ("upgrade",         0.15), ("expansion",       0.12),
    ("improve",         0.12), ("improvement",     0.12),
    ("boost",           0.15), ("stimulus",        0.15),
    ("easing",          0.18), ("eases",           0.15),
    ("credit expansion",    0.18), ("credit grow",       0.15),
    ("approve",         0.10), ("approved",        0.10),
    ("optimistic",      0.15), ("support",         0.08),
]

_FIN_NEGATIVE: List[Tuple[str, float]] = [
    # Market decline
    ("falls",           0.18), ("fall",            0.15), ("fallen",        0.15),
    ("drops",           0.18), ("drop",            0.15),
    ("declines",        0.18), ("decline",         0.15),
    ("crash",           0.30), ("crashes",         0.30),
    ("selloff",         0.25), ("sell-off",        0.25),
    ("slump",           0.20), ("plunge",          0.25),
    ("bearish",         0.22), ("pessimistic",     0.15),
    # Liquidity / rates — negative direction
    ("liquidity crunch",    0.28), ("liquidity shortage", 0.25),
    ("liquidity tight",     0.22), ("liquidity pressure", 0.20),
    ("liquidity tighten",   0.22),
    ("rate hike",           0.22), ("rate increase",     0.15),
    ("rate rise",           0.15), ("rate rises",        0.15),
    ("interest rate rise",  0.20), ("interest rate hike", 0.22),
    ("tighten",         0.18), ("tightening",      0.18),
    # Inflation (compound-scoped to prevent firing on 'inflation eases')
    ("inflation rise",      0.22), ("inflation rises",    0.22),
    ("inflation high",      0.20), ("inflation surge",    0.25),
    ("inflation surges",    0.25), ("inflation pressure", 0.20),
    ("inflation hit",       0.20), ("inflation soar",     0.25),
    # Banking stress
    ("npa rise",            0.22), ("bad loan",          0.22),
    ("non-performing",      0.18), ("npa increase",      0.20),
    ("banking crisis",      0.28), ("bank failure",      0.30),
    # Regulatory / institutional
    ("restrict",        0.15), ("ban",             0.12),
    ("suspend",         0.20), ("penalty",         0.15),
    ("directive",       0.08), ("warning",         0.12),
    # Macro stress
    ("deficit",         0.15), ("debt crisis",     0.25),
    ("foreign reserve fall", 0.20), ("reserve decline", 0.18),
    ("slowdown",        0.15), ("contraction",     0.20),
    ("loss",            0.15), ("losses",          0.15),
    ("weak",            0.12), ("weakness",        0.12),
    ("concern",         0.10), ("deteriorate",     0.18),
    ("credit contraction",  0.20), ("credit slow",       0.15),
    ("crisis",          0.25), ("crunch",          0.25),
]


def calculate_sentiment(text: str) -> float:
    """
    Hybrid sentiment scorer for Nepal financial headlines.

    TextBlob alone returns near-zero polarity for factual business statements
    like "NEPSE index rises 15 points" because those sentences lack subjective
    language. This function blends TextBlob with a domain-specific Nepal
    financial keyword table to produce meaningful directional scores.

    Scoring:
      1. TextBlob base polarity (20% weight) — picks up adjectives & adverbs
      2. Keyword delta: sum(positive matches) - sum(negative matches) (80% weight)
      3. Clamp result to [-1.0, +1.0]
    """
    if not text:
        return 0.0

    base_score = TextBlob(text).sentiment.polarity
    t = text.lower()

    pos_delta = sum(w for kw, w in _FIN_POSITIVE if kw in t)
    neg_delta = sum(w for kw, w in _FIN_NEGATIVE if kw in t)
    kw_delta  = pos_delta - neg_delta

    blended = (base_score * 0.20) + (kw_delta * 0.80)
    return max(-1.0, min(1.0, blended))


def get_mood_label(score: float) -> str:
    """
    Simple 3-tier label — used for Market Pulse (which is fairly binary).
    - score >  0.1  => Bullish
    - score < -0.1  => Bearish
    - otherwise     => Neutral
    """
    if score > 0.1:
        return "Bullish"
    elif score < -0.1:
        return "Bearish"
    return "Neutral"


def get_tiered_mood_label(score: float) -> str:
    """
    5-tier calibrated label for the macro climate signal.
    Prevents overstating conviction for small positive/negative scores.

    - score >  0.25 => Bullish
    - score >  0.10 => Mildly Bullish
    - score >= -0.10 => Neutral
    - score >= -0.25 => Mildly Bearish
    - score <  -0.25 => Bearish
    """
    if score > 0.25:
        return "Bullish"
    elif score > 0.10:
        return "Mildly Bullish"
    elif score >= -0.10:
        return "Neutral"
    elif score >= -0.25:
        return "Mildly Bearish"
    return "Bearish"


def score_relevance(title: str) -> int:
    """
    Scores how directly financially relevant a Nepal headline is (0–10).

    Tier-1 (+2 each): core Nepal finance institutions / markets by name
    Tier-2 (+1 each): generic but financial terms in a Nepal context

    Articles scoring < 2 are treated as noise.
    """
    t = title.lower()
    score = 0

    tier1 = [
        "nepse", "nepal rastra bank", "nrb", "sebon",
        "nepal stock", "nepal share", "nepal index",
        "nepal inflation", "nepal gdp", "nepal economy",
        "nepal liquidity", "nepal interest rate", "nepal monetary",
        "nepal budget", "nepal fiscal", "nepal banking",
    ]
    for kw in tier1:
        if kw in t:
            score += 2

    tier2 = [
        "stock market", "share market", "ipo", "dividend", "bonus share",
        "right share", "trading", "liquidity", "interest rate", "inflation",
        "monetary policy", "repo rate", "deposit rate", "loan rate",
        "financial policy", "fiscal policy", "central bank",
        "microfinance", "remittance", "forex", "foreign exchange",
        "balance of payment", "credit growth", "npa", "bfi",
        "development bank", "cooperative bank", "insurance",
    ]
    for kw in tier2:
        if kw in t:
            score += 1

    return score


def score_article_priority(article: Dict[str, Any]) -> float:
    """
    Priority score (higher = selected first) for dynamic article selection.
    Combines relevance score with a recency weight derived from published_at.

    Recency decay:
      "X minutes ago"  → 1.0  (very fresh)
      "X hours ago"    → linear 1.0→0.50 over 24 h
      "X days ago"     → linear 0.40→0.05 over 6 d
      "X weeks ago"    → 0.02
      unknown/missing  → 0.30 (neutral baseline)
    """
    title     = article.get("title", "")
    relevance = score_relevance(title)

    published = (article.get("published_at") or "").lower().strip()
    recency   = 0.30  # default for unknown dates

    m = re.search(r"(\d+)\s+minute", published)
    if m:
        recency = 1.0

    m = re.search(r"(\d+)\s+hour", published)
    if m:
        hours   = int(m.group(1))
        recency = max(0.50, 1.0 - (hours / 48.0))   # 1.0 @ 0h → 0.5 @ 24h

    m = re.search(r"(\d+)\s+day", published)
    if m:
        days    = int(m.group(1))
        recency = max(0.05, 0.40 - (days * 0.06))   # 0.40 @ 1d → 0.04 @ 6d+

    m = re.search(r"(\d+)\s+week", published)
    if m:
        weeks   = int(m.group(1))
        recency = max(0.02, 0.10 - (weeks * 0.03))

    return relevance * recency


# ── Time-window constants ──────────────────────────────────────────────────────
_WINDOW_CORE_H  = 72   # Primary analysis window (3 days)
_WINDOW_MAX_H   = 168  # Absolute outer limit  (7 days)
_MIN_CORE_COUNT = 8    # Minimum articles needed before falling back to 7-day window


def _parse_article_age_hours(article: Dict[str, Any]) -> Optional[float]:
    """
    Parse `published_at` into an age in hours relative to now.

    Handles two formats that exist in the pipeline:
      1. Relative strings ── "2 hours ago", "3 days ago", "1 week ago"
         (produced by OnlineKhabar scraper)
      2. ISO-8601 strings ── "2025-04-18T09:30:00Z"
         (produced by NewsData.io)

    Returns None when the timestamp is absent, unparseable, or in the future.
    """
    raw = (article.get("published_at") or "").strip()
    if not raw:
        return None

    lo = raw.lower()

    # ── Relative strings ──────────────────────────────────────────────────────
    m = re.search(r"(\d+)\s+minute", lo)
    if m:
        return int(m.group(1)) / 60.0

    m = re.search(r"(\d+)\s+hour", lo)
    if m:
        return float(m.group(1))

    m = re.search(r"(\d+)\s+day", lo)
    if m:
        return float(m.group(1)) * 24.0

    m = re.search(r"(\d+)\s+week", lo)
    if m:
        return float(m.group(1)) * 168.0

    # ── ISO-8601 / datetime strings ───────────────────────────────────────────
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
            return age_h if age_h >= 0 else None
        except ValueError:
            continue

    return None   # unparseable


def _age_to_recency_weight(age_h: float) -> float:
    """
    Four-tier recency decay weight:
      0 – 24 h  → 1.0
      24 – 48 h → 0.7
      48 – 72 h → 0.5
      72 – 168 h → 0.2   (extended / fallback window only)
    """
    if age_h <= 24:   return 1.0
    if age_h <= 48:   return 0.7
    if age_h <= 72:   return 0.5
    return 0.2


def apply_recency_filter_and_weight(
    articles: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Filter `articles` to a time window and stamp each with `_recency_weight`.

    Strategy:
      1. Attempt to include only articles from the last 72 hours (core window).
      2. If fewer than _MIN_CORE_COUNT survive, expand to 7 days (fallback window).
      3. Drop everything older than 7 days entirely.
      4. Articles without a parseable timestamp are included with weight 0.3
         (neutral baseline — they cannot be verified as stale or fresh).

    Returns:
      (filtered_articles, window_label)
        window_label ∈ {"core_72h", "extended_7d", "no_filter"}
    """
    import logging
    log = logging.getLogger(__name__)

    if not articles:
        return [], "no_filter"

    # Stamp every article with its parsed age and initial recency weight
    for art in articles:
        age = _parse_article_age_hours(art)
        if age is None:
            art["_age_h"]         = None
            art["_recency_weight"] = 0.30   # neutral baseline for unknown dates
        else:
            art["_age_h"]         = age
            art["_recency_weight"] = _age_to_recency_weight(age)

    # Core window: articles ≤ 72 h  OR  unknown-age articles
    core = [
        a for a in articles
        if a["_age_h"] is None or a["_age_h"] <= _WINDOW_CORE_H
    ]

    if len(core) >= _MIN_CORE_COUNT:
        log.info(
            f"Recency filter: {len(core)}/{len(articles)} articles inside "
            f"72-hour core window."
        )
        return core, "core_72h"

    # Fallback: expand to 7 days; drop strictly older
    extended = [
        a for a in articles
        if a["_age_h"] is None or a["_age_h"] <= _WINDOW_MAX_H
    ]
    dropped  = len(articles) - len(extended)
    log.info(
        f"Recency filter: core window too thin ({len(core)} articles); "
        f"expanding to 7-day fallback — {len(extended)} kept, {dropped} dropped."
    )
    return extended, "extended_7d"


def score_materiality(title: str) -> float:
    """
    Returns a materiality weight (0.3 / 0.7 / 1.0) for a Nepal financial headline.

    High materiality (1.0):
      - monetary policy, rate changes, liquidity crisis, NRB regulatory action,
        major market moves, inflation data, banking instability

    Medium materiality (0.7):
      - NEPSE index/turnover movement, budget impact, sector rotation,
        ADB/IMF forecasts, governor appointments

    Low materiality (0.3):
      - general informational NRB stories, appointments, profiles,
        logistics/operational content, non-directional updates
    """
    t = title.lower()

    # --- High materiality signals ---
    high_patterns = [
        # Monetary / rates
        "interest rate", "rate hike", "rate cut", "policy rate", "repo rate",
        "monetary policy", "crd", "cash reserve ratio",
        # Liquidity
        "liquidity crunch", "liquidity surplus", "liquidity injection",
        "liquidity tightening", "liquidity shortage", "liquidity pressure",
        # NRB regulatory action
        "nrb directive", "nrb regulation", "nrb action", "nrb circular",
        "nrb order", "nrb bans", "nrb restricts", "nrb allowance",
        "nrb policy", "rastra bank directive", "rastra bank policy",
        # Inflation
        "inflation", "consumer price", "cpi",
        # Market emergency / crisis
        "circuit breaker", "market crash", "market halt", "market suspension",
        "trading halt", "nepse crash", "banking crisis", "bank failure",
        # Macro shocks
        "budget deficit", "debt crisis", "foreign reserve", "bop crisis",
        "balance of payment", "current account deficit",
    ]
    if any(kw in t for kw in high_patterns):
        return 1.0

    # --- Medium materiality signals ---
    medium_patterns = [
        # NEPSE movement / market data
        "nepse", "market index", "turnover", "market capitalization",
        "market rise", "market fall", "market gain", "market drop",
        "sector gain", "sector rotation", "sector performance",
        # Budget / fiscal
        "budget", "fiscal policy", "revenue collection", "tax revenue",
        # International forecasts
        "adb", "imf", "world bank", "growth forecast", "gdp forecast",
        # Banking leadership / direction
        "governor", "deputy governor", "nrb governor", "central bank chief",
        "bank ceo", "bank merger", "bank acquisition",
        # Credit / loan movements
        "credit growth", "loan disbursement", "npa", "bad loan",
        # Remittance
        "remittance",
    ]
    if any(kw in t for kw in medium_patterns):
        return 0.7

    # --- Low materiality (default) ---
    return 0.3


def categorize_article(title: str) -> str:
    """
    Classifies a Nepal financial headline into 'market', 'banking', 'macro', or 'noise'.

    Two-stage filter:
    1. Hard-block non-financial topics.
    2. Relevance gate: score < 2 => noise.
    3. Category assignment: market > banking > macro.
    """
    t = title.lower()

    # --- Stage 1: Hard noise block ---
    hard_noise = [
        "cricket", "football", "volleyball", "tennis", "badminton", "athletics",
        "hockey", "wrestling", "marathon", "olympic",
        "movie", "film", "actor", "actress", "celebrity", "music", "concert",
        "festival", "culture", "trekking", "travel", "pilgrimage",
        "religion", "temple", "monastery", "puja",
        "weather", "earthquake", "flood", "landslide", "disaster", "rainfall",
        "election", "parliament", "party", "vote", "coalition", "prime minister",
        "minister resigns", "cabinet", "ambassador",
        "education", "school", "university", "scholarship", "hospital",
        "disease", "covid", "vaccine", "health",
        "crime", "murder", "theft", "police", "arrest", "scam", "fraud",
        "highway", "road project", "bridge", "railway", "airport construction",
        "trade corridor", "trade route", "barabanki", "bahraich",
        "border crossing", "transit route", "bilateral",
    ]
    if any(kw in t for kw in hard_noise):
        return "noise"

    # --- Stage 2: Relevance gate ---
    if score_relevance(title) < 2:
        return "noise"

    # --- Stage 3: Category classification (most specific first) ---
    market_signals = [
        "nepse", "sebon", "stock market", "share market", "share price",
        "ipo", "dividend", "bonus share", "right share",
        "listed compan", "trading session", "market index", "bull", "bear",
        "demat", "floorsheet",
    ]
    if any(kw in t for kw in market_signals):
        return "market"

    banking_signals = [
        "nepal rastra bank", "nrb", "commercial bank", "microfinance", "bfi",
        "liquidity", "interest rate", "deposit rate", "lending rate",
        "monetary policy", "repo rate", "ccd ratio", "npa", "credit growth",
        "loan", "interbank rate", "remittance", "forex", "foreign exchange",
    ]
    if any(kw in t for kw in banking_signals):
        return "banking"

    macro_signals = [
        "nepal economy", "nepal gdp", "nepal inflation", "fiscal policy",
        "nepal budget", "revenue", "trade deficit", "balance of payment",
        "adb", "imf", "world bank", "economic growth", "economic slowdown",
        "foreign investment", "fdi", "cooperative", "insurance",
        "development bank", "financial stability",
    ]
    if any(kw in t for kw in macro_signals):
        return "macro"

    return "macro"


def extract_drivers(title: str, sentiment_score: float, category: str, materiality: float = 0.7) -> List[str]:
    """
    Extracts 0–2 investor-grade semantic driver phrases from a headline.

    Rules:
    - Uses contextual compound patterns (not single keywords).
    - Only generates fallback tags for medium/high materiality articles.
    - Returns human-readable investor phrases, not system tags.
    """
    t = title.lower()
    drivers = []

    # ---- High-signal bearish patterns ----
    if "liquidity" in t and any(w in t for w in ["crunch", "tight", "pressure", "shortage", "drain"]):
        drivers.append("liquidity pressure")
    if ("nrb" in t or "rastra bank" in t) and any(w in t for w in ["tighten", "restrict", "directive", "circular", "ban", "suspend", "warn"]):
        drivers.append("regulatory tightening")
    if ("monetary policy" in t or "policy rate" in t or "repo rate" in t) and any(w in t for w in ["hike", "increase", "tighten", "raise"]):
        drivers.append("monetary policy tightening")
    if "inflation" in t and any(w in t for w in ["rise", "high", "pressure", "surge"]):
        drivers.append("inflation pressure")
    if "interest rate" in t and any(w in t for w in ["rise", "hike", "increase", "high", "surge"]):
        drivers.append("rising interest rates")
    if any(w in t for w in ["npa", "bad loan", "non-performing"]) and any(w in t for w in ["rise", "increase", "grow", "high"]):
        drivers.append("rising non-performing loans")
    if "nepse" in t and any(w in t for w in ["fall", "drop", "decline", "slump", "crash", "selloff"]):
        drivers.append("continued NEPSE weakness")
    if "credit" in t and any(w in t for w in ["slow", "contract", "tight", "weak", "decline"]):
        drivers.append("credit contraction")
    if "growth" in t and any(w in t for w in ["slow", "decline", "contract", "weak", "concern"]):
        drivers.append("growth slowdown concerns")
    if "reserve" in t and any(w in t for w in ["fall", "decline", "drop", "low", "pressure"]):
        drivers.append("foreign reserve pressure")

    # ---- High-signal bullish patterns ----
    if "liquidity" in t and any(w in t for w in ["ease", "improve", "recover", "surplus", "inject"]):
        drivers.append("liquidity easing")
    if ("monetary policy" in t or "policy rate" in t or "repo rate" in t) and any(w in t for w in ["cut", "ease", "reduce", "lower", "accommodat"]):
        drivers.append("monetary policy support")
    if ("nrb" in t or "rastra bank" in t) and any(w in t for w in ["ease", "support", "allow", "stimulus", "cut"]):
        drivers.append("regulatory easing")
    if "interest rate" in t and any(w in t for w in ["cut", "fall", "low", "reduce", "decline"]):
        drivers.append("falling interest rates")
    if "nepse" in t and any(w in t for w in ["rise", "gain", "rally", "recover", "surge", "momentum"]):
        drivers.append("market recovery momentum")
    if "remittance" in t and any(w in t for w in ["rise", "increase", "strong", "record", "surge", "high"]):
        drivers.append("strong remittance inflows")
    if "gdp" in t and any(w in t for w in ["grow", "expand", "strong", "recover", "forecast"]):
        drivers.append("economic growth signal")
    if "credit" in t and any(w in t for w in ["grow", "expand", "recover", "rise", "increase"]):
        drivers.append("credit expansion")
    if "dividend" in t or "bonus share" in t:
        drivers.append("dividend announcements")

    # ---- Medium-signal contextual patterns ----
    if ("governor" in t or "deputy governor" in t or "ceo" in t) and ("nrb" in t or "rastra bank" in t or "bank" in t):
        drivers.append("banking leadership change")
    if "budget" in t and any(w in t for w in ["market", "sector", "stock", "fiscal"]):
        drivers.append("budget-linked market impact")
    if "adb" in t or "imf" in t or "world bank" in t:
        drivers.append("international growth outlook")
    if "sector" in t and any(w in t for w in ["rotat", "shift", "gain", "lead"]):
        drivers.append("sector rotation")
    if "merger" in t or "acquisition" in t:
        drivers.append("banking consolidation")

    # ---- Fallback: only for high/medium materiality articles with clear direction ----
    if not drivers and materiality >= 0.7 and abs(sentiment_score) > 0.1:
        if category == "market":
            drivers.append("NEPSE directional pressure" if sentiment_score < 0 else "market sentiment improving")
        elif category == "banking":
            drivers.append("banking sector caution" if sentiment_score < 0 else "banking sector stability")
        else:
            drivers.append("macro headwinds" if sentiment_score < 0 else "macro backdrop support")

    return drivers


def compute_macro_confidence(
    analyzed_count: int,
    categories_breakdown: Dict[str, int],
    high_materiality_count: int = 0
) -> str:
    """
    Computes a macro signal confidence level.

    Confidence must reflect:
    - Article volume
    - Category diversity
    - Materiality mix (many low-impact articles don't inflate confidence)

    Levels:
    - High:   8+ articles AND 2+ categories AND 2+ high-materiality articles
    - Medium: 4+ articles AND (2+ categories OR 1+ high-materiality)
    - Low:    otherwise
    """
    active_categories = sum(1 for v in categories_breakdown.values() if v > 0)

    if analyzed_count >= 8 and active_categories >= 2 and high_materiality_count >= 2:
        return "High"
    elif analyzed_count >= 4 and (active_categories >= 2 or high_materiality_count >= 1):
        return "Medium"
    return "Low"


def get_category_coverage_note(categories_breakdown: Dict[str, int]) -> str:
    """
    Returns an investor-facing note if the macro signal is dominated by one category.
    Returns empty string if coverage is reasonably balanced.
    """
    total = sum(categories_breakdown.values())
    if total == 0:
        return ""

    market_share  = categories_breakdown.get("market",  0) / total
    banking_share = categories_breakdown.get("banking", 0) / total
    macro_share   = categories_breakdown.get("macro",   0) / total

    if market_share >= 0.80:
        return "Macro signal is currently based mostly on market headlines and has limited banking or macroeconomic confirmation."
    if banking_share >= 0.80:
        return "Macro signal draws heavily from banking sector coverage with limited broader economic context."
    if macro_share >= 0.80:
        return "Signal is driven primarily by macroeconomic coverage. Market and banking confirmation is limited."

    return ""


def build_macro_reasoning(
    label: str,
    score: float,
    drivers: List[str],
    confidence: str,
    article_count: int,
    change: float,
    has_previous: bool,
    prev_label: str,
    categories_breakdown: Dict[str, int],
) -> str:
    """
    Generates an investor-grade explanation of the current macro state.
    Changes on every pipeline run — never static placeholder text.

    Format:
      "Macro climate is {label} based on {N} Nepal financial headlines ({confidence}
       confidence). Signal is {direction}. Key driver(s): {...}. {state_change_text}"
    """
    direction = (
        "improving"  if change >  0.01 else
        "weakening"  if change < -0.01 else
        "broadly unchanged"
    )

    # Driver text
    if len(drivers) == 0:
        driver_text = " No strong directional drivers identified."
    elif len(drivers) == 1:
        driver_text = f" Key driver: {drivers[0]}."
    else:
        driver_text = f" Key drivers: {drivers[0]} and {drivers[1]}."

    # State transition text
    state_text = ""
    if has_previous and abs(change) >= 0.02:
        change_str = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
        if prev_label != label:
            state_text = f" Signal shifted from {prev_label} to {label} ({change_str})."
        else:
            state_text = f" Score moved {change_str} from prior reading."

    # Dominant category
    dominant = (
        max(categories_breakdown, key=categories_breakdown.get)
        if categories_breakdown else "market"
    )

    count_text = f"{article_count} Nepal financial headline{'s' if article_count != 1 else ''}"

    return (
        f"Macro climate is {label} based on {count_text} ({confidence.lower()} confidence). "
        f"Signal is {direction}, with dominant coverage from the {dominant} sector."
        f"{driver_text}{state_text}"
    )


def build_delta_intelligence(
    new_label: str,
    prev_label: str,
    new_score: float,
    prev_score: float,
    new_drivers: List[str],
    prev_drivers: List[str],
    article_delta: Dict[str, int],
) -> str:
    """
    Generate natural-language delta explanation comparing the current macro
    state with the previous one. Used as the primary delta_summary text.

    Produces coherent analyst-style sentences with connective language rather
    than stacked em-dash fragments. Mixed-signal states are surfaced honestly
    using contrasting connectors like "although", "though", or "despite".

    Magnitude mapping (abs score change):
      < 0.02   → "remained broadly unchanged" / "held steady"
      0.02–0.08 → "slightly" / "modestly"
      0.08–0.15 → "moderately"
      > 0.15   → "sharply"

    Examples:
      "Sentiment weakened slightly (-0.06), although early recovery signals
       remain visible in select segments (4 new articles reviewed)."
      "Signal remained broadly unchanged, with regulatory tightening
       offsetting pockets of recovery."
      "Market climate shifted from Neutral to Mildly Bullish, supported by
       emerging liquidity easing (4 new articles reviewed)."
    """
    score_change = new_score - prev_score
    abs_change   = abs(score_change)

    # ── 1. Determine primary direction ────────────────────────────────────────
    label_changed = (new_label != prev_label)

    if label_changed:
        # Rank labels so we can tell if the shift is up or down
        _rank = {
            "Bearish":      0, "Mildly Bearish": 1, "Neutral":       2,
            "Mildly Bullish": 3, "Bullish":      4,
        }
        prev_rank   = _rank.get(prev_label, 2)
        new_rank    = _rank.get(new_label,  2)
        is_improved = new_rank > prev_rank
        is_weakened = new_rank < prev_rank
        primary     = f"Market climate shifted from {prev_label} to {new_label}"
    elif abs_change >= 0.02:
        if score_change > 0:
            if score_change < 0.08:    magnitude = "slightly" if score_change < 0.05 else "modestly"
            elif score_change < 0.15:  magnitude = "moderately"
            else:                      magnitude = "sharply"
            primary     = f"Sentiment improved {magnitude} ({score_change:+.2f})"
            is_improved, is_weakened = True, False
        else:
            if abs_change < 0.08:    magnitude = "slightly" if abs_change < 0.05 else "modestly"
            elif abs_change < 0.15:  magnitude = "moderately"
            else:                    magnitude = "sharply"
            primary     = f"Sentiment weakened {magnitude} ({score_change:+.2f})"
            is_improved, is_weakened = False, True
    else:
        primary      = "Signal held steady"
        is_improved  = False
        is_weakened  = False

    # ── 2. Classify emerged/faded drivers as aligned or opposing ──────────────
    new_d_set = set(new_drivers)
    prev_d_set = set(prev_drivers)
    emerged   = list(new_d_set - prev_d_set)
    faded     = list(prev_d_set - new_d_set)

    # Keyword sets for quick polarity detection inside a driver phrase
    _positive_kw = {
        "recovery", "easing", "growth", "inflow", "rally", "improvement",
        "momentum", "expansion", "catalysts", "liquidity", "rebound",
    }
    _negative_kw = {
        "tightening", "weakness", "pressure", "contraction", "concern",
        "regulatory", "slowdown", "friction", "risk", "caution", "uncertainty",
        "stress", "restriction",
    }

    def _polarity(driver: str) -> int:
        """Return +1 (positive), -1 (negative), or 0 (neutral) for a driver."""
        words = set(driver.lower().split())
        pos = bool(words & _positive_kw)
        neg = bool(words & _negative_kw)
        if pos and not neg: return  1
        if neg and not pos: return -1
        return 0

    def _clean_driver(raw: str) -> str:
        """Strip leading 'market ' to prevent doubled nouns (e.g. 'market recovery momentum')."""
        import re
        cleaned = re.sub(r'^market\s+', '', raw.strip(), flags=re.IGNORECASE)
        return cleaned

    driver_clause = ""
    if emerged:
        top = _clean_driver(emerged[0])
        pol = _polarity(top)
        if is_weakened and pol > 0:
            # Score fell but a positive driver emerged — classic mixed state
            driver_clause = f"although early recovery signals remain visible in select segments"
        elif is_improved and pol < 0:
            # Score rose but negative driver emerged — caution warranted
            driver_clause = f"though {top} still limits broader conviction"
        elif is_weakened and pol < 0:
            # Score fell and driver is also negative — aligned downside
            driver_clause = f"with {top} reinforcing the downside pressure"
        elif is_improved and pol > 0:
            # Score rose and driver is positive — aligned upside
            driver_clause = f"supported by {top}"
        elif not is_improved and not is_weakened:
            # Signal flat but a driver shifted — surface it as context
            driver_clause = f"with {top} reshaping the underlying mix"
        else:
            driver_clause = f"with {top} emerging as a contributing factor"
    elif faded:
        top = _clean_driver(faded[0])
        driver_clause = f"as {top} fades from the primary signal"

    # ── 3. Article volume — kept as a parenthetical, not a headline clause ────
    new_count = article_delta.get("new", 0)
    rem_count = article_delta.get("removed", 0)
    if new_count > 0 and rem_count > 0:
        volume_note = f"{new_count} new / {rem_count} removed articles"
    elif new_count > 0:
        volume_note = f"{new_count} new article{'s' if new_count > 1 else ''} reviewed"
    elif rem_count > 0:
        volume_note = f"{rem_count} article{'s' if rem_count > 1 else ''} dropped from pool"
    else:
        volume_note = ""

    # ── 4. Assemble: one coherent sentence ────────────────────────────────────
    sentence = primary
    if driver_clause:
        sentence = f"{primary}, {driver_clause}"
    if volume_note:
        sentence = f"{sentence} ({volume_note})"

    return sentence.rstrip(".") + "."

