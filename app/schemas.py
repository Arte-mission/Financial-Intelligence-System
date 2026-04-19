from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class Drift(BaseModel):
    status: str
    change: float

class MacroLayerStatus(BaseModel):
    market_pulse: str            # Bullish / Neutral / Bearish — always computed from market headlines
    macro_climate: str           # Bullish / Neutral / Bearish / "Insufficient Coverage"
    market_article_count: int
    macro_article_count: int     # banking + macro combined
    coverage_sufficient: bool    # True if banking+macro >= MIN_MACRO_ARTICLES (3)

class SignalTimelineEntry(BaseModel):
    """One distinct macro state transition stored in the in-memory timeline."""
    timestamp: str
    score: float
    label: str
    drivers: List[str] = []
    article_count: int = 0
    change_reason: Optional[str] = None


class MarketMoodResponse(BaseModel):
    average_score: float
    mood_label: str
    headline_count: int
    latest_headlines: List[str]
    macro_sentiment: str
    score: float
    drivers: List[str]
    last_checked: Optional[str] = None
    last_updated: Optional[str] = None
    refresh_interval_seconds: Optional[int] = 180
    drift: Optional[Drift] = None
    delta_summary: Optional[str] = None
    macro_confidence: Optional[str] = None
    macro_layer_status: Optional[MacroLayerStatus] = None
    # --- System performance metadata ---
    cache_hit: Optional[bool] = None
    data_freshness: Optional[str] = None  # "fresh" | "valid" | "stale"
    raw_score: Optional[float] = None
    smoothed_score: Optional[float] = None
    previous_score: Optional[float] = None
    filtered_article_count: Optional[int] = None
    analyzed_article_count: Optional[int] = None
    query_used: Optional[str] = None
    categories_breakdown: Optional[Dict[str, int]] = None
    analyzed_headlines: Optional[List[str]] = None
    source_breakdown: Optional[Dict[str, int]] = None
    source_change_status: Optional[str] = None
    source_change: Optional[bool] = None
    # ── Sprint 2: timeline, reasoning, article delta ────────────────────────
    signal_timeline: Optional[List["SignalTimelineEntry"]] = None
    macro_reasoning: Optional[str] = None
    article_delta: Optional[Dict[str, int]] = None
    # ── Sprint 5: System transparency and refresh cycle ─────────────────────
    raw_article_count: Optional[int] = None
    relevant_article_count: Optional[int] = None
    freshest_article_at: Optional[str] = None
    next_check_at: Optional[str] = None

class MarketMoodSnapshotResponse(BaseModel):
    id: int
    average_score: float
    mood_label: str
    headline_count: int
    created_at: datetime

class Headline(BaseModel):
    title: str
    url: Optional[str] = None

class CompanySignal(BaseModel):
    source: str
    title: str
    url: Optional[str] = None

class CompanyAnalysisItem(BaseModel):
    text: str
    sentiment: str
    materiality: str
    weight: float

class CompanyAnalysis(BaseModel):
    items: List[CompanyAnalysisItem]
    weighted_company_score: float

class AICompanySignal(BaseModel):
    label: str
    score: float
    impact_summary: str

class SignalAlignment(BaseModel):
    status: str
    note: str

class TechnicalIndicators(BaseModel):
    ma5:    Optional[str] = None    # 5-day moving average price
    ma20:   Optional[str] = None    # 20-day moving average price
    ma180:  Optional[str] = None    # 180-day moving average price
    ma_signal: Optional[str] = None  # "Bullish" | "Bearish" | "Neutral" | "Mixed"
    ma_note:   Optional[str] = None  # human-readable interpretation
    ma_score:  Optional[float] = None  # [-1, 1] contribution to final sentiment

class MADetail(BaseModel):
    """Single moving-average level with scraped or inferred signal."""
    value: Optional[float] = None
    signal: Optional[str] = None   # "Bullish" | "Neutral" | "Bearish"


class MovingAnalysis(BaseModel):
    """All three MA levels from the Sharesansar Moving Analysis table."""
    ma5:   Optional[MADetail] = None
    ma20:  Optional[MADetail] = None
    ma180: Optional[MADetail] = None


class PivotSupport(BaseModel):
    s1: Optional[float] = None
    s2: Optional[float] = None
    s3: Optional[float] = None


class PivotResistance(BaseModel):
    r1: Optional[float] = None
    r2: Optional[float] = None
    r3: Optional[float] = None


class PivotAnalysis(BaseModel):
    """Classic pivot point levels scraped from Sharesansar."""
    pp:         Optional[float] = None
    support:    Optional[PivotSupport] = None
    resistance: Optional[PivotResistance] = None


class TechnicalAnalysis(BaseModel):
    """
    Combined technical intelligence block returned at the top level of StockResponse.
    moving_score   = weighted blend of MA5/MA20/MA180 signals
    pivot_score    = price position relative to pivot/support/resistance levels
    technical_score = 0.65 * moving_score + 0.35 * pivot_score
    """
    moving_analysis:  Optional[MovingAnalysis] = None
    pivot_analysis:   Optional[PivotAnalysis]  = None
    moving_score:     Optional[float] = None
    pivot_score:      Optional[float] = None
    pivot_signal:     Optional[str]   = None   # "Mildly Bullish" | "Neutral" | etc.
    technical_score:  Optional[float] = None
    technical_summary: Optional[str]  = None


class Alert(BaseModel):
    """Proactive notification of meaningful state changes (Sentiment flip, Technical breakout, etc)."""
    type: str       # "sentiment" | "technical" | "news"
    message: str
    severity: str   # "high" | "medium" | "low"

class WhatChanged(BaseModel):
    """Concise explanation of deltas between current and previous analysis states."""
    summary: str
    changes: List[str]

class Insight(BaseModel):
    stock_sentiment: str
    confidence: str
    confidence_context: Optional[str] = None
    reasoning: str
    drivers: List[str]
    company_analysis: Optional[CompanyAnalysis] = None
    ai_company_signal: Optional[AICompanySignal] = None
    signal_alignment: Optional[SignalAlignment] = None
    technical_indicators: Optional[TechnicalIndicators] = None

class StockResponse(BaseModel):
    ticker: str
    company_name: str
    sector: str
    current_price: Optional[str] = None
    price_change: Optional[str] = None
    price_change_percent: Optional[str] = None
    fifty_two_week_high: Optional[str] = None
    fifty_two_week_low: Optional[str] = None
    ma5:   Optional[str] = None   # 5-day moving average
    ma20:  Optional[str] = None   # 20-day moving average
    ma180: Optional[str] = None   # 180-day moving average
    headline: Optional[Headline] = None
    company_signal: Optional[CompanySignal] = None
    insight: Optional[Insight] = None
    # ── Sprint 3: technical analysis block ───────────────────────────────
    technical_analysis: Optional[TechnicalAnalysis] = None
    # ── Sprint 4: Active Intelligence ────────────────────────────────────
    alerts: List[Alert] = []
    what_changed: Optional[WhatChanged] = None
    # ── System-level performance metadata ──────────────────────────────────
    cache_hit: Optional[bool] = None       # True when served from in-memory cache
    last_updated: Optional[str] = None    # ISO 8601 UTC timestamp of when data was last computed
    source_change: Optional[bool] = None  # True when Sharesansar returned fresh data vs cache reuse
    data_freshness: Optional[str] = None  # "fresh" | "valid" | "stale" — tells frontend data age tier

class CompanyResponse(BaseModel):
    symbol: str
    company_name: str
    sector: str
    instrument_type: str

class NewsArticleCreate(BaseModel):
    title: str
    source: str
    published_at: Optional[str] = None
    sentiment_score: Optional[float] = None
    query_used: Optional[str] = None

# ── Sprint 5: Watchlist Intelligence ───────────────────────────────
class WatchlistFeedItem(BaseModel):
    ticker: str
    headline: str
    summary: str
    severity: str # "high" | "medium" | "low"
    timestamp: str # last_updated

class WatchlistFeedRequest(BaseModel):
    tickers: List[str]

class WatchlistFeedResponse(BaseModel):
    items: List[WatchlistFeedItem]
