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

class MarketMoodResponse(BaseModel):
    average_score: float
    mood_label: str
    headline_count: int
    latest_headlines: List[str]
    macro_sentiment: str
    score: float
    drivers: List[str]
    last_updated: Optional[str] = None
    refresh_interval_seconds: Optional[int] = 180
    drift: Optional[Drift] = None
    delta_summary: Optional[str] = None
    macro_confidence: Optional[str] = None
    macro_layer_status: Optional[MacroLayerStatus] = None
    # --- Diagnostic / debug fields ---
    cache_hit: Optional[bool] = None
    raw_score: Optional[float] = None
    smoothed_score: Optional[float] = None
    previous_score: Optional[float] = None
    filtered_article_count: Optional[int] = None
    analyzed_article_count: Optional[int] = None
    query_used: Optional[str] = None
    categories_breakdown: Optional[Dict[str, int]] = None
    analyzed_headlines: Optional[List[str]] = None
    source_breakdown: Optional[Dict[str, int]] = None
    source_change_status: Optional[str] = None   # "No new relevant articles" | "New relevant coverage detected"

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
    cache_hit: Optional[bool] = None   # True when served from in-memory cache

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
