from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any
import logging
import time
import threading

from app.schemas import (
    StockResponse, CompanyResponse, TechnicalIndicators,
    TechnicalAnalysis, MovingAnalysis, MADetail,
    PivotAnalysis, PivotSupport, PivotResistance,
    Alert, WhatChanged,
    WatchlistFeedItem, WatchlistFeedRequest, WatchlistFeedResponse
)
from app.models import Company
from app.services.sharesansar_service import get_company_data, _fetch_summary
from app.services.ticker_service import get_company_by_ticker, get_all_companies, search_companies
from app.database import get_db, SessionLocal

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Stock Info"])

# ── Stock response cache ──────────────────────────────────────────────────
# Keyed by ticker (uppercase). Avoids re-running the full scrape + AI pipeline
# on every request. TTL is set to 12 minutes — fast enough to catch intra-day
# company events while eliminating redundant Playwright/Gemini calls.
#
# Stale-while-revalidate strategy:
#   [0 .. STOCK_CACHE_TTL]          → "valid"  — return instantly
#   [STOCK_CACHE_TTL .. STALE_SERVE_TTL] → "stale" — return instantly + background refresh
#   [> STALE_SERVE_TTL]             → expired  — must block and recompute (cold start only)
STOCK_CACHE_TTL    = 720    # seconds (12 min)  — data considered fully fresh
STALE_SERVE_TTL    = 1440   # seconds (24 min)  — stale data still served while refresh runs
STOCK_CACHE: Dict[str, Dict] = {}   # { ticker: {"data": StockResponse dict, "ts": float} }

# ── Background revalidation lock ─────────────────────────────────────────
# Prevents multiple threads from scraping the same ticker simultaneously.
_REVALIDATING: set = set()   # set of ticker_upper strings currently being refreshed
_REVALIDATE_LOCK = threading.Lock()

AI_CACHE_TTL = 1800     # seconds (30 min)
AI_SIGNAL_CACHE: Dict[str, Dict] = {}   # { title_key: {"signal": dict, "ts": float} }

# ── Sprint 4: Active Intelligence History ─────────────────────────────
# Stores the PREVIOUS state of a stock analysis to allow delta comparison.
# This enables the "What Changed" engine and "Alerts" triggers.
# History is updated whenever a stale cache entry is revalidated.
STOCK_HISTORY: Dict[str, StockResponse] = {} # { ticker: StockResponse }


@router.get("/tickers/debug", response_model=Dict[str, Any])
def debug_tickers(db: Session = Depends(get_db)):
    """
    Returns debug information about the seeded companies.
    """
    total = db.query(Company).count()
    first_20_records = db.query(Company).limit(20).all()
    first_20 = [c.symbol for c in first_20_records]
    
    sectors_count = db.query(Company.sector, func.count(Company.id)).group_by(Company.sector).all()
    sector_summary = {sector if sector else "Unknown": count for sector, count in sectors_count}
    
    return {
        "total_companies": total,
        "first_20_symbols": first_20,
        "count_by_sector": sector_summary
    }

@router.get("/tickers", response_model=List[CompanyResponse])
def get_tickers(db: Session = Depends(get_db)):
    """
    Returns the full allowed ticker list (equity only).
    """
    companies = get_all_companies(db)
    return companies

@router.get("/tickers/search", response_model=List[CompanyResponse])
def search_tickers(q: str, db: Session = Depends(get_db)):
    """
    Returns matching tickers for autocomplete by symbol or company name.
    """
    companies = search_companies(db, q)
    return companies

@router.post("/watchlist/intelligence", response_model=WatchlistFeedResponse)
def get_watchlist_intelligence(req: WatchlistFeedRequest):
    """
    Aggregates existing alerts and deltas for a list of tickers.
    Strict cache-only lookup; no background revalidation here to ensure speed.
    """
    items = []
    for ticker in req.tickers:
        t_upper = ticker.upper()
        cached = STOCK_CACHE.get(t_upper)
        if not cached:
            continue
            
        data = cached["data"]
        # Alerts and what_changed might be dicts (if from cache) or objects
        alerts = data.get("alerts", [])
        wc = data.get("what_changed")
        
        # Only include stocks that actually have meaningful intelligence
        if not alerts and not wc:
            continue
            
        # Prioritize Alert message for Feed Headline
        headline = ""
        if alerts:
            # Type-safe access whether it's an object or a dict
            first_alert = alerts[0]
            headline = first_alert.message if hasattr(first_alert, "message") else first_alert.get("message", "System Alert")
        elif wc:
            headline = wc.summary if hasattr(wc, "summary") else wc.get("summary", "Analysis Updated")
        else:
            headline = "No recent material changes"

        # Summary detail
        summary_txt = ""
        if wc:
            summary_txt = wc.summary if hasattr(wc, "summary") else wc.get("summary", "")
        elif alerts:
            summary_txt = alerts[0].message if hasattr(alerts[0], "message") else alerts[0].get("message", "")

        # Highest Severity Mapping
        severity = "low"
        def get_sev(a): return a.severity if hasattr(a, "severity") else a.get("severity", "low")
        
        if any(get_sev(a) == "high" for a in alerts):
            severity = "high"
        elif any(get_sev(a) == "medium" for a in alerts):
            severity = "medium"
            
        items.append(WatchlistFeedItem(
            ticker=t_upper,
            headline=headline,
            summary=summary_txt,
            severity=severity,
            timestamp=data.get("last_updated", "")
        ))
        
    # Sort logic: High Severity first, then by timestamp recency
    severity_score = {"high": 3, "medium": 2, "low": 1}
    items.sort(key=lambda x: (severity_score.get(x.severity, 0), x.timestamp), reverse=True)
    
    return WatchlistFeedResponse(items=items[:10])

@router.get("/stock/{ticker}", response_model=StockResponse)
def get_stock_data(ticker: str, db: Session = Depends(get_db)):
    """
    Retrieves and scores company data from Sharesansar.
    """
    ticker_upper = ticker.upper()

    company = get_company_by_ticker(db, ticker)
    if not company:
        raise HTTPException(
            status_code=400,
            detail="This instrument is not supported. Only equity stocks are allowed."
        )

    cached = STOCK_CACHE.get(ticker_upper)
    if cached:
        age = time.time() - cached["ts"]
        
        # ── Tier 1: Valid (Fresh) ──
        if age < STOCK_CACHE_TTL:
            logger.info(f"Stock cache hit for {ticker_upper} (age {int(age)}s, FRESH)")
            result = cached["data"].copy()

            # LAYER B: Lightweight real-time market sync
            try:
                live_market = _fetch_summary(ticker_upper)
                if live_market:
                    result["current_price"] = live_market.get("current_price") or result.get("current_price")
                    result["price_change"] = live_market.get("day_change") or result.get("price_change")
                    result["price_change_percent"] = live_market.get("day_change_pct") or result.get("price_change_percent")
                    result["ma5"] = live_market.get("ma5") or result.get("ma5")
                    result["ma20"] = live_market.get("ma20") or result.get("ma20")
                    result["ma180"] = live_market.get("ma180") or result.get("ma180")
                    
                    new_ta = _compute_live_technical_analysis(live_market, ticker_upper)
                    if new_ta:
                        result["technical_analysis"] = new_ta.model_dump()
            except Exception as e:
                logger.warning(f"Failed to stitch live market layer for {ticker_upper}: {e}")

            result["cache_hit"] = True
            result["source_change"] = False
            result["data_freshness"] = "fresh"
            return StockResponse(**result)
            
        # ── Tier 2: Stale, but serve while revalidating ──
        if age < STALE_SERVE_TTL:
            logger.info(f"Stock cache stale for {ticker_upper} (age {int(age)}s, REVALIDATING)")
            
            with _REVALIDATE_LOCK:
                if ticker_upper not in _REVALIDATING:
                    _REVALIDATING.add(ticker_upper)
                    
                    # ── Sprint 4: Save current state to history before revalidation ──
                    if ticker_upper in STOCK_CACHE:
                        try:
                            prev_data = STOCK_CACHE[ticker_upper]["data"]
                            STOCK_HISTORY[ticker_upper] = StockResponse(**prev_data)
                            logger.info(f"Previous state moved to history for {ticker_upper}")
                        except Exception as h_err:
                            logger.warning(f"Failed to save {ticker_upper} history: {h_err}")

                    thread = threading.Thread(
                        target=_revalidate_stock_async,
                        args=(ticker, ticker_upper, company.company_name, company.sector)
                    )
                    thread.start()
                    
            result = cached["data"].copy()

            # LAYER B: Lightweight real-time market sync
            try:
                live_market = _fetch_summary(ticker_upper)
                if live_market:
                    result["current_price"] = live_market.get("current_price") or result.get("current_price")
                    result["price_change"] = live_market.get("day_change") or result.get("price_change")
                    result["price_change_percent"] = live_market.get("day_change_pct") or result.get("price_change_percent")
                    result["ma5"] = live_market.get("ma5") or result.get("ma5")
                    result["ma20"] = live_market.get("ma20") or result.get("ma20")
                    result["ma180"] = live_market.get("ma180") or result.get("ma180")
                    
                    new_ta = _compute_live_technical_analysis(live_market, ticker_upper)
                    if new_ta:
                        result["technical_analysis"] = new_ta.model_dump()
            except Exception as e:
                logger.warning(f"Failed to stitch live market layer for {ticker_upper}: {e}")

            result["cache_hit"] = True
            result["source_change"] = False
            result["data_freshness"] = "stale"
            return StockResponse(**result)

    # ── Tier 3: Expired or Cold Start (Blocking Mode) ──
    logger.info(f"Stock cache expired/missing for {ticker_upper} — computing synchronously.")
    return _process_stock_data(ticker, ticker_upper, company.company_name, company.sector, db)


def _revalidate_stock_async(ticker: str, ticker_upper: str, company_name: str, sector: str):
    """Background thread to refresh stock data with its own DB session."""
    try:
        db = SessionLocal()
        try:
            _process_stock_data(ticker, ticker_upper, company_name, sector, db)
            logger.info(f"Background revalidation complete for {ticker_upper}")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Background revalidation failed for {ticker_upper}: {e}")
    finally:
        with _REVALIDATE_LOCK:
            _REVALIDATING.discard(ticker_upper)


def _process_stock_data(ticker: str, ticker_upper: str, company_name: str, sector: str, db: Session) -> StockResponse:
    data = get_company_data(ticker)

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Company with ticker '{ticker_upper}' data unavailable on Sharesansar."
        )
        
    # Inject database metadata into the response
    data["company_name"] = company_name
    data["sector"] = sector
    # Affirmatively ensure the properties are listed even if missing
    if "headline" not in data:
        data["headline"] = None
    if "company_signal" not in data:
        data["company_signal"] = None

    from app.models import MarketMoodSnapshot
    
    # 1. Market Sentiment Input
    latest_mood = db.query(MarketMoodSnapshot).order_by(MarketMoodSnapshot.created_at.desc()).first()
    market_score = latest_mood.average_score if latest_mood else 0.0
    
    if market_score > 0.2:
        market_sentiment = "Bullish"
    elif market_score < -0.2:
        market_sentiment = "Bearish"
    else:
        market_sentiment = "Neutral"
        
    drivers = [f"Market sentiment: {market_sentiment}"]
    
    # 2. Company Event Classification (Based on AGM Agenda/Primary Signal)
    headline_sentiment = "Neutral"
    event_reason = "No recent signals to deeply impact sentiment."
    company_source = data.get("company_signal", {}).get("source", "company signal") if data.get("company_signal") else "company signal"
    company_analysis = None
    if data.get("company_signal") and data["company_signal"].get("title"):
        raw_title = data["company_signal"]["title"]
        
        # Typo Normalization Step (Correcting typical Sharesansar scrape errors before processing)
        obvious_typos = {
            "Acquisiyion": "Acquisition",
            "acquisiyion": "acquisition"
        }
        for wrong, right in obvious_typos.items():
            raw_title = raw_title.replace(wrong, right)
            
        data["company_signal"]["title"] = raw_title
        # Splitting using exhaustive delimiters: enumerators, commas, explicit semicolons, or sentence breaks
        import re
        raw_items = re.split(r'\d+\.\s|;|\s*,\s*|(?<=[a-z])\.\s', raw_title)
        
        items = []
        total_weight = 0.0
        weighted_score = 0.0
        material_event_desc = []
        
        for item in raw_items:
            text = item.strip()
            if len(text) < 5:
                continue
                
            text_lower = text.lower()
            
            # Determine Materiality
            if any(w in text_lower for w in ["dividend", "bonus", "right", "rights", "merger", "acquisition", "capital", "penalty", "loss", "warning"]):
                materiality = "High"
                weight = 1.0
            elif any(w in text_lower for w in ["financial", "governance", "strategic", "restructuring", "profit"]):
                materiality = "Medium"
                weight = 0.5
            else:
                materiality = "Low"
                weight = 0.1
                
            # Determine Sentiment
            if any(w in text_lower for w in ["dividend", "bonus", "profit", "right share", "ipo"]):
                sentiment = "Positive"
                score_val = 1.0
            elif any(w in text_lower for w in ["merger", "acquisition"]):
                sentiment = "Positive"
                score_val = 0.5
            elif any(w in text_lower for w in ["issue", "loss", "penalty", "decline", "suspend", "resignation"]):
                sentiment = "Negative"
                score_val = -1.0
            else:
                sentiment = "Neutral"
                score_val = 0.0
                
            if materiality == "High" or (materiality == "Medium" and sentiment != "Neutral"):
                material_event_desc.append(text)
                
            weighted_score += (score_val * weight)
            total_weight += weight
            
            items.append({
                "text": text,
                "sentiment": sentiment,
                "materiality": materiality,
                "weight": weight
            })
            
        final_weighted_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Decide overall signal explicitly
        if final_weighted_score >= 0.3:
            headline_sentiment = "Positive"
        elif final_weighted_score <= -0.3:
            headline_sentiment = "Negative"
        else:
            headline_sentiment = "Neutral"
            
        if material_event_desc:
            event_reason = f"Material constraints detected: {material_event_desc[0]}."
            drivers.append("Company signal: Contains actionable material factors")
        else:
            event_reason = "Agenda consists primarily of routine/informational items."
            drivers.append("Company signal: Routine or procedural")
            
        company_analysis = {
            "items": items,
            "weighted_company_score": round(final_weighted_score, 2)
        }
    else:
        drivers.append("Company signal: None recent")
        
    # 4. Moving Average Technical Score
    # ── MA Signal Logic ────────────────────────────────────────────────────
    # Three layers, each capped at ±1.0:
    #   Short-term : price vs MA5 / MA20        (weight 0.40)
    #   Cross      : MA5 vs MA20 (golden/death) (weight 0.35)
    #   Long-term  : price vs MA180             (weight 0.25)
    # Final ma_score in [-1, 1] — contributes ma_weight=0.15 to final score.
    ma_score      = 0.0
    ma_signal     = "Neutral"
    ma_note       = None
    ma_layers_hit = 0
    technical_indicators = None

    try:
        cp_str   = data.get("current_price")
        ma5_str  = data.get("ma5")
        ma20_str = data.get("ma20")
        ma180_str= data.get("ma180")

        cp    = float(cp_str.replace(",", ""))   if cp_str   else None
        ma5   = float(ma5_str.replace(",", ""))  if ma5_str  else None
        ma20  = float(ma20_str.replace(",", "")) if ma20_str else None
        ma180 = float(ma180_str.replace(",", ""))if ma180_str else None

        layer_score = 0.0

        # Short-term: price vs MA5 and MA20
        if cp and ma5 and ma20:
            ma_layers_hit += 1
            above5  = cp > ma5
            above20 = cp > ma20
            if above5 and above20:
                layer_score += 0.40 * 1.0    # price above both → bullish
            elif above5 and not above20:
                layer_score += 0.40 * 0.2    # only above MA5 → mildly bullish
            elif not above5 and above20:
                layer_score += 0.40 * -0.2   # only below MA5 → mildly bearish
            else:
                layer_score += 0.40 * -1.0   # below both → bearish

        # Cross signal: MA5 vs MA20
        if ma5 and ma20:
            ma_layers_hit += 1
            if ma5 > ma20 * 1.01:            # golden cross (1% buffer to avoid noise)
                layer_score += 0.35 * 1.0
            elif ma5 < ma20 * 0.99:          # death cross
                layer_score += 0.35 * -1.0
            # else: MAs converging → no contribution

        # Long-term: price vs MA180
        if cp and ma180:
            ma_layers_hit += 1
            if cp > ma180 * 1.05:            # >5% above 180d MA → bullish trend
                layer_score += 0.25 * 1.0
            elif cp > ma180:
                layer_score += 0.25 * 0.4    # slightly above
            elif cp < ma180 * 0.95:          # >5% below → bearish trend
                layer_score += 0.25 * -1.0
            else:
                layer_score += 0.25 * -0.4   # slightly below

        if ma_layers_hit > 0:
            ma_score = max(-1.0, min(1.0, layer_score))   # clamp

            # Label
            if ma_score >= 0.40:
                ma_signal = "Bullish"
            elif ma_score >= 0.15:
                ma_signal = "Mildly Bullish"
            elif ma_score <= -0.40:
                ma_signal = "Bearish"
            elif ma_score <= -0.15:
                ma_signal = "Mildly Bearish"
            else:
                ma_signal = "Neutral"

            # Human-readable note
            parts = []
            if cp and ma5:   parts.append(f"Price {'above' if cp > ma5 else 'below'} MA5")
            if cp and ma20:  parts.append(f"{'above' if cp > ma20 else 'below'} MA20")
            if ma5 and ma20: parts.append(f"MA5 {'>' if ma5 > ma20 else '<'} MA20")
            if cp and ma180: parts.append(f"price {'above' if cp > ma180 else 'below'} MA180")
            ma_note = "; ".join(parts) if parts else None

            drivers.append(f"MA Signal: {ma_signal} ({ma_note or 'partial data'})")
            logger.info(f"MA score for {ticker_upper}: {ma_score:.3f} ({ma_signal})")

        technical_indicators = TechnicalIndicators(
            ma5    = ma5_str,
            ma20   = ma20_str,
            ma180  = ma180_str,
            ma_signal = ma_signal,
            ma_note   = ma_note,
            ma_score  = round(ma_score, 3)
        )

    except Exception as ma_err:
        logger.warning(f"MA scoring failed for {ticker_upper}: {ma_err}")
        ma_score = 0.0

    # ── Sprint 3: Build structured moving_analysis block ─────────────────────
    # Uses Sharesansar-scraped signals when available; falls back to price-vs-MA.
    def _infer_ma_signal(price_val, ma_val):
        if not price_val or not ma_val:
            return None
        try:
            diff_pct = (price_val - ma_val) / ma_val
            if diff_pct > 0.01:    return "Bullish"
            elif diff_pct < -0.01: return "Bearish"
            return "Neutral"
        except Exception:
            return None

    moving_analysis_obj = None
    try:
        _cp_f = float(data.get("current_price", "0").replace(",", "")) if data.get("current_price") else None
        _ma5_f  = float(data.get("ma5",  "0").replace(",", "")) if data.get("ma5")  else None
        _ma20_f = float(data.get("ma20", "0").replace(",", "")) if data.get("ma20") else None
        _ma180_f= float(data.get("ma180","0").replace(",", "")) if data.get("ma180") else None

        moving_analysis_obj = MovingAnalysis(
            ma5  = MADetail(
                value  = _ma5_f,
                signal = data.get("ma5_signal")  or _infer_ma_signal(_cp_f, _ma5_f),
            ),
            ma20 = MADetail(
                value  = _ma20_f,
                signal = data.get("ma20_signal") or _infer_ma_signal(_cp_f, _ma20_f),
            ),
            ma180= MADetail(
                value  = _ma180_f,
                signal = data.get("ma180_signal")or _infer_ma_signal(_cp_f, _ma180_f),
            ),
        )
    except Exception as mao_err:
        logger.warning(f"moving_analysis_obj build failed for {ticker_upper}: {mao_err}")

    # ── Sprint 3: Pivot analysis — scrape-based score + structured response ──
    pivot_score    = 0.0
    pivot_signal   = "Neutral"
    pivot_analysis_obj = None
    pivot_explanation = ""
    try:
        def _pf(key):  # safe float parser for pivot fields
            v = data.get(key)
            return float(v.replace(",", "")) if v else None

        pp  = _pf("pivot_pp")
        ps1 = _pf("pivot_s1"); ps2 = _pf("pivot_s2"); ps3 = _pf("pivot_s3")
        pr1 = _pf("pivot_r1"); pr2 = _pf("pivot_r2"); pr3 = _pf("pivot_r3")

        pivot_analysis_obj = PivotAnalysis(
            pp=pp,
            support    = PivotSupport(s1=ps1, s2=ps2, s3=ps3),
            resistance = PivotResistance(r1=pr1, r2=pr2, r3=pr3),
        )

        cp_pf = _pf("current_price") or (float(data["current_price"].replace(",","")) if data.get("current_price") else None)

        if cp_pf and pp:
            if pr2 and cp_pf >= pr2:
                pivot_score = -0.30;  pivot_signal = "Mildly Bearish"   # at/above R2 = overbought zone
                pivot_explanation = f"price is trading well above resistance ({pr2}), indicating an overbought zone. This suggests short-term momentum may be overextended, warning of potential exhaustion despite the rally."
            elif pr1 and cp_pf >= pr1:
                pivot_score = -0.15;  pivot_signal = "Mildly Bearish"   # near R1 = resistance
                pivot_explanation = f"price is trading near first resistance ({pr1}). Upside momentum may face friction here, requiring a clear breakout to justify stronger confidence."
            elif cp_pf >= pp:
                pivot_score =  0.20;  pivot_signal = "Mildly Bullish"   # above PP = bullish bias
                pr1_note = f" (below {pr1})" if pr1 else ""
                pivot_explanation = f"price is trading above the central pivot point ({pp}){pr1_note}. This reflects a constructive short-term trend, though clearing overhead resistance is needed to fully confirm the upside bias."
            elif ps1 and cp_pf >= ps1:
                pivot_score = -0.20;  pivot_signal = "Mildly Bearish"   # below PP above S1
                pivot_explanation = f"price is trading below the central pivot ({pp}), pointing to mild near-term weakness. However, holding above first support ({ps1}) limits immediate downside risk."
            elif ps2 and cp_pf >= ps2:
                pivot_score =  0.10;  pivot_signal = "Neutral"          # near S1/S2 support
                pivot_explanation = f"price is trading near structural support levels ({ps2}), which is cushioning the current decline. Buyers successfully defending this zone would improve broader technical stability."
            else:
                pivot_score =  0.25;  pivot_signal = "Mildly Bullish"   # deep support zone
                sup_limit = ps2 or ps1 or 'support'
                pivot_explanation = f"price is trading in a deep support zone (below {sup_limit}). While momentum is decidedly weak, this extreme level may eventually cap further downside if market sentiment stabilizes."
            pivot_score = max(-1.0, min(1.0, pivot_score))
            logger.info(f"Pivot score for {ticker_upper}: {pivot_score:.3f} ({pivot_signal})")

    except Exception as piv_err:
        logger.warning(f"Pivot computation failed for {ticker_upper}: {piv_err}")

    # ── Sprint 3: Combined technical score (MA 65%, Pivot 35%) ───────────────
    # MA is weighted higher because it reflects multi-day trend conviction;
    # pivot is single-day reference useful for intraday zone context.
    has_pivot = pivot_analysis_obj and pivot_analysis_obj.pp is not None
    if ma_layers_hit > 0 and has_pivot:
        technical_score = (ma_score * 0.65) + (pivot_score * 0.35)
    elif ma_layers_hit > 0:
        technical_score = ma_score          # pivot data missing: rely on MA only
    elif has_pivot:
        technical_score = pivot_score       # MA data missing: rely on pivot only
    else:
        technical_score = 0.0
    technical_score = max(-1.0, min(1.0, technical_score))
    logger.info(f"Technical score for {ticker_upper}: {technical_score:.3f}")

    # ── Sprint 3: Technical summary ───────────────────────────────────────────
    def _tech_summary(m_signal, m_note, p_signal, t_score):
        parts = []
        if m_signal and m_signal != "Neutral":
            parts.append(f"Moving averages note a {m_signal.lower()} trend" + (f" ({m_note})" if m_note else "") + ", reflecting broader momentum conditions")
        elif m_note:
            parts.append(f"Moving averages note: {m_note}")
        if pivot_explanation:
            parts.append(pivot_explanation)
        elif has_pivot and p_signal and p_signal != "Neutral":
            parts.append(f"price trades in a {p_signal.lower()} pivot zone, leaving technical conviction moderate")
        elif has_pivot:
            parts.append("price near pivot level")
        if not parts:
            return "Technical indicators are broadly neutral, suggesting waiting for clearer momentum signals."
        return ". ".join(p.capitalize() for p in parts) + "."

    technical_summary_text = _tech_summary(ma_signal, ma_note, pivot_signal, technical_score)

    # Remove duplicate price-position block (keep only the clean version below)

    # 5. Price Position Score (0 to 1 mapped to -0.5 to 0.5)
    price_position_label = "Mid Range"
    price_score = 0.0
    try:
        if data.get("current_price") and data.get("fifty_two_week_high") and data.get("fifty_two_week_low"):
            cp = float(data["current_price"].replace(',', ''))
            high = float(data["fifty_two_week_high"].replace(',', ''))
            low = float(data["fifty_two_week_low"].replace(',', ''))

            if high > low and cp > 0:
                position = (cp - low) / (high - low)
                price_score = (position - 0.5)
                if position > 0.7:
                    price_position_label = "Near High"
                elif position < 0.3:
                    price_position_label = "Near Low"
                drivers.append(f"Price: {price_position_label}")
            else:
                drivers.append("Price: Mid range")
        else:
            drivers.append("Price: Unknown range")
    except (TypeError, ValueError):
        drivers.append("Price: Unknown range")

    # 6. AI Company Signal — gated behind AI_SIGNAL_CACHE to avoid redundant Gemini calls
    ai_company_signal = None
    ai_score = 0.0
    if data.get("company_signal") and data["company_signal"].get("title"):
        title_text = data["company_signal"]["title"]
        title_key  = title_text.lower().strip()

        # Check AI cache first
        ai_cached = AI_SIGNAL_CACHE.get(title_key)
        if ai_cached and (time.time() - ai_cached["ts"] < AI_CACHE_TTL):
            logger.info(f"AI signal cache hit for '{title_key[:40]}...'")
            ai_company_signal = ai_cached["signal"]
            ai_score = ai_company_signal.get("score", 0.0)
        else:
            try:
                from app.services.ai_signal_service import generate_ai_company_signal
                ai_company_signal = generate_ai_company_signal(title_text)
                ai_score = ai_company_signal.get("score", 0.0)
                # Store in AI cache
                AI_SIGNAL_CACHE[title_key] = {"signal": ai_company_signal, "ts": time.time()}
                logger.info(f"AI company signal generated and cached for '{title_key[:40]}...'")
            except Exception as ai_err:
                logger.warning(f"Gemini failed, falling back to simulated logic: {ai_err}")
                title_lower = title_text.lower()
                if any(w in title_lower for w in ["dividend", "bonus", "profit", "right share", "ipo"]):
                    ai_score = 0.45; ai_label = "Positive"
                    ai_summary = "Signal includes strategic or material corporate actions (e.g. dividend expansion) interpreted as mildly positive for shareholder equity."
                elif any(w in title_lower for w in ["issue", "loss", "penalty", "decline", "suspend", "resignation"]):
                    ai_score = -0.50; ai_label = "Negative"
                    ai_summary = "Signal flags restructuring or negative risk factors that constrain forward momentum."
                else:
                    ai_score = 0.0; ai_label = "Neutral"
                    ai_summary = "Signal largely pertains to governance and procedural approvals representing baseline operations."
                ai_company_signal = {"label": ai_label, "score": ai_score, "impact_summary": ai_summary}
    
    # ── Sprint 3: Final score — 4-component weighted blend ────────────────────
    # Weights per spec: Market 25%, Company/news 35%, AI 25%, Technical 15%.
    # technical_score unifies MA + pivot; price_position is now handled by pivot.
    # AI weight is fixed at 0.25 (spec); market fills the remainder.
    market_weight        = 0.25
    company_score_weight = 0.35
    ai_weight            = 0.25
    technical_weight     = 0.15
    # Validate sum (sanity check — should always be 1.0)
    assert abs(market_weight + company_score_weight + ai_weight + technical_weight - 1.0) < 1e-9

    company_val = company_analysis["weighted_company_score"] if company_analysis else 0.0

    time_decay_multiplier = 1.0
    company_val *= time_decay_multiplier
    if getattr(ai_company_signal, "score", None) is not None:
        ai_score *= time_decay_multiplier

    final_sentiment_score = (
        (market_score   * market_weight) +
        (company_val    * company_score_weight) +
        (ai_score       * ai_weight) +
        (technical_score * technical_weight)
    )
    
    if final_sentiment_score >= 0.25:
        stock_sentiment = "Bullish"
    elif final_sentiment_score >= 0.10:
        stock_sentiment = "Mildly Bullish"
    elif final_sentiment_score <= -0.25:
        stock_sentiment = "Bearish"
    elif final_sentiment_score <= -0.10:
        stock_sentiment = "Mildly Bearish"
    else:
        stock_sentiment = "Neutral"
        
    has_high_materiality = data.get("company_signal") and "High" in [item["materiality"] for item in company_analysis.get("items", [])] if company_analysis else False
            
    # 6. Confidence Calibration & Alignment Constraints
    confidence_points = 0
    alignment_status = "Mixed"
    alignment_note = ""
    
    company_is_bullish = company_val > 0.1
    company_is_bearish = company_val < -0.1
    market_is_bullish = market_sentiment == "Bullish"
    market_is_bearish = market_sentiment == "Bearish"
    ai_is_bullish = ai_score > 0.1
    ai_is_bearish = ai_score < -0.1
    
    # ── Sprint 3: Technical alignment check ──────────────────────────────────
    tech_is_bullish = technical_score > 0.15
    tech_is_bearish = technical_score < -0.15

    if (company_is_bullish and ai_is_bullish and market_is_bullish) or (company_is_bearish and ai_is_bearish and market_is_bearish):
        alignment_status = "Aligned"
        alignment_note = "Company's recent developments are fully supported by the broader market direction."
        confidence_points += 3
    elif (company_is_bullish and market_is_bearish) or (company_is_bearish and market_is_bullish):
        alignment_status = "Contradictory"
        alignment_note = f"Company developments are {'positive' if company_is_bullish else 'negative'}, but the wider market backdrop does not support this signal."
        confidence_points -= 1
    elif (company_is_bullish and ai_is_bearish) or (company_is_bearish and ai_is_bullish):
        alignment_status = "Contradictory"
        alignment_note = "Company developments suggest one direction, but broader evaluation flags opposing risks."
        confidence_points -= 1
    else:
        alignment_status = "Mixed"
        comp_status = "positive" if company_is_bullish else ("negative" if company_is_bearish else "neutral")
        has_procedural_note = (
            " and largely consist of routine updates"
            if company_analysis and any(item.get("materiality") == "Low" for item in company_analysis.get("items", []))
            else ""
        )
        if comp_status != "neutral" and market_sentiment == "Neutral":
            alignment_note = f"Company updates are {comp_status}, but broader market conditions remain uncertain{has_procedural_note}."
        else:
            alignment_note = f"Key indicators currently send mixed signals{has_procedural_note}."

    # Technical alignment adjustment
    if tech_is_bullish and company_is_bullish:
        confidence_points += 1
        alignment_note += f" Technical indicators ({ma_signal}) confirm upward conviction."
    elif tech_is_bearish and company_is_bearish:
        confidence_points += 1
        alignment_note += f" Technical indicators ({ma_signal}) confirm downward pressure."
    elif tech_is_bearish and company_is_bullish:
        confidence_points -= 1
        alignment_note += f" Moving averages ({ma_signal}) limit near-term technical conviction."
    elif tech_is_bullish and company_is_bearish:
        confidence_points -= 1
        alignment_note += f" Moving averages ({ma_signal}) signal upward pressure despite weak fundamentals."

    if has_high_materiality: confidence_points += 1
    if market_sentiment != "Neutral": confidence_points += 1
    
    if company_source == "AGM Agenda" and not has_high_materiality:
        confidence_points -= 1
    
    if confidence_points >= 3:
        confidence = "High"
        confidence_context = "Confidence is high because strong company developments are confirmed by a supportive broader market."
    elif confidence_points >= 1:
        confidence = "Medium"
        if has_high_materiality and market_sentiment == "Neutral":
             confidence_context = "Confidence is medium because company signals are positive, but broader market conditions are not fully supportive."
        else:
             confidence_context = "Confidence is moderate because the available signals do not align strongly in a single direction."
    else:
        confidence = "Low"
        confidence_context = "Confidence is low because the available signals are mixed and do not point to a strong directional view."
        
    # Explicit AGM Conservatism Cap
    if company_source == "AGM Agenda" and market_sentiment == "Neutral" and alignment_status != "Aligned" and confidence != "High":
        if stock_sentiment == "Bullish":
            stock_sentiment = "Mildly Bullish"
        elif stock_sentiment == "Bearish":
            stock_sentiment = "Mildly Bearish"
        
    # Reasoning Generation
    actionable_factors = []
    if company_analysis:
        for item in company_analysis.get("items", []):
            if item["materiality"] == "High":
                actionable_factors.append(item["text"])
                
    market_str = market_sentiment.lower()
    
    if actionable_factors:
        joined_factors = " and ".join([f[:60].lower() for f in actionable_factors[:2]])
        procedural_note = " and several agenda items are procedural" if len(company_analysis.get("items", [])) > len(actionable_factors) else ""
        reasoning = f"The stock appears {stock_sentiment.lower()} because the {company_source} includes {joined_factors}, although the broader market remains {market_str}{procedural_note}."
    else:
        reasoning = f"The stock appears {stock_sentiment.lower()} because the latest {company_source} consists primarily of routine governance matters, while the broader market remains {market_str}."

    # Sprint 3: Append technical note to reasoning
    if ma_layers_hit > 0:
        if ma_signal in ("Bullish", "Mildly Bullish") and company_is_bullish:
            reasoning += f" Technical momentum ({ma_signal}) confirms the positive signal, strengthening overall conviction."
        elif ma_signal in ("Bearish", "Mildly Bearish") and company_is_bullish:
            reasoning += f" However, weak moving averages ({ma_signal}) urge caution, limiting near-term technical conviction despite positive news."
        elif ma_signal in ("Bullish", "Mildly Bullish") and company_is_bearish:
            reasoning += f" Interestingly, broader moving averages remain supportive ({ma_signal}), adding mixed signals that cushion extreme pessimism."
        elif ma_signal and ma_signal != "Neutral":
            reasoning += f" Broader technical analysis remains {ma_signal.lower()}."
            
    if pivot_explanation and pivot_signal != "Neutral":
        reasoning += f" Furthermore, {pivot_explanation}"
    elif has_pivot and pivot_signal != "Neutral":
        reasoning += f" Price action also rests in a {pivot_signal.lower()} pivot zone."
        
    # Final Synthesis Line
    tech_bullish = ma_signal in ("Bullish", "Mildly Bullish") or pivot_signal in ("Bullish", "Mildly Bullish")
    tech_bearish = ma_signal in ("Bearish", "Mildly Bearish") or pivot_signal in ("Bearish", "Mildly Bearish")
    
    if confidence == "High":
        if stock_sentiment in ("Bullish", "Mildly Bullish"):
            if tech_bearish:
                synth_line = f"Overall, strong fundamental tracking points to a clear near-term {stock_sentiment.lower()} tilt, even with slight technical friction requiring structural confirmation."
            else:
                synth_line = f"Overall, clear alignment between robust fundamentals and technicals reinforces a strong near-term {stock_sentiment.lower()} outlook."
        elif stock_sentiment in ("Bearish", "Mildly Bearish"):
            if tech_bullish:
                synth_line = f"Overall, strong fundamentals drive a clear near-term {stock_sentiment.lower()} trajectory, despite temporary technical support holding."
            else:
                synth_line = f"Overall, clear alignment between fundamentals and technicals reinforces a strong near-term {stock_sentiment.lower()} outlook."
        else:
            synth_line = "Overall, strong tracking confirms a lack of directional catalysts, leaving the near-term setup firmly neutral."
            
    elif confidence == "Medium":
        if stock_sentiment in ("Bullish", "Mildly Bullish"):
            if tech_bearish:
                synth_line = f"Overall, the setup reflects a {stock_sentiment.lower()} fundamental tilt, though near-term technical weakness limits clarity unless overhead levels are firmly reclaimed."
            else:
                synth_line = f"Overall, developing alignment across signals suggests a cohesive near-term {stock_sentiment.lower()} outlook, providing momentum continues compounding."
        elif stock_sentiment in ("Bearish", "Mildly Bearish"):
            if tech_bullish:
                 synth_line = f"Overall, the interpretation leans {stock_sentiment.lower()}, though conflicting technical bounds limit short-term downside clarity."
            else:
                 synth_line = f"Overall, technicals and fundamentals suggest a developing near-term {stock_sentiment.lower()} outlook, looking for sustained structural confirmation."
        else:
            synth_line = "Overall, the absence of strong fundamental triggers leaves the near-term setup broadly neutral."
            
    else: # Low / Uncertain
        synth_line = "Overall, mixed signals keep the near-term direction highly uncertain, with clearer conviction requiring stronger alignment across the board."
        
    reasoning += f" {synth_line}"
    
    # Rebuild explicit driver maps
    drivers = []
    
    # Top Drivers - Most material only
    added_drivers = 0
    if company_analysis:
        for item in company_analysis.get("items", []):
            if item['materiality'] == "High" and added_drivers < 2:
                drivers.append(f"Company signal: {item['text'][:50]} (high materiality)")
                added_drivers += 1
                
    if added_drivers == 0 and company_analysis:
       for item in company_analysis.get("items", []):
            if item['materiality'] == "Medium" and added_drivers < 1:
                drivers.append(f"Company signal: {item['text'][:50]} (medium materiality)")
                added_drivers += 1

    if not drivers:
        drivers.append(f"Company signal: {company_source} detected")
        
    drivers.append(f"Market sentiment: {market_sentiment}")
    drivers.append(f"Signal strength: {company_source}, {confidence.lower()} confidence")
    drivers.append(f"Price position: {price_position_label}")

    # Sprint 3: Add technical drivers
    if ma_layers_hit > 0:
        drivers.append(f"Technical (MA): {ma_signal} ({ma_note or 'MA analysis'})")
    if has_pivot and pivot_signal != "Neutral":
        drivers.append(f"Technical (Pivot): {pivot_signal} zone")

    # ── Sprint 3: Build TechnicalAnalysis response object ────────────────────
    technical_analysis_obj = TechnicalAnalysis(
        moving_analysis  = moving_analysis_obj,
        pivot_analysis   = pivot_analysis_obj,
        moving_score     = round(ma_score, 3),
        pivot_score      = round(pivot_score, 3),
        pivot_signal     = pivot_signal,
        technical_score  = round(technical_score, 3),
        technical_summary = technical_summary_text,
    )

    data["insight"] = {
        "stock_sentiment": stock_sentiment,
        "confidence": confidence,
        "confidence_context": confidence_context,
        "reasoning": reasoning,
        "drivers": drivers,
        "company_analysis": company_analysis,
        "ai_company_signal": ai_company_signal,
        "signal_alignment": {
            "status": alignment_status,
            "note": alignment_note
        },
        "technical_indicators": technical_indicators
    }
    # Promote MA fields to top-level for direct frontend access
    data["ma5"]  = data.get("ma5")
    data["ma20"] = data.get("ma20")
    data["ma180"]= data.get("ma180")
    # Sprint 3: top-level technical_analysis block
    data["technical_analysis"] = technical_analysis_obj

    # ─────────────────────────────────────────────────────────────────────────
    # SPRINT 4: ACTIVE INTELLIGENCE (Alerts & "What Changed")
    # ─────────────────────────────────────────────────────────────────────────
    alerts_list = []
    what_changed_info = WhatChanged(summary="No material change since last analysis", changes=[])

    old_res = STOCK_HISTORY.get(ticker_upper)
    if old_res and old_res.insight:
        delta_points = []
        
        # 1. Sentiment Change
        old_sent = old_res.insight.stock_sentiment or "Neutral"
        if old_sent != stock_sentiment:
            alerts_list.append(Alert(
                type="sentiment",
                message=f"Sentiment shifted from {old_sent} to {stock_sentiment}",
                severity="high"
            ))
            delta_points.append(f"Sentiment shift: {old_sent} → {stock_sentiment}")

        # 2. Technical Breakouts
        try:
            prev_price = float(old_res.current_price.replace(",", "")) if old_res.current_price else 0.0
            curr_price = cp_pf or 0.0
            
            if pivot_analysis_obj and pivot_analysis_obj.pp:
                pp = pivot_analysis_obj.pp
                if prev_price < pp <= curr_price:
                    alerts_list.append(Alert(type="technical", message="Price crossed above main Pivot Point", severity="medium"))
                    delta_points.append("Crossed above central pivot")
                elif prev_price > pp >= curr_price:
                    alerts_list.append(Alert(type="technical", message="Price fell below main Pivot Point", severity="medium"))
                    delta_points.append("Fell below central pivot")
                
                r1 = pivot_analysis_obj.resistance.r1 if pivot_analysis_obj.resistance else None
                s1 = pivot_analysis_obj.support.s1 if pivot_analysis_obj.support else None
                if r1 and prev_price < r1 <= curr_price:
                    alerts_list.append(Alert(type="technical", message="Breakout: Price broke R1 resistance", severity="high"))
                    delta_points.append("Broke above R1 resistance")
                elif s1 and prev_price > s1 >= curr_price:
                    alerts_list.append(Alert(type="technical", message="Breakdown: Price dropped below S1 support", severity="high"))
                    delta_points.append("Dropped below S1 support")
        except: pass

        # 3. Message Detection
        old_sig_title = old_res.company_signal.title if old_res.company_signal else None
        new_sig_title = data.get("company_signal", {}).get("title")
        if new_sig_title and new_sig_title != old_sig_title:
            alerts_list.append(Alert(type="news", message=f"New signal: {new_sig_title[:40]}...", severity="medium"))
            delta_points.append("Fresh company signal detected")

        if delta_points:
            joined = ", ".join(delta_points).lower().capitalize()
            summary_str = f"Intelligence summary: {joined}."
            what_changed_info = WhatChanged(summary=summary_str, changes=delta_points)

    data["alerts"] = [a.dict() for a in alerts_list]
    data["what_changed"] = what_changed_info.dict()

    # ── System-level performance metadata ────────────────────────────────
    from datetime import datetime, timezone
    _now_iso = datetime.now(timezone.utc).isoformat()
    data["cache_hit"]    = False
    data["last_updated"]  = _now_iso
    data["source_change"] = True
    data["data_freshness"] = "fresh"

    # ── Populate cache ────────────────────────────────────────────────────
    STOCK_CACHE[ticker_upper] = {"data": data.copy(), "ts": time.time()}
    logger.info(f"Stock cache populated for {ticker_upper} (last_updated={_now_iso})")

    return StockResponse(**data)


def _compute_live_technical_analysis(data: dict, ticker_upper: str):
    from typing import Optional
    from app.schemas import TechnicalAnalysis, MovingAnalysis, MADetail, PivotAnalysis, PivotSupport, PivotResistance
    try:
        def _infer_ma_signal(price_val, ma_val):
            if not price_val or not ma_val: return None
            try:
                diff_pct = (price_val - ma_val) / ma_val
                if diff_pct > 0.01:    return "Bullish"
                elif diff_pct < -0.01: return "Bearish"
                return "Neutral"
            except Exception:
                return None

        _cp_f    = float(data.get("current_price", "0").replace(",", "")) if data.get("current_price") else None
        _ma5_f   = float(data.get("ma5",  "0").replace(",", "")) if data.get("ma5")  else None
        _ma20_f  = float(data.get("ma20", "0").replace(",", "")) if data.get("ma20") else None
        _ma180_f = float(data.get("ma180","0").replace(",", "")) if data.get("ma180") else None

        moving_analysis_obj = MovingAnalysis(
            ma5  = MADetail(value=_ma5_f,   signal=data.get("ma5_signal")   or _infer_ma_signal(_cp_f, _ma5_f)),
            ma20 = MADetail(value=_ma20_f,  signal=data.get("ma20_signal")  or _infer_ma_signal(_cp_f, _ma20_f)),
            ma180= MADetail(value=_ma180_f, signal=data.get("ma180_signal") or _infer_ma_signal(_cp_f, _ma180_f)),
        )

        def _pf(key):
            v = data.get(key)
            return float(v.replace(",", "")) if v else None
            
        pp  = _pf("pivot_pp")
        pivot_analysis_obj = PivotAnalysis(
            pp=pp,
            support    = PivotSupport(s1=_pf("pivot_s1"), s2=_pf("pivot_s2"), s3=_pf("pivot_s3")),
            resistance = PivotResistance(r1=_pf("pivot_r1"), r2=_pf("pivot_r2"), r3=_pf("pivot_r3")),
        )

        # Basic label derivation to supply technical_summary_text string
        ma_signal = moving_analysis_obj.ma20.signal or "Neutral"
        pivot_signal = "Neutral"
        cp_pf = _cp_f
        
        # Simplified live summary
        parts = []
        if ma_signal != "Neutral":
            parts.append(f"Moving averages point to a {ma_signal.lower()} trend based on recent price action")
            
        if cp_pf and pp:
            pr2 = _pf("pivot_r2")
            pr1 = _pf("pivot_r1")
            ps1 = _pf("pivot_s1")
            ps2 = _pf("pivot_s2")
            if pr2 and cp_pf >= pr2:
                pivot_signal = "Bearish"
                parts.append(f"however, price is deeply overbought above resistance ({pr2})")
            elif pr1 and cp_pf >= pr1:
                pivot_signal = "Mildly Bearish"
                parts.append(f"with momentum facing friction near overhead resistance ({pr1})")
            elif cp_pf >= pp:
                pivot_signal = "Mildly Bullish"
                parts.append("while maintaining a constructive bias above the central pivot")
            elif ps1 and cp_pf >= ps1:
                pivot_signal = "Mildly Bearish"
                parts.append(f"though currently resting below the pivot, limited by support ({ps1})")
            elif ps2 and cp_pf >= ps2:
                pivot_signal = "Neutral"
                parts.append(f"currently absorbing pressure near structural support levels ({ps2})")
            
        if not parts:
            parts.append("Technical indicators remain broadly neutral across current levels")
            
        summary_text = ", ".join(parts).capitalize() + "."
            
        return TechnicalAnalysis(
            moving_analysis=moving_analysis_obj,
            pivot_analysis=pivot_analysis_obj,
            technical_summary_text=summary_text
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to compute live technical analysis for {ticker_upper}: {e}")
        return None
