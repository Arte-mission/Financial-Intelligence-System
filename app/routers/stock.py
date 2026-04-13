from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any
import logging
import time

from app.schemas import StockResponse, CompanyResponse, TechnicalIndicators
from app.models import Company
from app.services.sharesansar_service import get_company_data
from app.services.ticker_service import get_company_by_ticker, get_all_companies, search_companies
from app.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Stock Info"])

# ── Stock response cache ──────────────────────────────────────────────────
# Keyed by ticker (uppercase). Avoids re-running the full scrape + AI pipeline
# on every request. TTL is set to 12 minutes — fast enough to catch intra-day
# company events while eliminating redundant Playwright/Gemini calls.
STOCK_CACHE_TTL = 720   # seconds (12 min)
STOCK_CACHE: Dict[str, Dict] = {}   # { ticker: {"data": StockResponse dict, "ts": float} }

# ── AI signal cache ───────────────────────────────────────────────────────
# Gemini calls are the most expensive part of the pipeline (latency + cost).
# Cache by company_signal title text so we only call Gemini when the
# underlying company event actually changes.
AI_CACHE_TTL = 1800     # seconds (30 min)
AI_SIGNAL_CACHE: Dict[str, Dict] = {}   # { title_key: {"signal": dict, "ts": float} }

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

@router.get("/stock/{ticker}", response_model=StockResponse)
def get_stock_data(ticker: str, db: Session = Depends(get_db)):
    """
    Retrieves and scores company data from Sharesansar.
    Results are cached for STOCK_CACHE_TTL seconds (12 min) per ticker to avoid
    redundant scraping and Gemini calls on every frontend refresh.
    """
    ticker_upper = ticker.upper()

    # ── Fast path: return cached response ────────────────────────────────
    cached = STOCK_CACHE.get(ticker_upper)
    if cached and (time.time() - cached["ts"] < STOCK_CACHE_TTL):
        logger.info(f"Stock cache hit for {ticker_upper} (age {int(time.time() - cached['ts'])}s)")
        result = cached["data"].copy()
        result["cache_hit"] = True
        return StockResponse(**result)

    # ── Slow path: full scrape + analysis ────────────────────────────────
    company = get_company_by_ticker(db, ticker)
    if not company:
        raise HTTPException(
            status_code=400,
            detail="This instrument is not supported. Only equity stocks are allowed."
        )

    data = get_company_data(ticker)

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"Company with ticker '{ticker_upper}' data unavailable on Sharesansar."
        )
        
    # Inject database metadata into the response
    data["company_name"] = company.company_name
    data["sector"] = company.sector
    
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

    try:
        if data.get("current_price") and data.get("fifty_two_week_high") and data.get("fifty_two_week_low"):
            cp = float(data["current_price"].replace(',', ''))
            high = float(data["fifty_two_week_high"].replace(',', ''))
            low = float(data["fifty_two_week_low"].replace(',', ''))
            
            if high > low and cp > 0:
                position = (cp - low) / (high - low)
                # Map position (0 to 1) to sentiment score (-0.5 to 0.5)
                # high = usually overvalued/momentum, low = undervalued/reversal
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
    
    # 7. Final Stock Insight Calculation — 5-component weighted score
    # Weights rebalanced to include MA technical layer (ma_weight=0.15)
    company_source = data.get("company_signal", {}).get("source", "News") if data.get("company_signal") else "News"

    company_score_weight = 0.30    # down from 0.35 to make room for MA
    price_weight         = 0.10
    ma_weight            = 0.15    # NEW: moving average technical score

    if company_source == "AGM Agenda":
        ai_weight = 0.10
    elif company_source == "Latest Event":
        ai_weight = 0.18
    else:
        ai_weight = 0.25

    market_weight = 1.0 - company_score_weight - ai_weight - price_weight - ma_weight

    company_val = company_analysis["weighted_company_score"] if company_analysis else 0.0

    # 7a. Time-decay multiplier (framework initialized; apply once date parsing is native)
    time_decay_multiplier = 1.0
    company_val *= time_decay_multiplier
    if getattr(ai_company_signal, "score", None) is not None:
        ai_score *= time_decay_multiplier

    final_sentiment_score = (
        (market_score       * market_weight) +
        (company_val        * company_score_weight) +
        (ai_score           * ai_weight) +
        (ma_score           * ma_weight) +
        (price_score        * price_weight)
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
    
    if (company_is_bullish and ai_is_bullish and market_is_bullish) or (company_is_bearish and ai_is_bearish and market_is_bearish):
        alignment_status = "Aligned"
        alignment_note = "The company's recent developments are fully supported by the broader market direction."
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
            if company_analysis and any(
                item.get("materiality") == "Low"
                for item in company_analysis.get("items", [])
            )
            else ""
        )
        
        if comp_status != "neutral" and market_sentiment == "Neutral":
            alignment_note = f"Company updates are {comp_status}, but broader market conditions remain uncertain{has_procedural_note}."
        else:
            alignment_note = f"Key indicators currently send mixed signals{has_procedural_note}."
            
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
        # Join max 2 factors for brevity
        joined_factors = " and ".join([f[:60].lower() for f in actionable_factors[:2]])
        procedural_note = " and several agenda items are procedural" if len(company_analysis.get("items", [])) > len(actionable_factors) else ""
        reasoning = f"The stock appears {stock_sentiment.lower()} because the {company_source} includes {joined_factors}, although the broader market remains {market_str}{procedural_note}."
    else:
        reasoning = f"The stock appears {stock_sentiment.lower()} because the latest {company_source} consists primarily of routine governance matters, while the broader market remains {market_str}."
    
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
    data["cache_hit"] = False

    # ── Populate cache ────────────────────────────────────────────────────
    STOCK_CACHE[ticker_upper] = {"data": data.copy(), "ts": time.time()}
    logger.info(f"Stock cache populated for {ticker_upper}")

    return StockResponse(**data)
