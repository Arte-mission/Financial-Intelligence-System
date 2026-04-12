"""
sentiment_service.py — Nepal financial sentiment analysis layer.

Key functions:
  - calculate_sentiment(): TextBlob polarity score
  - get_mood_label(): 3-tier label (Bullish/Neutral/Bearish)
  - get_tiered_mood_label(): 5-tier calibrated label for macro signals
  - score_relevance(): relevance gate (0–10 score, gate at >= 2)
  - categorize_article(): market / banking / macro / noise
  - score_materiality(): article importance weight (High/Medium/Low)
  - extract_drivers(): semantic investor-grade driver phrases
  - compute_macro_confidence(): confidence with materiality awareness
  - get_category_coverage_note(): investor note on single-category dominance
"""

from textblob import TextBlob
from typing import Tuple, List, Dict


def calculate_sentiment(text: str) -> float:
    """
    Calculates the sentiment polarity of a given text using TextBlob.
    Returns a float between -1.0 (very negative) and 1.0 (very positive).
    """
    if not text:
        return 0.0
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


def get_mood_label(score: float) -> str:
    """
    Simple 3-tier label — used for Market Pulse (which is fairly binary).
    - score > 0.1  => Bullish
    - score < -0.1 => Bearish
    - otherwise    => Neutral
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

    # --- Low materiality (default for anything else that passed relevance gate) ---
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

    market_share = categories_breakdown.get("market", 0) / total
    banking_share = categories_breakdown.get("banking", 0) / total
    macro_share = categories_breakdown.get("macro", 0) / total

    if market_share >= 0.80:
        return "Macro signal is currently based mostly on market headlines and has limited banking or macroeconomic confirmation."
    if banking_share >= 0.80:
        return "Macro signal draws heavily from banking sector coverage with limited broader economic context."
    if macro_share >= 0.80:
        return "Signal is driven primarily by macroeconomic coverage. Market and banking confirmation is limited."

    return ""
