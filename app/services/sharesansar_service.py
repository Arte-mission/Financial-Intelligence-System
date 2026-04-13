import requests
from bs4 import BeautifulSoup
import logging
import re
import time
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# ── Two-level scrape cache ────────────────────────────────────────────────────
#
# Summary cache  — prices + sector  (cheap HTTP + BS4 parse)
#   TTL: 5 min  — prices update intraday; short TTL keeps data fresh
#
# Signal cache   — company events + headlines  (expensive Playwright launch)
#   TTL: 20 min — AGM agenda / board events change at most once per session;
#                 Playwright is the single most expensive call in the stack
#
# Decision tree in get_company_data():
#   Both valid           → return merged dict instantly (zero I/O, zero browser)
#   Only summary expired → re-run HTTP + BS4 only; keep cached Playwright result
#   Signal expired       → full re-scrape (HTTP + Playwright); update both caches
#
_SUMMARY_CACHE: Dict[str, Dict] = {}  # { TICKER: {"data": dict, "ts": float} }
_SIGNAL_CACHE:  Dict[str, Dict] = {}  # { TICKER: {"data": dict, "ts": float} }

SUMMARY_CACHE_TTL = 300    # 5 minutes  — price/sector refresh cadence
SIGNAL_CACHE_TTL  = 1200   # 20 minutes — Playwright / company event refresh cadence

_SHARESANSAR_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sharesansar.com/"
}


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: HTTP + BeautifulSoup summary scrape
# Returns price + sector fields, or None if ticker is invalid.
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_summary(ticker: str) -> Optional[Dict[str, Any]]:
    """
    HTTP GET + BeautifulSoup parse of the Sharesansar company page.
    Extracts: current_price, price_change, price_change_percent,
              fifty_two_week_high, fifty_two_week_low, sector.
    Returns None on 404 or 'Company Not Found'.
    """
    ticker_lower = ticker.lower()
    url = f"https://www.sharesansar.com/company/{ticker_lower}"

    logger.info(f"Sharesansar summary fetch: {url}")
    try:
        response = requests.get(url, headers=_SHARESANSAR_HEADERS, timeout=15)

        if response.status_code == 404:
            return None
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        if soup.find(string=re.compile(r"Company Not Found", re.I)):
            return None

        current_price = None
        price_change = None
        price_change_percent = None
        fifty_two_week_high = None
        fifty_two_week_low = None
        sector = None
        ma5   = None
        ma20  = None
        ma180 = None

        # 1. Price Info (Top summary block near "As on:")
        as_on_elem = soup.find(string=re.compile(r"As on\s*:", re.I))
        if as_on_elem:
            logger.info(f"Summary block found for {ticker}")
            parent_container = as_on_elem.parent.parent.parent
            if parent_container:
                raw_text = parent_container.get_text(separator=' ').strip()
                logger.info(f"DEBUG: raw summary text: {raw_text}")

                cleaned_text = re.sub(r'\d{4}-\d{2}-\d{2}', '', raw_text)
                cleaned_text = re.sub(r'\d{2}:\d{2}:\d{2}', '', cleaned_text)
                logger.info(f"DEBUG: cleaned summary text: {cleaned_text}")

                percent_regex = r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%'
                number_regex  = r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'

                percent_matches = re.findall(percent_regex, cleaned_text)
                if percent_matches:
                    raw_percent_token = percent_matches[0]
                    price_change_percent = re.sub(r"\s+", "", raw_percent_token)
                    cleaned_text = cleaned_text.replace(raw_percent_token, "")

                number_matches = re.findall(number_regex, cleaned_text)
                logger.info(f"DEBUG: extracted numeric tokens: {number_matches}")
                logger.info(f"DEBUG: extracted percent token: {price_change_percent}")

                if len(number_matches) >= 2:
                    current_price = number_matches[0]
                    price_change  = number_matches[1]
                elif len(number_matches) == 1:
                    current_price = number_matches[0]
        else:
            logger.warning(f"Summary block NOT found for {ticker}")

        # 2. Sector
        sector_label = soup.find(string=re.compile(r"Sector\s*:", re.I))
        if sector_label:
            parent_text = sector_label.parent.parent.get_text(separator=' ')
            match = re.search(r'Sector\s*:\s*([A-Za-z\s]+)', parent_text, re.I)
            if match:
                sector = match.group(1).strip().split('\n')[0].strip()
        else:
            logger.warning(f"Sector block NOT found for {ticker}")

        # 3. 52-Week High / Low
        full_page_text = soup.get_text(separator=' ')
        high_low_match = re.search(
            r"52\s*Week\s*High-Low\s*:?\s*([0-9,]+\.\d+)\s*-\s*([0-9,]+\.\d+)",
            full_page_text, re.I
        )
        if high_low_match:
            fifty_two_week_high = high_low_match.group(1).replace(",", "")
            fifty_two_week_low  = high_low_match.group(2).replace(",", "")
            logger.info(f"52-week matched: high={fifty_two_week_high}, low={fifty_two_week_low}")
        else:
            high_low_elem = soup.find(string=re.compile(r"52\s*Weeks\s*High\s*/\s*Low", re.I))
            if high_low_elem:
                parent_text = high_low_elem.parent.parent.get_text(separator=' ')
                match = re.search(r"52\s*Weeks\s*High\s*/\s*Low\s*(.*)", parent_text, re.I)
                if match:
                    nums = re.findall(r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?', match.group(1))
                    if len(nums) >= 2:
                        fifty_two_week_high = nums[0]
                        fifty_two_week_low  = nums[1]
            if not high_low_match:
                logger.warning(f"52-Week block NOT found for {ticker}")

        # 4. Moving Averages (MA5, MA20, MA180)
        # Sharesansar shows these as "Average (5):", "Average (20):", "Average (180):"
        # or "5 Day MA:", "20 Day MA:", "180 Day MA:" depending on page version.
        # Flexible regex covers both formats.
        ma_patterns = [
            ("ma5",   r"(?:5\s*[Dd]ay\s*MA|Average\s*\(?5\)?)\s*:?\s*([0-9,]+(?:\.[0-9]+)?)"),
            ("ma20",  r"(?:20\s*[Dd]ay\s*MA|Average\s*\(?20\)?)\s*:?\s*([0-9,]+(?:\.[0-9]+)?)"),
            ("ma180", r"(?:180\s*[Dd]ay\s*MA|Average\s*\(?180\)?)\s*:?\s*([0-9,]+(?:\.[0-9]+)?)"),
        ]
        ma_values = {}
        for ma_key, pattern in ma_patterns:
            m = re.search(pattern, full_page_text, re.I)
            if m:
                ma_values[ma_key] = m.group(1).replace(",", "")
                logger.info(f"{ma_key.upper()} found: {ma_values[ma_key]}")
            else:
                logger.debug(f"{ma_key.upper()} not found on page")
        ma5   = ma_values.get("ma5")
        ma20  = ma_values.get("ma20")
        ma180 = ma_values.get("ma180")

        return {
            "ticker":               ticker.upper(),
            "company_name":         None,
            "sector":               sector,
            "current_price":        current_price,
            "price_change":         price_change,
            "price_change_percent": price_change_percent,
            "fifty_two_week_high":  fifty_two_week_high,
            "fifty_two_week_low":   fifty_two_week_low,
            "ma5":                  ma5,
            "ma20":                 ma20,
            "ma180":                ma180,
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching summary for {ticker}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching summary for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: Playwright signal + headline scrape
# Returns {headline, company_signal} — both may be None on failure.
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_signal(ticker: str) -> Dict[str, Any]:
    """
    Playwright browser launch to extract AGM agenda / event signals and
    the latest company headline from Sharesansar.
    This is the most expensive call in the stack (~15s per ticker).
    Results are cached for SIGNAL_CACHE_TTL (20 min).
    """
    url = f"https://www.sharesansar.com/company/{ticker.lower()}"
    headline       = None
    company_signal = None

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page    = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=15000)
            logger.info(f"Playwright session opened for {ticker}")

            # ── AGM History ─────────────────────────────────────────────────
            try:
                logger.info("Trying AGM History extraction first")
                agm_history_tab = page.get_by_text("AGM History", exact=True)
                if agm_history_tab.count() > 0:
                    first_tab = agm_history_tab.first
                    logger.info("AGM History tab located")
                    first_tab.scroll_into_view_if_needed()
                    page.wait_for_timeout(500)
                    try:
                        first_tab.click(force=True)
                    except Exception:
                        first_tab.evaluate("el => el.click()")
                    logger.info("Clicked AGM History tab")

                    try:
                        page.wait_for_selector("table#myTableAgm tbody tr", timeout=5000)
                        agm_rows  = page.locator("table#myTableAgm tbody tr")
                        row_count = agm_rows.count()
                        logger.info(f"AGM rows found: {row_count}")

                        if row_count > 0:
                            first_row   = agm_rows.first
                            agenda_cell = first_row.locator("td").last
                            if agenda_cell.locator("p").count() > 0:
                                agenda_cell_text = agenda_cell.locator("p").inner_text()
                            else:
                                agenda_cell_text = agenda_cell.inner_text()

                            agenda_text = re.sub(r'\s+', ' ', agenda_cell_text).strip()
                            if len(agenda_text) > 5:
                                company_signal = {
                                    "source": "AGM Agenda",
                                    "title":  agenda_text,
                                    "url":    None
                                }
                                logger.info(f"AGM agenda extracted: {agenda_text[:60]}")
                    except Exception:
                        pass

                if not company_signal:
                    logger.info("AGM History extraction yielded no results")
            except Exception as e:
                logger.warning(f"AGM History extraction failed: {e}")

            # ── Events tab ──────────────────────────────────────────────────
            if not company_signal:
                logger.info("Falling back to Events tab extraction")
            logger.info("Trying Events extraction")
            try:
                events_tab = page.get_by_text("Events", exact=True).first
                events_tab.click()
                logger.info("Clicked Events tab")
                page.wait_for_timeout(1500)
                page.wait_for_selector("table tbody tr", timeout=3000)
            except Exception as e:
                logger.warning(f"Failed to click Events tab or wait for table: {e}")

            rows = page.locator(".tab-content .active table tbody tr")
            if rows.count() == 0:
                rows = page.locator(".dataTables_wrapper table tbody tr")
            if rows.count() == 0:
                rows = page.locator("table tbody tr")

            count = rows.count()
            logger.info(f"Rendered event rows found: {count}")

            if count == 0:
                try:
                    debug_snippet = (
                        page.locator(".tab-content").inner_html()
                        if page.locator(".tab-content").count() > 0
                        else page.content()
                    )
                    logger.debug(f"Events container HTML: {debug_snippet[:2000]}...")
                except Exception:
                    pass
            else:
                for i in range(count):
                    row  = rows.nth(i)
                    cols = row.locator("td")
                    if cols.count() >= 2:
                        title_cell = cols.nth(1)
                        anchor     = title_cell.locator("a")
                        if anchor.count() > 0:
                            text     = anchor.first.inner_text().strip()
                            link_url = anchor.first.get_attribute("href")
                        else:
                            text     = title_cell.inner_text().strip()
                            link_url = None

                        if text and len(text) > 5:
                            if link_url and not link_url.startswith('http'):
                                link_url = (
                                    'https://www.sharesansar.com' + link_url
                                    if link_url.startswith('/')
                                    else 'https://www.sharesansar.com/' + link_url
                                )
                            headline = {"title": text, "url": link_url}
                            logger.info(f"First event extracted: {text} | URL: {link_url}")
                            break

            # ── News fallback ───────────────────────────────────────────────
            if not headline:
                news_section = page.locator("text=Company News").locator("..")
                if news_section.count() == 0:
                    news_section = page.locator("text=Related News").locator("..")
                if news_section.count() > 0:
                    anchors = news_section.locator("a")
                    for i in range(anchors.count()):
                        text     = anchors.nth(i).inner_text().strip()
                        link_url = anchors.nth(i).get_attribute("href")
                        if len(text) > 20 and link_url:
                            if not link_url.startswith('http'):
                                link_url = (
                                    'https://www.sharesansar.com' + link_url
                                    if link_url.startswith('/')
                                    else 'https://www.sharesansar.com/' + link_url
                                )
                            headline = {"title": text, "url": link_url}
                            logger.info(f"News headline found: {text} | URL: {link_url}")
                            break

            # If we still have no company_signal, derive it from the headline
            if not company_signal and headline:
                company_signal = {
                    "source": "Latest Event",
                    "title":  headline["title"],
                    "url":    headline.get("url")
                }

            browser.close()

    except Exception as e:
        logger.error(f"Playwright error for {ticker}: {e}")
        headline       = None
        company_signal = None

    return {"headline": headline, "company_signal": company_signal}


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────
def get_company_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Returns merged company data for a given ticker.

    Two-level cache strategy:
    ┌──────────────────────────────────────────────────────────────┐
    │  _SUMMARY_CACHE  (5 min)  — prices / sector                 │
    │  _SIGNAL_CACHE   (20 min) — Playwright events / headlines   │
    ├──────────────────────────────────────────────────────────────┤
    │  Both valid     → instant dict merge, zero I/O              │
    │  Summary stale  → HTTP + BS4 only; Playwright result reused │
    │  Signal stale   → full rescrape (HTTP + Playwright)         │
    └──────────────────────────────────────────────────────────────┘
    """
    ticker_upper = ticker.upper()
    now = time.time()

    sum_entry = _SUMMARY_CACHE.get(ticker_upper)
    sig_entry = _SIGNAL_CACHE.get(ticker_upper)

    sum_valid = bool(sum_entry and (now - sum_entry["ts"] < SUMMARY_CACHE_TTL))
    sig_valid = bool(sig_entry and (now - sig_entry["ts"] < SIGNAL_CACHE_TTL))

    # ── Fast path: both layers valid ─────────────────────────────────────────
    if sum_valid and sig_valid:
        sum_age = int(now - sum_entry["ts"])
        sig_age = int(now - sig_entry["ts"])
        logger.info(
            f"Sharesansar full cache hit for {ticker_upper} "
            f"(summary {sum_age}s, signal {sig_age}s)"
        )
        return {**sum_entry["data"], **sig_entry["data"]}

    # ── Summary stale, signal still valid ────────────────────────────────────
    if not sum_valid and sig_valid:
        logger.info(f"Sharesansar summary cache miss for {ticker_upper} — re-fetching prices only")
        summary = _fetch_summary(ticker)
        if summary is None:
            return None
        _SUMMARY_CACHE[ticker_upper] = {"data": summary, "ts": now}
        logger.info(f"Summary cache updated for {ticker_upper} (signal reused)")
        return {**summary, **sig_entry["data"]}

    # ── Signal stale (with or without summary) — full rescrape ───────────────
    logger.info(
        f"Sharesansar signal cache miss for {ticker_upper} — full rescrape "
        f"(summary_valid={sum_valid}, signal_valid={sig_valid})"
    )

    # Always re-fetch summary too when doing a full rescrape
    summary = _fetch_summary(ticker)
    if summary is None:
        return None
    _SUMMARY_CACHE[ticker_upper] = {"data": summary, "ts": now}

    signal = _fetch_signal(ticker)
    _SIGNAL_CACHE[ticker_upper] = {"data": signal, "ts": now}
    logger.info(f"Both caches updated for {ticker_upper}")

    return {**summary, **signal}
