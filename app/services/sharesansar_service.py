import requests
from bs4 import BeautifulSoup
import logging
import re
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def get_company_data(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Scrapes company data from sharesansar.com for a given ticker.
    Uses flexible text extraction to avoid brittle CSS selectors.
    """
    ticker_lower = ticker.lower()
    url = f"https://www.sharesansar.com/company/{ticker_lower}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.sharesansar.com/"
    }
    
    logger.info(f"Fetching Sharesansar URL: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        
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
        headline = None
        company_signal = None
        
        # 1. Extract Price Info (Top summary block near "As on:")
        as_on_elem = soup.find(string=re.compile(r"As on\s*:", re.I))
        if as_on_elem:
            logger.info(f"Summary block found for {ticker}")
            
            # Navigate to a common parent container
            parent_container = as_on_elem.parent.parent.parent
            if parent_container:
                raw_text = parent_container.get_text(separator=' ').strip()
                logger.info(f"DEBUG: raw summary text: {raw_text}")
                
                # Remove Date patterns like YYYY-MM-DD
                cleaned_text = re.sub(r'\d{4}-\d{2}-\d{2}', '', raw_text)
                # optionally remove time HH:MM:SS if present
                cleaned_text = re.sub(r'\d{2}:\d{2}:\d{2}', '', cleaned_text)
                logger.info(f"DEBUG: cleaned summary text: {cleaned_text}")
                
                # Regex for explicit percentages and raw floats respectively
                percent_regex = r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*%'
                number_regex = r'-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
                
                percent_matches = re.findall(percent_regex, cleaned_text)
                if percent_matches:
                    raw_percent_token = percent_matches[0]
                    price_change_percent = re.sub(r"\s+", "", raw_percent_token)
                    # Strip it cleanly out to prevent it appearing as a regular number
                    cleaned_text = cleaned_text.replace(raw_percent_token, "")
                
                # Now extract exactly the float-like values for price and absolute change
                number_matches = re.findall(number_regex, cleaned_text)
                
                logger.info(f"DEBUG: extracted numeric tokens: {number_matches}")
                logger.info(f"DEBUG: extracted percent token: {price_change_percent}")

                # Ensure we safely grab price and change if we extracted valid tokens
                # Swap nothing incorrectly, if parsing fails, they safely stay None
                if len(number_matches) >= 2:
                    current_price = number_matches[0]
                    price_change = number_matches[1]
                elif len(number_matches) == 1:
                    current_price = number_matches[0]

        else:
            logger.warning(f"Summary block NOT found for {ticker}")

        # 2. Extract Sector Details
        sector_label = soup.find(string=re.compile(r"Sector\s*:", re.I))
        if sector_label:
            logger.info(f"Sector block found for {ticker}")
            
            # Find text in the vicinity
            parent_text = sector_label.parent.parent.get_text(separator=' ')
            
            # Attempt to split after "Sector :" or search by regex
            match = re.search(r'Sector\s*:\s*([A-Za-z\s]+)', parent_text, re.I)
            if match:
                sector = match.group(1).strip()
                # Clean up if it grabbed too much
                sector = sector.split('\n')[0].strip()
        else:
            logger.warning(f"Sector block NOT found for {ticker}")
            
        # 3. Extract 52W High and Low
        full_page_text = soup.get_text(separator=' ')
        high_low_match = re.search(r"52\s*Week\s*High-Low\s*:?\s*([0-9,]+\.\d+)\s*-\s*([0-9,]+\.\d+)", full_page_text, re.I)
        if high_low_match:
            fifty_two_week_high = high_low_match.group(1).replace(",", "")
            fifty_two_week_low = high_low_match.group(2).replace(",", "")
            logger.info(f"52-week regex matched: high={fifty_two_week_high}, low={fifty_two_week_low}")
        else:
            # Fallback legacy regex
            high_low_elem = soup.find(string=re.compile(r"52\s*Weeks\s*High\s*/\s*Low", re.I))
            if high_low_elem:
                parent_text = high_low_elem.parent.parent.get_text(separator=' ')
                match = re.search(r"52\s*Weeks\s*High\s*/\s*Low\s*(.*)", parent_text, re.I)
                if match:
                    post_text = match.group(1)
                    number_regex = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
                    nums = re.findall(number_regex, post_text)
                    if len(nums) >= 2:
                        fifty_two_week_high = nums[0]
                        fifty_two_week_low = nums[1]
            if not high_low_match:
                logger.warning(f"52-Week block NOT found for {ticker}")

        # 4. Extract Headlines/Events via Playwright
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=15000)
                
                logger.info("Browser session opened for Playwright extraction")
                
                try:
                    # 1. AGM History Extraction
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
                            # Evaluate bypass fallback
                            first_tab.evaluate("el => el.click()")
                            
                        logger.info("Clicked AGM History tab successfully")
                        
                        logger.info("Waiting for AGM table #myTableAgm")
                        try:
                            # Strict wait for strict table specifically for AGM History
                            page.wait_for_selector("table#myTableAgm tbody tr", timeout=5000)
                            logger.info("AGM table found")
                            
                            agm_rows = page.locator("table#myTableAgm tbody tr")
                            row_count = agm_rows.count()
                            logger.info(f"AGM rows found: {row_count}")
                            
                            if row_count > 0:
                                first_row = agm_rows.first
                                agenda_cell = first_row.locator("td").last
                                
                                # Prefer paragraph inside cell if available
                                if agenda_cell.locator("p").count() > 0:
                                    agenda_cell_text = agenda_cell.locator("p").inner_text()
                                else:
                                    agenda_cell_text = agenda_cell.inner_text()
                                    
                                agenda_text = re.sub(r'\s+', ' ', agenda_cell_text).strip()
                                
                                if len(agenda_text) > 5:
                                    company_signal = {
                                        "source": "AGM Agenda",
                                        "title": agenda_text,
                                        "url": None
                                    }
                                    logger.info(f"AGM agenda extracted: {agenda_text[:60]}")
                        except Exception:
                            pass
                    
                    if not company_signal:
                        logger.info("AGM History extraction didn't yield results.")
                except Exception as e:
                    logger.warning(f"AGM History extraction failed: {e}")

                if not company_signal:
                    logger.info("Falling back to Events only because AGM History extraction failed")
                
                logger.info("Trying Events extraction")
                try:
                    # Securely click events tab
                    events_tab = page.get_by_text("Events", exact=True).first
                    events_tab.click()
                    logger.info("Clicked Events tab successfully")
                    
                    # Wait for table
                    page.wait_for_timeout(1500)
                    page.wait_for_selector("table tbody tr", timeout=3000)
                except Exception as e:
                    logger.warning(f"Failed to click Events tab or wait for table via Playwright: {e}")
                
                # Broaden the event rows locator context avoiding multiple tables
                # Focus on the active datatable in the tab content
                rows = page.locator(".tab-content .active table tbody tr")
                if rows.count() == 0:
                    rows = page.locator(".dataTables_wrapper table tbody tr")
                    if rows.count() == 0:
                        rows = page.locator("table tbody tr")
                        
                count = rows.count()
                logger.info(f"Rendered event rows found: {count}")
                
                if count == 0:
                    try:
                        # Massive dump debug mode
                        debug_snippet = page.locator(".tab-content").inner_html() if page.locator(".tab-content").count() > 0 else page.content()
                        logger.debug(f"Events container HTML: {debug_snippet[:2000]}...")
                    except Exception:
                        pass
                else:
                    # Validate row headers and find correct item
                    for i in range(count):
                        row = rows.nth(i)
                        cols = row.locator("td")
                        
                        if cols.count() >= 2:
                            title_cell = cols.nth(1)
                            anchor = title_cell.locator("a")
                            
                            if anchor.count() > 0:
                                text = anchor.first.inner_text().strip()
                                link_url = anchor.first.get_attribute("href")
                            else:
                                text = title_cell.inner_text().strip()
                                link_url = None
                                
                            if text and len(text) > 5:  # ensure it's not a dummy cell
                                if link_url and not link_url.startswith('http'):
                                    link_url = 'https://www.sharesansar.com' + link_url if link_url.startswith('/') else 'https://www.sharesansar.com/' + link_url
                                    
                                headline = {
                                    "title": text,
                                    "url": link_url
                                }
                                logger.info(f"First event extracted: {text} | URL: {link_url}")
                                break
                
                # Playwright fallback to News if events fail
                if not headline:
                    news_section = page.locator("text=Company News").locator("..")
                    if news_section.count() == 0:
                        news_section = page.locator("text=Related News").locator("..")
                    
                    if news_section.count() > 0:
                        anchors = news_section.locator("a")
                        for i in range(anchors.count()):
                            text = anchors.nth(i).inner_text().strip()
                            link_url = anchors.nth(i).get_attribute("href")
                            if len(text) > 20 and link_url:
                                if not link_url.startswith('http'):
                                    link_url = 'https://www.sharesansar.com' + link_url if link_url.startswith('/') else 'https://www.sharesansar.com/' + link_url
                                headline = {"title": text, "url": link_url}
                                logger.info(f"First news headline found: {text} | URL: {link_url}")
                                break
                                
                # Handle Fallback for Company Signal
                if not company_signal and headline:
                    company_signal = {
                        "source": "Latest Event",
                        "title": headline["title"],
                        "url": headline.get("url")
                    }
                                
                browser.close()

                    
        except Exception as e:
            logger.error(f"Error extracting headlines for {ticker}: {e}")
            headline = None
            
        return {
            "ticker": ticker.upper(),
            "company_name": None,  # Will be populated by router
            "sector": sector,
            "current_price": current_price,
            "price_change": price_change,
            "price_change_percent": price_change_percent,
            "fifty_two_week_high": fifty_two_week_high,
            "fifty_two_week_low": fifty_two_week_low,
            "headline": headline,
            "company_signal": company_signal
        }
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error while scraping {ticker}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while scraping {ticker}: {e}")
        return None
