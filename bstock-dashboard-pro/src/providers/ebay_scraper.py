from __future__ import annotations
import re, time, math, os, sys, asyncio
from typing import Dict, Any, List
from urllib.parse import quote_plus
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

# prefer headless to reduce windowing issues
HEADLESS = os.getenv("BIDDER_HEADLESS", "true").lower() == "true"
SCRAPE_DELAY_SEC = float(os.getenv("SCRAPE_DELAY_SEC", "3.0"))

PRICE_RE = re.compile(r"[\\$£€]?[\\s]*([0-9]+(?:[,][0-9]{3})*(?:\\.[0-9]{1,2})?)")

def _parse_price(text: str) -> float | None:
    parts = re.findall(PRICE_RE, text.replace(",", ""))
    if not parts:
        return None
    nums = [float(p) for p in parts]
    if len(nums) >= 2 and " to " in text:
        return (nums[0] + nums[1]) / 2.0
    return nums[0]

def _collect(page, url: str) -> List[Dict[str, Any]]:
    page.goto(url, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_timeout(1200)
    try:
        page.locator('button:has-text("Close")').first.click(timeout=1500)
    except Exception:
        pass
    items = []
    rows = page.locator(".s-item").all()
    if not rows:
        rows = page.locator("[data-testid='item-card']").all()
    for r in rows:
        try:
            title = r.locator(".s-item__title").first.inner_text(timeout=1500)
        except Exception:
            try:
                title = r.locator("[data-testid='item-card-title']").first.inner_text(timeout=1500)
            except Exception:
                title = ""
        try:
            price_text = r.locator(".s-item__price").first.inner_text(timeout=1500)
        except Exception:
            try:
                price_text = r.locator("[data-testid='ux-price-values']").first.inner_text(timeout=1500)
            except Exception:
                price_text = ""
        price = _parse_price(price_text) if price_text else None
        if price:
            items.append({"title": title, "price": float(price)})
    return items

def ebay_scrape_comps(keyword: str) -> Dict[str, Any]:
    sold_url = f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(keyword)}&LH_Sold=1&LH_Complete=1"
    active_url = f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(keyword)}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=HEADLESS)
            context = browser.new_context()
            page = context.new_page()
            sold_items = _collect(page, sold_url)
            time.sleep(SCRAPE_DELAY_SEC)
            active_items = _collect(page, active_url)
            browser.close()
        return {"sold_items": sold_items, "active_items": active_items}
    except PWTimeoutError as e:
        raise RuntimeError(f"Playwright timeout reaching eBay: {e}")
    except Exception as e:
        # common Windows loop issue
        raise RuntimeError(f"Playwright failed to launch browser: {e}")
