from __future__ import annotations
"""Simple eBay Finding API wrapper used for comps gathering.

The implementation focuses on the "findCompletedItems" and
"findItemsAdvanced" operations and normalises the payload into a
light-weight structure consumed by the pricing module.

Network calls are performed with ``requests`` and are intentionally kept
serial – the service is fast enough for the small payloads used in unit
tests.  When running in production the caller may execute the coroutine
in an executor or migrate to ``httpx.AsyncClient``.
"""

from typing import Dict, Any, List
import os
import time
import requests

EBAY_ENDPOINT = "https://svcs.ebay.com/services/search/FindingService/v1"


def _call(op: str, params: Dict[str, Any], app_id: str, page: int = 1) -> Dict[str, Any]:
    """Call the eBay Finding API and return JSON response."""
    headers = {"X-EBAY-SOA-OPERATION-NAME": op, "X-EBAY-SOA-SECURITY-APPNAME": app_id,
               "X-EBAY-SOA-RESPONSE-DATA-FORMAT": "JSON"}
    query = {"paginationInput.entriesPerPage": 100, "paginationInput.pageNumber": page}
    query.update(params)
    resp = requests.get(EBAY_ENDPOINT, params=query, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _extract_items(resp: Dict[str, Any], sold: bool, include_shipping: bool) -> List[Dict[str, Any]]:
    """Normalise items returned by eBay Finding API."""
    key = "findCompletedItemsResponse" if sold else "findItemsAdvancedResponse"
    resp_root = resp.get(key, [{}])[0]
    search = resp_root.get("searchResult", [{}])[0]
    items = []
    for it in search.get("item", []):
        title = (it.get("title") or [""])[0]
        item_id = (it.get("itemId") or [""])[0]
        selling = (it.get("sellingStatus") or [{}])[0]
        price_d = (selling.get("currentPrice") or [{}])[0]
        price = float(price_d.get("__value__", 0))
        currency = price_d.get("@currencyId")
        ship_cost = 0.0
        ship_d = (it.get("shippingInfo") or [{}])[0]
        if include_shipping:
            ship_cost = float((ship_d.get("shippingServiceCost") or [{}])[0].get("__value__", 0))
        condition = (it.get("condition") or [{}])[0].get("conditionDisplayName", [None])[0]
        listing_info = (it.get("listingInfo") or [{}])[0]
        end_time = (listing_info.get("endTime") or [None])[0]
        best_offer = (listing_info.get("bestOfferAccepted") or ["false"])[0].lower() == "true"
        url = (it.get("viewItemURL") or [None])[0]
        items.append({
            "title": title,
            "price": price + ship_cost,
            "shipping": ship_cost,
            "currency": currency,
            "condition": condition,
            "listingId": item_id,
            "endTime": end_time,
            "bestOfferAccepted": best_offer,
            "url": url,
        })
    return items


def _collect(op: str, keyword: str, app_id: str, include_shipping: bool, max_pages: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        resp = _call(op, {"keywords": keyword}, app_id=app_id, page=page)
        items = _extract_items(resp, sold=(op == "findCompletedItems"), include_shipping=include_shipping)
        if not items:
            break
        out.extend(items)
        # Stop if we received less than a full page
        if len(items) < 100:
            break
        time.sleep(0.1)
    return out


def ebay_api_comps(keyword: str, app_id: str | None = None, include_shipping: bool = True) -> Dict[str, Any]:
    """Return sold and active comps for ``keyword`` via the eBay API."""
    app_id = app_id or os.getenv("EBAY_APP_ID")
    if not app_id:
        raise RuntimeError("EBAY_APP_ID not configured")
    sold = _collect("findCompletedItems", keyword, app_id, include_shipping)
    active = _collect("findItemsAdvanced", keyword, app_id, include_shipping)
    return {"sold_items": sold, "active_items": active}

