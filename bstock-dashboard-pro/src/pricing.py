from __future__ import annotations
from typing import Dict, Any, Optional, List
import numpy as np
from rapidfuzz import fuzz
from pathlib import Path
import json, time, os

from .providers.ebay_scraper import ebay_scrape_comps

# ---- simple comps cache by keyword ----
_CACHE_FILE = Path("outputs/comps_raw_cache.json")
_CACHE_TTL_SEC = float(os.getenv("COMPS_CACHE_TTL_SEC", str(7 * 24 * 3600)))  # default 7 days
_MEM: Dict[str, Any] = {}

def _ensure_cache_dir():
    _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

def _load_cache() -> Dict[str, Any]:
    if _CACHE_FILE.exists():
        try:
            return json.load(open(_CACHE_FILE, "r", encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_cache(db: Dict[str, Any]) -> None:
    _ensure_cache_dir()
    json.dump(db, open(_CACHE_FILE, "w", encoding="utf-8"))

def _get_raw_for_keyword(kw: str) -> Dict[str, Any]:
    now = time.time()
    if kw in _MEM:
        return _MEM[kw]
    db = _load_cache()
    hit = db.get(kw)
    if hit and (now - hit.get("ts", 0) < _CACHE_TTL_SEC):
        _MEM[kw] = hit["data"]
        return hit["data"]
    raw = ebay_scrape_comps(kw)
    db[kw] = {"ts": now, "data": raw}
    _save_cache(db)
    _MEM[kw] = raw
    return raw

def _choose_keyword(title: str, upc: Optional[str]) -> str:
    if upc and str(upc).strip():
        return str(upc).strip()
    t = title.replace("/", " ").replace("|", " ").strip()
    return " ".join(t.split()[:8])

def _filter_by_similarity(items: List[Dict[str,Any]], manifest_title: str, min_ratio: int=70) -> List[Dict[str,Any]]:
    m = manifest_title.lower()
    out = []
    for it in items:
        t = (it.get("title") or "").lower()
        ratio = fuzz.token_set_ratio(m, t)
        if ratio >= min_ratio:
            out.append(it)
    return out

def comp_stats_from_items(sold_items: List[Dict[str,Any]], active_items: List[Dict[str,Any]]) -> Dict[str, Any]:
    sold_prices = np.array([x["price"] for x in sold_items if x.get("price") is not None], dtype=float)
    active_prices = np.array([x["price"] for x in active_items if x.get("price") is not None], dtype=float)
    stats: Dict[str, Any] = {}
    if sold_prices.size:
        stats.update({
            "sold_count": int(sold_prices.size),
            "sold_p10": float(np.percentile(sold_prices,10)),
            "sold_p50": float(np.percentile(sold_prices,50)),
            "sold_p90": float(np.percentile(sold_prices,90)),
        })
    else:
        stats.update({"sold_count":0,"sold_p10":None,"sold_p50":None,"sold_p90":None})
    if active_prices.size:
        stats.update({
            "active_count": int(active_prices.size),
            "active_p10": float(np.percentile(active_prices,10)),
            "active_p50": float(np.percentile(active_prices,50)),
        })
    else:
        stats.update({"active_count":0,"active_p10":None,"active_p50":None})
    denom = max(stats["active_count"], 1)
    stats["sell_through_proxy"] = stats["sold_count"] / denom
    return stats

def get_comps(title: str, upc: Optional[str], min_similarity: int=70) -> Dict[str, Any]:
    kw = _choose_keyword(title, upc)
    raw = _get_raw_for_keyword(kw)  # cached raw sold/active for this keyword
    sold_f = _filter_by_similarity(raw.get("sold_items", []), title, min_similarity)
    active_f = _filter_by_similarity(raw.get("active_items", []), title, min_similarity)
    stats = comp_stats_from_items(sold_f, active_f)
    stats["used_similarity_threshold"] = min_similarity
    stats["keyword"] = kw
    return stats

def apply_price_strategy(stats: Dict[str, Any], strategy: str="median", pct: float=0.0) -> Dict[str, Any]:
    """
    strategy: "median", "undercut_pct", "premium_pct"
    pct: percentage for undercut or premium (e.g., 5 = 5%)
    Returns dict with listing_price and floor applied.
    """
    p50 = stats.get("sold_p50")
    if p50 is None:
        return {"listing_price": None, "floor": None}
    if strategy == "median":
        lp = p50
    elif strategy == "undercut_pct":
        lp = p50 * (1 - pct/100.0)
    elif strategy == "premium_pct":
        lp = p50 * (1 + pct/100.0)
    else:
        lp = p50
    # Competitor floor = active P10 (if exists)
    floor = stats.get("active_p10")
    if floor is not None:
        lp = max(lp, floor)
    return {"listing_price": float(lp), "floor": float(floor) if floor is not None else None}

def velocity_adjustment(stats: Dict[str, Any], listing_price: float) -> float:
    """
    Return a days adjustment based on price position vs sold distribution.
    Negative -> faster, positive -> slower. Heuristic.
    """
    p10, p50, p90 = stats.get("sold_p10"), stats.get("sold_p50"), stats.get("sold_p90")
    if not all([p50]):
        return 0.0
    if p10 is None: p10 = 0.85*p50
    if p90 is None: p90 = 1.15*p50
    if listing_price <= p10:
        return -8.0
    if listing_price >= p90:
        return +15.0
    span = p90 - p10 if p90>p10 else 1.0
    x = (listing_price - p50) / span
    return float(20.0 * x)
