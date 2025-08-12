# B-Stock Auction Dashboard — PRO (no-API eBay ready)

Upgrades:
- **Robust column mapping** (fuzzy + heuristics) for messy manifests
- **Competitiveness controls**: list at median / undercut / premium
- **Elasticity**: adjusts sell-through speed with your price choice
- **Comps filtering**: UPC-first + fuzzy match to drop irrelevant results
- **Competitor floor**: respects active P10 as a floor
- **Category+condition tables**: per-category discounts & defect rates
- Works with **eBay API** if present, *or* **Playwright scraping** if not

## Install (Windows)
```bat
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install
```

## Run
```bat
streamlit run app.py
```

## Notes
- To use eBay API later, create `.env` with `EBAY_APP_ID=...` (optional). Else, it scrapes politely.
- Keep manifests modest (≤20 rows) when scraping to avoid rate/antibot friction.
