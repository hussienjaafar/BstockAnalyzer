# src/parsing.py
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --------- Synonyms & helpers ---------
TITLE_SYNONYMS = [
    "title", "item", "item name", "product", "product name", "product title",
    "description", "item description", "desc", "product description",
]
UPC_SYNONYMS = ["upc", "ean", "gtin", "barcode", "upc/ean", "upc code", "gtin-12", "barcode number"]
QTY_SYNONYMS = ["qty", "quantity", "qty ordered", "units", "unit qty", "pieces"]
UNIT_COST_SYNONYMS = ["unit cost", "unit price", "cost", "price", "estimated price", "est price", "unit_value"]
COND_SYNONYMS = ["condition", "item condition", "grade", "status"]
SKU_SYNONYMS = ["sku", "item id", "id", "line id", "internal id"]

def _norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.strip().lower())

def _score_column(col: str, syns: list[str]) -> int:
    c = _norm_col(col)
    # exact match wins
    if c in syns: return 100
    # startswith/contains are decent
    if any(c.startswith(s) for s in syns): return 80
    if any(s in c for s in syns): return 60
    return 0

def looks_like_id(col_values: pd.Series, sample: int = 50) -> bool:
    vals = [str(v) for v in list(col_values.dropna().head(sample))]
    hits = 0
    for v in vals:
        v2 = v.strip()
        if (
            re.fullmatch(r"[A-Z]{2,}\d{3,}", v2)              # e.g., LPGS100644
            or (re.fullmatch(r"[A-Za-z0-9\-_/.]{6,}", v2) and " " not in v2)  # long code-ish, no spaces
        ):
            hits += 1
    return hits >= max(5, int(0.6 * max(1, len(vals))))  # 60%+ look like IDs

def _guess(df: pd.DataFrame, syns: list[str], penalize_id_for_title: bool = False) -> Optional[str]:
    best = (None, -1)
    for col in df.columns:
        s = _score_column(col, syns)
        if penalize_id_for_title and s > 0 and looks_like_id(df[col]):
            s -= 50  # strongly push away ID-like columns for title/description
        if s > best[1]:
            best = (col, s)
    return best[0]

def _try_get(mapping: Dict[str,str], key: str) -> Optional[str]:
    v = mapping.get(key)
    return v if v and v != "(none)" else None

def _clean_string_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def _clean_upc_series(s: pd.Series) -> pd.Series:
    # keep digits only; preserve leading zeros
    s2 = s.astype(str).str.replace(r"\D", "", regex=True)
    # normalize all-empty to NaN
    s2 = s2.replace("", np.nan)
    return s2

def _clean_numeric_series(s: pd.Series, default: float | int = 0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)

# --------- Mapping store ---------
def _load_saved_maps(path: str) -> Dict[str, Dict[str, str]]:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def _save_maps(path: str, maps: Dict[str, Dict[str, str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(maps, indent=2), encoding="utf-8")

# --------- Main interactive loader ---------
def load_manifest_df(
    df_raw: pd.DataFrame,
    interactive: bool = True,
    store_path: str = "outputs/column_maps.json",
) -> pd.DataFrame:
    """
    Ensures the manifest has canonical columns:
      - title (required)
      - upc (optional)
      - qty (required; default 1 if missing)
      - unit_cost (optional)
      - condition (optional)
      - sku (optional)
    If interactive=True, shows a UI to map columns and persists the mapping.
    """
    # Normalize column names for selection
    columns = list(df_raw.columns)

    saved = _load_saved_maps(store_path)
    # Use a single default mapping bucket keyed as "default"
    prev = saved.get("default", {})

    # Heuristic guesses (used as defaults in the UI)
    guess_title = _guess(df_raw, [_norm_col(x) for x in TITLE_SYNONYMS], penalize_id_for_title=True)
    guess_upc = _guess(df_raw, [_norm_col(x) for x in UPC_SYNONYMS])
    guess_qty = _guess(df_raw, [_norm_col(x) for x in QTY_SYNONYMS])
    guess_cost = _guess(df_raw, [_norm_col(x) for x in UNIT_COST_SYNONYMS])
    guess_cond = _guess(df_raw, [_norm_col(x) for x in COND_SYNONYMS])
    guess_sku  = _guess(df_raw, [_norm_col(x) for x in SKU_SYNONYMS])

    # Fall back to previously saved mapping if present
    def default_pick(k, g):
        return prev.get(k, g) if prev.get(k, g) in columns else g

    default_map = {
        "title": default_pick("title", guess_title or columns[0]),
        "upc": default_pick("upc", guess_upc) or "(none)",
        "qty": default_pick("qty", guess_qty or "(none)"),
        "unit_cost": default_pick("unit_cost", guess_cost) or "(none)",
        "condition": default_pick("condition", guess_cond) or "(none)",
        "sku": default_pick("sku", guess_sku) or "(none)",
    }

    if interactive:
        st.subheader("🧭 Map columns")
        st.caption("Choose which headers correspond to each field. This is remembered for next time.")
        c1, c2 = st.columns([1,1])

        with c1:
            title_col = st.selectbox("Title / Description **(required)**", columns, index=columns.index(default_map["title"]))
            upc_col = st.selectbox("UPC / EAN / GTIN (optional)", ["(none)"] + columns, index=(["(none)"] + columns).index(default_map["upc"]))
            qty_col = st.selectbox("Quantity", ["(none)"] + columns, index=(["(none)"] + columns).index(default_map["qty"]))
        with c2:
            unit_cost_col = st.selectbox("Unit Cost / Price (optional)", ["(none)"] + columns, index=(["(none)"] + columns).index(default_map["unit_cost"]))
            cond_col = st.selectbox("Condition / Grade (optional)", ["(none)"] + columns, index=(["(none)"] + columns).index(default_map["condition"]))
            sku_col = st.selectbox("SKU / Item ID (optional)", ["(none)"] + columns, index=(["(none)"] + columns).index(default_map["sku"]))

        # Preview
        preview_cols = [c for c in {title_col, upc_col, qty_col, unit_cost_col, cond_col, sku_col} if c and c != "(none)"]
        st.write("**Preview (first 8 rows):**")
        st.dataframe(df_raw[preview_cols].head(8))

        remember = st.checkbox("Remember this mapping for future uploads", value=True)
        go = st.button("✅ Use this mapping")
        if not go:
            st.stop()

        mapping = {
            "title": title_col,
            "upc": _try_get({"upc": upc_col}, "upc"),
            "qty": _try_get({"qty": qty_col}, "qty"),
            "unit_cost": _try_get({"unit_cost": unit_cost_col}, "unit_cost"),
            "condition": _try_get({"condition": cond_col}, "condition"),
            "sku": _try_get({"sku": sku_col}, "sku"),
        }

        if remember:
            saved["default"] = {k: (v or "(none)") for k, v in mapping.items()}
            _save_maps(store_path, saved)

    else:
        # Non-interactive: just apply heuristic/defaults
        mapping = {
            "title": default_map["title"],
            "upc": None if default_map["upc"] == "(none)" else default_map["upc"],
            "qty": None if default_map["qty"] == "(none)" else default_map["qty"],
            "unit_cost": None if default_map["unit_cost"] == "(none)" else default_map["unit_cost"],
            "condition": None if default_map["condition"] == "(none)" else default_map["condition"],
            "sku": None if default_map["sku"] == "(none)" else default_map["sku"],
        }

    # --------- Build normalized DataFrame ---------
    out = pd.DataFrame()
    # title
    out["title"] = _clean_string_series(df_raw[mapping["title"]]) if mapping.get("title") else ""

    # upc
    if mapping.get("upc"):
        out["upc"] = _clean_upc_series(df_raw[mapping["upc"]])
    else:
        out["upc"] = pd.Series([np.nan] * len(df_raw))

    # qty
    if mapping.get("qty"):
        out["qty"] = _clean_numeric_series(df_raw[mapping["qty"]], default=1)
        out.loc[out["qty"] <= 0, "qty"] = 1
    else:
        out["qty"] = 1

    # unit_cost
    if mapping.get("unit_cost"):
        out["unit_cost"] = _clean_numeric_series(df_raw[mapping["unit_cost"]], default=np.nan)
    else:
        out["unit_cost"] = np.nan

    # condition
    if mapping.get("condition"):
        out["condition"] = _clean_string_series(df_raw[mapping["condition"]])
    else:
        out["condition"] = ""

    # sku
    if mapping.get("sku"):
        out["sku"] = _clean_string_series(df_raw[mapping["sku"]])
    else:
        out["sku"] = ""

    # Preserve any other columns (optional, for reference)
    # out = pd.concat([out, df_raw], axis=1)

    return out
