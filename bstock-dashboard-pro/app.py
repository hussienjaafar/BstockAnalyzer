# app.py
from __future__ import annotations

import os, io, sys, json, time, hashlib, logging, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---- Windows Playwright fix ----
import asyncio
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

# ---- Logging ----
ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = Path("outputs/logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"run_{ts}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger("bstock")

# ---- Local modules ----
from src.parsing import load_manifest_df
from src.pricing import get_comps, apply_price_strategy, velocity_adjustment
from src.model import simulate_lot, compute_max_bid, CostAssumptions
from src.utils import money

load_dotenv()
st.set_page_config(page_title="B-Stock PRO Dashboard", layout="wide")
st.title("📦 B-Stock PRO — Auction Analyzer")

# ---- Sidebar ----
with st.sidebar:
    st.header("Settings")
    n_runs = st.slider("Monte Carlo runs", 500, 6000, 3000, step=500)
    min_sim = st.slider("Comps similarity threshold", 50, 95, 70, step=5,
                        help="Fuzzy title match; higher=cleaner comps, lower=more results")
    max_rows = st.slider("Rows to process (first N)", 1, 500, 50, step=1,
                         help="Throttles scraping; re-run for more.")
    provider_mode = st.selectbox("eBay provider", ["auto", "api", "scrape"], index=0)
    evid_n = st.slider("Evidence items", 1, 10, 3, step=1)
    st.markdown("### Pricing strategy")
    strat = st.radio("Strategy", ["Median market price", "Undercut market", "Premium to market"], index=0)
    pct = 0.0
    if strat != "Median market price":
        pct = st.slider("% amount", 1, 25, 5, step=1)
    st.markdown("### Risk")
    risk_ev = st.slider("Risk buffer (% of EV for downside rule)", 0, 20, 5, step=1)
    st.caption("Used in Max Bid (Downside): P25 − risk% of EV")

tabs = st.tabs(["Analyze", "History", "Assumptions"])

# ---- History helpers ----
HIST_PATH = os.path.join("outputs", "history.jsonl")
os.makedirs("outputs", exist_ok=True)
def save_history(entry: dict):
    with open(HIST_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
def load_history():
    if not os.path.exists(HIST_PATH): return []
    rows = []
    with open(HIST_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows

# =========================
# Tab 1 — Analyze
# =========================
with tabs[0]:
    st.subheader("Upload Manifest (CSV/XLSX)")
    up = st.file_uploader("Choose a file", type=["csv","xlsx","xls"], key="manifest_uploader")

    if up is None:
        st.info("Upload a manifest to map columns and analyze.")
        st.stop()

    # Identify this upload and persist bytes so we don't lose it on re-run
    up_bytes = up.getvalue()
    upload_id = hashlib.md5((up.name + str(len(up_bytes))).encode("utf-8")).hexdigest()

    # If a new file was uploaded, reset mapping state
    if st.session_state.get("upload_id") != upload_id:
        st.session_state["upload_id"] = upload_id
        st.session_state.pop("mapping_confirmed", None)
        st.session_state.pop("df_mapped", None)

    # Show manual mapping ONCE per upload
    if not st.session_state.get("mapping_confirmed"):
        df_raw = pd.read_csv(io.BytesIO(up_bytes)) if up.name.lower().endswith(".csv") else pd.read_excel(io.BytesIO(up_bytes))
        df = load_manifest_df(df_raw, interactive=True, store_path="outputs/column_maps.json")
        # Once the user clicks "Use this mapping", load_manifest_df returns here:
        st.session_state["df_mapped"] = df
        st.session_state["mapping_confirmed"] = True
        st.success(f"Mapping saved • {len(df)} lines • {int(df['qty'].sum())} units")

    # Use the already-mapped DataFrame from session
    df = st.session_state["df_mapped"]

    cols = st.columns([1,1,2])
    with cols[0]:
        remap = st.button("🔁 Remap columns")
    with cols[1]:
        run_btn = st.button("▶️ Run Analysis", type="primary")

    if remap:
        # Force mapping UI to show again on next run
        st.session_state["mapping_confirmed"] = False
        st.rerun()

    if not run_btn:
        st.stop()

    # ---- Proceed with analysis ----
    df_proc = df.head(max_rows)

    with st.status("Fetching comps and computing strategies...", expanded=True) as status:
        ebay_stats = {}
        strategy_by_row = {}
        errors = []
        prog = st.progress(0.0)
        total = len(df_proc)

        for i, (idx, r) in enumerate(df_proc.iterrows()):
            title = str(r["title"])
            upc = r.get("upc", None)
            start = time.perf_counter()
            st.write(f"🔎 {title[:70]}{'...' if len(title)>70 else ''}")
            try:
                stats = get_comps(title, upc, min_similarity=min_sim,
                                  provider_mode=provider_mode, n_evidence=evid_n)
                took = time.perf_counter() - start
                logger.info(
                    f"row={idx} OK sold={stats.get('sold_count')} active={stats.get('active_count')} "
                    f"sold_p50={stats.get('sold_p50')} act_p10={stats.get('active_p10')} kw={stats.get('keyword')} "
                    f"mode={stats.get('query_mode')} cache={stats.get('cache_hit')} raw={stats.get('counts_raw')} took={took:.2f}s"
                )
            except Exception as e:
                err = f"Row {idx} comps failed: {e}"
                errors.append(err)
                logger.exception(err)
                st.warning(err)
                stats = {"sold_count":0,"sold_p50":None,"sold_p10":None,"sold_p90":None,
                         "active_count":0,"active_p10":None,"active_p50":None,"sell_through_proxy":None,
                         "evidence":{"sold":[],"active":[]}, "query_mode":"error",
                         "keyword":str(upc) if upc else title,
                         "counts_raw":{"sold":0,"active":0}, "cache_hit":False}

            with st.expander(f"Evidence (row {idx}) — mode={stats.get('query_mode')} kw={stats.get('keyword')}"):
                ev = stats.get("evidence", {})
                sold_e = ev.get("sold", []); active_e = ev.get("active", [])
                st.write(f"**Sold matches (top {len(sold_e)}):**")
                for it in sold_e:
                    st.markdown(f"- {it.get('title','(no title)')} — ${it.get('price')}  " +
                                (f"[link]({it.get('url')})" if it.get('url') else ""))
                st.write(f"**Active matches (top {len(active_e)}):**")
                for it in active_e:
                    st.markdown(f"- {it.get('title','(no title)')} — ${it.get('price')}  " +
                                (f"[link]({it.get('url')})" if it.get('url') else ""))
                st.download_button(
                    "Download raw comps JSON",
                    data=json.dumps(ev, indent=2),
                    file_name=f"row_{idx}_comps.json",
                    mime="application/json",
                )

            ebay_stats[idx] = stats

            # Pricing strategy
            if strat == "Median market price":
                applied = apply_price_strategy(stats, "median", 0.0)
            elif strat == "Undercut market":
                applied = apply_price_strategy(stats, "undercut_pct", pct)
            else:
                applied = apply_price_strategy(stats, "premium_pct", pct)

            lp = applied.get("listing_price")
            floor = applied.get("floor")
            days_adj = velocity_adjustment(stats, lp) if lp else 0.0
            strategy_by_row[idx] = {"listing_price": lp, "floor": floor, "days_adj": days_adj}
            prog.progress((i + 1) / max(1, total))

        status.update(label="Running Monte Carlo...", state="running")
        costs = CostAssumptions()
        sim_stats = simulate_lot(df_proc, costs,
                                 ebay_stats=ebay_stats,
                                 strategy_by_row=strategy_by_row,
                                 n_runs=n_runs)
        status.update(label="Done", state="complete", expanded=False)

    # ---- Summary metrics ----
    ev, p25, p75, p05 = sim_stats['ev_net'], sim_stats['p25_net'], sim_stats['p75_net'], sim_stats['p05_net']
    months = sim_stats['months_to_80_med']
    max_bid = compute_max_bid(sim_stats, risk_pct_of_ev=risk_ev/100.0, rule="downside_protected")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("EV Net", money(ev))
    k2.metric("P25 Net", money(p25))
    k3.metric("P75 Net", money(p75))
    k4.metric("Max Bid (Downside)", money(max_bid))
    st.caption(f"Median months to 80% sell-through: **{months:.1f}**")

    # ---- Distribution chart ----
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.hist(sim_stats['runs'], bins=30)
    plt.title("Distribution of Net Proceeds (Monte Carlo)")
    plt.xlabel("Net Proceeds ($)")
    plt.ylabel("Frequency")
    st.pyplot(fig, clear_figure=True)

    # ---- Per-line report ----
    rep = df_proc.copy()
    rep["ebay_sold_count"] = [ebay_stats[i].get("sold_count") for i in df_proc.index]
    rep["ebay_sold_p50"] = [ebay_stats[i].get("sold_p50") for i in df_proc.index]
    rep["ebay_active_count"] = [ebay_stats[i].get("active_count") for i in df_proc.index]
    rep["ebay_active_p10"] = [ebay_stats[i].get("active_p10") for i in df_proc.index]
    rep["sell_through_proxy"] = [ebay_stats[i].get("sell_through_proxy") for i in df_proc.index]
    rep["listing_price"] = [strategy_by_row[i].get("listing_price") for i in df_proc.index]
    rep["price_floor_used"] = [strategy_by_row[i].get("floor") for i in df_proc.index]
    rep["days_adj_from_price"] = [strategy_by_row[i].get("days_adj") for i in df_proc.index]
    rep["query_mode"] = [ebay_stats[i].get("query_mode") for i in df_proc.index]
    rep["kw_used"] = [ebay_stats[i].get("keyword") for i in df_proc.index]
    rep["raw_counts"] = [str(ebay_stats[i].get("counts_raw")) for i in df_proc.index]
    st.subheader("Line Items (first N rows)")
    st.dataframe(rep)

    # ---- Save report + history ----
    ts2 = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(up.name)[0]
    out_csv = os.path.join("outputs", f"{base_name}_{ts2}_report_first{len(df_proc)}.csv")
    os.makedirs("outputs", exist_ok=True)
    rep.to_csv(out_csv, index=False)
    run_id = hashlib.md5(f"{base_name}_{ts2}".encode("utf-8")).hexdigest()[:10]

    save_history({
        "id": run_id, "ts": ts2, "file": up.name,
        "units": int(df_proc['qty'].sum()), "lines": int(len(df_proc)),
        "ev_net": ev, "p25": p25, "p75": p75, "p05": p05,
        "months80": months, "max_bid": max_bid, "output_csv": out_csv,
        "strategy": strat, "pct": pct, "min_similarity": min_sim, "risk_ev": risk_ev,
        "log_path": str(LOG_PATH),
    })

    st.success(f"Saved report: {out_csv}")
    with open(out_csv, "rb") as f:
        st.download_button("Download CSV", data=f.read(), file_name=os.path.basename(out_csv))

    with st.expander("Run log (tail)"):
        try:
            tail = open(LOG_PATH, "r", encoding="utf-8").read()[-8000:]
            st.code(tail)
        except Exception:
            st.write("No log available.")
    st.caption(f"Full log saved to: {LOG_PATH}")

# =========================
# Tab 2 — History
# =========================
with tabs[1]:
    st.subheader("Past Uploads")
    rows = load_history()
    if not rows:
        st.info("No history yet.")
    else:
        rows = sorted(rows, key=lambda r: r["ts"], reverse=True)
        for r in rows:
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3,2,2,2])
                c1.markdown(f"**{r['file']}**"); c1.caption(r['ts'])
                c2.metric("EV Net", money(r['ev_net']))
                c3.metric("P25", money(r['p25']))
                c4.metric("Max Bid", money(r['max_bid']))
                cc1, cc2 = st.columns([1,1])
                if r.get("output_csv") and os.path.exists(r["output_csv"]):
                    with open(r["output_csv"], "rb") as f:
                        cc1.download_button("Download report", data=f.read(),
                                            file_name=os.path.basename(r["output_csv"]), key=r["id"])
                if r.get("log_path") and os.path.exists(r["log_path"]):
                    try:
                        text = open(r["log_path"], "r", encoding="utf-8").read()[-4000:]
                        st.expander("View log").code(text)
                    except Exception:
                        pass

# =========================
# Tab 3 — Assumptions
# =========================
with tabs[2]:
    st.subheader("Model Assumptions (summary)")
    st.markdown("""
- **Condition-aware** price discounts & defect rates vary by category (controllers vs peripherals).
- **Competitiveness:** Price baselines from eBay solds (scraped), filtered by fuzzy similarity (skipped for UPC modes). Active P10 sets a **floor**.
- **Pricing strategy:** Choose `median`, `undercut`, or `premium`; the model adjusts **velocity** with a simple elasticity heuristic.
- **Fees & costs:** includes marketplace + payment + outbound shipping + labor + storage + inbound freight + overhead.
- **Max Bid (Downside):** `P25 − risk% of EV`.
    """)
