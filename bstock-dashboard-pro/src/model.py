from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, Optional
from .utils import clamp

# Per-category baseline resale as fraction of MSRP
CATEGORY_BASE_MULTIPLIER = {
    'game_controller': 0.60,
    'pc_peripheral': 0.50,
    None: 0.45
}

# Per-category condition discounts (lower price when worse condition)
CONDITION_DISCOUNT_BY_CAT = {
    'game_controller': {
        'New': 0.00, 'Open Box': 0.05, 'Grade A': 0.10, 'Grade B': 0.20, 'Grade C': 0.35, 'Salvage': 0.60
    },
    'pc_peripheral': {
        'New': 0.00, 'Open Box': 0.07, 'Grade A': 0.12, 'Grade B': 0.22, 'Grade C': 0.38, 'Salvage': 0.65
    },
    None: {
        'New': 0.00, 'Open Box': 0.06, 'Grade A': 0.11, 'Grade B': 0.21, 'Grade C': 0.36, 'Salvage': 0.62
    }
}

# Per-category/condition defect and return rates
P_DEAD_BY_CAT = {
    'game_controller': {'New':0.02,'Open Box':0.04,'Grade A':0.08,'Grade B':0.15,'Grade C':0.30,'Salvage':0.60},
    'pc_peripheral': {'New':0.02,'Open Box':0.05,'Grade A':0.09,'Grade B':0.16,'Grade C':0.32,'Salvage':0.60},
    None: {'New':0.02,'Open Box':0.05,'Grade A':0.09,'Grade B':0.17,'Grade C':0.33,'Salvage':0.60},
}
P_RETURN_BY_CAT = {
    'game_controller': 0.05,
    'pc_peripheral': 0.06,
    None: 0.07
}

@dataclass
class CostAssumptions:
    marketplace_fee: float = 0.13
    payment_fee: float = 0.029
    payment_fixed: float = 0.3
    outbound_ship: float = 6.5
    labor_per_unit: float = 3.0
    storage_per_unit_month: float = 0.25
    inbound_freight_flat: float = 250.0
    overhead_pct: float = 0.05

def _expected_price_prior(msrp: float | None, category: str | None, condition: str) -> float:
    base_mult = CATEGORY_BASE_MULTIPLIER.get(category, CATEGORY_BASE_MULTIPLIER[None])
    disc_map = CONDITION_DISCOUNT_BY_CAT.get(category, CONDITION_DISCOUNT_BY_CAT[None])
    disc = disc_map.get(condition, 0.20)
    msrp = msrp if msrp and msrp > 5 else 40.0
    price = msrp * base_mult * (1 - disc)
    return clamp(price, 8.0, msrp * 0.9)

def blend_price(ebay_p50: Optional[float], msrp: Optional[float], category: Optional[str], condition: str) -> float:
    prior = _expected_price_prior(msrp, category, condition)
    if ebay_p50 is None:
        return prior
    return float(0.7 * ebay_p50 + 0.3 * prior)

def simulate_lot(
    df: pd.DataFrame,
    costs: CostAssumptions,
    ebay_stats: Dict[int, Dict[str, Any]] | None = None,
    strategy_by_row: Dict[int, Dict[str, Any]] | None = None,
    n_runs: int = 3000,
    random_seed: int = 42
) -> Dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    lines = []
    for idx, r in df.iterrows():
        cat = r.get('category', None) if pd.notna(r.get('category', None)) else None
        cond = str(r['condition']).strip()
        msrp = r.get('msrp', None)
        qty = int(r['qty'])
        # Base price from comps+prior
        if ebay_stats and idx in ebay_stats and ebay_stats[idx].get('sold_p50'):
            p50_market = blend_price(ebay_stats[idx]['sold_p50'], msrp, cat, cond)
        else:
            p50_market = _expected_price_prior(msrp, cat, cond)

        # Apply pricing strategy (listing price) and elasticity adjustment
        listing_price = p50_market
        floor = None
        days_adj_from_price = 0.0
        if strategy_by_row and idx in strategy_by_row:
            st = strategy_by_row[idx]
            # st: {'listing_price': value, 'days_adj': value, 'floor': value}
            if st.get('listing_price'):
                listing_price = float(st['listing_price'])
            if st.get('days_adj'):
                days_adj_from_price = float(st['days_adj'])
            floor = st.get('floor')

        p_dead = P_DEAD_BY_CAT.get(cat, P_DEAD_BY_CAT[None]).get(cond, 0.2)
        p_ret = P_RETURN_BY_CAT.get(cat, P_RETURN_BY_CAT[None])
        # Draw price distribution around listing_price (lognormal noise)
        cv = 0.22  # a bit tighter once you set your price
        sigma = np.sqrt(np.log(cv**2 + 1))
        mu = np.log(max(listing_price, 1e-6))

        # Velocity baseline
        base_days = 25 if cat == 'game_controller' else 35
        if ebay_stats and idx in ebay_stats and ebay_stats[idx].get('sell_through_proxy') is not None:
            stp = ebay_stats[idx]['sell_through_proxy']
            # market competitiveness factor
            base_days += (5 - 25*(1.0 - stp)) if 0.2 < stp < 1.0 else (-6 if stp>=1.0 else 10)
        base_days += days_adj_from_price  # elasticity effect

        lines.append(dict(
            idx=idx, qty=qty, category=cat, condition=cond,
            p50_market=p50_market, listing_price=listing_price, mu=mu, sigma=sigma,
            p_dead=p_dead, p_ret=p_ret, base_days=base_days
        ))

    proceeds_runs = np.zeros(n_runs)
    months_to_80_runs = np.zeros(n_runs)
    for k in range(n_runs):
        net = 0.0
        units_total = 0
        sold_days = []
        for ln in lines:
            units_total += ln['qty']
            prices = np.exp(rng.normal(ln['mu'], ln['sigma'], size=ln['qty']))
            is_dead = rng.random(ln['qty']) < ln['p_dead']
            is_return = rng.random(ln['qty']) < ln['p_ret']
            cond_pen = {'New':0,'Open Box':2,'Grade A':5,'Grade B':12,'Grade C':25,'Salvage':40}.get(ln['condition'],10)
            d2s = rng.normal(ln['base_days'] + cond_pen, 6, size=ln['qty'])
            d2s = np.clip(d2s, 5, 120)
            for i, p in enumerate(prices):
                if is_dead[i]:
                    salvage = 0.2 * ln['p50_market']
                    net += salvage - 0.5 * 3.0
                    sold_days.append(d2s[i] * 0.5)
                    continue
                if is_return[i]:
                    net += - (p * (costs.marketplace_fee + costs.payment_fee) + costs.payment_fixed + costs.outbound_ship + costs.labor_per_unit)
                    sold_days.append(d2s[i] * 1.2)
                    continue
                fees = p * (costs.marketplace_fee + costs.payment_fee) + costs.payment_fixed
                net += p - fees - costs.outbound_ship - costs.labor_per_unit
                sold_days.append(d2s[i])
        if units_total > 0:
            t80 = float(np.sort(sold_days)[max(1, int(0.8 * units_total)) - 1])
            months = t80 / 30.0
        else:
            months = 0.0
        storage_cost = units_total * costs.storage_per_unit_month * months
        net -= storage_cost
        net -= costs.inbound_freight_flat
        net = net * (1 - costs.overhead_pct)
        proceeds_runs[k] = net
        months_to_80_runs[k] = months

    ev = float(np.mean(proceeds_runs))
    return {
        'ev_net': ev,
        'p25_net': float(np.percentile(proceeds_runs, 25)),
        'p75_net': float(np.percentile(proceeds_runs, 75)),
        'p05_net': float(np.percentile(proceeds_runs, 5)),
        'months_to_80_med': float(np.median(months_to_80_runs)),
        'runs': proceeds_runs
    }

def compute_max_bid(stats: Dict[str, Any], risk_pct_of_ev: float = 0.05, target_margin: float = 0.20, rule: str = "downside_protected") -> float:
    rp = max(0.0, risk_pct_of_ev * max(0.0, stats['ev_net']))
    if rule == "downside_protected":
        return max(0.0, stats['p25_net'] - rp)
    return max(0.0, (stats['ev_net'] - rp) / (1.0 + target_margin))
