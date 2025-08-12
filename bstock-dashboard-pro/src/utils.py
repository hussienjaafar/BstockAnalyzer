from __future__ import annotations
import math

def money(x: float) -> str:
    return f"${x:,.0f}" if abs(x) >= 1000 else f"${x:,.2f}"

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def months_from_days(days: float) -> float:
    return days / 30.0
