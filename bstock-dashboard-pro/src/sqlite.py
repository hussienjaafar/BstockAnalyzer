from __future__ import annotations
"""Lightweight SQLite helpers for caching provider responses.

The cache lives at ``outputs/cache.db`` with a single table:
``cache(key TEXT PRIMARY KEY, payload TEXT, ts REAL)``.
"""

from pathlib import Path
import sqlite3
import json
import time
from typing import Tuple, Any

DB_PATH = Path("outputs/cache.db")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, payload TEXT, ts REAL)"
    )
    return conn


def get_cache(key: str, ttl: float) -> Tuple[Any | None, bool]:
    """Return cached payload if present and not expired."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT payload, ts FROM cache WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None, False
    payload, ts_val = row
    if time.time() - ts_val > ttl:
        return None, False
    return json.loads(payload), True


def set_cache(key: str, payload: Any) -> None:
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "REPLACE INTO cache(key, payload, ts) VALUES (?, ?, ?)",
        (key, json.dumps(payload), time.time()),
    )
    conn.commit()
    conn.close()
