from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

HEADLESS = os.getenv("BIDDER_HEADLESS", "false").lower() == "true"
SCRAPE_DELAY_SEC = float(os.getenv("SCRAPE_DELAY_SEC", "2.0"))
EBAY_APP_ID = os.getenv("EBAY_APP_ID")
