import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE))

from src.providers import ebay_api

FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_completed_items():
    data = json.load(open(FIXTURES / "findCompletedItems.json"))
    items = ebay_api._extract_items(data, sold=True, include_shipping=True)
    assert len(items) == 2
    assert items[0]["price"] == 12.0  # 10 + 2 shipping
    assert items[1]["bestOfferAccepted"] is True


def test_extract_active_items():
    data = json.load(open(FIXTURES / "findItemsAdvanced.json"))
    items = ebay_api._extract_items(data, sold=False, include_shipping=True)
    assert len(items) == 1
    assert items[0]["listingId"] == "789"
    assert items[0]["price"] == 18.0
