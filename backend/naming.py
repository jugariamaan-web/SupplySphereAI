"""
naming.py
------------------------------------------------------------
Backend DISPLAY ALIAS layer.

MIGRATION NOTE (2024-2025 synthetic dataset):
The previous dataset shipped no real product names, categories, or
supplier IDs, so this module used to hold a hand-typed "SKU_1 -> Wireless
Mouse" style guess table, and ml_service.py repurposed the `category`
field to secretly hold a supplier ID with this module aliasing it into a
fake business name. The new dataset ships REAL product/warehouse/supplier
names in product_master.csv / warehouse_master.csv / supplier_master.csv,
so product_name, category, and supplier_id are now stored directly and
correctly in the database - no guessing or repurposing needed.

This module now does ONE remaining job: aliasing a warehouse_id
(WH_01..WH_05), which is stored as the DB's `store_id` for historical
compatibility with the frontend/API contract, to its real warehouse_name
for display. Product names/categories/supplier IDs are already correct in
the DB and need no aliasing - callers should read them directly.

Both lookup functions are total (never raise): an ID outside the known
set falls back to returning the original ID unchanged, so unknown/future
IDs degrade gracefully instead of breaking a response.
------------------------------------------------------------
"""

import csv
from pathlib import Path

_MASTER_DIR = Path(__file__).parent.parent / "data" / "raw" / "supplysphere_dataset"


def _load_map(filename: str, key_col: str, value_col: str) -> dict:
    path = _MASTER_DIR / filename
    mapping = {}
    if not path.exists():
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mapping[row[key_col]] = row[value_col]
    return mapping


# Loaded once at import time from the master reference CSVs (also mirrored
# in the `warehouses` / `suppliers` MySQL tables via ml/save_to_db.py).
WAREHOUSE_DISPLAY_NAMES = _load_map("warehouse_master.csv", "warehouse_id", "warehouse_name")
SUPPLIER_DISPLAY_NAMES = _load_map("supplier_master.csv", "supplier_id", "supplier_name")


def warehouse_display_name(warehouse_id: str) -> str:
    """Returns the business-friendly name for a warehouse_id (e.g. WH_01 -> 'North Hub')."""
    if not warehouse_id:
        return warehouse_id
    return WAREHOUSE_DISPLAY_NAMES.get(warehouse_id, warehouse_id)


def supplier_display_name(supplier_id: str) -> str:
    """Returns the business-friendly name for a supplier_id (e.g. SUP_01 -> 'Nimbus Global Trading')."""
    if not supplier_id:
        return supplier_id
    return SUPPLIER_DISPLAY_NAMES.get(supplier_id, supplier_id)
