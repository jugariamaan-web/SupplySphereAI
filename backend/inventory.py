"""
inventory.py
------------------------------------------------------------
Pure BUSINESS LOGIC & INVENTORY FORMULAS.

Calculates classic supply chain formulas:
    1. Average Daily Demand   -> from product_stats
    2. Safety Stock           -> Z * std_dev * sqrt(lead_time)
    3. Reorder Point (ROP)    -> (avg_demand * lead_time) + safety_stock
    4. Economic Order Qty     -> sqrt((2 * annual_demand * order_cost) / holding_cost)
    5. Recommended Order Qty  -> EOQ if stock < ROP else 0
    6. Days Remaining         -> stock / avg_daily_demand
    7. Inventory Health       -> Critical / Warning / Healthy
    8. Stockout Probability   -> derived from inventory_risk table / formula
------------------------------------------------------------
"""

import math
from typing import Optional
from ml_service import get_product_stats, get_inventory_risk

# MIGRATION NOTE: the old dataset had no real supplier lead-time data, so
# every product used this same hardcoded constant. The new dataset
# provides real Supplier_Lead_Time_Days per row, aggregated per product
# into product_stats.avg_supplier_lead_time_days (see ml/preprocess.py).
# calculate_inventory_plan() now uses that real value when available and
# falls back to this constant only if it's missing (e.g. against an
# un-migrated database), so the formulas themselves are unchanged.
DEFAULT_LEAD_TIME_DAYS = 7
SERVICE_Z_SCORE = 1.65  # 95% service level
ORDERING_COST = 50.0
HOLDING_COST_PER_UNIT = 2.0


def classify_inventory_health(avg_daily_demand: float, current_stock: float, lead_time_days: float):
    """
    Pure function (no DB access): classifies inventory health from runway
    (days_remaining) vs. lead time - i.e. "will stock last until the next
    order arrives?" - rather than a raw stock vs. reorder-point comparison,
    which flagged almost every product as Critical/Warning since a snapshot
    average is nearly always below its own reorder point.

    Extracted out of calculate_inventory_plan() so callers that already
    have product_stats rows in hand (e.g. the dashboard's health-breakdown
    loop in main.py, which needs this for every product) can classify
    health WITHOUT an extra per-product database round trip. Returns
    (inventory_health: str, days_remaining: float).
    """
    if avg_daily_demand <= 0:
        return "Healthy", 0.0

    days_remaining = round(current_stock / avg_daily_demand, 1)
    if days_remaining < lead_time_days:
        inventory_health = "Critical"
    elif days_remaining < lead_time_days * 1.5:
        inventory_health = "Warning"
    else:
        inventory_health = "Healthy"
    return inventory_health, days_remaining


def calculate_inventory_plan(product_id: str, current_stock: Optional[float] = None):
    """
    Calculates inventory optimization parameters for a product.
    """
    if not product_id:
        return {"error": "Product ID is required."}

    stats = get_product_stats(product_id)
    if not stats:
        return {"error": f"No statistics found for product {product_id}."}

    avg_daily_demand = stats["avg_daily_demand"]
    demand_std_dev = stats["demand_std_dev"]
    default_stock = stats["avg_inventory_level"]
    lead_time_days = stats.get("avg_supplier_lead_time_days") or DEFAULT_LEAD_TIME_DAYS

    if current_stock is None or current_stock < 0:
        current_stock = default_stock

    if avg_daily_demand <= 0:
        return {"error": "Average daily demand must be greater than zero."}

    # 1. Safety Stock
    safety_stock = SERVICE_Z_SCORE * demand_std_dev * math.sqrt(lead_time_days)

    # 2. Reorder Point
    reorder_point = (avg_daily_demand * lead_time_days) + safety_stock

    # 3. Economic Order Quantity (EOQ)
    annual_demand = avg_daily_demand * 365
    eoq = math.sqrt((2 * annual_demand * ORDERING_COST) / HOLDING_COST_PER_UNIT)

    # 4. Recommended Order Quantity
    if current_stock < reorder_point:
        recommended_order_qty = eoq
    else:
        recommended_order_qty = 0.0

    # 5. Days Remaining & 6. Inventory Health (shared with the dashboard's
    # bulk classifier - see classify_inventory_health() above)
    inventory_health, days_remaining = classify_inventory_health(avg_daily_demand, current_stock, lead_time_days)

    risk_records = get_inventory_risk(product_id)
    if risk_records:
        avg_risk_score = sum(r["risk_score"] for r in risk_records) / len(risk_records)
        stockout_probability = round(avg_risk_score * 100, 1)
    else:
        # Fallback heuristic, kept consistent with the health buckets above
        if inventory_health == "Critical":
            stockout_probability = 85.0
        elif inventory_health == "Warning":
            stockout_probability = 45.0
        else:
            stockout_probability = 10.0

    return {
        "product_id": stats["product_id"],
        "product_name": stats["product_name"],
        "category": stats["category"],
        "avg_daily_demand": round(avg_daily_demand, 2),
        "demand_std_dev": round(demand_std_dev, 2),
        "lead_time_days": round(lead_time_days, 1),
        "current_stock": round(current_stock, 1),
        "eoq": round(eoq, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "recommended_order_qty": round(recommended_order_qty, 2),
        "days_remaining": days_remaining,
        "inventory_health": inventory_health,
        "stockout_probability": stockout_probability,
    }

