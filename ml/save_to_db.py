"""
save_to_db.py
============================================================
SupplySphere AI - Database Seeding & Result Saving
============================================================
Central place that writes ML pipeline outputs into MySQL. Each stage of
the pipeline calls the relevant function here directly (preprocess.py ->
save_reference_tables/save_daily_sales/save_product_stats,
train_forecast.py -> save_demand_forecast, train_risk.py ->
save_inventory_risk) so results land in the DB immediately after being
computed, without needing a separate CSV round-trip.

Running this file directly (`python save_to_db.py`) re-loads everything
from data/processed/*.csv and data/raw/supplysphere_dataset/*_master.csv
and re-saves it all - useful for re-seeding MySQL without re-running the
full ML training.

MIGRATION NOTE (2024-2025 synthetic dataset):
Added save_reference_tables() for the new `warehouses` / `suppliers`
tables. product_stats and inventory_risk now carry a real `supplier_id`
column (previously the old dataset had no real supplier IDs, so
`category` was secretly repurposed to hold one - see naming.py history).
============================================================
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

from utils import get_mysql_connection

# Resolve all paths relative to this file's location (project/ml/save_to_db.py),
# NOT the current working directory. This makes the script work identically
# whether it's launched from project/, project/ml/, run_project.bat, or an IDE.
# Same pattern already used in preprocess.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "supplysphere_dataset"


def save_reference_tables(warehouses_df: pd.DataFrame, suppliers_df: pd.DataFrame):
    """Writes the static warehouses and suppliers reference tables."""
    connection = get_mysql_connection()
    if connection is None:
        print("[ERROR] Cannot connect to MySQL. Check your .env credentials.")
        return
    cursor = connection.cursor()

    print("\n--- Saving warehouses ---")
    cursor.execute("DELETE FROM warehouses")
    cursor.executemany(
        """
        INSERT INTO warehouses (warehouse_id, warehouse_name, region, demand_scale_factor)
        VALUES (%s, %s, %s, %s)
        """,
        [
            (str(r["warehouse_id"]), str(r["warehouse_name"]), str(r["region"]), float(r["demand_scale_factor"]))
            for _, r in warehouses_df.iterrows()
        ],
    )
    print(f"Inserted {len(warehouses_df)} rows into 'warehouses'.")

    print("--- Saving suppliers ---")
    cursor.execute("DELETE FROM suppliers")
    cursor.executemany(
        """
        INSERT INTO suppliers (supplier_id, supplier_name, mean_lead_time_days, lead_time_std_days)
        VALUES (%s, %s, %s, %s)
        """,
        [
            (str(r["supplier_id"]), str(r["supplier_name"]), float(r["mean_lead_time_days"]), float(r["lead_time_std_days"]))
            for _, r in suppliers_df.iterrows()
        ],
    )
    print(f"Inserted {len(suppliers_df)} rows into 'suppliers'.")

    connection.commit()
    cursor.close()
    connection.close()


def save_daily_sales(sales_df: pd.DataFrame):
    """Writes the daily_sales table (1 row/date)."""
    connection = get_mysql_connection()
    if connection is None:
        print("[ERROR] Cannot connect to MySQL. Check your .env credentials.")
        return
    cursor = connection.cursor()
    now = datetime.now()

    print("--- Saving daily_sales ---")
    cursor.execute("DELETE FROM daily_sales")
    cursor.executemany(
        """
        INSERT INTO daily_sales (date, total_units_sold, avg_inventory_level, generated_at)
        VALUES (%s, %s, %s, %s)
        """,
        [
            (str(r["date"])[:10], int(r["total_units_sold"]), float(r["avg_inventory_level"]), now)
            for _, r in sales_df.iterrows()
        ],
    )
    print(f"Inserted {len(sales_df)} rows into 'daily_sales'.")
    connection.commit()
    cursor.close()
    connection.close()


def save_product_stats(stats_df: pd.DataFrame):
    """Writes the product_stats table (1 row/SKU)."""
    connection = get_mysql_connection()
    if connection is None:
        print("[ERROR] Cannot connect to MySQL. Check your .env credentials.")
        return
    cursor = connection.cursor()
    now = datetime.now()

    print("--- Saving product_stats ---")
    cursor.execute("DELETE FROM product_stats")
    cursor.executemany(
        """
        INSERT INTO product_stats
            (product_id, product_name, category, supplier_id,
             avg_daily_demand, demand_std_dev, avg_inventory_level,
             avg_units_ordered, avg_supplier_lead_time_days, generated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        [
            (
                str(r["product_id"]), str(r["product_name"]), str(r["category"]), str(r["supplier_id"]),
                float(r["avg_daily_demand"]), float(r["demand_std_dev"]), float(r["avg_inventory_level"]),
                float(r["avg_units_ordered"]), float(r["avg_supplier_lead_time_days"]), now,
            )
            for _, r in stats_df.iterrows()
        ],
    )
    print(f"Inserted {len(stats_df)} rows into 'product_stats'.")
    connection.commit()
    cursor.close()
    connection.close()


def save_demand_forecast(forecast_df: pd.DataFrame):
    """Writes the demand_forecast table (next-30-day aggregate forecast)."""
    connection = get_mysql_connection()
    if connection is None:
        print("[ERROR] Cannot connect to MySQL. Check your .env credentials.")
        return
    cursor = connection.cursor()
    now = datetime.now()

    print("--- Saving demand_forecast ---")
    cursor.execute("DELETE FROM demand_forecast")
    has_pi_cols = "lower_bound" in forecast_df.columns and "upper_bound" in forecast_df.columns
    cursor.executemany(
        """
        INSERT INTO demand_forecast (date, predicted_demand, lower_bound, upper_bound, winning_model, generated_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        [
            (
                str(r["date"])[:10], float(r["predicted_demand"]),
                float(r["lower_bound"]) if has_pi_cols else None,
                float(r["upper_bound"]) if has_pi_cols else None,
                str(r.get("winning_model", "Unknown")), now,
            )
            for _, r in forecast_df.iterrows()
        ],
    )
    print(f"Inserted {len(forecast_df)} rows into 'demand_forecast'.")
    connection.commit()
    cursor.close()
    connection.close()


def save_inventory_risk(risk_df: pd.DataFrame):
    """Writes the inventory_risk table (1 row per SKU x warehouse pair)."""
    connection = get_mysql_connection()
    if connection is None:
        print("[ERROR] Cannot connect to MySQL. Check your .env credentials.")
        return
    cursor = connection.cursor()
    now = datetime.now()

    print("--- Saving inventory_risk ---")
    cursor.execute("DELETE FROM inventory_risk")
    cursor.executemany(
        """
        INSERT INTO inventory_risk
            (product_id, store_id, category, supplier_id, risk_score, risk_level,
             confidence, suggested_action, top_risk_factors_json, generated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        [
            (
                str(r["product_id"]), str(r["store_id"]), str(r["category"]), str(r["supplier_id"]),
                float(r["risk_score"]), str(r["risk_level"]), float(r["confidence"]),
                str(r["suggested_action"]),
                json.dumps(r["top_risk_factors"]) if not isinstance(r["top_risk_factors"], str) else r["top_risk_factors"],
                now,
            )
            for _, r in risk_df.iterrows()
        ],
    )
    print(f"Inserted {len(risk_df)} rows into 'inventory_risk'.")
    connection.commit()
    cursor.close()
    connection.close()


def save_data_to_mysql():
    """Standalone re-seed entry point: reads everything back from disk and re-saves it."""
    warehouse_csv = RAW_DATASET_DIR / "warehouse_master.csv"
    supplier_csv = RAW_DATASET_DIR / "supplier_master.csv"
    if not warehouse_csv.exists() or not supplier_csv.exists():
        print(f"[ERROR] Raw dataset not found under {RAW_DATASET_DIR}")
        print("        Check that data/raw/supplysphere_dataset/ exists at the project root.")
        return

    warehouses_df = pd.read_csv(warehouse_csv)
    suppliers_df = pd.read_csv(supplier_csv)
    save_reference_tables(warehouses_df, suppliers_df)

    sales_path = PROCESSED_DIR / "daily_sales.csv"
    stats_path = PROCESSED_DIR / "product_stats.csv"
    forecast_path = PROCESSED_DIR / "demand_forecast.csv"
    risk_path = PROCESSED_DIR / "inventory_risk.csv"

    if sales_path.exists():
        save_daily_sales(pd.read_csv(sales_path))
    else:
        print(f"[WARNING] {sales_path} not found - run preprocess.py first.")

    if stats_path.exists():
        save_product_stats(pd.read_csv(stats_path))
    else:
        print(f"[WARNING] {stats_path} not found - run preprocess.py first.")

    if forecast_path.exists():
        save_demand_forecast(pd.read_csv(forecast_path))
    else:
        print(f"[WARNING] {forecast_path} not found - run train_forecast.py first.")

    if risk_path.exists():
        risk_df = pd.read_csv(risk_path)
        # inventory_risk.csv already stores this column as "top_risk_factors";
        # only remap it if some other version of the CSV used the "_json" suffix.
        if "top_risk_factors" not in risk_df.columns and "top_risk_factors_json" in risk_df.columns:
            risk_df["top_risk_factors"] = risk_df["top_risk_factors_json"]
        save_inventory_risk(risk_df)
    else:
        print(f"[WARNING] {risk_path} not found - run train_risk.py first.")

    print("\nAll data committed to MySQL successfully.")


if __name__ == "__main__":
    save_data_to_mysql()