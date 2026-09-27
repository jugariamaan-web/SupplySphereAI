"""
preprocess.py
------------------------------------------------------------
STEP 1 of the SupplySphere AI ML pipeline.

MIGRATION NOTE (2024-2025 synthetic dataset):
This dataset (data/raw/supplysphere_dataset/) is fully feature-engineered
already - it ships prediction-time features (demand_lag1, demand_roll14_mean,
days_of_supply, etc.) and a validated forward-looking risk label
(stockout_next_7_days). This script therefore does NOT need to hand-roll
those features (train_forecast.py / train_risk.py read the raw CSV directly
for that). Its job is:

    1. Load the raw dataset + its 3 master tables (product/warehouse/supplier)
       and the replenishment orders log.
    2. Re-validate structural integrity (dates, duplicates, missing values,
       inventory-balance sanity) - belt-and-braces, since the bundle already
       ships its own validation report, but we don't trust a file blindly.
    3. Build the aggregated tables the FastAPI backend actually reads:
         - warehouses / suppliers (static reference data)
         - daily_sales   (1 row/date, summed across all SKUs x warehouses)
         - product_stats (1 row/SKU, aggregated across all 5 warehouses)
    4. Write those to data/processed/*.csv AND push them into MySQL via
       ml/save_to_db.py.

This script does NOT modify supplysphere_synthetic_2024_2025.csv itself -
that dataset is treated as the validated, immutable source of truth.
------------------------------------------------------------
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from utils import ensure_folder_exists
import save_to_db

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "supplysphere_dataset"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"

MAIN_CSV = RAW_DIR / "supplysphere_synthetic_2024_2025.csv"
PRODUCT_MASTER_CSV = RAW_DIR / "product_master.csv"
WAREHOUSE_MASTER_CSV = RAW_DIR / "warehouse_master.csv"
SUPPLIER_MASTER_CSV = RAW_DIR / "supplier_master.csv"
REPLENISHMENT_CSV = RAW_DIR / "replenishment_orders_log.csv"


def load_raw_data():
    print("[preprocess] Loading raw dataset files ...")
    main_df = pd.read_csv(MAIN_CSV, parse_dates=["date"])
    product_master = pd.read_csv(PRODUCT_MASTER_CSV)
    warehouse_master = pd.read_csv(WAREHOUSE_MASTER_CSV)
    supplier_master = pd.read_csv(SUPPLIER_MASTER_CSV)
    replenishment = pd.read_csv(REPLENISHMENT_CSV, parse_dates=["order_date", "expected_arrival_date"])
    print(f"[preprocess] Main dataset: {len(main_df):,} rows, {main_df['date'].min().date()} -> {main_df['date'].max().date()}")
    return main_df, product_master, warehouse_master, supplier_master, replenishment


def validate_structural_integrity(df: pd.DataFrame):
    """Re-validates the core invariants the dataset bundle claims to hold."""
    print("[preprocess] Validating structural integrity ...")
    issues = []

    n_dup_rows = df.duplicated().sum()
    if n_dup_rows:
        issues.append(f"{n_dup_rows} duplicate rows")

    n_dup_keys = df.duplicated(subset=["sku_id", "warehouse_id", "date"]).sum()
    if n_dup_keys:
        issues.append(f"{n_dup_keys} duplicate sku_id x warehouse_id x date keys")

    n_missing = df.isna().sum().sum()
    if n_missing:
        issues.append(f"{n_missing} missing values")

    n_neg_inv = (df["closing_inventory"] < 0).sum()
    if n_neg_inv:
        issues.append(f"{n_neg_inv} negative closing_inventory rows")

    balance_check = df["opening_inventory"] + df["stock_received"] - df["units_sold"] - df["closing_inventory"]
    n_balance_violations = (balance_check.abs() > 1e-6).sum()
    if n_balance_violations:
        issues.append(f"{n_balance_violations} inventory balance violations")

    n_skus = df["sku_id"].nunique()
    n_wh = df["warehouse_id"].nunique()
    n_sup = df["supplier_id"].nunique()

    if issues:
        print("[preprocess] WARNING - validation issues found:")
        for i in issues:
            print(f"   - {i}")
    else:
        print(f"[preprocess] OK: {n_skus} SKUs, {n_wh} warehouses, {n_sup} suppliers, 0 integrity issues.")

    return {
        "rows": len(df),
        "skus": n_skus,
        "warehouses": n_wh,
        "suppliers": n_sup,
        "issues": issues,
    }


def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar date, summed/averaged across ALL SKUs x warehouses."""
    print("[preprocess] Building daily_sales ...")
    grouped = df.groupby("date").agg(
        total_units_sold=("units_sold", "sum"),
        avg_inventory_level=("closing_inventory", "mean"),
    ).reset_index()
    grouped["generated_at"] = datetime.now()
    return grouped


def build_product_stats(df: pd.DataFrame, product_master: pd.DataFrame, replenishment: pd.DataFrame) -> pd.DataFrame:
    """One row per SKU, aggregated across all 5 warehouses and both years."""
    print("[preprocess] Building product_stats ...")

    # Daily total demand/inventory PER SKU, summed across all 5 warehouses
    # first, THEN averaged over days. Both avg_daily_demand and
    # avg_inventory_level must use the SAME "summed across warehouses"
    # aggregation, or days_remaining (inventory / demand) is comparing a
    # single-warehouse average stock level against a 5-warehouse-total
    # demand figure - which understates runway ~5x and misclassifies
    # every product as Critical.
    per_sku_daily = df.groupby(["sku_id", "date"]).agg(
        units_sold=("units_sold", "sum"),
        closing_inventory=("closing_inventory", "sum"),
    ).reset_index()
    demand_stats = per_sku_daily.groupby("sku_id").agg(
        avg_daily_demand=("units_sold", "mean"),
        demand_std_dev=("units_sold", "std"),
        avg_inventory_level=("closing_inventory", "mean"),
    ).reset_index()

    inv_stats = df.groupby("sku_id").agg(
        avg_supplier_lead_time_days=("lead_time_days", "mean"),
    ).reset_index()

    order_stats = replenishment.groupby("sku_id").agg(
        avg_units_ordered=("order_qty", "mean"),
    ).reset_index()

    stats = demand_stats.merge(inv_stats, on="sku_id", how="left")
    stats = stats.merge(order_stats, on="sku_id", how="left")
    stats = stats.merge(
        product_master[["sku_id", "product_name", "category", "supplier_id"]],
        on="sku_id", how="left",
    )

    stats["demand_std_dev"] = stats["demand_std_dev"].fillna(0.0)
    stats["avg_units_ordered"] = stats["avg_units_ordered"].fillna(0.0)
    stats["generated_at"] = datetime.now()

    stats = stats.rename(columns={"sku_id": "product_id"})
    cols = [
        "product_id", "product_name", "category", "supplier_id",
        "avg_daily_demand", "demand_std_dev", "avg_inventory_level",
        "avg_units_ordered", "avg_supplier_lead_time_days", "generated_at",
    ]
    return stats[cols]


def main():
    ensure_folder_exists(str(PROCESSED_DIR))
    main_df, product_master, warehouse_master, supplier_master, replenishment = load_raw_data()

    validation = validate_structural_integrity(main_df)

    daily_sales = build_daily_sales(main_df)
    product_stats = build_product_stats(main_df, product_master, replenishment)

    daily_sales.to_csv(PROCESSED_DIR / "daily_sales.csv", index=False)
    product_stats.to_csv(PROCESSED_DIR / "product_stats.csv", index=False)
    print(f"[preprocess] Wrote {PROCESSED_DIR / 'daily_sales.csv'} ({len(daily_sales)} rows)")
    print(f"[preprocess] Wrote {PROCESSED_DIR / 'product_stats.csv'} ({len(product_stats)} rows)")

    save_to_db.save_reference_tables(warehouse_master, supplier_master)
    save_to_db.save_daily_sales(daily_sales)
    save_to_db.save_product_stats(product_stats)

    print("[preprocess] Done.")
    return validation


if __name__ == "__main__":
    main()
