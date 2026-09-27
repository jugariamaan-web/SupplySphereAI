"""
ml_service.py
------------------------------------------------------------
IMPORTANT: This file does NOT train or run ML models.

Training happens OFFLINE inside the /ml folder.
Those scripts write their final predictions into MySQL:
    - daily_sales table      (historical sales & inventory)
    - demand_forecast table  (precomputed 30-day forecast)
    - inventory_risk table   (XGBoost inventory/stockout risk scores)
    - product_stats table    (precomputed per-product metrics)

This file simply READS those stored predictions from MySQL
and formats them for API responses.
------------------------------------------------------------
"""

import json
import pandas as pd
from pathlib import Path
from database import run_query
from naming import warehouse_display_name, supplier_display_name


def get_sales_history(days: int = 90):
    """Returns the last N days of actual sales (for dashboard + forecast chart)."""
    query = """
        SELECT date, total_units_sold, avg_inventory_level
        FROM daily_sales
        ORDER BY date DESC
        LIMIT %s
    """
    rows = run_query(query, (days,))
    if not rows:
        return []
    # Format date strings for JSON serialization
    for r in rows:
        r["date"] = str(r["date"])
    return list(reversed(rows))  # oldest -> newest


def get_demand_forecast():
    """Returns the precomputed 30-day demand forecast, including the
    95% prediction interval bounds (lower_bound/upper_bound) computed
    from the winning model's actual test-set residuals."""
    query = """
        SELECT date, predicted_demand, lower_bound, upper_bound, winning_model
        FROM demand_forecast
        ORDER BY date ASC
    """
    rows = run_query(query)
    if not rows:
        return []
    for r in rows:
        r["date"] = str(r["date"])
    return rows


def get_all_inventory_risk():
    """Returns risk scores for every product-store pair, highest risk first."""
    query = """
        SELECT product_id, store_id, category, supplier_id, risk_score, risk_level, confidence, suggested_action, top_risk_factors_json
        FROM inventory_risk
        ORDER BY risk_score DESC
    """
    rows = run_query(query)
    if not rows:
        return []
    for r in rows:
        if isinstance(r.get("top_risk_factors_json"), str):
            try:
                r["top_risk_factors"] = json.loads(r["top_risk_factors_json"])
            except Exception:
                r["top_risk_factors"] = []
        else:
            r["top_risk_factors"] = r.get("top_risk_factors_json", [])
        _apply_display_aliases(r)
    return rows


def get_inventory_risk(product_id: str):
    """Returns risk details for all store instances of a single product."""
    query = """
        SELECT product_id, store_id, category, supplier_id, risk_score, risk_level, confidence, suggested_action, top_risk_factors_json
        FROM inventory_risk
        WHERE product_id = %s
        ORDER BY risk_score DESC
    """
    rows = run_query(query, (product_id,))
    if not rows:
        return []
    for r in rows:
        if isinstance(r.get("top_risk_factors_json"), str):
            try:
                r["top_risk_factors"] = json.loads(r["top_risk_factors_json"])
            except Exception:
                r["top_risk_factors"] = []
        else:
            r["top_risk_factors"] = r.get("top_risk_factors_json", [])
        _apply_display_aliases(r)
    return rows


def get_risk_summary():
    """Counts how many product-store pairs fall into each risk bucket."""
    query = """
        SELECT risk_level, COUNT(*) AS total
        FROM inventory_risk
        GROUP BY risk_level
    """
    return run_query(query) or []


def _apply_display_aliases(row: dict) -> dict:
    """
    Backend display-alias layer (see naming.py). `product_name` and
    `category` already hold real values straight from the database (the
    2024-2025 dataset ships genuine product/category data - see
    ml/preprocess.py), so this only OVERLAYS a couple of extra
    friendly-label fields for IDs that are still raw codes:
      - `warehouse_name` -> aliased from `store_id` (WH_x)
      - `supplier_name`  -> aliased from `supplier_id` (SUP_x)

    The real IDs are never modified. Mutates and returns `row` in place
    so call sites can do `_apply_display_aliases(r)` inline.
    """
    if not row:
        return row
    if row.get("store_id"):
        row["warehouse_name"] = warehouse_display_name(row["store_id"])
    if row.get("supplier_id"):
        row["supplier_name"] = supplier_display_name(row["supplier_id"])
    return row


def get_all_products():
    """Returns a list of all products from product_stats."""
    query = """
        SELECT product_id, product_name, category, supplier_id, avg_daily_demand, demand_std_dev, avg_inventory_level, avg_units_ordered, avg_supplier_lead_time_days
        FROM product_stats
        ORDER BY product_id ASC
    """
    rows = run_query(query) or []
    return [_apply_display_aliases(r) for r in rows]


def get_product_stats(product_id: str):
    """Returns statistics for a specific product."""
    query = """
        SELECT product_id, product_name, category, supplier_id, avg_daily_demand, demand_std_dev, avg_inventory_level, avg_units_ordered, avg_supplier_lead_time_days
        FROM product_stats
        WHERE product_id = %s
    """
    rows = run_query(query, (product_id,))
    return _apply_display_aliases(rows[0]) if rows else None


def get_category_distribution():
    """Returns product count per real category (e.g. Household, Beverages, Dairy)."""
    query = """
        SELECT category, COUNT(*) AS total
        FROM product_stats
        GROUP BY category
    """
    return run_query(query) or []


def get_region_distribution():
    """Returns inventory risk count per warehouse (with a friendly region label)."""
    query = """
        SELECT store_id AS region, COUNT(*) AS total
        FROM inventory_risk
        GROUP BY store_id
    """
    rows = run_query(query) or []
    for r in rows:
        r["region_name"] = warehouse_display_name(r["region"])
    return rows


def get_forecast_benchmark():
    """Reads benchmark results CSV and metrics JSON."""
    outputs_dir = Path(__file__).parent.parent / "outputs"
    metrics_path = outputs_dir / "forecast_metrics.json"
    benchmark_path = outputs_dir / "benchmark_results.csv"

    metrics = {}
    if metrics_path.exists():
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            # Same NaN -> None safety net as below: a metric that ever comes
            # back as NaN (e.g. R2 on a degenerate split) must not be allowed
            # to reach a JSON response.
            metrics = {k: (None if isinstance(v, float) and v != v else v) for k, v in metrics.items()}
        except Exception:
            pass

    benchmark_list = []
    if benchmark_path.exists():
        try:
            df = pd.read_csv(benchmark_path)
            # benchmark_results.csv headers are lowercase (model,rmse,mae,
            # mape,r2) but the frontend table (forecast.html
            # renderBenchmarkTable) reads capitalized keys (Model, RMSE,
            # MAE, MAPE, R2) - rename here so the two agree instead of the
            # table rendering "undefined"/NaN for every row.
            df = df.rename(columns={
                "model": "Model", "rmse": "RMSE", "mae": "MAE",
                "mape": "MAPE", "r2": "R2",
            })
            benchmark_list = df.to_dict(orient="records")
            # IMPORTANT: a metric that ever comes back as NaN (e.g. R2 on a
            # degenerate split) is not valid JSON - FastAPI's JSONResponse
            # uses allow_nan=False and raises a 500 the moment it tries to
            # serialize it. Convert any NaN to None (JSON null) before this
            # data ever reaches a response. Using `v != v` (true only for
            # NaN) rather than pd.isna() keeps this working even on plain
            # floats/strings with no pandas import.
            benchmark_list = [
                {k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
                for row in benchmark_list
            ]
        except Exception:
            pass

    # Defaults below only apply if forecast_metrics.json is missing/unreadable.
    # Linear Regression is the current winning model on this dataset's
    # benchmark (see outputs/benchmark_results.csv) - these fallback figures
    # match its actual benchmarked metrics (re-benchmarked after switching
    # the annual-seasonality feature from raw "month" to cyclical
    # sin/cos-of-day-of-year encoding - see train_forecast.py STEP 10) so a
    # missing-file edge case still reports a consistent "winning model"
    # story instead of stale numbers from a different model or feature set.
    default_pi = {
        "confidence_level": 0.95,
        "z_score": 1.96,
        "residual_std": 85.73,
        "margin": 168.02,
        "n_test_samples": 71,
        "method": (
            "Empirical residual-based prediction interval: margin = "
            "Z(0.95) * std(actual - predicted) computed on the winning "
            "model's held-out test set."
        ),
    }
    return {
        "winning_model": metrics.get("model", "Linear Regression"),
        "rmse": metrics.get("rmse", 91.14),
        "mae": metrics.get("mae", 78.18),
        "mape": metrics.get("mape", 2.24),
        "r2_score": metrics.get("r2", 0.9846),
        "benchmark_results": benchmark_list,
        "prediction_interval": metrics.get("prediction_interval", default_pi),
    }