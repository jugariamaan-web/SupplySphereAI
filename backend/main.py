"""
main.py
------------------------------------------------------------
FastAPI entry point for SupplySphere AI.

This file defines API routes and serves the frontend templates.
All real logic lives in:
    - ml_service.py   -> reads precomputed results from MySQL
    - inventory.py    -> calculates EOQ / safety stock live
    - copilot.py      -> handles AI Copilot & report generation

Run locally with:
    uvicorn main:app --reload
------------------------------------------------------------
"""

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import logging
import os

import ml_service
import inventory
import copilot
import db_migrations
import database
import report_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("supplysphere")

app = FastAPI(title="SupplySphere AI API", version="1.0")

# CORS: allow_origins=["*"] is fine for a local diploma-project demo but is
# too permissive for anything reachable outside localhost - it lets ANY
# website the user's browser visits make authenticated-looking requests to
# this API and read the JSON back. Read the allowed origin(s) from an env
# var instead (comma-separated), defaulting to the two ports Vite/uvicorn
# dev setups commonly use, so nothing breaks locally but a deployment can
# (and should) set CORS_ALLOWED_ORIGINS to its real frontend URL(s).
_default_origins = "http://localhost:8000,http://127.0.0.1:8000"
CORS_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.on_event("startup")
def run_startup_schema_check():
    """
    Self-heals a common deployment drift: a previously-created MySQL
    volume that predates the `avg_supplier_lead_time_days` column on
    product_stats (see db_migrations.py for the full explanation).
    Runs once per process start; each check is a cheap no-op once the
    schema is current, so this is safe to leave in place permanently.
    """
    db_migrations.ensure_schema_compatibility()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak a raw stack trace to the client - log it and return a clean message."""
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on the server. Please check the backend logs."},
    )

templates_dir = Path(__file__).parent.parent / "supplytemplates"

# Served files (JS + HTML pages) change frequently during development/grading.
# FileResponse sets Last-Modified but no Cache-Control by default, which lets
# some browsers apply heuristic caching and keep serving an OLD cached copy
# even after the file on disk has been updated. Force revalidation on every
# request so the browser always checks for (and gets) the latest version.
NO_CACHE_HEADERS = {"Cache-Control": "no-cache, must-revalidate"}

@app.get("/api.js")
async def serve_api_js():
    return FileResponse(templates_dir / "api.js", headers=NO_CACHE_HEADERS)

@app.get("/components.js")
async def serve_components_js():
    return FileResponse(templates_dir / "components.js", headers=NO_CACHE_HEADERS)

@app.get("/theme.css")
async def serve_theme_css():
    return FileResponse(templates_dir / "theme.css", headers=NO_CACHE_HEADERS)

# ---------- Request models ----------

class CopilotRequest(BaseModel):
    message: str

class ReportRequest(BaseModel):
    report_type: str  # "procurement" | "inventory" | "risk" | "executive"

class ReportPdfRequest(BaseModel):
    report_type: str  # "procurement" | "inventory" | "risk" | "executive" | ...
    title: Optional[str] = None
    content: Optional[str] = None  # if provided (e.g. the report already shown on screen), reuse it instead of regenerating via the LLM

class InventoryRequest(BaseModel):
    product_id: str
    current_stock: Optional[float] = None


# ---------- Helper: build context for Copilot ----------

def _build_context_summary() -> dict:
    """
    Builds the ONLY data the AI Copilot / Report Generator ever sees.
    Whatever is NOT included here literally does not exist for the LLM,
    so this function is also the enforcement point for data-grounding:
    if a number isn't computed and placed into one of these strings,
    the model has no basis to reference it (and the system prompts in
    copilot.py explicitly forbid inventing it instead).

    NOTE ON SCOPE: this system has no cost, price, revenue, or margin
    data wired into any endpoint the AI reads (Unit_Cost/Price/Margin_Pct
    exist only inside ml/preprocess.py as an internal ML feature for the
    risk-classifier model - they are never surfaced here). So there is
    no way for the AI to legitimately state dollar-denominated savings,
    profit exposure, gross margin, or lost-sales figures. `financial_data`
    below says so explicitly, so the model is told this rather than
    left to infer/guess it.
    """
    forecast = ml_service.get_demand_forecast()
    risk_list = ml_service.get_all_inventory_risk()
    products = ml_service.get_all_products()

    # ---- Procurement detail for the products actually at risk ----
    # Previously this only gave the LLM ONE arbitrary "sample" product's
    # EOQ/safety stock, so any report that needed to recommend order
    # quantities for at-risk items had nothing real to draw on and filled
    # the gap with invented numbers (e.g. "1,200 units"). Instead, compute
    # real inventory-plan figures (same formulas used by the Inventory
    # page: safety stock, ROP, EOQ, recommended order qty, lead time,
    # days remaining, stockout probability) for the highest-risk products,
    # so recommendations can cite real per-product numbers.
    inventory_lines = []
    seen_products = set()
    high_risk_ids = [r["product_id"] for r in risk_list if r.get("risk_level") == "High"]
    for pid in high_risk_ids:
        if pid in seen_products:
            continue
        seen_products.add(pid)
        plan = inventory.calculate_inventory_plan(pid)
        if "error" in plan:
            continue
        inventory_lines.append(
            f"- {plan['product_name']} ({plan['product_id']}): current stock {plan['current_stock']} units, "
            f"avg daily demand {plan['avg_daily_demand']}, lead time {plan['lead_time_days']} days, "
            f"safety stock {plan['safety_stock']}, reorder point {plan['reorder_point']}, "
            f"EOQ {plan['eoq']}, recommended order qty {plan['recommended_order_qty']} units, "
            f"days of stock remaining {plan['days_remaining']}, inventory health {plan['inventory_health']}, "
            f"stockout probability {plan['stockout_probability']}%."
        )
        if len(inventory_lines) >= 5:
            break

    if inventory_lines:
        inventory_summary = (
            "Per-product inventory plan (EOQ / safety stock / reorder point are calculated "
            "live from actual demand, lead time, and demand variability - not estimates):\n"
            + "\n".join(inventory_lines)
        )
    elif products:
        # Fallback: no High-risk products right now, still ground on one real product.
        sample_prod = products[0]["product_id"]
        plan = inventory.calculate_inventory_plan(sample_prod)
        if "error" not in plan:
            inventory_summary = (
                f"Sample Product ({plan['product_name']}): avg daily demand {plan['avg_daily_demand']}, "
                f"safety stock {plan['safety_stock']}, reorder point {plan['reorder_point']}, EOQ {plan['eoq']}, "
                f"recommended order qty {plan['recommended_order_qty']} units."
            )
        else:
            inventory_summary = "No inventory data available."
    else:
        inventory_summary = "No inventory data available."

    if forecast:
        avg_forecast = sum(r["predicted_demand"] for r in forecast) / len(forecast)
        winning = forecast[0].get("winning_model", "Linear Regression")
        forecast_summary = (
            f"Winning model '{winning}' forecasts average demand of {avg_forecast:.1f} "
            f"units/day over the next {len(forecast)} days."
        )
    else:
        forecast_summary = "No forecast data available."

    if risk_list:
        high_risk = [r for r in risk_list if r["risk_level"] == "High"]
        medium_risk = [r for r in risk_list if r["risk_level"] == "Medium"]
        risk_summary = (
            f"Total product-store pairs monitored: {len(risk_list)}. "
            f"High risk pairs: {len(high_risk)}. Medium risk pairs: {len(medium_risk)}."
        )
    else:
        risk_summary = "No inventory risk data available."

    financial_data = (
        "No cost, price, margin, revenue, or profit data is available to this system. "
        "Any dollar figure, percentage margin, cost savings, or profit/revenue impact must be "
        "reported as unavailable rather than estimated."
    )

    return {
        "forecast_summary": forecast_summary,
        "risk_summary": risk_summary,
        "inventory_summary": inventory_summary,
        "financial_data": financial_data,
    }


# ---------- Frontend HTML Template Routes ----------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    p = templates_dir / "dashboard.html"
    return FileResponse(p, headers=NO_CACHE_HEADERS) if p.exists() else HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)

@app.get("/forecast", response_class=HTMLResponse)
async def serve_forecast():
    p = templates_dir / "forecast.html"
    return FileResponse(p, headers=NO_CACHE_HEADERS) if p.exists() else HTMLResponse("<h1>Forecast page not found</h1>", status_code=404)

@app.get("/inventory", response_class=HTMLResponse)
async def serve_inventory_page():
    p = templates_dir / "inventory.html"
    return FileResponse(p, headers=NO_CACHE_HEADERS) if p.exists() else HTMLResponse("<h1>Inventory page not found</h1>", status_code=404)

@app.get("/supplier-risk", response_class=HTMLResponse)
async def serve_inventory_risk_page():
    p = templates_dir / "supplier-risk.html"
    return FileResponse(p, headers=NO_CACHE_HEADERS) if p.exists() else HTMLResponse("<h1>Inventory Risk page not found</h1>", status_code=404)

@app.get("/copilot", response_class=HTMLResponse)
async def serve_copilot_page():
    p = templates_dir / "copilot.html"
    return FileResponse(p, headers=NO_CACHE_HEADERS) if p.exists() else HTMLResponse("<h1>Copilot page not found</h1>", status_code=404)

@app.get("/reports", response_class=HTMLResponse)
async def serve_reports_page():
    p = templates_dir / "reports.html"
    return FileResponse(p, headers=NO_CACHE_HEADERS) if p.exists() else HTMLResponse("<h1>Reports page not found</h1>", status_code=404)

@app.get("/api/health")
def health_check():
    db_status = database.check_connection()
    return {
        "status": "ok" if db_status["connected"] else "degraded",
        "message": "SupplySphere AI backend is running.",
        "database": db_status,
    }


# ---------- 1. Dashboard API ----------

@app.get("/api/dashboard")
def get_dashboard():
    products = ml_service.get_all_products()
    sales = ml_service.get_sales_history(days=90)
    forecast = ml_service.get_demand_forecast()

    total_products = len(products)
    total_inventory = round(sum(p["avg_inventory_level"] for p in products), 1) if products else 0.0
    units_sold = sum(s["total_units_sold"] for s in sales) if sales else 0

    # Calculate inventory health & low stock counts.
    # PERFORMANCE NOTE: this used to call inventory.calculate_inventory_plan()
    # per product, which issues 2 extra SQL queries (product_stats +
    # inventory_risk) EACH - on 50 products that's ~100 avoidable round trips
    # on every dashboard load, just to read a field (inventory_health) that
    # only needs data already sitting in `products` above. Using the shared
    # classify_inventory_health() helper directly reduces this loop to 0
    # additional queries (see inventory.py for the shared formula).
    healthy_count = 0
    warning_count = 0
    critical_count = 0
    low_stock_count = 0

    for p in products:
        lead_time_days = p.get("avg_supplier_lead_time_days") or inventory.DEFAULT_LEAD_TIME_DAYS
        h, _ = inventory.classify_inventory_health(p["avg_daily_demand"], p["avg_inventory_level"], lead_time_days)
        if h == "Healthy":
            healthy_count += 1
        elif h == "Warning":
            warning_count += 1
            low_stock_count += 1
        elif h == "Critical":
            critical_count += 1
            low_stock_count += 1

    # Risk summary map
    risk_summary_list = ml_service.get_risk_summary()
    risk_summary_map = {r["risk_level"]: r["total"] for r in risk_summary_list}

    # Forecast summary
    if forecast:
        total_fcst = sum(f["predicted_demand"] for f in forecast)
        avg_fcst = round(total_fcst / len(forecast), 1)
        winning_model = forecast[0].get("winning_model", "Linear Regression")
    else:
        total_fcst = 0.0
        avg_fcst = 0.0
        winning_model = "N/A"

    return {
        "total_products": total_products,
        "total_inventory": total_inventory,
        "units_sold": units_sold,
        "inventory_health": {
            "healthy": healthy_count,
            "warning": warning_count,
            "critical": critical_count,
        },
        "forecast_summary": {
            "total_predicted_demand": round(total_fcst, 1),
            "avg_daily_demand": avg_fcst,
            "winning_model": winning_model,
        },
        "low_stock_products": low_stock_count,
        "inventory_risk_summary": {
            "High": risk_summary_map.get("High", 0),
            "Medium": risk_summary_map.get("Medium", 0),
            "Low": risk_summary_map.get("Low", 0),
        },
        "category_distribution": ml_service.get_category_distribution(),
        "region_distribution": ml_service.get_region_distribution(),
        "sales_history": sales,
        "forecast_preview": forecast[:7] if forecast else [],
    }


# ---------- 2. Forecast API ----------

@app.get("/api/forecast")
def get_forecast():
    """
    NOTE on per_product_forecast:
    The forecasting pipeline (see /ml/train_forecast.py, currently won by
    Linear Regression on this dataset - Random Forest, XGBoost, and LSTM
    remain as benchmark comparisons) trains
    on TOTAL daily demand only - `demand_forecast` has no product_id
    column (see database/schema.sql), so there is no per-SKU model to
    read from. Training a separate forecast per product would be an
    architecture change (a different model per SKU), which is out of
    scope here.

    Instead, each product's 30-day forecast is that product's OWN
    historical share of total demand (avg_daily_demand from
    product_stats, computed per-product in ml/preprocess.py) applied to
    the aggregate forecast curve. This is a standard "top-down"
    proportional allocation, not an arbitrary/equal scaling - a product
    that historically sells 2x another's volume gets 2x the forecast.
    It does NOT capture a product's own day-to-day trend deviations
    from the aggregate; the `method` field below makes that explicit
    for API consumers instead of silently implying a per-SKU model.
    """
    forecast_data = ml_service.get_demand_forecast()
    benchmark_info = ml_service.get_forecast_benchmark()
    products = ml_service.get_all_products()

    # Build per-product breakdown
    per_product_forecast = []
    if products and forecast_data:
        total_avg_demand = sum(p["avg_daily_demand"] for p in products) or 1.0
        for p in products:
            share = p["avg_daily_demand"] / total_avg_demand
            product_30d_forecast = round(sum(f["predicted_demand"] * share for f in forecast_data), 1)
            per_product_forecast.append({
                "product_id": p["product_id"],
                "product_name": p["product_name"],
                "category": p["category"],
                "predicted_30d_demand": product_30d_forecast,
                "avg_daily_demand": round(p["avg_daily_demand"], 1),
                "method": "historical-share-allocation"
            })

    return {
        "combined_forecast": forecast_data,
        "per_product_forecast": per_product_forecast,
        "forecast_methodology": (
            "Per-product figures allocate the aggregate model forecast using "
            "each product's own historical average daily demand share."
        ),
        "winning_model": benchmark_info["winning_model"],
        "rmse": benchmark_info["rmse"],
        "mae": benchmark_info["mae"],
        "mape": benchmark_info["mape"],
        "r2_score": benchmark_info["r2_score"],
        "benchmark_results": benchmark_info["benchmark_results"],
        "prediction_interval": benchmark_info["prediction_interval"],
        "history": ml_service.get_sales_history(days=90),
    }


# ---------- 3. Inventory API ----------

@app.get("/api/inventory")
def get_inventory_get(product_id: Optional[str] = Query(None), current_stock: Optional[float] = Query(None)):
    products = ml_service.get_all_products()
    if not products:
        raise HTTPException(status_code=404, detail="No product data found in database.")

    target_pid = product_id or products[0]["product_id"]
    plan = inventory.calculate_inventory_plan(target_pid, current_stock)

    if "error" in plan:
        raise HTTPException(status_code=400, detail=plan["error"])
    return plan


@app.post("/api/inventory")
def get_inventory_post(req: InventoryRequest):
    plan = inventory.calculate_inventory_plan(req.product_id, req.current_stock)
    if "error" in plan:
        raise HTTPException(status_code=400, detail=plan["error"])
    return plan


# ---------- 4. Inventory Risk API ----------

@app.get("/api/risk")
def get_all_risk():
    risk_items = ml_service.get_all_inventory_risk()
    formatted = []
    for r in risk_items:
        formatted.append({
            "product_id": r["product_id"],
            "store_id": r["store_id"],
            "warehouse_name": r.get("warehouse_name", r["store_id"]),
            "category": r["category"],
            "supplier_id": r.get("supplier_id"),
            "supplier_name": r.get("supplier_name", r.get("supplier_id")),
            "risk_score": r["risk_score"],
            "risk_level": r["risk_level"],
            "confidence": r["confidence"],
            "suggested_action": r["suggested_action"],
            "top_risk_factors": r.get("top_risk_factors", [])
        })
    return {"inventory_risk": formatted}


@app.get("/api/risk/{product_id}")
def get_single_risk(product_id: str):
    results = ml_service.get_inventory_risk(product_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"No risk records found for product '{product_id}'.")
    return {"product_id": product_id, "risk_records": results}


# ---------- 5. Products Dropdown API ----------

@app.get("/api/products")
def get_all_products_route():
    return {"products": ml_service.get_all_products()}


# ---------- 6. AI Copilot Route ----------

@app.post("/api/copilot")
def ask_copilot_route(request: CopilotRequest):
    context_data = _build_context_summary()
    reply = copilot.ask_copilot(request.message, context_data)
    return {"reply": reply}


# ---------- 7. AI Report Route ----------

@app.post("/api/report")
def generate_report_route(request: ReportRequest):
    context_data = _build_context_summary()
    report_text = copilot.generate_report(request.report_type, context_data)
    return {"report_type": request.report_type, "content": report_text}


REPORT_TITLE_MAP = {
    "executive": "Executive Summary & AI Overview",
    "sales": "Sales Summary Report",
    "demand": "Demand Summary Report",
    "inventory": "Inventory Replenishment & Stock Report",
    "risk": "Inventory & Stockout Risk Analysis Report",
    "insights": "AI Insights Summary Report",
    "procurement": "Procurement & Purchase Order Report",
}


# ---------- 8. AI Report PDF Download ----------
# Returns a REAL PDF file (application/pdf, Content-Disposition: attachment)
# generated server-side by report_pdf.py, so the frontend can trigger a
# direct file download from a single API call - no new browser tab/page,
# no window.print() dependency, no client-side PDF library needed.
@app.post("/api/report/pdf")
def download_report_pdf_route(request: ReportPdfRequest):
    context_data = _build_context_summary()

    # Reuse the report text already generated/shown on screen when the
    # caller provides it (avoids a second, possibly different, LLM call
    # for the same report) - otherwise generate it fresh.
    report_text = request.content or copilot.generate_report(request.report_type, context_data)
    title = request.title or REPORT_TITLE_MAP.get(request.report_type, f"{request.report_type.title()} Report")

    pdf_bytes = report_pdf.build_report_pdf(title, report_text)
    filename = f"supplysphere_{request.report_type}_report.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )