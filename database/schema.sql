-- ============================================================
-- SupplySphere AI — Database Schema
-- ============================================================
-- Dataset  : data/raw/supplysphere_dataset/supplysphere_synthetic_2024_2025.csv
--            (+ product_master.csv, warehouse_master.csv, supplier_master.csv,
--             replenishment_orders_log.csv)
-- Warehouses : WH_01 – WH_05 | SKUs : SKU_001 – SKU_050 | Suppliers : SUP_01 – SUP_10
-- Date range : 2024-01-01 to 2025-12-31 (182,750 SKU x warehouse x day rows)
--
-- MIGRATION NOTE (2024-2025 synthetic dataset migration):
-- The previous dataset had no real product names, categories, or supplier
-- IDs, so the backend repurposed `product_stats.category` to secretly hold
-- a supplier ID and relied on a hardcoded naming.py alias table to fake
-- display names (see git history / naming.py comments). The new dataset
-- ships REAL product names, categories, supplier IDs and warehouse names
-- (see data/raw/supplysphere_dataset/*_master.csv), so that hack has been
-- removed: `category` now holds the genuine product category, and a real
-- `supplier_id` column has been added wherever the old category-overload
-- lived. Table/column names consumed by the FastAPI backend (product_id,
-- store_id, category, product_name, avg_daily_demand, etc.) are otherwise
-- UNCHANGED so backend/*.py did not need to be rewritten.
--
-- DESIGN RULES:
--   • Only processed / precomputed tables. No raw per-row data tables.
--   • All values come from ML scripts — never hardcoded.
--
-- HOW TO USE:
--   1. Create the database :  CREATE DATABASE supplysphere;
--   2. Run this file       :  mysql -u root -p supplysphere < schema.sql
--   3. Run ml/preprocess.py     -> populates warehouses, suppliers, daily_sales, product_stats
--   4. Run ml/train_forecast.py -> populates demand_forecast
--   5. Run ml/train_risk.py     -> populates inventory_risk
--   (ml/save_to_db.py is called by all three above to do the actual writes)
-- ============================================================

CREATE DATABASE IF NOT EXISTS supplysphere;
USE supplysphere;

-- ----------------------------------------------------------------
-- TABLE 1 : warehouses
-- ----------------------------------------------------------------
-- Static reference data from warehouse_master.csv.
-- ----------------------------------------------------------------
DROP TABLE IF EXISTS warehouses;
CREATE TABLE warehouses (
    warehouse_id        VARCHAR(16)  NOT NULL PRIMARY KEY,   -- e.g. WH_01
    warehouse_name       VARCHAR(128) NOT NULL,               -- e.g. North Hub
    region               VARCHAR(64)  NOT NULL,
    demand_scale_factor  FLOAT        NOT NULL
);

-- ----------------------------------------------------------------
-- TABLE 2 : suppliers
-- ----------------------------------------------------------------
-- Static reference data from supplier_master.csv.
-- ----------------------------------------------------------------
DROP TABLE IF EXISTS suppliers;
CREATE TABLE suppliers (
    supplier_id          VARCHAR(16)  NOT NULL PRIMARY KEY,   -- e.g. SUP_01
    supplier_name         VARCHAR(128) NOT NULL,
    mean_lead_time_days    FLOAT        NOT NULL,
    lead_time_std_days      FLOAT        NOT NULL
);

-- ----------------------------------------------------------------
-- TABLE 3 : daily_sales
-- ----------------------------------------------------------------
-- What it stores :
--   Aggregated daily inventory metrics across ALL warehouses/SKUs.
--   One row per calendar date (2024-01-01 .. 2025-12-31).
--
-- Populated by : ml/preprocess.py (via ml/save_to_db.py)
-- Used by      : Dashboard sales chart, Demand Forecast page history
-- ----------------------------------------------------------------
DROP TABLE IF EXISTS daily_sales;
CREATE TABLE daily_sales (
    date                DATE         NOT NULL PRIMARY KEY,
    total_units_sold    INT          NOT NULL,   -- sum of units_sold across all warehouses/SKUs that day
    avg_inventory_level FLOAT        NOT NULL,   -- average closing_inventory across all rows that day
    generated_at        DATETIME     NOT NULL
);

-- ----------------------------------------------------------------
-- TABLE 4 : demand_forecast
-- ----------------------------------------------------------------
-- What it stores :
--   The winning ML model's predicted AGGREGATE daily demand for the
--   NEXT 30 days (2026-01-01 onward). Completely overwritten every
--   time train_forecast.py is run.
--
-- Populated by : ml/train_forecast.py (via ml/save_to_db.py)
-- Used by      : Dashboard forecast card, Forecast page chart, Copilot
-- ----------------------------------------------------------------
DROP TABLE IF EXISTS demand_forecast;
CREATE TABLE demand_forecast (
    date             DATE        NOT NULL PRIMARY KEY,
    predicted_demand FLOAT       NOT NULL,
    lower_bound      FLOAT       NULL,       -- 95% prediction interval lower bound (residual-based)
    upper_bound      FLOAT       NULL,       -- 95% prediction interval upper bound (residual-based)
    winning_model    VARCHAR(64) NOT NULL,   -- e.g. 'Linear Regression', 'XGBoost'
    generated_at     DATETIME    NOT NULL
);

-- ----------------------------------------------------------------
-- TABLE 5 : inventory_risk
-- ----------------------------------------------------------------
-- What it stores :
--   7-day-forward stockout risk prediction for every SKU x warehouse
--   pair (250 combinations), based on that pair's most recent fully
--   observed feature row. Completely overwritten every time
--   train_risk.py is run.
--
-- Populated by : ml/train_risk.py (via ml/save_to_db.py)
-- Used by      : Inventory Risk page, Dashboard risk summary, Copilot
-- ----------------------------------------------------------------
DROP TABLE IF EXISTS inventory_risk;
CREATE TABLE inventory_risk (
    id                     INT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
    product_id             VARCHAR(16)  NOT NULL,   -- e.g. SKU_001
    store_id               VARCHAR(16)  NOT NULL,   -- warehouse_id, e.g. WH_01
    category               VARCHAR(64)  NOT NULL,   -- real product category (e.g. Household)
    supplier_id            VARCHAR(16)  NOT NULL,   -- real supplier ID, e.g. SUP_01
    risk_score             FLOAT        NOT NULL,   -- 0.0 (safe) to 1.0 (critical) - model probability of stockout in next 7 days
    risk_level             VARCHAR(10)  NOT NULL,   -- 'Low', 'Medium', or 'High'
    confidence              FLOAT        NOT NULL,   -- model probability confidence (0.0-1.0)
    suggested_action        TEXT         NOT NULL,
    top_risk_factors_json    TEXT         NOT NULL,   -- JSON array of top 3 contributing features
    generated_at            DATETIME     NOT NULL,
    UNIQUE KEY uq_product_store (product_id, store_id)
);

-- ----------------------------------------------------------------
-- TABLE 6 : product_stats
-- ----------------------------------------------------------------
-- What it stores :
--   Precomputed historical statistics per SKU (aggregated across all
--   5 warehouses). These power the Inventory Optimization formulas:
--   EOQ, Safety Stock, Reorder Point, Days Remaining.
--
-- Populated by : ml/preprocess.py (via ml/save_to_db.py)
-- Used by      : Inventory Optimization page, Copilot, Dashboard
-- ----------------------------------------------------------------
DROP TABLE IF EXISTS product_stats;
CREATE TABLE product_stats (
    product_id                  VARCHAR(16)  NOT NULL PRIMARY KEY,   -- SKU_001..SKU_050
    product_name                VARCHAR(255) NOT NULL,               -- real name, e.g. "Dish Wash Liquid 500ml"
    category                    VARCHAR(64)  NOT NULL,               -- real category, e.g. "Household"
    supplier_id                 VARCHAR(16)  NOT NULL,               -- primary supplier for this SKU
    avg_daily_demand            FLOAT        NOT NULL,   -- average units_sold per day, summed across warehouses
    demand_std_dev              FLOAT        NOT NULL,   -- standard deviation of daily units_sold (for safety stock)
    avg_inventory_level         FLOAT        NOT NULL,   -- average closing_inventory across warehouses
    avg_units_ordered           FLOAT        NOT NULL,   -- average replenishment order_qty (replenishment_orders_log.csv)
    avg_supplier_lead_time_days FLOAT        NOT NULL,   -- real avg realized lead_time_days - used by inventory.py EOQ/ROP formulas
    generated_at                 DATETIME     NOT NULL
);

-- ============================================================
-- END OF SCHEMA
-- ============================================================
