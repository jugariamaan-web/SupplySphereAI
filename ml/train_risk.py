"""
train_risk.py
============================================================
SupplySphere AI - Inventory / Stockout Risk Model (2024-2025 dataset)
============================================================
Target: stockout_next_7_days (whether a SKU x warehouse pair goes out of
stock at any point in the next 7 days).

IMPORTANT: rows with label_fully_observed == 0 (the final 7 days of the
dataset, where the forward-looking label can't be fully computed) are
EXCLUDED from training/evaluation entirely - never used as if they had a
valid label.

STRICT chronological split (matches train_forecast.py):
    Train      : 2024-01-01 -> 2024-12-31
    Validation : 2025-01-01 -> 2025-06-30
    Final test : 2025-07-01 -> 2025-12-31

Only prediction-time features are used (no stockout_next_7_days or any
future-derived column in the feature matrix). Model: XGBoost classifier.
The validated benchmark for this dataset is ~86.8% final test accuracy
with a ~1.1pp train/test gap - this script does not force a higher
number; it reports whatever the real trained model achieves.

Outputs:
    outputs/risk_metrics.json        - accuracy, train/test gap, etc.
    saved_models/risk_model.pkl      - trained XGBoost classifier
    data/processed/inventory_risk.csv - 1 row per SKU x warehouse (250 rows)
    MySQL `inventory_risk` table (via save_to_db.save_inventory_risk)
============================================================
"""

import os
import json
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

from utils import ensure_folder_exists
import save_to_db

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RAW_DIR = os.path.join("..", "data", "raw", "supplysphere_dataset")
MAIN_CSV = os.path.join(RAW_DIR, "supplysphere_synthetic_2024_2025.csv")
PROCESSED_DIR = os.path.join("..", "data", "processed")
SAVED_MODELS_DIR = os.path.join("..", "saved_models")
OUTPUTS_DIR = os.path.join("..", "outputs")
for d in [PROCESSED_DIR, SAVED_MODELS_DIR, OUTPUTS_DIR]:
    ensure_folder_exists(d)

TRAIN_END = "2024-12-31"
VAL_START, VAL_END = "2025-01-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2025-12-31"

NUMERIC_FEATURES = [
    "demand_lag1", "demand_lag7_mean", "demand_roll14_mean", "demand_volatility_14d",
    "recent_demand_trend", "closing_inventory", "days_of_supply", "gap_to_reorder_point",
    "lead_time_recent_30d", "expected_lead_time_days", "supplier_reliability",
    "recent_replenishment_qty_30d", "recent_stock_received_days_14d",
    "stockout_recent_7d", "demand_spike_recent_7d", "promotion_flag",
    "day_of_week", "is_weekend",
]
CATEGORY_COL = "category"
TARGET = "stockout_next_7_days"

print("[train_risk] Loading dataset ...")
cols_needed = ["date", "sku_id", "warehouse_id", "category", "supplier_id",
               "label_fully_observed"] + NUMERIC_FEATURES + [TARGET]
df = pd.read_csv(MAIN_CSV, usecols=cols_needed, parse_dates=["date"])
print(f"[train_risk] Loaded {len(df):,} rows.")

# ---------------------------------------------------------------
# STEP 1: Drop rows without a fully-observed forward label
# ---------------------------------------------------------------
df = df[df["label_fully_observed"] == 1].copy()
print(f"[train_risk] {len(df):,} rows with a fully-observed 7-day-forward label.")

# One-hot encode category (a legitimate prediction-time attribute, not leakage)
category_dummies = pd.get_dummies(df[CATEGORY_COL], prefix="cat")
FEATURES = NUMERIC_FEATURES + list(category_dummies.columns)
df = pd.concat([df, category_dummies], axis=1)

# ---------------------------------------------------------------
# STEP 2: Strict chronological split
# ---------------------------------------------------------------
train_mask = df["date"] <= TRAIN_END
val_mask = (df["date"] >= VAL_START) & (df["date"] <= VAL_END)
test_mask = (df["date"] >= TEST_START) & (df["date"] <= TEST_END)

X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET]
X_val, y_val = df.loc[val_mask, FEATURES], df.loc[val_mask, TARGET]
X_test, y_test = df.loc[test_mask, FEATURES], df.loc[test_mask, TARGET]
print(f"[train_risk] train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}")

# ---------------------------------------------------------------
# STEP 3: Train XGBoost classifier
# ---------------------------------------------------------------
print("[train_risk] Training XGBoost classifier ...")
model = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", random_state=SEED, n_jobs=-1,
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))
gap = (train_acc - test_acc) * 100
print(f"[train_risk] Train accuracy: {train_acc*100:.2f}%  Test accuracy: {test_acc*100:.2f}%  Gap: {gap:.2f}pp")

risk_metrics = {
    "model": "XGBoost",
    "train_accuracy_pct": round(train_acc * 100, 2),
    "test_accuracy_pct": round(test_acc * 100, 2),
    "train_test_gap_pct_points": round(gap, 2),
    "target": TARGET,
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
}
with open(os.path.join(OUTPUTS_DIR, "risk_metrics.json"), "w") as f:
    json.dump(risk_metrics, f, indent=2)
joblib.dump(model, os.path.join(SAVED_MODELS_DIR, "risk_model.pkl"))

# ---------------------------------------------------------------
# STEP 4: Global feature importance (used for top_risk_factors)
# ---------------------------------------------------------------
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
top_global_factors = list(importances.head(5).index)


def suggested_action(risk_level: str, days_of_supply: float) -> str:
    if risk_level == "High":
        return f"Reorder immediately - only ~{days_of_supply:.1f} days of supply remain at current demand."
    if risk_level == "Medium":
        return f"Monitor closely and consider placing a replenishment order (~{days_of_supply:.1f} days of supply remaining)."
    return "No action needed - stock levels are healthy relative to demand."


def pick_top_factors(row) -> list:
    """Top-3 factors from the globally most important features, using this
    row's own values as the human-readable evidence."""
    picks = []
    for feat in top_global_factors:
        if feat.startswith("cat_"):
            continue
        picks.append(f"{feat.replace('_', ' ')}: {round(float(row[feat]), 2)}")
        if len(picks) == 3:
            break
    return picks


# ---------------------------------------------------------------
# STEP 5: Predict risk for every SKU x warehouse pair using its MOST
# RECENT fully-observed row (i.e. "as of today")
# ---------------------------------------------------------------
print("[train_risk] Scoring latest snapshot for every SKU x warehouse pair ...")
latest = df.sort_values("date").groupby(["sku_id", "warehouse_id"]).tail(1).reset_index(drop=True)
X_latest = latest[FEATURES]
probs = model.predict_proba(X_latest)[:, 1]

risk_rows = []
for i, row in latest.iterrows():
    score = float(probs[i])
    if score >= 0.66:
        level = "High"
    elif score >= 0.33:
        level = "Medium"
    else:
        level = "Low"
    confidence = round(abs(score - 0.5) * 2, 3)  # distance from decision boundary, scaled to 0..1
    risk_rows.append({
        "product_id": row["sku_id"],
        "store_id": row["warehouse_id"],
        "category": row["category"],
        "supplier_id": row["supplier_id"],
        "risk_score": round(score, 4),
        "risk_level": level,
        "confidence": confidence,
        "suggested_action": suggested_action(level, row["days_of_supply"]),
        "top_risk_factors": pick_top_factors(row),
    })

risk_df = pd.DataFrame(risk_rows)
risk_csv_path = os.path.join(PROCESSED_DIR, "inventory_risk.csv")
risk_df.to_csv(risk_csv_path, index=False)
print(f"[train_risk] Wrote {risk_csv_path} ({len(risk_df)} rows)")
print(risk_df["risk_level"].value_counts())

save_to_db.save_inventory_risk(risk_df)
print("[train_risk] Done.")
