# SupplySphere Synthetic Inventory Dataset — 2024–2025

## 1. Dataset origin

This is a **synthetic, simulation-generated inventory and demand dataset** created specifically for the SupplySphere AI prototype. It is not collected from a real retailer, warehouse network, supplier, or customer database, and it should not be presented as real-world observations.

The entity masters retain the original synthetic project architecture:

- 50 SKUs
- 5 warehouses
- 10 suppliers
- 250 SKU × warehouse combinations
- Daily observations from **2024-01-01 through 2025-12-31**

Because 2024 is a leap year, the correct row count is:

`366 days × 50 SKUs × 5 warehouses + 365 days × 50 SKUs × 5 warehouses = 182,750 rows`

The originally requested 182,500-row arithmetic would only apply if both years had 365 days.

## 2. Generation methodology

Each SKU has a persistent synthetic profile containing:

- product/category
- demand tier
- unit cost and unit price
- baseline demand
- demand coefficient of variation
- annual demand trend
- nominal supplier lead time
- risk tier
- initial inventory reference

Each warehouse has its own demand scale factor. Each supplier has its own nominal lead time and lead-time variability.

### Demand generation

Demand is generated as a stochastic process rather than a straight line.

The expected demand combines:

1. SKU-specific baseline demand
2. warehouse-specific scale
3. category-specific weekday effects
4. category-specific monthly/seasonal effects
5. SKU-specific annual trend
6. SKU-specific latent demand noise with temporal persistence
7. promotions
8. occasional organic demand spikes
9. occasional demand drops
10. overdispersed count noise using a Gamma-Poisson / negative-binomial-style draw

2025 is generated independently from 2024 using different stochastic draws and a modified SKU-specific trend/regime component. It is therefore **not a copy of 2024**.

Some SKUs increase in popularity and some decline. In the final simulation, 14 SKUs grew by more than 2% in average daily demand from 2024 to 2025, while 21 declined by more than 2%.

### Promotions

Each SKU receives separate promotion windows in each year. Promotion timing and magnitude are randomly generated and are not copied from 2024 into 2025.

A promotion can produce a substantial uplift, while some promotion days intentionally fizzle and produce no uplift. This prevents promotions from being a deterministic risk label.

### Inventory simulation

Inventory is simulated as a running physical state:

`Opening inventory`
`→ stock received`
`→ customer demand`
`→ units sold`
`→ closing inventory`
`→ reorder decision`
`→ supplier lead time`
`→ future stock arrival`

The inventory balance is enforced as:

`closing_inventory = opening_inventory + stock_received - units_sold`

Units sold cannot exceed the available inventory after that day's receipt. When customer demand exceeds available inventory, the inventory remains at zero and the unmet quantity is recorded as `backorder_units`.

Inventory is therefore not generated as an independent random value on each row.

### Replenishment logic

After daily demand is fulfilled, a reorder decision is evaluated using information available at that time.

The reorder point is based on:

- recent demand estimate
- recent demand volatility
- current/recent supplier lead time
- a tier-specific safety buffer

A replenishment order is placed when available inventory plus outstanding order quantity is insufficient relative to the reorder point.

The order quantity is based on an order-up-to target rather than a fixed random quantity.

Supplier lead time is sampled from supplier-specific distributions. Occasional supplier disruptions add additional delay, creating realistic replenishment uncertainty.

### Risk tiers

Five of the 50 SKUs are classified as **constrained**. They have thinner inventory buffers and therefore a higher natural stockout frequency.

The remaining 45 SKUs are standard-buffer SKUs.

The constrained tier is intentionally a business simulation assumption, not a hidden target variable. Risk still varies within both groups because demand, promotions, seasonality, warehouse scale, and supplier delays are stochastic.

## 3. Risk-label methodology

The primary target is:

`stockout_next_7_days`

For a row at time `T`, the label is 1 if that SKU × warehouse combination experiences a stockout on any of the following seven days:

`T+1 ... T+7`

Otherwise it is 0.

The future stockout observations are used **only** to construct this target. They are not used as model features.

The final seven days of every SKU × warehouse series do not have a fully observable seven-day future. These rows are retained in the raw dataset with:

- `label_fully_observed = 0`
- `stockout_next_7_days = -1`

They must be excluded from supervised model training/evaluation.

The observed stockout-day rate is approximately **3.80%**, within the requested 3–8% event range. The seven-day forward-risk label is naturally higher at approximately **14.95%**, because a seven-day horizon can contain a stockout even when the current row itself is not a stockout.

## 4. Prediction-time features

The dataset includes legitimate prediction-time features such as:

- `demand_lag1`
- `demand_lag7`
- `demand_lag7_mean`
- `demand_roll14_mean`
- `demand_volatility_14d`
- `recent_demand_trend`
- `days_of_supply`
- `gap_to_reorder_point`
- `lead_time_recent_30d`
- `expected_lead_time_days`
- `supplier_reliability`
- `recent_replenishment_qty_30d`
- `recent_stock_received_days_14d`
- `stockout_recent_7d`
- `demand_spike_recent_7d`
- promotion and calendar features
- current inventory state

All rolling/lagged demand and replenishment features are shifted so that they only use information available before the prediction timestamp.

The future target is not used to construct these features.

## 5. Forecasting methodology

Forecasting predicts aggregate daily demand.

The candidate models are:

- Linear Regression
- Random Forest
- XGBoost
- LSTM

Forecasting uses a strict chronological split:

- **Training:** 2024
- **Validation:** 2025-01-01 through 2025-06-30
- **Final test:** 2025-07-01 through 2025-12-31

No random shuffling is used.

The forecasting features use lagged demand, rolling demand statistics, calendar effects, and recent variability.

## 6. Risk-model methodology

The risk classifier uses XGBoost with a small positive-class weighting adjustment to account for the natural class imbalance.

Chronological split:

- **Training:** 2024
- **Validation:** 2025-01-01 through 2025-06-30
- **Final test:** 2025-07-01 through 2025-12-31

No future rows are used as training features for earlier prediction timestamps.

The final test accuracy is approximately **86.8%**, with a train/test accuracy gap of approximately **1.14 percentage points**.

Because accuracy alone can hide minority-class weakness, precision, recall, F1 and balanced accuracy are also reported.

## 7. Validation summary

### Structural validation

- Rows: 182,750
- Duplicate rows: 0
- Duplicate SKU × warehouse × date keys: 0
- Missing values: 0
- SKUs: 50
- Warehouses: 5
- Suppliers: 10
- Date range: 2024-01-01 → 2025-12-31
- Negative inventory rows: 0
- Inventory balance violations: 0
- Demand < 0 violations: 0
- Units sold > available inventory violations: 0
- Price ≤ cost violations: 0
- Replenishment-order logic violations: 0

### Demand statistics

Row-level demand:

- Mean: ~12.97 units
- Standard deviation: ~11.25
- Coefficient of variation: ~0.87
- Median: 10
- 25th percentile: 4
- 75th percentile: 18
- Maximum: 233

Aggregate daily demand:

- Mean: ~3,244 units/day
- Standard deviation: ~170 units/day
- CV: ~0.0525
- Lag-1 autocorrelation: ~0.570
- Lag-7 autocorrelation: ~0.506

The positive autocorrelation at both lag 1 and lag 7 indicates temporal structure without making the series deterministic.

### Inventory statistics

- Mean opening inventory: ~93.1 units
- Mean closing inventory: ~93.0 units
- Mean reorder point: ~105.8 units
- Mean days of supply: ~8.15 days
- Closing inventory P10/P50/P90: ~10 / 65 / 210 units

### Stockout statistics

- Overall stockout-day rate: ~3.80%
- Constrained SKU stockout-day rate: ~8.05%
- Standard SKU stockout-day rate: ~3.33%
- Constrained SKUs: 5 / 50 = 10%
- Observed seven-day forward-risk positive rate: ~14.95%

The constrained group has higher risk, but it is not the sole source of stockouts. Standard SKUs can still experience risk from demand spikes, promotions, seasonality, and supplier delays.

### Lead time statistics

- Minimum realized/current lead time: 2 days
- Maximum: 18 days
- Mean: ~6.41 days
- Standard deviation: ~2.17 days

The upper tail includes simulated supplier disruptions.

### Pricing

- Mean unit cost: ~54.24
- Mean unit price: ~76.89
- Mean margin percentage: ~28.5%

## 8. Forecasting results

Final late-2025 test results:

| Model | RMSE | MAE | MAPE | R² |
|---|---:|---:|---:|---:|
| Linear Regression | ~115.6 | ~94.0 | ~2.90% | ~0.521 |
| Random Forest | ~121.5 | ~99.3 | ~3.09% | ~0.470 |
| XGBoost | ~126.5 | ~101.7 | ~3.17% | ~0.426 |
| LSTM | ~151.9 | ~119.8 | ~3.70% | ~0.172 |

The forecasting problem therefore contains meaningful but imperfect signal. Linear Regression is the strongest of these four candidates on the final test split, while the LSTM does not automatically outperform simpler models.

## 9. Risk-model results

Final XGBoost risk model:

| Metric | Train | Validation | Final Test |
|---|---:|---:|---:|
| Accuracy | ~87.93% | ~85.99% | **~86.79%** |
| Precision | ~64.76% | ~55.69% | ~56.56% |
| Recall | ~44.09% | ~35.86% | ~36.87% |
| F1 | ~52.47% | ~43.62% | ~44.64% |
| Balanced Accuracy | ~69.91% | — | ~66.04% |

Final test confusion matrix:

`[[36047, 1810], [4036, 2357]]`

The train/test accuracy gap is approximately **1.14 percentage points**.

The model is not artificially forced to 85%+. The observed ~86.8% test accuracy emerges from the simulated inventory state, demand behavior, replenishment dynamics and forward stockout outcomes.

## 10. Leakage audit

No target leakage was identified in the final model feature set.

The important distinction is:

`CURRENT STATE → future 7-day outcome`

not:

`future outcome → current features`

Examples of safe feature construction:

- rolling demand uses shifted historical demand
- recent replenishment quantity uses prior receipts
- recent stockout history uses prior stockout observations
- recent lead time uses prior realized lead times
- days of supply uses current inventory and historical demand estimate

The target itself is generated from future stockout outcomes and is excluded from the feature matrix.

## 11. Assumptions

This simulation assumes:

- daily demand is observable at the SKU × warehouse level
- customer demand is counted even when inventory cannot fulfill it
- backorders represent unmet demand
- one outstanding replenishment order is allowed per SKU × warehouse in the simulator
- suppliers have stable base lead-time distributions with occasional disruptions
- promotions influence demand probabilistically
- seasonal effects are category-specific
- warehouse demand scale remains structurally different
- inventory does not become negative
- unit prices remain above unit costs
- demand is non-negative

## 12. Limitations

This dataset is **synthetic** and therefore cannot establish real-world predictive performance.

Important limitations include:

1. Supplier behavior is simulated rather than learned from actual supplier delivery histories.
2. Demand seasonality is designed from plausible business assumptions rather than estimated from proprietary sales data.
3. The constrained SKU tier is a simulation assumption.
4. The risk label measures simulated stockout exposure, not real business loss or service-level impact.
5. Forecasting is performed on aggregate daily demand in the benchmark.
6. Real businesses can have substitution, cancellations, returns, multi-echelon inventory, transportation constraints, and supplier allocation effects that are not fully represented.
7. Model accuracy should not be interpreted as expected production accuracy on real SupplySphere customers.

## 13. Integration readiness

The dataset is suitable as a **synthetic development and demonstration dataset** for SupplySphere AI.

It is structurally validated and contains the requested 2024–2025 coverage, running inventory state, replenishment behavior, stockouts, prediction-time features, forward risk labels, and chronological model benchmarks.

It should **not** yet be treated as evidence of production performance.

This final bundle intentionally does not modify or integrate with the existing SupplySphere AI application.
