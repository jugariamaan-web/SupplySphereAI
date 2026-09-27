# SupplySphere AI
## Intelligent Supply Chain Management System Using AI

This project is an AI-powered Supply Chain Management System built with FastAPI, MySQL, and Machine Learning (LSTM & XGBoost).

### ✨ Features
- **Dashboard** - live KPIs, inventory health breakdown, demand trend chart, category/store distribution.
- **Demand Forecast** - 30-day forecast with a 4-model benchmark (Linear Regression, Random Forest, XGBoost, LSTM) and per-product breakdown.
- **Inventory Plan** - EOQ, safety stock, reorder point, and stockout probability per product, with a "what-if" manual stock override.
- **Inventory Risk** - XGBoost-classified stockout risk per product-store pair, with search, filters, pagination, and a detail modal showing top risk drivers.
- **AI Copilot** - chat assistant grounded in the system's real data (Groq), with a professional offline fallback.
- **AI Reports** - generates executive/inventory/risk/procurement reports, exportable as CSV or PDF (via print).

### 🏗 Architecture
```
Browser (supplytemplates/*.html + api.js)
        │  fetch() calls
        ▼
FastAPI backend (backend/main.py)
   ├── ml_service.py   → reads precomputed forecast/risk/stats from MySQL
   ├── inventory.py    → computes EOQ/safety stock/ROP live, on demand
   └── copilot.py      → calls Groq, with a data-grounded fallback
        │
        ▼
MySQL (daily_sales, demand_forecast, inventory_risk, product_stats)
        ▲
ml/ scripts (preprocess.py → train_forecast.py → train_risk.py)
   train OFFLINE and write their results into MySQL + saved_models/
```
The API never trains or loads TensorFlow/XGBoost at request time - it only reads
already-computed results, which keeps the backend fast and lightweight.

### 📊 Dataset
`data/raw/supplysphere_dataset/supplysphere_synthetic_2024_2025.csv` - a
**synthetic** (not real-world observed) daily SKU x warehouse inventory
dataset: 182,750 rows covering 50 SKUs (SKU_001-SKU_050) across 5 warehouses
(WH_01-WH_05) and 10 suppliers (SUP_01-SUP_10), from 2024-01-01 to 2025-12-31
(2 full years). It ships real product names/categories (see
`product_master.csv`), real warehouse names/regions (`warehouse_master.csv`),
real supplier names/lead-time stats (`supplier_master.csv`), a replenishment
order log (`replenishment_orders_log.csv`), and pre-engineered prediction-time
features (demand lags/rolling stats, days-of-supply, gap-to-reorder-point,
supplier reliability, etc.) plus a validated forward-looking risk label
(`stockout_next_7_days`).

**Migration note (from the original 1-year generic retail dataset):** unlike
that dataset, this one has REAL product categories and supplier IDs, so the
old `category`-secretly-holds-a-supplier-ID workaround (and the hardcoded
name-lookup table it needed) has been removed - `product_stats.category`
and `inventory_risk.category` now hold genuine categories (e.g. "Household",
"Beverages", "Dairy"), and a real `supplier_id` column was added alongside
them. See the migration notes at the top of `database/schema.sql`,
`ml/preprocess.py`, `ml/train_forecast.py`, and `ml/train_risk.py` for full
details.

**ML methodology:** both `ml/train_forecast.py` and `ml/train_risk.py` use a
strict chronological split - train on 2024, validate on 2025 H1, final-test
on 2025 H2 - with no shuffling. The risk model excludes the final 7 days of
the dataset (where the forward-looking 7-day label can't be fully observed)
from training/evaluation entirely.

### 🚀 Quick Start

#### 1. Prerequisites
- Python 3.11 (recommended) or any Python 3.10+, installed globally - **no virtual environment needed or used anywhere in this project.**
- MySQL Server (or Docker - see below)
- Groq API Key (for AI Copilot & Reports) - free at https://console.groq.com/keys

#### 2. Install dependencies globally
Everything runs against your system/global Python installation - one command installs every package this project needs:
```bash
pip install fastapi==0.115.0 uvicorn==0.30.6 mysql-connector-python==9.0.0 python-dotenv==1.0.1 "groq>=1.6.0" pydantic==2.9.2 pandas numpy matplotlib scikit-learn xgboost tensorflow reportlab
```
(Equivalent shortcut: `pip install -r backend/requirements.txt` - same package list, kept in the repo for reference/reproducibility.)

#### 3. Configure `.env`
- Open `backend/.env` and replace `GROQ_API_KEY=your_api_key_here` with your real key.
- Update MySQL credentials if necessary.

#### 4. Database Setup
```bash
mysql -u your_user -p -e "CREATE DATABASE supplysphere;"
mysql -u your_user -p supplysphere < database/schema.sql
```

#### 5. Run the ML Pipeline
```bash
cd ml
python preprocess.py
python train_forecast.py
python train_risk.py
```
*These scripts populate the MySQL database with historical data and predictions - run them once before starting the backend, and again any time the dataset changes.*

#### 6. Start the Backend
```bash
cd backend
uvicorn main:app --reload
```
Then open `http://localhost:8000` in your browser.

**One-click alternative (Windows):** double-click `run_project.bat` in the project root. It uses your global Python installation directly (no virtual environment), installs any missing packages, checks MySQL connectivity, starts the API, and opens your browser automatically. It prefers Python 3.11 via the `py` launcher if available, otherwise falls back to whatever `python` resolves to on PATH.

#### 7. Stop the Application
- **Manual run:** press `Ctrl+C` in the terminal running `uvicorn`.
- **run_project.bat:** close the separate "Backend Logs" window it opened (or press `Ctrl+C` inside it).
- **Docker:** `docker compose down` (see below).

### 🐳 Running with Docker

Docker is an alternative to the steps above - it does not replace them, and either
can be used independently.

```bash
# 1. Copy the root env file and fill in your Groq key
cp .env.example .env

# 2. Build and start both containers (MySQL + backend)
docker compose up --build
```
- The backend is served at `http://localhost:8000`.
- MySQL runs inside its own container and is seeded automatically from
  `database/schema.sql` on first start.
- You still need to run the ML pipeline once (`ml/preprocess.py`,
  `ml/train_forecast.py`, `ml/train_risk.py`) so the tables have data -
  either locally against `localhost:3307` (the container's exposed MySQL
  port) or from inside the container with `docker compose exec backend bash`.

### 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Backend health check |
| GET | `/api/dashboard` | KPIs, inventory health, forecast summary, category/region charts |
| GET | `/api/forecast` | 30-day demand forecast, benchmark metrics, per-product breakdown |
| GET/POST | `/api/inventory` | EOQ, safety stock, ROP, stockout probability for a product |
| GET | `/api/risk` | Inventory risk for every product-store pair |
| GET | `/api/risk/{product_id}` | Risk records for a single product |
| GET | `/api/products` | Product list (for dropdowns/selectors) |
| POST | `/api/copilot` | AI Copilot chat, grounded in live data |
| POST | `/api/report` | Generates a text report (executive/sales/demand/inventory/risk/insights/procurement) |

### 🛠 Tech Stack
- **Frontend**: Vanilla JS, Tailwind CSS (CDN), Chart.js
- **Backend**: FastAPI
- **Database**: MySQL
- **ML Models**:
  - **Demand Forecasting**: benchmarks Linear Regression, Random Forest, XGBoost, and LSTM (TensorFlow/Keras) on aggregate daily demand with a strict chronological 2024-train / 2025-H1-validation / 2025-H2-test split; automatically selects the model with the lowest final-test RMSE (see `outputs/benchmark_results.csv` and `outputs/forecast_metrics.json` for the current winner), and reports a 95% prediction interval from that model's actual test-set residuals.
  - **XGBoost**: 7-day-forward Stockout Risk Prediction (see `outputs/risk_metrics.json` for current accuracy)
- **AI**: Groq (LLM integration - official Python SDK)

### 📂 Project Structure
- `backend/`: FastAPI routes and business logic.
- `ml/`: Sequential ML scripts (Colab-style).
- `data/`: Raw (`data/raw/supplysphere_dataset/supplysphere_synthetic_2024_2025.csv` + master/reference CSVs) and processed dataset files.
- `supplytemplates/`: Professional responsive UI templates.
- `database/`: SQL schema.
- `saved_models/`: Trained model files.
- `outputs/`: Visualization charts and evaluation metrics.

### 🖼 Screenshots

Add your own screenshots of the running app to `docs/screenshots/` and reference
them here, e.g.:

```markdown
![Dashboard](docs/screenshots/dashboard.png)
![Demand Forecast](docs/screenshots/forecast.png)
![Inventory Risk](docs/screenshots/inventory-risk.png)
```

### 🔭 Future Scope
- Add authentication/roles so multiple users can have their own saved views.
- Automate the ML pipeline on a schedule (e.g. nightly retrain) instead of manual runs.
- Add historical trend charts per product on the Inventory page.
- Add a supplier/PO tracking module to close the loop after "Order Now" is triggered.
