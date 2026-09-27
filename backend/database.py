"""
database.py
------------------------------------------------------------
Handles the MySQL connection for SupplySphere AI.

We ONLY store processed / model-output data here:
    1. daily_sales      -> historical daily order count & revenue
    2. demand_forecast   -> LSTM's next-30-day predictions (precomputed)
    3. supplier_risk      -> XGBoost's risk score per supplier (precomputed)

No raw per-row dataset tables. No chat/report tables.
Models are trained OFFLINE (see /ml folder) and their results are
written into these tables. FastAPI only READS from them -> fast,
lightweight, no TensorFlow loaded inside the API process.
------------------------------------------------------------
"""

import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load DB credentials from a local .env file (never hard-code passwords)
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "supplysphere"),
}


def get_connection():
    """
    Opens and returns a new MySQL connection.
    A fresh connection per request is simple and totally fine
    at diploma-project scale (no connection pooling needed).
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        print(f"[database.py] Could not connect to MySQL: {e}")
        raise


def check_connection() -> dict:
    """
    Lightweight DB connectivity probe used by /api/health.

    Every query in this app already fails "quietly" (run_query returns []
    instead of raising - see below) so that one bad table doesn't 500 an
    entire page. The tradeoff is that a totally unreachable/misconfigured
    database looks identical to "this product legitimately has no risk
    records yet": every page just silently shows 0s and empty charts with
    no visible error. This gives the frontend something concrete to check
    at load time instead, so a wrong DB_PASSWORD or a MySQL that isn't
    running yet surfaces as an actionable banner instead of a wall of
    zeros that looks like a bug in the app itself.
    """
    try:
        connection = mysql.connector.connect(**DB_CONFIG, connection_timeout=3)
        connection.close()
        return {"connected": True}
    except Error as e:
        hint = "Check DB_HOST/DB_USER/DB_PASSWORD/DB_NAME in backend/.env against your actual MySQL setup."
        if e.errno == 1045:
            hint = "Access denied - DB_PASSWORD in backend/.env doesn't match your MySQL root (or app user) password."
        elif e.errno == 1049:
            hint = "Unknown database - run database/schema.sql to create it first."
        elif e.errno in (2003, 2005):
            hint = "Can't reach the MySQL server - is it running, and is DB_HOST/DB_PORT correct?"
        return {"connected": False, "error": str(e), "hint": hint}


def run_query(query: str, params: tuple = None, fetch: bool = True):
    """
    Small helper to run a SELECT query and return rows as a list of dicts.
    Keeping this in one place avoids repeating boilerplate in every route.

    If the database is unreachable or the query fails (e.g. MySQL isn't
    running yet, or the schema/tables haven't been seeded), this returns
    an empty list instead of raising - callers already handle "no data"
    gracefully, so pages degrade instead of returning a 500. The real
    error is still logged to the console for debugging.
    """
    try:
        connection = get_connection()
    except Error:
        return [] if fetch else None

    cursor = connection.cursor(dictionary=True)  # dictionary=True -> rows come back as dicts
    try:
        cursor.execute(query, params or ())
        if fetch:
            result = cursor.fetchall()
        else:
            connection.commit()
            result = None
        return result
    except Error as e:
        hint = ""
        if e.errno == 1054:  # ER_BAD_FIELD_ERROR - "Unknown column ... in 'field list'"
            hint = (
                "\n  [hint] This looks like schema drift between the code and the live "
                "database (e.g. an old MySQL volume that predates a column added to "
                "database/schema.sql). The backend runs a startup self-heal for the "
                "known 'product_stats.avg_supplier_lead_time_days' case - see "
                "db_migrations.py. Restart the backend, or run database/schema.sql "
                "and ml/save_to_db.py again, to fully resync."
            )
        print(f"[database.py] Query failed: {e}\n  Query: {query.strip()[:200]}{hint}")
        return [] if fetch else None
    finally:
        cursor.close()
        connection.close()
