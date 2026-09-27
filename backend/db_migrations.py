"""
db_migrations.py
------------------------------------------------------------
Idempotent, startup-time schema compatibility check.

WHY THIS EXISTS:
docker-compose.yml mounts database/schema.sql into MySQL's
/docker-entrypoint-initdb.d/. MySQL only executes files in that folder
the FIRST time a container starts against an EMPTY data volume. If a
`mysql_data` volume already existed from an earlier run of this project
(e.g. from before the 2024-2025 dataset migration, when the schema had
no `warehouses`/`suppliers` tables and no `supplier_id` columns), MySQL
will skip initialization entirely and keep serving the OLD table shape -
even though schema.sql, ml/preprocess.py, and ml/save_to_db.py have all
already been updated to the new shape. That drift is exactly what
produces "Unknown column ... in 'field list'" errors on every query in
ml_service.py, which database.run_query() silently swallows (by design,
so pages degrade instead of 500ing) - which is why the symptom shows up
as "0" KPIs and empty dropdowns rather than a visible error.

This module inspects the live database at backend startup and patches
any drift it finds, so the API works correctly whether or not the
volume/DB was fully re-migrated. It never drops or overwrites existing
data - only adds what's missing.
------------------------------------------------------------
"""

import logging
from mysql.connector import Error
from database import get_connection, DB_CONFIG

logger = logging.getLogger("supplysphere")

DEFAULT_LEAD_TIME_DAYS = 7.0


def _table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM information_schema.tables
        WHERE table_schema = %s AND table_name = %s
        """,
        (DB_CONFIG["database"], table_name),
    )
    return cursor.fetchone()["total"] > 0


def _column_exists(cursor, table_name: str, column_name: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*) AS total
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s AND column_name = %s
        """,
        (DB_CONFIG["database"], table_name, column_name),
    )
    return cursor.fetchone()["total"] > 0


def _ensure_column(cursor, connection, table: str, column: str, ddl_type: str, default_sql: str, backfill_msg: str):
    if not _table_exists(cursor, table):
        logger.info(f"[db_migrations] '{table}' table not found yet - skipping.")
        return
    if _column_exists(cursor, table, column):
        return
    logger.warning(f"[db_migrations] '{table}.{column}' is missing (pre-migration DB volume). {backfill_msg}")
    cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type} {default_sql}")
    connection.commit()
    logger.info(f"[db_migrations] Added '{column}' to '{table}'.")


def ensure_schema_compatibility():
    """
    Checks the live database for the tables/columns the 2024-2025 dataset
    migration introduced (warehouses, suppliers, product_stats.supplier_id,
    inventory_risk.supplier_id) and patches any drift it finds. Safe to
    call on every startup - each check is a no-op once the schema is
    current.
    """
    try:
        connection = get_connection()
    except Error as e:
        logger.warning(f"[db_migrations] Skipping schema check - MySQL unreachable: {e}")
        return

    cursor = connection.cursor(dictionary=True)
    try:
        for table in ("warehouses", "suppliers"):
            if not _table_exists(cursor, table):
                logger.warning(
                    f"[db_migrations] '{table}' table not found - run database/schema.sql "
                    "then ml/preprocess.py to create and populate it."
                )

        _ensure_column(
            cursor, connection, "product_stats", "supplier_id", "VARCHAR(16)", "NOT NULL DEFAULT ''",
            "Adding it now with an empty default; re-run `python ml/preprocess.py` to backfill real values.",
        )
        _ensure_column(
            cursor, connection, "product_stats", "avg_supplier_lead_time_days", "FLOAT",
            f"NOT NULL DEFAULT {DEFAULT_LEAD_TIME_DAYS}",
            "Adding it now with a default so the API keeps working; re-run `python ml/preprocess.py` to backfill.",
        )
        _ensure_column(
            cursor, connection, "inventory_risk", "supplier_id", "VARCHAR(16)", "NOT NULL DEFAULT ''",
            "Adding it now with an empty default; re-run `python ml/train_risk.py` to backfill real values.",
        )
        for col in ("lower_bound", "upper_bound"):
            _ensure_column(
                cursor, connection, "demand_forecast", col, "FLOAT", "NULL",
                "Adding it now as a nullable column; re-run `python ml/train_forecast.py` to backfill.",
            )

        logger.info("[db_migrations] Schema compatibility check complete.")

    except Error as e:
        logger.error(f"[db_migrations] Schema check failed: {e}")
    finally:
        cursor.close()
        connection.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ensure_schema_compatibility()
