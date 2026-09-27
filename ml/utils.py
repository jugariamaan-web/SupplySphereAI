"""
utils.py
------------------------------------------------------------
Small shared helper functions used by BOTH training scripts
(train_forecast.py and train_risk.py). Kept here only because
these two things are genuinely duplicated - nothing else.

Contains:
    1. get_mysql_connection() -> reusable DB connection for saving results
    2. ensure_folder_exists()  -> creates output folders if missing
------------------------------------------------------------
"""

import os
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

load_dotenv()  # reads DB credentials from a local .env file


def get_mysql_connection():
    """
    Connects to the SAME MySQL database used by the FastAPI backend.
    Training scripts WRITE results here; FastAPI only READS them.
    Returns None if the connection fails (caller should handle this).
    """
    try:
        connection = mysql.connector.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 3306)),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "supplysphere"),
        )
        return connection
    except Error as e:
        print(f"[ERROR] Could not connect to MySQL: {e}")
        return None


def ensure_folder_exists(folder_path: str):
    """Creates a folder (and parent folders) if it doesn't already exist."""
    os.makedirs(folder_path, exist_ok=True)
