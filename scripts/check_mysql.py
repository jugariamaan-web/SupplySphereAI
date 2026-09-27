"""
check_mysql.py
------------------------------------------------------------
Small standalone connectivity check used by run_project.bat.
Never raises - it only prints a clear OK / WARNING message,
so the launcher can continue starting the server either way.
------------------------------------------------------------
"""
import os
import sys
from pathlib import Path

# Make "backend" importable and load its .env file regardless of
# which directory this script is invoked from.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "backend"))

# These imports used to sit at module scope, which broke the "never
# raises" contract above: if python-dotenv or mysql-connector-python
# weren't actually installed in the active venv (e.g. requirements.txt
# install got interrupted partway through), this script crashed with
# a raw traceback instead of printing the intended WARNING - and
# run_project.bat doesn't check this script's exit code, so the crash
# was silently swallowed while looking like an unhandled failure.
try:
    from dotenv import load_dotenv  # noqa: E402
    load_dotenv(project_root / "backend" / ".env")
    import mysql.connector  # noqa: E402
except ImportError as e:
    print(f"      [WARNING] Skipping MySQL check - a required package isn't installed: {e}")
    print("      Re-run run_project.bat to install missing dependencies, or run:")
    print(f"      venv\\Scripts\\pip install -r backend\\requirements.txt")
    sys.exit(0)

try:
    conn = mysql.connector.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 3306)),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_NAME", "supplysphere"),
        connection_timeout=5,
    )
    conn.close()
    print("      MySQL connection OK.")
except Exception as e:
    print(f"      [WARNING] Could not connect to MySQL: {e}")
    print("      The app will still start, but pages will show 'no data' until MySQL is reachable.")
    print("      Run database\\schema.sql and the ml\\ scripts, then check backend\\.env.")
