@echo off
setlocal enabledelayedexpansion
title SupplySphere AI - Local Launcher
cd /d "%~dp0"

echo ============================================================
echo   SupplySphere AI - Starting Local Environment
echo ============================================================
echo.

REM ------------------------------------------------------------
REM 1. Locate a Python interpreter (no virtual environment).
REM    Prefers the Python 3.11 launcher (py -3.11) if installed,
REM    then falls back to whatever "python" resolves to on PATH.
REM    Every package this project needs is installed globally
REM    into that interpreter - see README for the single pip
REM    install command.
REM ------------------------------------------------------------
set "PYEXE="
py -3.11 -c "print(1)" >nul 2>nul
if not errorlevel 1 (
    set "PYEXE=py -3.11"
    echo [1/4] Using Python 3.11 via the "py" launcher.
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] No Python interpreter found on PATH, and "py -3.11" is not
        echo         available. Install Python 3.11 ^(or any Python 3.10+^) and
        echo         make sure it is on PATH, then re-run this script.
        pause
        exit /b 1
    )
    set "PYEXE=python"
    echo [1/4] Python 3.11 launcher not found - using "python" from PATH.
)

REM ------------------------------------------------------------
REM 2. Install dependencies globally only if something is missing
REM    (no venv - packages install straight into the system/user
REM    Python site-packages).
REM ------------------------------------------------------------
echo [2/4] Checking Python dependencies...
%PYEXE% -c "import fastapi, uvicorn, mysql.connector, dotenv, pandas, numpy, matplotlib, sklearn, xgboost, tensorflow, groq, pydantic, reportlab" 2>nul
if errorlevel 1 (
    echo       Missing packages detected - installing globally from requirements.txt ...
    %PYEXE% -m pip install --disable-pip-version-check -q -r backend\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    REM Belt-and-suspenders: re-verify immediately after install instead of
    REM trusting pip's exit code alone. A large multi-package install (this
    REM project pulls in tensorflow) can be interrupted or partially fail
    REM in ways that don't always surface as a non-zero pip exit code on
    REM Windows - if that happens, reinstall just the missing piece instead
    REM of silently limping forward into a ModuleNotFoundError later.
    %PYEXE% -c "import fastapi, uvicorn, mysql.connector, dotenv, pandas, numpy, matplotlib, sklearn, xgboost, tensorflow, groq, pydantic, reportlab" 2>nul
    if errorlevel 1 (
        echo       [WARNING] Still missing packages after install - retrying individually ...
        %PYEXE% -m pip install --disable-pip-version-check -q --force-reinstall --no-deps python-dotenv==1.0.1
        %PYEXE% -m pip install --disable-pip-version-check -q -r backend\requirements.txt
    )
) else (
    echo       All core packages already installed globally.
)

REM ------------------------------------------------------------
REM 3. Verify MySQL connectivity (uses backend\.env if present)
REM ------------------------------------------------------------
echo [3/5] Checking MySQL connectivity...
%PYEXE% "%~dp0scripts\check_mysql.py"
if errorlevel 1 (
    echo       [WARNING] check_mysql.py did not complete cleanly - continuing anyway.
    echo       The backend will still start; pages will show 'no data' until MySQL is reachable.
)

REM ------------------------------------------------------------
REM 4. Re-seed MySQL from the existing ML pipeline outputs
REM    (data/processed/*.csv and data/raw/supplysphere_dataset/*_master.csv).
REM    save_to_db.py resolves all paths from its own script location
REM    (project/ml/), so this works no matter where it's launched from.
REM    This is a fast reseed, NOT a retrain - it does not touch the
REM    trained models. To regenerate the CSVs/models themselves, run
REM    preprocess.py -> train_forecast.py -> train_risk.py manually
REM    first (see README); each of those already calls into
REM    save_to_db.py as it runs, so this step just keeps MySQL in
REM    sync with whatever is currently on disk every time you launch.
REM ------------------------------------------------------------
echo [4/5] Seeding MySQL from ml/data outputs (save_to_db.py)...
%PYEXE% "%~dp0ml\save_to_db.py"
if errorlevel 1 (
    echo       [WARNING] save_to_db.py did not complete cleanly - continuing anyway.
    echo       The backend will still start; pages may show 'no data' until this succeeds.
)

REM ------------------------------------------------------------
REM 5. Start FastAPI (in a new window so logs stay visible),
REM    then wait a moment and open the browser.
REM ------------------------------------------------------------
echo [5/5] Starting FastAPI backend on http://localhost:8000 ...
start "SupplySphere AI - Backend Logs" cmd /k "cd /d "%~dp0backend" && %PYEXE% -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo       Waiting for the server to come up...
timeout /t 5 /nobreak >nul
start "" "http://localhost:8000"

echo.
echo ============================================================
echo   SupplySphere AI is running at http://localhost:8000
echo   Backend logs are in the separate "Backend Logs" window.
echo   Close that window (or press Ctrl+C in it) to stop the server.
echo ============================================================
pause