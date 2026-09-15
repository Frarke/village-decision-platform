@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Village Decision Platform V1

set "VENV_PY=.venv\Scripts\python.exe"
set "APP_PORT=8513"

if exist "%VENV_PY%" goto dependencies

echo [1/3] Checking Python and creating virtual environment...

set "BOOTSTRAP="
python --version >nul 2>nul
if not errorlevel 1 set "BOOTSTRAP=python"

if defined BOOTSTRAP goto create_venv

py -3 --version >nul 2>nul
if not errorlevel 1 set "BOOTSTRAP=py -3"

if defined BOOTSTRAP goto create_venv

echo.
echo [ERROR] Python was not found.
echo Please install 64-bit Python 3.10, 3.11, or 3.12.
echo Select "Add Python to PATH" during installation.
pause
exit /b 1

:create_venv
%BOOTSTRAP% -m venv .venv
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create the virtual environment.
    echo Make sure this folder is writable.
    pause
    exit /b 1
)

:dependencies
echo [2/3] Checking dependencies...

"%VENV_PY%" -c "import streamlit, plotly, pandas, numpy, shapely, pyproj, rasterio, fastapi, uvicorn, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies...
    "%VENV_PY%" -m pip install --upgrade pip
    if errorlevel 1 goto install_error

    "%VENV_PY%" -m pip install -r requirements.txt
    if errorlevel 1 goto install_error
)

echo [3/3] Starting spatial decision platform...
echo Address: http://localhost:%APP_PORT%
echo Keep this window open while using the platform.
echo.

"%VENV_PY%" -m streamlit run v1_app.py --server.port %APP_PORT% --server.address localhost

if errorlevel 1 (
    echo.
    echo [ERROR] The platform stopped unexpectedly.
    echo Please keep this window open and check the error above.
    pause
    exit /b 1
)

exit /b 0

:install_error
echo.
echo [ERROR] Failed to install dependencies.
echo Check the network connection and requirements.txt.
pause
exit /b 1
