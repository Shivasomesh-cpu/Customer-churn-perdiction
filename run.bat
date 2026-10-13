@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto activate

set "PY_CMD="
where py >nul 2>nul
if %errorlevel% equ 0 (
  set "PY_CMD=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel% equ 0 (
    set "PY_CMD=python"
  )
)

if "%PY_CMD%"=="" (
  echo Python 3.10 or newer is required. Install Python from https://www.python.org/downloads/
  pause
  exit /b 1
)

%PY_CMD% -m venv .venv
if errorlevel 1 (
  echo Failed to create virtual environment. Ensure Python 3.10 or newer is installed.
  pause
  exit /b 1
)

:activate
call ".venv\Scripts\activate.bat"
python -m pip install -r requirements.txt
python -m streamlit run app.py
