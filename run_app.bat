@echo off
rem Crux offline ranker — double-click launcher.
rem Installs dependencies on first run, then starts the app and opens your browser.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.9+ is required but was not found on PATH.
  echo Install it from https://www.python.org/downloads/ and re-run this file.
  pause
  exit /b 1
)

python -c "import streamlit, sentence_transformers, numpy" >nul 2>nul
if errorlevel 1 (
  echo First run: installing dependencies ^(one time, a few minutes^)...
  python -m pip install -r requirements.txt
)

echo Starting the Crux offline ranker...
python app.py
pause
