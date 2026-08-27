@echo off
REM Quick way to run/test the tool on Windows using Python, no compiling needed.
REM Requires Python 3.10+ installed from python.org (check "Add to PATH" during install).
setlocal
cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -q -r requirements.txt

echo Launching MotilityTracker...
python launcher.py

pause
