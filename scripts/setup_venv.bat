@echo off
REM Setup virtual environment for TTS Mini Assistant

echo 🐍 Setting up TTS Mini Assistant virtual environment...

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.10+
    exit /b 1
)

REM Create virtual environment
echo 📁 Creating virtual environment...
python -m venv .venv

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing dependencies...
pip install -r requirements.txt

echo ✅ Setup complete!
echo.
echo To activate the virtual environment in the future:
echo   .venv\Scripts\activate.bat
echo.
echo To run preflight checks:
echo   python scripts\preflight.py
echo.
echo To run tests:
echo   pytest -q
