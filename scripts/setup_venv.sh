#!/bin/bash
# Setup virtual environment for TTS Mini Assistant

set -e

echo "🐍 Setting up TTS Mini Assistant virtual environment..."

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Create virtual environment
echo "📁 Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment in the future:"
echo "  source .venv/bin/activate"
echo ""
echo "To run preflight checks:"
echo "  python scripts/preflight.py"
echo ""
echo "To run tests:"
echo "  pytest -q"
