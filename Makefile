.PHONY: help venv install preflight test test-tts fmt

help: ## Show this help message
	@echo "TTS Mini Assistant - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@echo "  make venv      - Create virtual environment"
	@echo "  make install   - Install dependencies"
	@echo "  make preflight - Run preflight checks"
	@echo "  make test      - Run basic tests (no TTS)"
	@echo "  make test-tts  - Run TTS smoke test"
	@echo "  make fmt       - Format code (placeholder)"
	@echo ""

venv: ## Create virtual environment
	@echo "🐍 Creating virtual environment..."
	@python -m venv .venv
	@echo "✅ Virtual environment created at .venv/"

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	@python -m pip install --upgrade pip
	@python -m pip install -r requirements.txt
	@echo "✅ Dependencies installed"

activate-unix: ## Show Unix/macOS activation command
	@echo "🐍 For Unix/macOS, activate with:"
	@echo "   source .venv/bin/activate"

activate-win: ## Show Windows activation command
	@echo "🐍 For Windows, activate with:"
	@echo "   .venv\\Scripts\\activate"

preflight: ## Run preflight checks
	@echo "🔍 Running preflight checks..."
	@python scripts/preflight.py

test: ## Run basic tests (no TTS)
	@echo "🧪 Running basic tests..."
	@python -m pytest -q

test-tts: ## Run TTS smoke test
	@echo "🎤 Running TTS smoke test..."
	@set RUN_TTS_TEST=1 && python -m pytest -q -k test_tts

fmt: ## Format code (placeholder)
	@echo "✨ Code formatting (placeholder for future black/ruff)"
