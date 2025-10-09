# Makefile for KYC Verification Service

.PHONY: help install run dev test clean models setup

help:
	@echo "Available commands:"
	@echo "  make setup    - First time setup (create dirs, copy .env)"
	@echo "  make models   - Download required models"
	@echo "  make install  - Install dependencies"
	@echo "  make run      - Run production server"
	@echo "  make dev      - Run development server with reload"
	@echo "  make test     - Run tests"
	@echo "  make clean    - Clean temp files"

setup:
	@echo "🔧 Setting up project..."
	mkdir -p models logs temp data/id data/realtime_photo
	[ -f .env ] || cp .env.example .env
	@echo "✓ Setup complete. Edit .env with your configuration."

models:
	@echo "📦 Downloading models..."
	python scripts/download_models.py

install:
	@echo "📦 Installing dependencies..."
	poetry install

run:
	@echo "🚀 Starting production server..."
	poetry run uvicorn api.api:app --host 0.0.0.0 --port 8000

dev:
	@echo "🔧 Starting development server..."
	poetry run uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload

test:
	@echo "🧪 Running tests..."
	poetry run pytest tests/ -v

clean:
	@echo "🧹 Cleaning temporary files..."
	rm -rf temp/*
	rm -rf logs/*.log
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "✓ Cleaned"