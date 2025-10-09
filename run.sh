#!/bin/bash
# run.sh - Quick start script for KYC service

set -e  # Exit on error

echo "🚀 Starting KYC Verification Service..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    cp .env.example .env
    echo "✓ Created .env file. Please edit it with your configuration."
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "📦 Creating models directory..."
    mkdir -p models
fi

# Check if yunet.onnx exists
if [ ! -f "models/yunet.onnx" ]; then
    echo "⚠️  YuNet model not found. Please run: python scripts/download_models.py"
    exit 1
fi

# Create required directories
echo "📁 Creating required directories..."
mkdir -p logs temp data/id data/realtime_photo

# Check if poetry is available
if command -v poetry &> /dev/null; then
    echo "🔧 Installing dependencies with Poetry..."
    poetry install
    
    echo "🌟 Starting FastAPI with uvicorn..."
    poetry run uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
else
    echo "⚠️  Poetry not found. Using pip..."
    pip install -e .
    
    echo "🌟 Starting FastAPI with uvicorn..."
    uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
fi