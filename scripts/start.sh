#!/bin/bash
# LoRA Fine-tune Agent - Startup Script

set -e

echo "================================"
echo "  LoRA Fine-tune Agent"
echo "================================"

# Check Python
python --version || python3 --version

# Create necessary directories
mkdir -p data/uploads data/processed outputs logs

# Check CUDA
echo ""
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Install dependencies if needed
if [ "$1" == "--install" ]; then
    echo ""
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Start server
echo ""
echo "Starting FastAPI server on http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""

cd "$(dirname "$0")/.."
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
