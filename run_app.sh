#!/bin/bash
# AI Image Detector - Linux/macOS Run Script

echo "Starting AI Image Detector..."

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    echo "Then run: source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Start API server in background
echo "Starting FastAPI Backend on port 8000..."
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# Wait for API to start
echo "Waiting for API to initialize..."
sleep 3

# Start Streamlit Dashboard
echo "Starting Streamlit Dashboard..."
echo ""
echo "========================================"
echo "  Open http://localhost:8501 in browser"
echo "========================================"
echo ""

# Trap to kill API when script exits
trap "kill $API_PID 2>/dev/null" EXIT

streamlit run dashboard.py
