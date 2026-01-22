# AI Image Detector - Run Script
Write-Host "Starting AI Image Detector..." -ForegroundColor Green

# Activate venv
& .\venv\Scripts\Activate.ps1

# Start API in background
Write-Host "Starting FastAPI Backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\venv\Scripts\Activate.ps1; uvicorn api.main:app --reload --host 127.0.0.1 --port 8000"

# Wait for API to start
Start-Sleep -Seconds 3

# Start Streamlit
Write-Host "Starting Streamlit Dashboard..." -ForegroundColor Yellow
& .\venv\Scripts\streamlit run dashboard.py
