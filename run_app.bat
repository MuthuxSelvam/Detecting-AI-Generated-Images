@echo off
REM AI Image Detector - Windows Batch Run Script
echo Starting AI Image Detector...

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    echo Then run: venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start API server in background
echo Starting FastAPI Backend on port 8000...
start "AI Detector API" cmd /k "venv\Scripts\python -m uvicorn api.main:app --host 127.0.0.1 --port 8000"

REM Wait for API to start
echo Waiting for API to initialize...
timeout /t 3 /nobreak >nul

REM Start Streamlit Dashboard
echo Starting Streamlit Dashboard...
echo.
echo ========================================
echo   Open http://localhost:8501 in browser
echo ========================================
echo.
venv\Scripts\streamlit run dashboard.py
