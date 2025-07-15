@echo off
echo 🌙 Starting Dream Analysis Dashboard - Optimized V2...
echo 📂 Working with data from: logs_optimized_v2/
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python not found in PATH
    echo Please ensure Python is installed and added to PATH
    pause
    exit /b 1
)

REM Check if logs_optimized_v2 directory exists
if not exist "logs_optimized_v2" (
    echo ❌ Error: Directory logs_optimized_v2 not found!
    echo Please ensure the logs_optimized_v2 directory exists with dream data.
    pause
    exit /b 1
)

echo ✅ Found logs_optimized_v2 directory
echo 🚀 Starting Streamlit app...
echo 📱 The app will open in your default web browser
echo 🔗 URL: http://localhost:8501
echo.
echo 📝 Note: Close this window or press Ctrl+C to stop
echo ==========================================
echo.

REM Run the streamlit app
python -m streamlit run streamlit_dream_analyzer_optimized.py

echo.
echo 👋 Streamlit app stopped
pause 