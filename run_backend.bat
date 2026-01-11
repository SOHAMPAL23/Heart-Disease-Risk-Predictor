@echo off
echo Starting Heart Disease Prediction Backend...
echo.
echo Make sure you have activated your virtual environment.
echo Backend API will be available at: http://localhost:8002
echo.
cd /d "%~dp0\backend"
python main.py
pause