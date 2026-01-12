@echo off
echo Starting Heart Disease Prediction Frontend...
echo.
echo Please make sure the backend server is running on another terminal.
echo Access the frontend at: http://localhost:3000
echo.
cd /d "%~dp0\frontend"
python -m http.server 3000
pause