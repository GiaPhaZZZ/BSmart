@echo off
setlocal
cd /d "%~dp0\.."
echo Starting BSmart Python Backend Server...
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload
pause
