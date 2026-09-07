@echo off
title DataMind AI Launcher
echo ===================================================
echo   Starting DataMind AI Platform
echo   - Backend:  http://127.0.0.1:8000
echo   - Frontend: http://localhost:5173
echo ===================================================

echo Starting Backend API Server...
start "DataMind Backend (FastAPI)" cmd /k "cd /d %~dp0backend && venv\Scripts\activate && python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000"

echo Starting Frontend Dev Server...
start "DataMind Frontend (Vite)" cmd /k "cd /d %~dp0frontend && npm run dev"

echo Both services launched in separate windows!
timeout /t 3 /nobreak >nul
start http://localhost:5173
