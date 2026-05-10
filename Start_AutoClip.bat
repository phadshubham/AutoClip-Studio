@echo off
title AutoClip Studio - Starting...
echo ========================================================
echo   AutoClip Studio: YouTube to Vertical Shorts
echo ========================================================
echo.
echo Starting local AI pipeline...
echo.

:: Check if python is installed
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python and add it to your PATH.
    pause
    exit /b
)

:: Run the start script
python start_server.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application failed to start.
    echo This might be due to missing dependencies.
    echo Try running: pip install -r requirements.txt
    echo.
    pause
)
