@echo off
cd /d "%~dp0"
python raec_gui.py
if errorlevel 1 (
    echo.
    echo === ERROR OCCURRED ===
    pause
)
