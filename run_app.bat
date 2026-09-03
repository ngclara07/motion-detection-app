@echo off

title CM3065 - Traffic Vision Lab

cd /d "%~dp0"

echo.
echo ============================================
echo   CM3065 - TRAFFIC VISION LAB
echo   Exercise 1.1 Vehicle Detection
echo ============================================
echo.

python -m streamlit run app.py

pause
