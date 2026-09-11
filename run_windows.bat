@echo off
call .venv\Scripts\activate
start "Network Collector" cmd /k python collector.py
timeout /t 2 >nul
start "Network Dashboard" cmd /k streamlit run app.py
echo Collector and dashboard started.
