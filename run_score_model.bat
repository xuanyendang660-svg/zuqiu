@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONPATH=src
python -m score_model.cli examples\matches_12_sample.json --pretty
echo.
pause
