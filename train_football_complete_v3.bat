@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=src
python -m score_model.training_cli --min-samples 3000
pause
