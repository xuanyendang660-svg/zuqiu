@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONPATH=src
py -m score_model.gui
if errorlevel 1 (
  python -m score_model.gui
)

