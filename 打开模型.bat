@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
set PYTHONPATH=src

if "%~1"=="" (
  set /p INPUT=请输入盈利玩法候选盘口 JSON 路径: 
) else (
  set INPUT=%~1
)

if "%INPUT%"=="" (
  echo 未提供输入文件，已退出。
  exit /b 2
)

py -m profit_model.cli "%INPUT%" --pretty
if errorlevel 1 (
  python -m profit_model.cli "%INPUT%" --pretty
)
