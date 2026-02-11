@echo off
setlocal

cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

if not exist ".\venv\Scripts\activate" (
  echo [ERROR] venv not found: .\venv\Scripts\activate
  pause
  exit /b 1
)

if not exist ".\anima_train_config.toml" (
  echo [ERROR] config not found: .\anima_train_config.toml
  pause
  exit /b 1
)

if not exist ".\anima_train_network.py" (
  echo [ERROR] script not found: .\anima_train_network.py
  pause
  exit /b 1
)

call ".\venv\Scripts\activate"

call ".\venv\Scripts\accelerate.exe" launch ^
  --dynamo_backend no ^
  --dynamo_mode default ^
  --mixed_precision bf16 ^
  --num_processes 1 ^
  --num_machines 1 ^
  --num_cpu_threads_per_process 2 ^
  ".\anima_train_network.py" ^
  --config_file ".\anima_train_config.toml"

if errorlevel 1 (
  echo.
  echo [ERROR] anima training failed.
  pause
  exit /b 1
)

echo.
echo [OK] anima training finished.
pause
exit /b 0
