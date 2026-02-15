@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ROOT_DIR=%~dp0.."
cd /d "%ROOT_DIR%"

if not exist ".\venv\Scripts\python.exe" (
  echo [ERROR] root venv not found: .\venv\Scripts\python.exe
  exit /b 1
)

set "UI_HOST=127.0.0.1"
set "UI_PORT=8765"
set "UI_URL=http://%UI_HOST%:%UI_PORT%"

echo [INFO] Starting ANIMA UI backend at %UI_URL%
start "ANIMA UI Backend" "%ROOT_DIR%\venv\Scripts\python.exe" -m ui.backend.app --host %UI_HOST% --port %UI_PORT%

timeout /t 2 /nobreak >nul
start "ANIMA UI" "%UI_URL%"

endlocal
exit /b 0
