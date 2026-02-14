@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--help" goto help
echo [ERROR] Unknown argument: %~1
echo [ERROR] Use --help for usage.
exit /b 1

:args_done
if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found. Run this script from the sd-scripts root folder.
  exit /b 1
)
set "CUSTOM_REPO_URL=https://github.com/gesen2egee/custom.git"
set "CUSTOM_TMP_DIR=%CD%\.tmp_custom_repo"

where git >nul 2>&1
if errorlevel 1 (
  echo [ERROR] git was not found in PATH.
  exit /b 1
)

where python >nul 2>&1
if %errorlevel%==0 (
  set "PYTHON_CMD=python"
) else (
  where py >nul 2>&1
  if %errorlevel%==0 (
    set "PYTHON_CMD=py"
  ) else (
    echo [ERROR] Python was not found in PATH.
    echo [ERROR] Install Python 3.10.x first: https://www.python.org/downloads/windows/
    exit /b 1
  )
)

echo [INFO] Pulling latest code...
git pull
if errorlevel 1 (
  echo [ERROR] git pull failed. Resolve conflicts or local changes, then run again.
  exit /b 1
)

if not exist "venv\Scripts\python.exe" (
  echo [INFO] venv not found. Creating virtual environment...
  %PYTHON_CMD% -m venv venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
)

call "venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  exit /b 1
)

echo [INFO] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip.
  exit /b 1
)

echo [INFO] Upgrading requirements.txt...
python -m pip install --use-pep517 --upgrade -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to upgrade requirements.
  exit /b 1
)

call :sync_custom_repo
if errorlevel 1 (
  exit /b 1
)

echo.
echo [SUCCESS] Update completed.
echo [INFO] Repo is up to date and environment dependencies were upgraded.
exit /b 0

:sync_custom_repo
echo [INFO] Syncing custom repo into venv site-packages...
if exist "%CUSTOM_TMP_DIR%\.git" (
  git -C "%CUSTOM_TMP_DIR%" pull --ff-only
) else (
  if exist "%CUSTOM_TMP_DIR%" rmdir /s /q "%CUSTOM_TMP_DIR%"
  git clone --depth 1 "%CUSTOM_REPO_URL%" "%CUSTOM_TMP_DIR%"
)
if errorlevel 1 (
  echo [ERROR] Failed to update custom repo: %CUSTOM_REPO_URL%
  exit /b 1
)

set "SITE_PACKAGES="
for /f "usebackq delims=" %%I in (`python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"`) do set "SITE_PACKAGES=%%I"
if not defined SITE_PACKAGES (
  echo [ERROR] Failed to detect site-packages path.
  exit /b 1
)

set "CUSTOM_SITE_DIR=%SITE_PACKAGES%\custom"
if exist "%CUSTOM_SITE_DIR%" (
  rmdir /s /q "%CUSTOM_SITE_DIR%"
  if exist "%CUSTOM_SITE_DIR%" (
    echo [ERROR] Failed to remove old custom package: %CUSTOM_SITE_DIR%
    exit /b 1
  )
)

mkdir "%CUSTOM_SITE_DIR%" >nul 2>nul
robocopy "%CUSTOM_TMP_DIR%" "%CUSTOM_SITE_DIR%" /E /XD .git >nul
set "ROBOCOPY_RC=%errorlevel%"
if %ROBOCOPY_RC% GEQ 8 (
  echo [ERROR] Failed to copy custom package to venv. robocopy exit code: %ROBOCOPY_RC%
  exit /b 1
)
echo [INFO] custom package synced: %CUSTOM_SITE_DIR%
exit /b 0

:help
echo Usage: update.bat
echo.
echo What it does:
echo   1. git pull
echo   2. create venv if missing
echo   3. activate venv
echo   4. pip install --upgrade pip
echo   5. pip install --use-pep517 --upgrade -r requirements.txt
echo   6. sync custom repo to venv site-packages/custom
exit /b 0
