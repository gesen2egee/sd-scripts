@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "INSTALL_XFORMERS="

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--with-xformers" set "INSTALL_XFORMERS=Y"
if /I "%~1"=="--without-xformers" set "INSTALL_XFORMERS=N"
if /I "%~1"=="--help" goto help
shift
goto parse_args

:args_done
if not exist "requirements.txt" (
  echo [ERROR] requirements.txt not found. Run this script from the sd-scripts root folder.
  exit /b 1
)
set "CUSTOM_REPO_URL=https://github.com/gesen2egee/custom.git"
set "CUSTOM_TMP_DIR=%CD%\.tmp_custom_repo"

if not defined TORCH_VERSION set "TORCH_VERSION=2.6.0"
if not defined TORCHVISION_VERSION set "TORCHVISION_VERSION=0.21.0"
set "CUDA_TAG_SOURCE=env"
if not defined CUDA_TAG (
  call :detect_cuda_tag
  if defined CUDA_TAG (
    set "CUDA_TAG_SOURCE=auto"
  ) else (
    set "CUDA_TAG=cu124"
    set "CUDA_TAG_SOURCE=default"
  )
)
set "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/%CUDA_TAG%"

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

echo [INFO] Python launcher: %PYTHON_CMD%
echo [INFO] torch=%TORCH_VERSION%, torchvision=%TORCHVISION_VERSION%
echo [INFO] cuda tag=%CUDA_TAG% ^(%CUDA_TAG_SOURCE%^)
if defined CUDA_DETECTED_VERSION echo [INFO] detected CUDA runtime: %CUDA_DETECTED_VERSION%
echo [INFO] Starting install in: %CD%

if not exist "venv\Scripts\python.exe" (
  echo [INFO] Creating virtual environment...
  %PYTHON_CMD% -m venv venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
  )
) else (
  echo [INFO] Reusing existing virtual environment.
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

echo [INFO] Installing PyTorch...
python -m pip install torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% --index-url %PYTORCH_INDEX_URL%
if errorlevel 1 (
  echo [ERROR] Failed to install PyTorch.
  exit /b 1
)

echo [INFO] Installing requirements.txt...
python -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to install requirements.
  exit /b 1
)

call :sync_custom_repo
if errorlevel 1 (
  exit /b 1
)

if not defined INSTALL_XFORMERS (
  set /p INSTALL_XFORMERS=Install xformers (optional)? [Y/n]:
  if "!INSTALL_XFORMERS!"=="" set "INSTALL_XFORMERS=Y"
)

if /I "!INSTALL_XFORMERS!"=="Y" (
  echo [INFO] Installing xformers...
  python -m pip install xformers --index-url %PYTORCH_INDEX_URL%
  if errorlevel 1 (
    echo [WARN] xformers install failed. Continuing without xformers.
  )
) else (
  echo [INFO] Skipping xformers.
)

echo.
echo [INFO] Recommended answers for "accelerate config":
echo   - This machine
echo   - No distributed training
echo   - NO
echo   - NO
echo   - NO
echo   - all ^(or 0 for single GPU if needed^)
echo   - fp16 ^(or bf16 if your GPU supports bf16^)
echo.
set "RUN_ACCELERATE="
set /p RUN_ACCELERATE=Run "accelerate config" now? [Y/n]:
if "!RUN_ACCELERATE!"=="" set "RUN_ACCELERATE=Y"

if /I "!RUN_ACCELERATE!"=="Y" (
  accelerate config
  if errorlevel 1 (
    echo [WARN] accelerate config did not complete. Run it later with:
    echo        accelerate config
  )
) else (
  echo [INFO] Skipped accelerate config.
)

echo.
echo [SUCCESS] Installation completed.
echo [INFO] Next time activate with: venv\Scripts\activate.bat
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
echo Usage: install.bat [--with-xformers ^| --without-xformers]
echo.
echo Environment overrides:
echo   TORCH_VERSION       default: 2.6.0
echo   TORCHVISION_VERSION default: 0.21.0
echo   CUDA_TAG            auto-detect from nvidia-smi, fallback: cu124
echo.
echo Example:
echo   set CUDA_TAG=cu121 ^&^& install.bat --with-xformers
exit /b 0

:detect_cuda_tag
set "CUDA_DETECTED_VERSION="
set "CUDA_MAJOR_NUM="
set "CUDA_MINOR_NUM="

for /f "usebackq delims=" %%V in (`powershell -NoProfile -Command "$ErrorActionPreference='SilentlyContinue'; $v=''; if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { $m = nvidia-smi | Select-String 'CUDA Version:\s*([0-9]+\.[0-9]+)' | Select-Object -First 1; if ($m) { $v = $m.Matches[0].Groups[1].Value } }; Write-Output $v"`) do set "CUDA_DETECTED_VERSION=%%V"

if not defined CUDA_DETECTED_VERSION exit /b 0

for /f "tokens=1,2 delims=." %%M in ("%CUDA_DETECTED_VERSION%") do (
  set /a CUDA_MAJOR_NUM=%%M
  set /a CUDA_MINOR_NUM=%%N
)

if not defined CUDA_MAJOR_NUM exit /b 0

if !CUDA_MAJOR_NUM! GTR 12 (
  set "CUDA_TAG=cu124"
  exit /b 0
)

if !CUDA_MAJOR_NUM! EQU 12 (
  if !CUDA_MINOR_NUM! GEQ 4 (
    set "CUDA_TAG=cu124"
  ) else if !CUDA_MINOR_NUM! GEQ 1 (
    set "CUDA_TAG=cu121"
  ) else (
    set "CUDA_TAG=cu118"
  )
  exit /b 0
)

if !CUDA_MAJOR_NUM! GEQ 11 (
  set "CUDA_TAG=cu118"
)
exit /b 0
