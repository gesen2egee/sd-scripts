@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not defined TORCHDYNAMO_DISABLE set TORCHDYNAMO_DISABLE=1

set "BASE_CONFIG=.\anima_train_config.toml"
set "TRAIN_SCRIPT=.\anima_train_network.py"
set "TEMP_DATASET_CONFIG=.\__anima_dataset_runtime.toml"

if not exist ".\venv\Scripts\activate" (
  echo [ERROR] venv not found: .\venv\Scripts\activate
  pause
  exit /b 1
)

if not exist "%BASE_CONFIG%" (
  echo [ERROR] config not found: %BASE_CONFIG%
  pause
  exit /b 1
)

if not exist "%TRAIN_SCRIPT%" (
  echo [ERROR] script not found: %TRAIN_SCRIPT%
  pause
  exit /b 1
)

call ".\venv\Scripts\activate"
call :run_all
set "RUN_EXIT=%errorlevel%"

if exist "%TEMP_DATASET_CONFIG%" del /q "%TEMP_DATASET_CONFIG%" >nul 2>nul

if not "%RUN_EXIT%"=="0" (
  echo.
  echo [ERROR] anima batch training failed.
  pause
  exit /b %RUN_EXIT%
)

echo.
echo [OK] anima batch training finished.
pause
exit /b 0

:run_all

call :run_profile "E:\NE\20_miss valentine" "miss valentine anima"
if errorlevel 1 exit /b 1

call :run_profile "D:\SDXL\ai-toolkit\datasets\1_wang yu wen" "wang yu wen anima"
if errorlevel 1 exit /b 1

call :run_profile "D:\SDXL\ai-toolkit\datasets\hong jin youn" "hong jin youn anima"
if errorlevel 1 exit /b 1

call :run_profile "G:\sera\100_{sera}" "sera anima"
if errorlevel 1 exit /b 1

call :run_profile "E:\hypno2\100_{mind control} {before and after}" "mind control before and after anima"
if errorlevel 1 exit /b 1

call :run_profile "D:\SDXL\ai-toolkit\datasets\loli" "loli anima"
if errorlevel 1 exit /b 1

exit /b 0

:run_profile
set "CURRENT_IMAGE_DIR=%~1"
set "CURRENT_LORA_BASE=%~2"
set "CURRENT_OUTPUT_NAME=!CURRENT_LORA_BASE!"

call :write_dataset_config "%TEMP_DATASET_CONFIG%" "!CURRENT_IMAGE_DIR!"
if errorlevel 1 exit /b 1

echo.
echo [INFO] Start training: !CURRENT_OUTPUT_NAME!
echo [INFO]   dataset: !CURRENT_IMAGE_DIR!
echo [INFO]   resolutions: 640/768/1024

call ".\venv\Scripts\accelerate.exe" launch ^
  --dynamo_backend no ^
  --dynamo_mode default ^
  --mixed_precision bf16 ^
  --num_processes 1 ^
  --num_machines 1 ^
  --num_cpu_threads_per_process 2 ^
  "%TRAIN_SCRIPT%" ^
  --config_file "%BASE_CONFIG%" ^
  --dataset_config "%TEMP_DATASET_CONFIG%" ^
  --output_name "!CURRENT_OUTPUT_NAME!"

if errorlevel 1 (
  echo [ERROR] Training failed: !CURRENT_OUTPUT_NAME!
  exit /b 1
)

exit /b 0

:write_dataset_config
set "CFG_FILE=%~1"
set "CFG_IMAGE_DIR=%~2"

> "%CFG_FILE%" (
  echo [general]
  echo shuffle_caption = false
  echo caption_extension = '.txt'
  echo.
  echo [[datasets]]
  echo resolution = [512, 512]
  echo batch_size = 1
  echo enable_bucket = true
  echo min_bucket_reso = 256
  echo max_bucket_reso = 768
  echo bucket_reso_steps = 16
  echo bucket_no_upscale = true
  echo.
  echo   [[datasets.subsets]]
  echo   image_dir = '%CFG_IMAGE_DIR%'
  echo   num_repeats = 1
  echo   caption_extension = '.txt'
  echo.
  echo [[datasets]]
  echo resolution = [640, 640]
  echo batch_size = 1
  echo enable_bucket = true
  echo min_bucket_reso = 256
  echo max_bucket_reso = 1024
  echo bucket_reso_steps = 16
  echo bucket_no_upscale = true
  echo.
  echo   [[datasets.subsets]]
  echo   image_dir = '%CFG_IMAGE_DIR%'
  echo   num_repeats = 1
  echo   caption_extension = '.txt'
  echo.
  echo [[datasets]]
  echo resolution = [768, 768]
  echo batch_size = 1
  echo enable_bucket = true
  echo min_bucket_reso = 384
  echo max_bucket_reso = 1152
  echo bucket_reso_steps = 16
  echo bucket_no_upscale = true
  echo.
  echo   [[datasets.subsets]]
  echo   image_dir = '%CFG_IMAGE_DIR%'
  echo   num_repeats = 1
  echo   caption_extension = '.txt'
  echo.
  echo [[datasets]]
  echo resolution = [1024, 1024]
  echo batch_size = 1
  echo enable_bucket = true
  echo min_bucket_reso = 512
  echo max_bucket_reso = 1280
  echo bucket_reso_steps = 16
  echo bucket_no_upscale = true
  echo.
  echo   [[datasets.subsets]]
  echo   image_dir = '%CFG_IMAGE_DIR%'
  echo   num_repeats = 1
  echo   caption_extension = '.txt'
)

if errorlevel 1 (
  echo [ERROR] Failed to write dataset config: %CFG_FILE%
  exit /b 1
)

exit /b 0
