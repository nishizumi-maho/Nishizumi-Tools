@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo === Nishizumi Tools single-EXE build ===
python --version >nul 2>&1
if errorlevel 1 (
    echo Python was not found in PATH.
    exit /b 1
)

set "DIST_DIR=%CD%\dist"
set "BUILD_DIR=%CD%\build"
set "SPEC_DIR=%BUILD_DIR%\specs"
set "EXE_NAME=NishizumiTools"
set "ICON_FILE_ICO=%CD%\nishizumi_tools_icon.ico"
set "ICON_FILE_PNG=%CD%\nishizumi_tools_icon.png"

if not exist "%ICON_FILE_ICO%" (
    echo Icon file was not found: "%ICON_FILE_ICO%"
    exit /b 1
)

python -m pip install --upgrade pip
python -m pip install pyinstaller pyside6 pyirsdk numpy requests pyyaml

if exist "%DIST_DIR%" rmdir /s /q "%DIST_DIR%"
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
mkdir "%DIST_DIR%"
mkdir "%BUILD_DIR%"
mkdir "%SPEC_DIR%"

python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --name "%EXE_NAME%" ^
  --icon "%ICON_FILE_ICO%" ^
  --distpath "%DIST_DIR%" ^
  --workpath "%BUILD_DIR%\%EXE_NAME%" ^
  --specpath "%SPEC_DIR%" ^
  --collect-all PySide6 ^
  --add-data "%ICON_FILE_PNG%;." ^
  --add-data "%ICON_FILE_ICO%;." ^
  --hidden-import Nishizumi_FuelMonitor ^
  --hidden-import nishizumi_pitcalibrator ^
  --hidden-import Nishizumi_TireWear ^
  --hidden-import Nishizumi_Traction ^
  --hidden-import Nishizumi_CautionOverlay ^
  menu.py

if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo.
echo Build finished.
echo Single EXE: "%DIST_DIR%\%EXE_NAME%.exe"
exit /b 0
