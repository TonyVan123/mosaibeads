@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv --system-site-packages
)
.venv\Scripts\python.exe -m pip install --upgrade pyinstaller
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean BeadSketchStudio.spec
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean BeadSketchStudioPortable.spec
echo.
echo Build finished: dist\BeadSketchStudio\BeadSketchStudio.exe
echo Portable EXE: dist\BeadSketchStudio_v2.3.exe
endlocal
