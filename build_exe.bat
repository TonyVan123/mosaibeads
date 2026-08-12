@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv --system-site-packages
)
.venv\Scripts\python.exe -m pip install --upgrade pyinstaller
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean MOSAIBeads.spec
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean MOSAIBeadsPortable.spec
echo.
echo Build finished: dist\MOSAIBeads\MOSAIBeads.exe
echo Portable EXE: dist\MOSAIBeads_v3.0.1.exe
endlocal
