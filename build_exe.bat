@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv --system-site-packages
)
.venv\Scripts\python.exe -m pip install --upgrade pyinstaller
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean MOSAIBEADS.spec
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean MOSAIBEADSPortable.spec
echo.
echo Build finished: dist\MOSAIBEADS\MOSAIBEADS.exe
echo Portable EXE: dist\MOSAIBEADS_v3.0.exe
endlocal
