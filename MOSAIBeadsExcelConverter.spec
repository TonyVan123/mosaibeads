# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['excel_converter_main.py'],
    pathex=[],
    binaries=[],
    datas=[('assets/palettes', 'assets/palettes')],
    hiddenimports=['openpyxl', 'openpyxl.cell._writer'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['scipy', 'sklearn', 'matplotlib', 'pandas'],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [],
    name='MOSAIBeads_Excel_Converter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
