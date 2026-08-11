# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules
import os
import cv2

hiddenimports = collect_submodules('cv2')
face_xml = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('assets', 'assets'), (face_xml, 'cv2/data')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pandas', 'scipy', 'sklearn', 'skimage', 'torch'],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BeadSketchStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BeadSketchStudio',
)
