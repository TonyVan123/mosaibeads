# -*- mode: python ; coding: utf-8 -*-
import os
import cv2

face_xml = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('assets', 'assets'), (face_xml, 'cv2/data')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'pandas', 'scipy', 'sklearn', 'skimage', 'torch', 'torchvision',
              'onnx', 'onnxruntime', 'pygame'],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MOSAIBeads_v3.0.1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)
