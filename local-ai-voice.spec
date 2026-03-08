# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules

datas = [('web/index.html', 'web')]
binaries = []
hiddenimports = ['browser_webrtc', 'av', 'noisereduce', 'webrtcvad', 'webview', 'hf_xet']
datas += collect_data_files('openvino')
datas += collect_data_files('openvino_genai')
datas += collect_data_files('openvino_tokenizers')
datas += collect_data_files('webview')
datas += collect_data_files('hf_xet')
binaries += collect_dynamic_libs('openvino')
binaries += collect_dynamic_libs('openvino_genai')
binaries += collect_dynamic_libs('openvino_tokenizers')
binaries += collect_dynamic_libs('hf_xet')
hiddenimports += collect_submodules('uvicorn')
hiddenimports += collect_submodules('websockets')


a = Analysis(
    ['local-ai-voice.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='local-ai-voice',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
