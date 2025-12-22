# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.building.build_main import Analysis, COLLECT, EXE, PYZ
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None

project_root = os.path.abspath(os.path.dirname(__file__))
source_dir = os.path.join(project_root, "iRacingTools")
main_script = os.path.join(source_dir, "main_launcher.pyw")

# Dependências com dados/recursos que o PyInstaller não detecta sozinho
_datas, _binaries, _hiddenimports = [], [], []
for pkg in ("customtkinter", "darkdetect"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    _datas += pkg_datas
    _binaries += pkg_binaries
    _hiddenimports += pkg_hidden

# pygame usa dados de fonte/recursos
_datas += collect_data_files("pygame")

# Dados do projeto
configs_dir = os.path.join(source_dir, "configs")
sons_dir = os.path.join(source_dir, "sons")
if os.path.isdir(configs_dir):
    _datas.append((configs_dir, "configs"))
if os.path.isdir(sons_dir):
    _datas.append((sons_dir, "sons"))


a = Analysis(
    [main_script],
    pathex=[source_dir],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="iRacingTools",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="iRacingTools",
)
