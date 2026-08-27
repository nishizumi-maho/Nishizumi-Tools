# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build of Nishizumi Caution Overlay.

PySide6 ships a great deal that a two number widgets overlay never touches and
PyInstaller collects most of it by default, which makes the one file build
about 39 MB.  Everything listed below is dropped from the bundle instead.

Nothing here is guessed blindly: the workflow runs the frozen executable with
``--selftest`` right after building it, so removing something that is actually
needed fails the build rather than shipping a broken .exe.
"""

import os

# Dropped by file name (the Qt libraries and plugins this app never loads).
BLOCKED_NAMES = (
    "opengl32sw.dll",  # software OpenGL fallback; widgets paint with raster
    "d3dcompiler_47.dll",  # only used by the OpenGL / Qt Quick backends
    "qt6network.dll",  # nothing in the overlay talks to the network
    "qtnetwork.pyd",
    "qt6svg.dll",  # no SVG is ever loaded
    "qtsvg.pyd",
    "libcrypto-3.dll",  # OpenSSL, dragged in by QtNetwork only
    "libcrypto-3-x64.dll",
    "libssl-3.dll",
    "libssl-3-x64.dll",
    "qdirect2d.dll",  # alternative platform plugin; qwindows is the one used
)

# Dropped by folder.
BLOCKED_DIRS = (
    "plugins/imageformats",  # no image file is ever decoded
    "plugins/tls",  # TLS backends belong to QtNetwork
    "plugins/networkinformation",
    "plugins/iconengines",
    "plugins/generic",
    "translations",  # the UI is English only
)


def keep(entry):
    destination = entry[0].replace("\\", "/")
    if os.path.basename(destination).lower() in BLOCKED_NAMES:
        return False
    return not any(blocked in destination for blocked in BLOCKED_DIRS)


a = Analysis(
    [os.path.join(SPECPATH, "Nishizumi_CautionOverlay.py")],
    pathex=[SPECPATH],
    binaries=[],
    datas=[],
    hiddenimports=["irsdk", "yaml"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6.QtNetwork",
        "PySide6.QtSvg",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtOpenGL",
        "PySide6.QtPrintSupport",
        "PySide6.QtDBus",
        "tkinter",
    ],
    noarchive=False,
    optimize=0,
)

a.binaries = [entry for entry in a.binaries if keep(entry)]
a.datas = [entry for entry in a.datas if keep(entry)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="Nishizumi_CautionOverlay",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
