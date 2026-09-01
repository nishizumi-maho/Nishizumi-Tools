Nishizumi Tools - Single EXE package

What changed
- Launcher version is now v9
- Caution Overlay is now part of the launcher: full course cautions and caution laps, opened like any other app
- The launcher builds into one EXE only: NishizumiTools.exe
- FuelMonitor, Pit Calibrator, TireWear, Traction, and Caution Overlay are all launched from inside that same EXE
- The custom icon is used for the EXE and for the launcher tray icon
- Closing the launcher can hide it to the system tray instead of exiting
- The launcher checks GitHub Releases for updates every 6 hours

How to build
1. Open this folder on Windows.
2. Run build_all.bat
3. The final file will be in dist\NishizumiTools.exe
4. Optional check: run "dist\NishizumiTools.exe --selftest" - it exits with code 0 when every app is bundled

Notes
- The launcher can still open and close each app individually.
- App data is still saved in %APPDATA%\NishizumiTools
- The individual Python files stay in this source package so PyInstaller can bundle them into the single EXE
- Nishizumi_CautionOverlay.py is a copy of apps\Nishizumi_CautionOverlay.py and CI checks that the two stay identical
