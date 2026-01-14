# Nishizumi Tools

Two small desktop helpers for iRacing pit operations.

## Apps

### Fuel Helper (`fuelhelper.py`)

Fuel Helper is a tkinter desktop app that watches live iRacing telemetry and helps you avoid missing fuel stops. It:

- Monitors fuel level, pit fuel target, estimated burn per lap, lap distance, and whether you need to pit.
- Detects when iRacing autofuel is armed and alerts if the configured pit fuel would exceed the tank capacity.
- Plays optional audio alerts for autofuel warnings and late-lap pit warnings (via `pygame`).
- Tracks per-car tank capacity overrides and stores settings in a config file in `%APPDATA%/FuelHelper/config.json`.

The app connects to iRacing via `irsdk` and shows basic telemetry status in the UI. It supports a configurable pit warning trigger (lap %), volume sliders, and audio file selection for alerts.

### Turbo Pit (`turbo_pit.py`)

Turbo Pit is a tkinter desktop app (with an optional CLI mode) for sending pit command broadcasts to iRacing. It:

- Connects to iRacing via `irsdk` on Windows and sends pit commands (tire changes, windshield tear-off, fuel, etc.).
- Provides a single toggle action that clears tires + windshield on first press, and re-enables them on the next.
- Supports auto-clearing tires + windshield when the car enters pit road if enabled.
- Stores configuration in `%APPDATA%/TurboPit/config.json` and exposes a built-in documentation tab.

Run in CLI mode with `python turbo_pit.py --cli` for a simple console menu.

## Requirements

- Python 3.9+
- Windows (iRacing + `irsdk` are required to connect to live telemetry)
- `irsdk` (iRacing SDK Python bindings)
- `pygame` (optional, used by Fuel Helper for audio playback)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running

```bash
python fuelhelper.py
python turbo_pit.py
```

## Notes

- These tools rely on iRacing telemetry and pit command broadcasts; they will show a disconnected status when iRacing is not running or the SDK is unavailable.
- Audio alerts in Fuel Helper are disabled if `pygame` is not installed.
