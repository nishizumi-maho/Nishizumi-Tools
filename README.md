# Nishizumi-Tools

A small collection of iRacing helper utilities built with Python + Tkinter. The repository currently contains two standalone apps:

- **Fuel Helper** (`fuelhelper (3).py`)
- **Turbo Pit** (`turbo_pit (4).py`)

Both tools use the iRacing SDK (`irsdk`) and are intended to be run on Windows while iRacing is active.

## Requirements

- Python 3.9+
- Windows (required for iRacing telemetry/broadcasts)
- iRacing SDK Python bindings (`irsdk`)
- Optional: `pygame` for audio playback in Fuel Helper

Install dependencies:

```bash
pip install -r requirements.txt
```

## Fuel Helper

**Purpose**

Fuel Helper monitors live iRacing telemetry and provides audio alerts for fuel-related events. It is designed to help you catch pit timing issues, especially when auto-fuel is active.

**What it does**

- Connects to iRacing telemetry (`irsdk`) and continuously reads fuel, lap, and pit data.
- Estimates fuel burn per lap from recent lap-to-lap fuel deltas, falling back to the SDK's fuel-per-hour field when needed.
- Triggers an **autofuel alert** if the current fuel + pending pit fuel will not reach the configured tank capacity.
- Triggers a **pit warning** once per lap when a configurable lap-percentage threshold is reached and the car is projected to need fuel.
- Stores user configuration (audio paths, volumes, tank capacity, pit trigger percent) under `%APPDATA%\FuelHelper\config.json`.

**UI highlights**

- Select separate audio clips for the autofuel alert and pit warning.
- Test buttons for both alert sounds.
- Volume sliders for each alert.
- Tank capacity input (with per-car overrides and a lock toggle).
- Live status + telemetry readout (car, track, fuel level, burn estimate, etc.).

**Run it**

```bash
python "fuelhelper (3).py"
```

## Turbo Pit

**Purpose**

Turbo Pit provides a simple UI (or optional CLI) to broadcast iRacing pit commands, focusing on quick toggling of tire changes and windshield service.

**What it does**

- Connects to iRacing’s broadcast interface and sends pit commands.
- Offers a **Toggle tires + windshield** action:
  - First press clears tire changes + windshield service.
  - Second press re-enables them.
- Supports **auto-clear on pit road** (based on the `OnPitRoad` telemetry flag).
- Loads and saves configuration under `%APPDATA%\TurboPit\config.json`.

**UI highlights**

- Connection status and IRSDK availability indicator.
- Single action button for the toggle behavior.
- Option to automatically clear on pit road entry.
- Built-in documentation tab explaining commands and behavior.

**Run it (GUI)**

```bash
python "turbo_pit (4).py"
```

**Run it (CLI)**

```bash
python "turbo_pit (4).py" --cli
```

## Notes

- Both apps depend on the iRacing SDK and will not function without iRacing running.
- If `pygame` is not installed, Fuel Helper will run but audio playback is disabled.
