# Nishizumi Tools

Professional, lightweight iRacing telemetry overlays written in Python.

## Project layout

```
Nishizumi-Tools/
├── apps/                        # runnable overlay applications
│   ├── Nishizumi_FuelMonitor.py
│   ├── Nishizumi_Pittime.py
│   ├── Nishizumi_TireWear.py
│   └── Nishizumi_Traction.py
├── docs/
│   └── fuel-monitor.md          # detailed FuelMonitor technical documentation
├── .github/workflows/
│   └── build.yml                # CI validation workflow
├── LICENSE
├── README.md
└── requirements.txt
```

## Applications

- **FuelMonitor**: live fuel burn, stint projection, and strategy helper.
- **PitTime**: pit-stop loss and traffic-safe rejoin estimator.
- **TireWear**: learned tire degradation model with PyQt overlay.
- **Traction**: traction-circle coaching overlay for grip usage analysis.

## Quick start

### 1) Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2) Run an overlay

```bash
python apps/Nishizumi_FuelMonitor.py
python apps/Nishizumi_Pittime.py
python apps/Nishizumi_TireWear.py
python apps/Nishizumi_Traction.py
```

## Runtime notes

- Main target platform: **Windows**.
- Saved state is written to `%APPDATA%/NishizumiTools` on Windows.
- Non-Windows fallback path: `~/.config/NishizumiTools`.
- Start iRacing and click **Drive** before launching overlays for live telemetry.

## Development checks

```bash
python -m py_compile apps/*.py
```

