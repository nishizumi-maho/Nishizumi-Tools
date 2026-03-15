# Nishizumi Tools

Nishizumi Tools is now a **tool collection repository** for iRacing overlays and telemetry helpers.
It includes multiple standalone Python apps focused on fuel planning, tire wear learning, pit-stop strategy,
and traction analysis.

## Included apps

### 1) Nishizumi Fuel (`Nishizumi_Fuel.py`)
A lightweight Tkinter overlay for fuel strategy.

**What it does**
- Tracks live fuel-per-lap usage.
- Estimates remaining laps from current fuel.
- Shows stint planning insights with configurable target consumption.
- Detects refuels and resets stint tracking automatically.

**Run**
```bash
python Nishizumi_Fuel.py
```

---

### 2) Nishizumi TireWear (`Nishizumi_TireWear.py`)
A PyQt5 overlay that learns tire wear behavior from live telemetry.

**What it does**
- Reads iRacing telemetry at high frequency.
- Learns tire wear trends per tire over stints.
- Uses load and environmental variables (track temp, air temp, humidity).
- Persists learned model data by track/config/car in JSON.

**Run**
```bash
python Nishizumi_TireWear.py
```

---

### 3) Nishizumi PitTime (`Nishizumi_PitTime.py`)
A Tkinter pit-stop loss and rejoin safety estimator.

**What it does**
- Estimates total pit time loss (base + tires + fueling).
- Displays pit window and rejoin safety status.
- Supports fuel-rate profiling and optional custom fuel max.
- Includes compact/minimal mode for quick in-race use.

**Run**
```bash
python Nishizumi_PitTime.py
```

---

### 4) Nishizumi Traction (`Nishizumi_Traction.py`)
A traction-circle telemetry overlay with coaching-focused feedback.

**What it does**
- Visualizes real-time traction circle from longitudinal/lateral acceleration.
- Learns reference grip usage from valid laps.
- Detects underuse segments and provides improvement hints.
- Can load external IBT reference data or use live adaptive reference.

**Run**
```bash
python Nishizumi_Traction.py
```

## Requirements

- Python 3.10+
- iRacing running with telemetry enabled
- Python packages:
  - `irsdk` (all tools)
  - `numpy` + `pyqt5` (TireWear)
  - `tkinter` (Fuel, PitTime, Traction; included with most Python installs)

## Notes

- These tools are standalone scripts; run the one you need.
- Most overlays are always-on-top windows intended for race sessions.
- Some tools persist local settings/model data JSON files next to the script or in your home directory.

## Documentation

- See [`DOCUMENTATION.md`](DOCUMENTATION.md) for detailed fuel overlay behavior.

## Repository structure

- `Nishizumi_Fuel.py`
- `Nishizumi_TireWear.py`
- `Nishizumi_PitTime.py`
- `Nishizumi_Traction.py`
- `README.md`
- `DOCUMENTATION.md`
