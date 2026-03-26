# Nishizumi Tools

➡️ **Latest Release:** [Get the newest release package](https://github.com/<YOUR-USER-OR-ORG>/Nishizumi-Tools/releases/latest)

Nishizumi Tools is a collection of standalone iRacing helper overlays written in Python. Each app focuses on one race-day task:

- **Nishizumi FuelMonitor**: fuel usage tracking, stint projection, and target delta.
- **Nishizumi Pit Calibrator**: one-stop pit timing calibration (total, service, base, fuel, and manual tire marker).
- **Nishizumi TireWear**: learned tire degradation model per car/track/conditions.
- **Nishizumi Traction**: grip-usage coaching overlay based on your telemetry.

---

## Requirements

### Required software

- **Windows** recommended (data is stored in `%APPDATA%/NishizumiTools` when available).
- **Python 3.10+**
- **iRacing** running with live telemetry.

Install dependencies:

```bash
pip install irsdk numpy pyqt5
```

Notes:

- `tkinter` is used by FuelMonitor, Pit Calibrator, and Traction (usually included with Python).
- `numpy` + `pyqt5` are required for TireWear.
- On Linux/macOS, apps use `~/.config/NishizumiTools` as fallback.

### Recommended launch order

1. Open iRacing.
2. Join a session.
3. Click **Drive** (telemetry must be live).
4. Launch the app you want.

---

## Data storage

Saved app data location:

- Windows: `%APPDATA%/NishizumiTools`
- Linux/macOS fallback: `~/.config/NishizumiTools`

Files currently used:

- `fuel_consumption_monitor.json` — FuelMonitor window position.
- `nishizumi_tirewear_model.json` — TireWear learned model.
- `nishizumi_tirewear_settings.json` — TireWear HUD settings.

---

## Included apps

## 1) Nishizumi FuelMonitor

**File:** `apps/Nishizumi_FuelMonitor.py`

```bash
python apps/Nishizumi_FuelMonitor.py
```

### What it does

FuelMonitor provides a compact always-on-top overlay for race fuel management:

- average fuel/lap
- delta vs your manual target
- current fuel in tank
- estimated laps remaining
- last completed lap usage
- stint length projection (target vs measured)

### Core usage

1. Launch while in the car.
2. Set **Target L/Lap**.
3. Drive a few representative laps.
4. Watch the delta and remaining laps.
5. Use **R** to reset stint tracking when needed.

---

## 2) Nishizumi Pit Calibrator

**File:** `apps/nishizumi_pitcalibrator.py`

```bash
python apps/nishizumi_pitcalibrator.py
```

### What it does (detailed)

Pit Calibrator is a **manual calibration app** to measure one pit stop cleanly and copy the values into your strategy workflow.

It does not build long-term profiles and does not auto-tune settings. Instead, it captures a single armed stop and freezes the result for easy note-taking.

During an armed stop, it tracks:

- **Live total**: full pit-road time while you are on pit lane.
- **Live service**: time with pit service active.
- **Live base**: `total - service` (entry/exit + non-service portion).
- **Live fuel added**: measured from tank delta.
- **Live fuel rate**: average from valid positive fueling samples.
- **Live manual tire time**: optional timestamp captured with the **TIRE** button.
- **Pending pit fuel**: requested fuel (`PitSvFuel`) value.

After you leave pit road, it freezes these as **Saved** values:

- Saved total
- Saved service
- Saved base
- Saved fuel added
- Saved fuel rate
- Saved tire time

### Full workflow

1. Start iRacing, enter the car, then launch Pit Calibrator.
2. Verify connection line shows iRacing connected.
3. Click **ARM** before the stop you want to measure.
4. Enter pit lane as normal.
5. Watch live values update in real time.
6. (Optional) Click **TIRE** at the exact moment tire service is done to store a manual tire marker.
7. Exit pit road.
8. The app automatically ends the armed stop and freezes the final values in the **Frozen result** section.
9. Copy those values into your notes/setup sheet.
10. Re-arm for another calibration stop if needed.

### UI and behavior details

- Borderless always-on-top floating window.
- Drag from the title area.
- **ARM** toggles whether the next stop is measured.
- **TIRE** works only while an armed stop is actively running.
- Designed for quick calibration sessions (practice, test day, race prep).

### Best use cases

- Building accurate pit-loss baselines for a track.
- Separating base pit-road loss from service time.
- Measuring real fueling rate under current session conditions.
- Creating reliable notes for race strategy tools.

---

## 3) Nishizumi TireWear

**File:** `apps/Nishizumi_TireWear.py`

```bash
python apps/Nishizumi_TireWear.py
```

### What it does

TireWear learns degradation behavior for your current car/track/config over multiple completed stints and estimates live tire condition with confidence feedback.

---

## 4) Nishizumi Traction

**File:** `apps/Nishizumi_Traction.py`

```bash
python apps/Nishizumi_Traction.py
```

### What it does

Traction shows live grip usage and coaching hints so you can identify where you are leaving performance on the table.

---

## Removed / archived app

- **Nishizumi_Pittime** was removed from active apps and moved to `OLD/Nishizumi_Pittime.py`.

---

## Quick app chooser

- Need fuel targets and stint projections → **FuelMonitor**
- Need to calibrate pit timings from real stops → **Pit Calibrator**
- Need long-run tire degradation learning → **TireWear**
- Need grip/coaching feedback → **Traction**

---

## Repository structure

- `README.md`
- `requirements.txt`
- `LICENSE`
- `apps/Nishizumi_FuelMonitor.py`
- `apps/nishizumi_pitcalibrator.py`
- `apps/Nishizumi_TireWear.py`
- `apps/Nishizumi_Traction.py`
- `OLD/Nishizumi_Pittime.py`
- `docs/fuel-monitor.md`

