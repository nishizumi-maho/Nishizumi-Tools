# FuelMonitor Documentation

## Overview

Nishizumi FuelMonitor is a Tkinter-based overlay that connects to iRacing via the `irsdk` Python
bindings. It tracks fuel usage across a stint, filters anomalous laps, and provides both
high-level and detailed stint planning insights.

## Data flow

1. **Initialization**: The app starts the `irsdk` connection and waits for iRacing telemetry.
2. **Stint tracking**: When valid telemetry is detected, a stint is created with starting
   fuel, lap, lap distance, and timestamp.
3. **Lap tracking**: Each lap completion records the fuel used, filters out yellow flag laps,
   and skips anomalous outliers.
4. **Averaging**: The display uses the filtered lap list when available, otherwise it falls
   back to a stint-wide average.
5. **Insights**: Planned vs. expected laps and +/-1 lap targets are derived from the current
   fuel level and target values.

## Telemetry inputs

The overlay reads the following telemetry keys from iRacing:

- `FuelLevel` (float)
- `LapDistPct` (float)
- `Lap` (int)
- `IsOnTrack` (bool)
- `SessionFlags` (int)
- `OnPitRoad` (bool)

If telemetry is missing or the driver is not on track, the UI displays a waiting state until
values are available.

## Calculations

### Stint progress

Stint progress is computed from lap count and lap distance percentage:

```
progress = (lap - stint_lap_start) + (lapdist - stint_lapdist_start)
```

The overlay waits until a minimum progress threshold (5% of a lap) before showing averages
so that early values are not overly noisy.

### Fuel usage average

- **Primary**: average of filtered lap consumptions.
- **Fallback**: stint-wide fuel used divided by stint progress.

### Anomaly filtering

When at least three laps are available, a lap is considered anomalous if it deviates by 30%
(or more) from the current average of recorded laps. Anomalous laps are excluded from the
filtered average.

### Refuel detection

If the fuel level increases by 0.3 liters or more between samples, a refuel event is assumed
and the stint data is reset.

### Target comparison

When a target is set, the overlay compares the average fuel usage to the target and applies:

- **Green** when usage is within the target.
- **Red** when usage exceeds the target.

## Advanced insights

The **Insights** view adds the following calculations:

- **Planned laps**: `floor(fuel / target)`
- **Expected laps**: `floor(fuel / average)`
- **+1 lap target**: `fuel / (expected_laps + 1)`
- **-1 lap target**: `fuel / (expected_laps - 1)` (only when at least 2 expected laps)

The interface also shows per-lap savings needed to gain one lap, and (when possible) the
additional fuel per lap that would cost a lap.

## UI behavior

- The window is always on top, slightly transparent, and frameless.
- Dragging is disabled while the target is locked to prevent accidental moves.
- Closing the window persists its position to `%APPDATA%/NishizumiTools/fuel_consumption_monitor.json` (or `~/.config/NishizumiTools/fuel_consumption_monitor.json` on non-Windows).

## Troubleshooting

- **"Waiting for iRacing..."**: iRacing is not running or telemetry is unavailable.
- **"Waiting for telemetry..."**: iRacing is running but the driver is not on track or
  telemetry fields are not yet populated.
- **No updates**: confirm the `irsdk` package is installed and that iRacing telemetry output
  is enabled in the simulator.
