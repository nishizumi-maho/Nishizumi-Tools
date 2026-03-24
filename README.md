# Nishizumi Tools

Nishizumi Tools is a collection of standalone iRacing helper overlays written in Python. Each app solves a different race-day problem:

- **Nishizumi FuelMonitor** helps you understand fuel consumption, target saving, and estimated laps left.
- **Nishizumi PitTime** estimates your total pit loss and whether you will rejoin traffic safely.
- **Nishizumi TireWear** learns tire degradation for the current car/track combination over time.
- **Nishizumi Traction** shows how much grip you are using and points out where you are leaving time on the table.

The goal of this README is to explain the apps in plain language so a new user can install them, launch them, and understand what they are looking at without extra guidance.

## What you need before using any app

### 1) Required software

- **Windows** is the main intended platform because the tools store their data in `%APPDATA%/NishizumiTools` when available.
- **Python 3.10+**.
- **iRacing** running with telemetry available.
- Required Python packages:

```bash
pip install irsdk numpy pyqt5
```

Notes:

- `tkinter` is used by FuelMonitor, PitTime, and Traction. It is included with most standard Python installs.
- `numpy` and `pyqt5` are required for TireWear.
- On Linux/macOS, the apps fall back to `~/.config/NishizumiTools` for saved data instead of `%APPDATA%/NishizumiTools`.

### 2) Start order

For the smoothest experience:

1. Launch iRacing.
2. Join a session.
3. Click **Drive** so live telemetry is flowing.
4. Launch the app you want.

If an overlay says it is waiting for telemetry or disconnected, iRacing is usually open but you have not entered the car yet.

## Where the apps save data

Several tools remember settings or learned values. They store them here:

- Windows: `%APPDATA%/NishizumiTools`
- Fallback on other systems: `~/.config/NishizumiTools`

Files currently used by this repo include:

- `fuel_consumption_monitor.json` — saved FuelMonitor overlay position.
- `nishizumi_pittime_profiles.json` — learned PitTime fuel rates and pit-loss profiles.
- `nishizumi_tirewear_model.json` — TireWear learned model data.
- `nishizumi_tirewear_settings.json` — TireWear overlay settings.

This means you can close and reopen the apps without losing everything each time.

## Included apps

## 1) Nishizumi FuelMonitor

**File:** `apps/Nishizumi_FuelMonitor.py`

Run:

```bash
python apps/Nishizumi_FuelMonitor.py
```

### What it does

Nishizumi FuelMonitor is a small always-on-top overlay that watches live fuel use and turns it into simple race information:

- current average fuel per lap
- difference versus your manual target
- current fuel remaining
- estimated laps remaining
- last completed lap usage
- comparison between your planned stint and estimated real stint

It also detects pit entry/refueling behavior and can briefly switch to a larger stint-average display while you are on pit road.

### What the overlay means

From top to bottom:

- **Large number** — your current average fuel consumption per lap.
- **Number in parentheses** — the delta versus the target you entered.
  - negative = you are saving more fuel than target
  - positive = you are using more fuel than target
- **Fuel** — how much fuel is currently in the car.
- **Remaining** — estimated laps left at the current average.
- **Last lap** — fuel used on the last valid completed lap.
- **Stint: (C) / (E)**
  - (C) = laps expected from the target consumption you typed in
  - (E) = laps estimated from your actual measured usage

### How to use it in a session

1. Start the app after entering the car in iRacing.
2. Drag the window to where you want it.
3. Watch a few green-flag laps so the app can build a useful average.
4. In **Target L/Lap**, type the fuel target you want.
5. If you do not want accidental edits, click the lock checkbox.
6. Use the overlay to see whether you are above or below target.

### Buttons and controls

- **R** — manually resets current stint tracking.
- **I** — toggles the advanced information panel.
- **Target field** — your planned fuel-per-lap target.
- **Lock checkbox** — freezes the target value and disables editing.

### Advanced panel explained

When you open the advanced panel, the app calculates what would be required to gain or lose one lap of stint length.

Quick actions:

- **+1 lap** — fills a more conservative target to stretch the stint one lap further.
- **-1 lap** — fills a more aggressive target to spend fuel faster and shorten by one lap.

Useful for quick answers such as:

- “How much do I need to save per lap to make this fuel number work?”
- “If I push harder, how much more fuel can I spend?”

### Important behavior to know

- The app ignores caution laps when building stored lap history.
- It rejects obvious anomalous lap values once enough history exists.
- It adapts to liters or gallons based on iRacing display units.
- Entering pit road shows a temporary large stint average display for ~10 seconds.
- Overlay position is saved automatically when moved/closed.

### Best use case

Use FuelMonitor during races where you need a quick answer to:

- whether you are on target
- how many laps you can realistically go
- whether one extra lap is possible with saving

## 2) Nishizumi PitTime

**File:** `apps/Nishizumi_Pittime.py`

Run:

```bash
python apps/Nishizumi_Pittime.py
```

### What it does

Nishizumi PitTime estimates two things in real time:

1. How much total time your next pit stop will cost.
2. How safe your rejoin will be relative to nearby traffic.

It combines:

- user-entered base pit loss
- optional tire-change time
- learned/stored fueling rate
- current fuel in car and tank size
- nearby-car estimated deltas from iRacing telemetry

### Main idea behind the calculation

The app computes:

`total pit loss = base pit loss + tire-change loss + fueling time`

Then it projects where you would rejoin and turns that into a simple status:

- **GREEN** — comfortable rejoin gap
- **YELLOW** — borderline
- **RED** — unsafe/tight rejoin

### How to set it up

In the pit-loss setup section:

- **Base pit loss (in + out, no service)**
- **Tire-change loss**
- **Fuel rate [L/s] (auto-learned)**
- **Use custom fuel tank max [L]**
- **Custom fuel max [L]**
- **Lock typed inputs (read-only)**
- **Race minimal mode**

### What the live output means

- **Fuel** — current fuel, detected max tank, and amount needed to fill.
- **Fuel time** — expected fill duration using the learned/stored rate.
- **Total pit time loss** — complete stop cost from current settings.
- **Window** — score and traffic-status color.
- **Projected rejoin gaps** — estimated front/rear gaps after stop loss.

### How to use it in practice

1. Launch the app while in an iRacing session.
2. Enter/fine-tune base pit loss for the current track.
3. If your series changes tires, enter tire-change loss.
4. Let the app observe one real pit stop so it can learn fuel rate.
5. During race runs, check rejoin score before deciding whether to pit now or wait.
6. Use minimal mode if you only want the high-value traffic window display.

### Profile saving behavior

PitTime remembers setup values per car and track. On return, it can restore:

- learned fuel rate
- base pit loss
- tire loss
- custom tank max

### Good workflow suggestion

- Practice session: do one or two pit stops so fuel rate gets learned.
- Before race: confirm base pit loss and tire loss.
- During race: use color status plus projected front/rear gaps to decide if the lap is a safe stop window.

## 3) Nishizumi TireWear

**File:** `apps/Nishizumi_TireWear.py`

Run:

```bash
python apps/Nishizumi_TireWear.py
```

### What it does

TireWear is a transparent overlay plus a background learning model. It watches complete stints, learns how quickly tires wear for a specific car + track + configuration, and estimates remaining tread live.

Unlike a static tire display, this app learns from:

- track temperature
- air temperature
- humidity
- driving load / energy per lap
- completed stint history on the same combo

### What you see on the overlay

The HUD displays:

- LF / RF / LR / RR tread percentages
- track and config name
- car identifier
- model confidence
- SDK online/offline state

Tire colors indicate condition:

- green — healthy tread
- yellow — moderate wear
- red — heavy wear / caution zone

### How the learning works

The app needs completed stint data before predictions stabilize:

1. Start a stint.
2. Drive normal laps.
3. Return to pit road.
4. The app compares start/end wear.
5. It stores the sample for current car/track/config.
6. Future runs on same combo become more accurate.

### Why model confidence matters

- **Low confidence** = not enough completed samples yet.
- **Higher confidence** = enough history for more reliable estimates.

Outlier-looking data can be rejected to avoid contaminating the model.

### Built-in overlay controls

At the top-right:

- **ℹ** — opens information dialog (connection state, dataset key, conditions, sample count, confidence, coefficients).
- **⚙** — opens settings (always-on-top, opacity, font size, width, height).
- **✕** — closes overlay.

### Quick start inside the app

1. Launch iRacing and join a session.
2. Start the app.
3. Drag overlay where you want it.
4. Open settings for size/opacity/font changes if needed.
5. Drive clean laps and complete stints.
6. Open info dialog to inspect sample count and confidence.
7. Use reset data only if you intentionally want to clear learning.

### Reset data button

Reset data clears:

- saved tire wear model file
- temporary runtime learning state

Use when:

- you want to restart learning from zero
- saved data became invalid
- you intentionally changed workflow and want a fresh dataset

### Best use case

TireWear is most useful for long runs and endurance prep where you want to know:

- which tire is wearing fastest
- how trustworthy the prediction is
- whether current conditions are harder on tires than previous runs

## 4) Nishizumi Traction

**File:** `apps/Nishizumi_Traction.py`

Run:

```bash
python apps/Nishizumi_Traction.py
```

### What it does

Traction is a coaching overlay built around a live traction circle. It measures longitudinal/lateral acceleration, estimates current grip usage, and identifies parts of the lap where you are below learned reference grip.

In plain language: it helps answer, “Where am I leaving grip on the table?”

### Main screen sections

- **Status row**
  - connection state
  - reminder that `M` toggles compact/detailed mode
  - buttons to load IBT reference, return to live reference, and show/hide quickstart
- **Quickstart panel**
  - short guide for first-time users
- **Coaching settings**
  - laps required before tips begin
  - option for incident-free laps only
- **Traction circle**
  - live dot for combined braking/acceleration/cornering load
- **Telemetry summary**
  - current long/lat/total g
  - estimated grip limit
  - coaching quality metrics
  - generated coaching advice

### How the app learns a reference

By default, Traction builds an adaptive reference from your own laps. It divides the lap into many segments, stores best grip usage in each segment while filtering outliers, and compares recent laps against that reference.

Coaching quality improves after a few representative laps.

### Incident-free laps option

The checkbox **Only count incident-free laps for learning and tips** controls dataset quality:

- Enabled: laps with off-tracks/incidents are ignored for learning.
- Disabled: every completed lap can contribute.

### Tips threshold

**Start tips after this many laps** controls when advice begins.

Example:

- set to `5`
- drive 5 valid/incident-free laps
- app begins comparison against learned baseline

### Understanding coaching output

The summary highlights biggest grip deficits first and can report:

- where in lap issue appears
- grip difference (`Δg`) versus reference
- likely phase (entry / mid-corner / exit)
- trend (improving / stable / declining)
- frequency across recent laps
- what to try next lap

Compact mode shows key advice only; detailed mode expands breakdown.

### Keyboard shortcut

Press **M** to switch compact/detailed coaching summaries.

### Using an IBT file as reference

Traction can use an external `.ibt` telemetry file instead of live-only reference.

Buttons:

- **Load IBT** — choose telemetry file and build reference.
- **Use Live** — discard IBT reference and return to live adaptive reference.

Useful to compare yourself against:

- an earlier personal best
- a stronger benchmark session
- a saved telemetry run from same combo

### Best workflow

1. Launch app after entering car.
2. Keep quickstart visible if first use.
3. Set laps required before coaching starts.
4. Choose whether incident-free laps only should count.
5. Drive representative laps.
6. Read top coaching items and apply one change at a time.
7. Press `M` for full detailed explanation if needed.
8. Optionally load `.ibt` benchmark reference.

## Which app should I use?

Quick chooser:

- I want to save fuel or know whether I can make another lap → **FuelMonitor**
- I want to know whether pitting now will rejoin into traffic → **PitTime**
- I want to understand long-run tire degradation → **TireWear**
- I want driving/coaching feedback about grip usage and lap execution → **Traction**

You can also run more than one overlay at once if your system and workflow are comfortable with it.

## Common troubleshooting

### The app says disconnected or waiting for telemetry

Usually one of these is true:

- iRacing is not running
- you joined a session but have not clicked **Drive**
- telemetry is unavailable to `irsdk`

Fix:

1. Open iRacing.
2. Join a session.
3. Click **Drive**.
4. Relaunch the app if needed.

### A tool is not remembering data

Check whether the data folder exists and is writable:

- `%APPDATA%/NishizumiTools`

If the folder cannot be written, settings/model persistence can fail.

### TireWear predictions look weak at first

Normal behavior: it needs completed stints on the same combo before confidence improves.

### FuelMonitor numbers look strange after pit stop or heavy cautions

Use the **R** reset button in FuelMonitor if the current stint should be recalculated from scratch.

### PitTime fuel rate is wrong

Do one complete normal refueling stop and let the app relearn. Because the rate is saved per car/track, old saved profiles can still influence output until a new stop is observed.

### Traction coaching is not useful yet

Check that you have:

- enough laps for your tips threshold
- enough clean laps if incident-free mode is enabled
- representative pace (not mostly out/in laps)

## Repository files

Current top-level repo files:

- `README.md`
- `requirements.txt`
- `LICENSE`
- `apps/Nishizumi_FuelMonitor.py`
- `apps/Nishizumi_Pittime.py`
- `apps/Nishizumi_TireWear.py`
- `apps/Nishizumi_Traction.py`
- `docs/fuel-monitor.md`

## Final note

These are standalone overlays, not one monolithic app. Pick the one matching the problem you want to solve in that session, and let it learn over time where applicable.

If you are new to the collection, a simple order to try them is:

1. FuelMonitor for immediate race usefulness
2. PitTime for strategy timing
3. Traction for driving improvement
4. TireWear for longer-run learning and endurance prep
