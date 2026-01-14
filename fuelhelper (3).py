#!/usr/bin/env python3
"""Fuel helper app for iRacing pit timing alerts."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Deque, Optional
from collections import deque

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import irsdk

try:
    import pygame

    HAS_PYGAME = True
except ImportError:
    pygame = None
    HAS_PYGAME = False

APP_NAME = "Fuel Helper"
APP_FOLDER = "FuelHelper"
CONFIG_FILENAME = "config.json"
ALERT_COOLDOWN_S = 30.0


def _config_path() -> str:
    base = os.getenv("APPDATA") or os.path.expanduser("~")
    folder = os.path.join(base, APP_FOLDER)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, CONFIG_FILENAME)


def load_config() -> dict:
    path = _config_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def save_config(data: dict) -> None:
    path = _config_path()
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
    except Exception:
        return


def read_var(ir: irsdk.IRSDK, name: str) -> Optional[float]:
    try:
        value = ir[name]
    except Exception:
        return None
    try:
        return float(value)
    except Exception:
        return None


def read_car_name(ir: irsdk.IRSDK) -> Optional[str]:
    try:
        driver_info = ir["DriverInfo"]
    except Exception:
        return None
    if not driver_info:
        return None
    try:
        idx = driver_info["DriverCarIdx"]
        drivers = driver_info.get("Drivers") or []
        if idx is None or idx >= len(drivers):
            return None
        car_name = drivers[idx].get("CarScreenName") or drivers[idx].get("CarPath")
        return str(car_name) if car_name else None
    except Exception:
        return None


def read_track_name(ir: irsdk.IRSDK) -> Optional[str]:
    try:
        weekend_info = ir["WeekendInfo"]
    except Exception:
        return None
    if not weekend_info:
        return None
    try:
        track_name = weekend_info.get("TrackDisplayName") or weekend_info.get("TrackName")
        return str(track_name) if track_name else None
    except Exception:
        return None


@dataclass
class TelemetrySnapshot:
    connected: bool
    autofuel_on: bool
    fuel_level: Optional[float]
    pit_fuel: Optional[float]
    burn_per_lap: Optional[float]
    lap_dist_pct: Optional[float]
    needs_pit: bool
    display_units: Optional[int]
    car_name: Optional[str]
    track_name: Optional[str]


class FuelBurnEstimator:
    def __init__(self, *, max_samples: int = 10, max_burn: float = 50.0) -> None:
        self._samples: Deque[float] = deque(maxlen=max_samples)
        self._last_lap: Optional[int] = None
        self._lap_start_fuel: Optional[float] = None
        self._max_burn = max_burn

    def reset(self) -> None:
        self._samples.clear()
        self._last_lap = None
        self._lap_start_fuel = None

    def update(self, *, lap: Optional[float], fuel_level: Optional[float]) -> None:
        if lap is None or fuel_level is None:
            return

        try:
            lap_i = int(lap)
            fuel_f = float(fuel_level)
        except Exception:
            return

        if self._last_lap is None:
            self._last_lap = lap_i
            self._lap_start_fuel = fuel_f
            return

        if lap_i != self._last_lap:
            if self._lap_start_fuel is not None:
                used = self._lap_start_fuel - fuel_f
                if 0 < used <= self._max_burn:
                    self._samples.append(float(used))
            self._last_lap = lap_i
            self._lap_start_fuel = fuel_f

    def estimate(self, *, fallback: Optional[float]) -> Optional[float]:
        if self._samples:
            return sum(self._samples) / len(self._samples)
        if fallback is not None and fallback > 0:
            return float(fallback)
        return None


class AudioAlert:
    def __init__(self) -> None:
        self._last_play = 0.0
        self._path: Optional[str] = None
        self._volume = 1.0
        self._lock = threading.Lock()

        if HAS_PYGAME:
            try:
                pygame.mixer.init()
            except Exception:
                pass

    def set_path(self, path: Optional[str]) -> None:
        with self._lock:
            self._path = path

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self._volume = max(0.0, min(1.0, float(volume)))

    def play(self, *, force: bool = False) -> None:
        if not HAS_PYGAME:
            return

        now = time.time()
        if not force and now - self._last_play < ALERT_COOLDOWN_S:
            return

        with self._lock:
            path = self._path
            volume = self._volume

        if not path or not os.path.isfile(path):
            return

        try:
            sound = pygame.mixer.Sound(path)
            sound.set_volume(volume)
            sound.play()
            self._last_play = now
        except Exception:
            return


class TelemetryWorker(threading.Thread):
    def __init__(
        self, *, audio: AudioAlert, pit_audio: AudioAlert, pit_trigger_pct, tank_capacity, callback
    ) -> None:
        super().__init__(daemon=True)
        self._audio = audio
        self._pit_audio = pit_audio
        self._pit_trigger_pct = pit_trigger_pct
        self._tank_capacity = tank_capacity
        self._callback = callback
        self._stop_event = threading.Event()
        self._burn = FuelBurnEstimator()
        self._pit_alert_lap: Optional[int] = None
        self._pit_alert_fired = False

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        ir = irsdk.IRSDK()
        last_connected = False

        while not self._stop_event.is_set():
            if not getattr(ir, "is_initialized", False):
                if not ir.startup():
                    if last_connected:
                        self._burn.reset()
                    last_connected = False
                    self._callback(
                        TelemetrySnapshot(
                            connected=False,
                            autofuel_on=False,
                            fuel_level=None,
                            pit_fuel=None,
                            burn_per_lap=None,
                            lap_dist_pct=None,
                            needs_pit=False,
                            display_units=None,
                            car_name=None,
                            track_name=None,
                        )
                    )
                time.sleep(0.5)
                continue

            last_connected = True

            fuel_level = read_var(ir, "FuelLevel")
            fuel_use_per_lap = read_var(ir, "FuelUsePerHour")
            pit_sv_fuel = read_var(ir, "PitSvFuel")
            pit_add_kg = read_var(ir, "dpFuelAddKg")
            autofill_active = read_var(ir, "dpFuelAutoFillActive")
            autofill_enabled = read_var(ir, "dpFuelAutoFillEnabled")
            lap_dist_pct = read_var(ir, "LapDistPct")
            display_units = read_var(ir, "DisplayUnits")
            car_name = read_car_name(ir)
            track_name = read_track_name(ir)

            lap = read_var(ir, "Lap")

            self._burn.update(lap=lap, fuel_level=fuel_level)
            burn_per_lap = self._burn.estimate(fallback=fuel_use_per_lap)

            pit_fuel = pit_sv_fuel if pit_sv_fuel is not None else pit_add_kg
            autofuel_on = bool((autofill_active or 0.0) > 0.5 and (autofill_enabled or 0.0) > 0.5)

            needs_pit = False
            if burn_per_lap is not None and fuel_level is not None:
                needs_pit = fuel_level < burn_per_lap

            snapshot = TelemetrySnapshot(
                connected=True,
                autofuel_on=autofuel_on,
                fuel_level=fuel_level,
                pit_fuel=pit_fuel,
                burn_per_lap=burn_per_lap,
                lap_dist_pct=lap_dist_pct,
                needs_pit=needs_pit,
                display_units=int(display_units) if display_units is not None else None,
                car_name=car_name,
                track_name=track_name,
            )
            self._callback(snapshot)

            if self._should_play_autofuel_alert(snapshot=snapshot):
                self._audio.play()

            self._maybe_play_pit_warning(snapshot, lap)

            time.sleep(0.2)

    def _maybe_play_pit_warning(self, snapshot: TelemetrySnapshot, lap: Optional[float]) -> None:
        if not snapshot.needs_pit or snapshot.lap_dist_pct is None:
            return

        trigger_pct = self._pit_trigger_pct()
        if trigger_pct is None:
            return

        trigger_pct = max(0.0, min(1.0, float(trigger_pct)))

        lap_i = None
        if lap is not None:
            try:
                lap_i = int(lap)
            except Exception:
                lap_i = None

        if lap_i is not None and lap_i != self._pit_alert_lap:
            self._pit_alert_lap = lap_i
            self._pit_alert_fired = False

        if self._pit_alert_fired:
            return

        if snapshot.lap_dist_pct >= trigger_pct:
            self._pit_audio.play()
            self._pit_alert_fired = True

    def _should_play_autofuel_alert(self, *, snapshot: TelemetrySnapshot) -> bool:
        if not snapshot.autofuel_on:
            return False
        if snapshot.pit_fuel is None or snapshot.fuel_level is None:
            return False
        if snapshot.pit_fuel <= 0.01:
            return False
        tank_capacity = self._tank_capacity()
        if tank_capacity is None:
            return False
        return (snapshot.fuel_level + snapshot.pit_fuel) <= tank_capacity


class FuelHelperApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.audio = AudioAlert()
        self.pit_audio = AudioAlert()
        self._snapshot: Optional[TelemetrySnapshot] = None
        self._current_car: Optional[str] = None

        self._config = load_config()
        audio_path = self._config.get("audio_path")
        if audio_path:
            self.audio.set_path(audio_path)
        pit_audio_path = self._config.get("pit_audio_path")
        if pit_audio_path:
            self.pit_audio.set_path(pit_audio_path)
        autofuel_volume = self._load_volume(self._config.get("audio_volume", 1.0))
        pit_volume = self._load_volume(
            self._config.get("pit_audio_volume", self._config.get("audio_volume", 1.0))
        )
        self.audio.set_volume(autofuel_volume)
        self.pit_audio.set_volume(pit_volume)
        pit_alert_pct = self._config.get("pit_alert_pct", 0.65)
        try:
            pit_alert_pct = float(pit_alert_pct)
        except Exception:
            pit_alert_pct = 0.65
        self._pit_alert_pct_value = pit_alert_pct
        self._tank_capacity_by_car = self._load_tank_capacity_map(
            self._config.get("tank_capacity_by_car")
        )
        self._tank_capacity_value = self._load_tank_capacity(self._config.get("tank_capacity"))

        self._build_ui(audio_path, pit_audio_path, pit_alert_pct, autofuel_volume, pit_volume)

        self._worker = TelemetryWorker(
            audio=self.audio,
            pit_audio=self.pit_audio,
            pit_trigger_pct=self._get_pit_alert_pct,
            tank_capacity=self._get_tank_capacity,
            callback=self._on_snapshot,
        )
        self._worker.start()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(
        self,
        audio_path: Optional[str],
        pit_audio_path: Optional[str],
        pit_alert_pct: float,
        autofuel_volume: float,
        pit_volume: float,
    ) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        ttk.Label(main, text="Fuel Helper", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w"
        )

        ttk.Label(main, text="Autofuel alert audio:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        self.audio_label = ttk.Label(
            main,
            text=audio_path or "No file selected",
            wraplength=360,
        )
        self.audio_label.grid(row=2, column=0, columnspan=2, sticky="w")

        ttk.Button(main, text="Choose MP3", command=self._choose_audio).grid(
            row=2, column=2, sticky="e"
        )
        ttk.Button(main, text="Test", command=self._test_audio).grid(row=2, column=3, sticky="e")

        ttk.Label(main, text="Pit warning audio:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.pit_audio_label = ttk.Label(
            main,
            text=pit_audio_path or "No file selected",
            wraplength=360,
        )
        self.pit_audio_label.grid(row=4, column=0, columnspan=2, sticky="w")

        ttk.Button(main, text="Choose Pit MP3", command=self._choose_pit_audio).grid(
            row=4, column=2, sticky="e"
        )
        ttk.Button(main, text="Test", command=self._test_pit_audio).grid(
            row=4, column=3, sticky="e"
        )

        ttk.Label(main, text="Autofuel alert volume:").grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )
        self.autofuel_volume_var = tk.DoubleVar(value=autofuel_volume)
        self.autofuel_volume_value_label = ttk.Label(
            main, text=format_percent(autofuel_volume)
        )
        self.autofuel_volume_value_label.grid(row=5, column=2, sticky="e")
        autofuel_volume_scale = ttk.Scale(
            main,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.autofuel_volume_var,
            command=self._on_autofuel_volume_change,
        )
        autofuel_volume_scale.grid(row=6, column=0, columnspan=3, sticky="ew")

        ttk.Label(main, text="Pit warning volume:").grid(
            row=7, column=0, sticky="w", pady=(8, 0)
        )
        self.pit_volume_var = tk.DoubleVar(value=pit_volume)
        self.pit_volume_value_label = ttk.Label(main, text=format_percent(pit_volume))
        self.pit_volume_value_label.grid(row=7, column=2, sticky="e")
        pit_volume_scale = ttk.Scale(
            main,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.pit_volume_var,
            command=self._on_pit_volume_change,
        )
        pit_volume_scale.grid(row=8, column=0, columnspan=3, sticky="ew")

        ttk.Label(main, text="Fuel tank capacity:").grid(
            row=9, column=0, sticky="w", pady=(8, 0)
        )
        self.tank_capacity_var = tk.StringVar(
            value="" if self._tank_capacity_value is None else f"{self._tank_capacity_value:.2f}"
        )
        self.tank_capacity_entry = ttk.Entry(main, textvariable=self.tank_capacity_var, width=12)
        self.tank_capacity_entry.grid(row=10, column=0, sticky="w")
        self.tank_capacity_entry.bind("<FocusOut>", self._on_tank_capacity_change)
        self.tank_capacity_entry.bind("<Return>", self._on_tank_capacity_change)
        self.tank_capacity_unit_label = ttk.Label(main, text="(units)")
        self.tank_capacity_unit_label.grid(row=10, column=1, sticky="w")
        self.tank_capacity_lock = tk.BooleanVar(
            value=bool(self._config.get("tank_capacity_locked", False))
        )
        self.tank_capacity_lock_check = ttk.Checkbutton(
            main,
            text="Lock",
            variable=self.tank_capacity_lock,
            command=self._on_tank_capacity_lock_toggle,
        )
        self.tank_capacity_lock_check.grid(row=10, column=2, sticky="w", padx=(8, 0))
        self._update_tank_capacity_entry_state()

        ttk.Label(main, text="Pit warning trigger (lap %):").grid(
            row=11, column=0, sticky="w", pady=(8, 0)
        )
        self.pit_alert_pct = tk.DoubleVar(value=pit_alert_pct)
        self.pit_alert_value_label = ttk.Label(main, text=format_percent(pit_alert_pct))
        self.pit_alert_value_label.grid(row=11, column=2, sticky="e")
        pit_scale = ttk.Scale(
            main,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.pit_alert_pct,
            command=self._on_pit_alert_change,
        )
        pit_scale.grid(row=12, column=0, columnspan=3, sticky="ew")

        self.status_label = ttk.Label(main, text="Status: waiting for telemetry...")
        self.status_label.grid(row=13, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.metrics_text = tk.Text(main, height=8, width=48, state="disabled")
        self.metrics_text.grid(row=14, column=0, columnspan=3, pady=(6, 0), sticky="nsew")

        main.columnconfigure(1, weight=1)
        main.rowconfigure(14, weight=1)

    def _choose_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Select alert audio",
            filetypes=[("Audio Files", "*.mp3 *.wav"), ("All Files", "*.*")],
        )
        if not path:
            return

        if not os.path.isfile(path):
            messagebox.showerror(APP_NAME, "Selected file does not exist.")
            return

        self.audio.set_path(path)
        self.audio_label.configure(text=path)
        self._config["audio_path"] = path
        save_config(self._config)

    def _choose_pit_audio(self) -> None:
        path = filedialog.askopenfilename(
            title="Select pit warning audio",
            filetypes=[("Audio Files", "*.mp3 *.wav"), ("All Files", "*.*")],
        )
        if not path:
            return

        if not os.path.isfile(path):
            messagebox.showerror(APP_NAME, "Selected file does not exist.")
            return

        self.pit_audio.set_path(path)
        self.pit_audio_label.configure(text=path)
        self._config["pit_audio_path"] = path
        save_config(self._config)

    def _test_audio(self) -> None:
        self.audio.play(force=True)

    def _test_pit_audio(self) -> None:
        self.pit_audio.play(force=True)

    def _get_pit_alert_pct(self) -> float:
        return float(self._pit_alert_pct_value)

    def _on_pit_alert_change(self, *_args: object) -> None:
        value = float(self.pit_alert_pct.get())
        self._pit_alert_pct_value = value
        self.pit_alert_value_label.configure(text=format_percent(value))
        self._config["pit_alert_pct"] = value
        save_config(self._config)

    def _on_autofuel_volume_change(self, *_args: object) -> None:
        value = self._load_volume(self.autofuel_volume_var.get())
        self.autofuel_volume_value_label.configure(text=format_percent(value))
        self.audio.set_volume(value)
        self._config["audio_volume"] = value
        save_config(self._config)

    def _on_pit_volume_change(self, *_args: object) -> None:
        value = self._load_volume(self.pit_volume_var.get())
        self.pit_volume_value_label.configure(text=format_percent(value))
        self.pit_audio.set_volume(value)
        self._config["pit_audio_volume"] = value
        save_config(self._config)

    def _on_tank_capacity_change(self, *_args: object) -> None:
        value = self.tank_capacity_var.get().strip()
        parsed = self._load_tank_capacity(value)
        self._tank_capacity_value = parsed
        if self._current_car:
            if parsed is None:
                self._tank_capacity_by_car.pop(self._current_car, None)
            else:
                self._tank_capacity_by_car[self._current_car] = parsed
        if parsed is None:
            self._config.pop("tank_capacity", None)
        else:
            self._config["tank_capacity"] = parsed
            self.tank_capacity_var.set(f"{parsed:.2f}")
        if self._tank_capacity_by_car:
            self._config["tank_capacity_by_car"] = dict(self._tank_capacity_by_car)
        else:
            self._config.pop("tank_capacity_by_car", None)
        save_config(self._config)

    def _load_tank_capacity(self, value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value) if value > 0 else None
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = float(text)
        except Exception:
            return None
        return parsed if parsed > 0 else None

    def _load_tank_capacity_map(self, value: Optional[object]) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        cleaned: dict[str, float] = {}
        for key, raw_value in value.items():
            parsed = self._load_tank_capacity(raw_value)
            if parsed is None:
                continue
            cleaned[str(key)] = parsed
        return cleaned

    def _load_volume(self, value: Optional[object]) -> float:
        try:
            parsed = float(value)
        except Exception:
            return 1.0
        return max(0.0, min(1.0, parsed))

    def _get_tank_capacity(self) -> Optional[float]:
        return self._tank_capacity_value

    def _on_snapshot(self, snapshot: TelemetrySnapshot) -> None:
        self._snapshot = snapshot
        self.root.after(0, self._refresh_ui)

    def _refresh_ui(self) -> None:
        snapshot = self._snapshot
        if snapshot is None:
            return

        self._update_current_car(snapshot.car_name)
        self._update_tank_capacity_units(snapshot.display_units)

        if not snapshot.connected:
            self.status_label.configure(text="Status: not connected to iRacing")
        else:
            status = "Autofuel armed" if snapshot.autofuel_on else "Monitoring"
            self.status_label.configure(text=f"Status: {status}")

        lines = [
            f"Car: {snapshot.car_name or '-'}",
            f"Track: {snapshot.track_name or '-'}",
            f"Autofuel on: {snapshot.autofuel_on}",
            f"Fuel level: {format_metric(snapshot.fuel_level)}",
            f"Pit fuel target: {format_metric(snapshot.pit_fuel)}",
            f"Burn per lap: {format_metric(snapshot.burn_per_lap)}",
            f"Lap dist: {format_percent(snapshot.lap_dist_pct)}",
            f"Needs pit: {snapshot.needs_pit}",
            f"Display units: {format_display_units(snapshot.display_units)}",
        ]

        self.metrics_text.configure(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, "\n".join(lines))
        self.metrics_text.configure(state="disabled")

    def _on_close(self) -> None:
        if self._worker:
            self._worker.stop()
        self.root.destroy()

    def _update_tank_capacity_units(self, display_units: Optional[int]) -> None:
        unit_label = units_label(display_units)
        if unit_label:
            self.tank_capacity_unit_label.configure(text=f"({unit_label})")
        else:
            self.tank_capacity_unit_label.configure(text="(units)")

    def _update_current_car(self, car_name: Optional[str]) -> None:
        if not car_name or car_name == self._current_car:
            return
        self._current_car = car_name
        car_value = self._tank_capacity_by_car.get(car_name)
        self._tank_capacity_value = car_value
        if car_value is None:
            self.tank_capacity_var.set("")
        else:
            self.tank_capacity_var.set(f"{car_value:.2f}")

    def _on_tank_capacity_lock_toggle(self) -> None:
        self._config["tank_capacity_locked"] = bool(self.tank_capacity_lock.get())
        save_config(self._config)
        self._update_tank_capacity_entry_state()

    def _update_tank_capacity_entry_state(self) -> None:
        state = "disabled" if self.tank_capacity_lock.get() else "normal"
        self.tank_capacity_entry.configure(state=state)


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def format_percent(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.0f}%"


def units_label(display_units: Optional[int]) -> Optional[str]:
    if display_units == 0:
        return "gal"
    if display_units == 1:
        return "L"
    return None


def format_display_units(display_units: Optional[int]) -> str:
    label = units_label(display_units)
    return label or "-"


def main() -> int:
    if not HAS_PYGAME:
        messagebox.showwarning(
            APP_NAME,
            "pygame is not installed. Audio playback will be disabled.",
        )

    root = tk.Tk()
    app = FuelHelperApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
