#!/usr/bin/env python3
"""Tkinter overlay for smart fuel consumption monitoring in iRacing."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import irsdk


def _get_appdata_dir() -> Path:
    root = Path(os.getenv("APPDATA") or Path.home() / ".config")
    path = root / "NishizumiTools"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class StintState:
    fuel_start: float
    lap_start: int
    lapdist_start: float
    started_at: float


class FuelConsumptionMonitor:
    WINDOW_WIDTH = 360
    WINDOW_HEIGHT_COLLAPSED = 190
    WINDOW_HEIGHT_EXPANDED = 310
    LITER_TO_GALLON = 0.2641720524

    def __init__(self) -> None:
        self.ir = irsdk.IRSDK()

        self.root = tk.Tk()
        self.root.title("Fuel Consumption Monitor")
        self.root.configure(bg="#0f1115")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self._is_dragging = False

        self._drag_offset_x = 0
        self._drag_offset_y = 0

        self._stint: Optional[StintState] = None
        self._last_fuel: Optional[float] = None
        self._last_lap: Optional[int] = None
        self._lap_start_fuel: Optional[float] = None
        self._last_lap_used: Optional[float] = None
        self._pit_hold_until: float = 0.0
        self._lap_consumptions: list[float] = []
        self._locked_target: Optional[float] = None
        self._last_on_pitroad: Optional[bool] = None
        self._pit_overlay_until: float = 0.0
        self._pit_overlay_value: Optional[float] = None
        self._display_units: Optional[int] = None
        self._unit_label = "L"
        self._connected = False

        self.refuel_threshold_l = 0.3
        self.avg_min_progress = 0.05
        self.anomaly_threshold = 0.3

        self.target_var = tk.StringVar(value="2.50")
        self.lock_target_var = tk.BooleanVar(value=False)
        self.show_advanced_var = tk.BooleanVar(value=False)
        self.advanced_toggle_text = tk.StringVar(value="I")

        self._position_path = _get_appdata_dir() / "fuel_consumption_monitor.json"
        self._apply_window_geometry(default_pos=(60, 60))

        self._plus_one_target: Optional[float] = None
        self._minus_one_target: Optional[float] = None
        self._plus_one_laps: Optional[int] = None
        self._minus_one_laps: Optional[int] = None

        self._build_ui()

        self.root.bind("<Escape>", self._ignore_escape)
        self.root.bind("<ButtonPress-1>", self._start_move)
        self.root.bind("<B1-Motion>", self._on_move)
        self.root.bind("<ButtonRelease-1>", self._stop_move)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._update_loop()

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg="#0f1115")
        self.top_frame = top
        top.pack(fill="x", padx=12, pady=(10, 4))

        self.avg_label = tk.Label(
            top,
            text="--.-- L/Lap",
            font=("Segoe UI", 20, "bold"),
            fg="#d8f8d8",
            bg="#0f1115",
        )
        self.avg_label.pack(side="left")

        self.delta_label = tk.Label(
            top,
            text="(+0.00)",
            font=("Segoe UI", 14, "bold"),
            fg="#a0f0a0",
            bg="#0f1115",
            padx=8,
        )
        self.delta_label.pack(side="left")

        bottom = tk.Frame(self.root, bg="#0f1115")
        self.bottom_frame = bottom
        bottom.pack(fill="x", padx=12, pady=(0, 6))

        self.fuel_label = tk.Label(
            bottom,
            text="Fuel: --.-- L",
            font=("Segoe UI", 12),
            fg="#f2f2f2",
            bg="#0f1115",
        )
        self.fuel_label.pack(anchor="w")

        self.laps_label = tk.Label(
            bottom,
            text="Remaining: --.- laps",
            font=("Segoe UI", 12),
            fg="#f2f2f2",
            bg="#0f1115",
        )
        self.laps_label.pack(anchor="w")

        self.lastlap_label = tk.Label(
            bottom,
            text="Last lap: --.- L",
            font=("Segoe UI", 12),
            fg="#f2f2f2",
            bg="#0f1115",
        )
        self.lastlap_label.pack(anchor="w")

        self.stint_label = tk.Label(
            bottom,
            text="Stint: (C) --; (E) --",
            font=("Segoe UI", 12),
            fg="#d4d4d4",
            bg="#0f1115",
        )
        self.stint_label.pack(anchor="w")

        controls = tk.Frame(self.root, bg="#0f1115")
        self.controls_frame = controls
        controls.pack(fill="x", padx=12, pady=(0, 4))

        button_column = tk.Frame(controls, bg="#0f1115")
        button_column.pack(side="left", anchor="n", padx=(0, 8))

        button_row = tk.Frame(button_column, bg="#0f1115")
        button_row.pack(side="top")

        tk.Button(
            button_row,
            text="R",
            command=self._manual_reset,
            font=("Segoe UI", 8),
            bg="#1c2533",
            fg="#e8e8e8",
            relief="flat",
            padx=2,
            pady=0,
            width=1,
        ).pack(side="left", padx=(0, 4))

        self.advanced_toggle_button = tk.Button(
            button_row,
            textvariable=self.advanced_toggle_text,
            command=self._toggle_advanced_info,
            font=("Segoe UI", 9),
            bg="#1c2533",
            fg="#e8e8e8",
            relief="flat",
            padx=6,
            pady=2,
            takefocus=False,
        )
        self.advanced_toggle_button.pack(side="left")

        controls_body = tk.Frame(controls, bg="#0f1115")
        controls_body.pack(side="left", fill="x", expand=True)

        tk.Label(
            controls_body,
            text="Target L/Lap:",
            font=("Segoe UI", 10),
            fg="#c4c4c4",
            bg="#0f1115",
        ).pack(side="left")

        self.target_entry = tk.Entry(
            controls_body,
            textvariable=self.target_var,
            width=6,
            font=("Segoe UI", 10),
            justify="center",
        )
        self.target_entry.pack(side="left", padx=(6, 12))

        tk.Checkbutton(
            controls_body,
            text="",
            variable=self.lock_target_var,
            command=self._toggle_target_lock,
            font=("Segoe UI", 9),
            fg="#c4c4c4",
            bg="#0f1115",
            activebackground="#0f1115",
            activeforeground="#e8e8e8",
            selectcolor="#1c2533",
            relief="flat",
        ).pack(side="left", padx=(0, 10))

        self._apply_window_geometry()

        self.advanced_frame = tk.Frame(self.root, bg="#0f1115")
        self.advanced_info_label = tk.Label(
            self.advanced_frame,
            text="",
            font=("Segoe UI", 11),
            fg="#d4d4d4",
            bg="#0f1115",
            justify="left",
            wraplength=320,
        )
        self.advanced_info_label.pack(anchor="w", pady=(0, 2))
        advanced_buttons = tk.Frame(self.advanced_frame, bg="#0f1115")
        advanced_buttons.pack(fill="x")

        self.plus_one_button = tk.Button(
            advanced_buttons,
            text="+1 lap",
            command=lambda: self._apply_advanced_target("plus"),
            font=("Segoe UI", 11, "bold"),
            bg="#1c2533",
            fg="#e8e8e8",
            relief="flat",
            padx=14,
            pady=4,
        )
        self.plus_one_button.pack(side="left", padx=(0, 12))

        self.minus_one_button = tk.Button(
            advanced_buttons,
            text="-1lap",
            command=lambda: self._apply_advanced_target("minus"),
            font=("Segoe UI", 11, "bold"),
            bg="#1c2533",
            fg="#e8e8e8",
            relief="flat",
            padx=14,
            pady=4,
        )
        self.minus_one_button.pack(side="left")

        self.advanced_stint_label = tk.Label(
            self.advanced_frame,
            text="",
            font=("Segoe UI", 9, "bold"),
            fg="#d4d4d4",
            bg="#0f1115",
            justify="left",
        )
        self.advanced_stint_label.pack(anchor="w", pady=(2, 0))

        self.status_label = tk.Label(
            self.root,
            text="Waiting for iRacing...",
            font=("Segoe UI", 9),
            fg="#8c8c8c",
            bg="#0f1115",
        )
        self.status_label.pack(anchor="w", padx=12, pady=(0, 6))

        self.pit_overlay_frame = tk.Frame(self.root, bg="#0f1115")
        self.pit_overlay_label = tk.Label(
            self.pit_overlay_frame,
            text="Stint avg\n--.-- L/Lap",
            font=("Segoe UI", 20, "bold"),
            fg="#6fe38f",
            bg="#0f1115",
            justify="center",
        )
        self.pit_overlay_label.pack(expand=True, fill="both", padx=12, pady=16)

    def _set_display_units(self, display_units: Optional[int]) -> None:
        if display_units not in (0, 1):
            return
        if self._display_units is None:
            self._display_units = display_units
            self._unit_label = "gal" if display_units == 0 else "L"
            return
        if display_units == self._display_units:
            return
        previous_units = self._display_units
        target_liters = self._parse_target_with_units(previous_units)
        self._display_units = display_units
        self._unit_label = "gal" if display_units == 0 else "L"
        if target_liters is not None:
            target_display = self._from_liters(target_liters)
            self.target_var.set(f"{target_display:.2f}")

    def _from_liters(self, value: float) -> float:
        if self._display_units == 0:
            return value * self.LITER_TO_GALLON
        return value

    def _to_liters(self, value: float) -> float:
        if self._display_units == 0:
            return value / self.LITER_TO_GALLON
        return value

    def _set_standby_state(self, status_text: str) -> None:
        unit = self._unit_label
        self.avg_label.config(text=f"--.-- {unit}/Lap", fg="#c8c8c8")
        self.delta_label.config(text="(--)", fg="#c8c8c8")
        self.fuel_label.config(text=f"Fuel: --.-- {unit}")
        self.laps_label.config(text="Remaining: --.- laps")
        self.lastlap_label.config(text=f"Last lap: --.- {unit}")
        self.stint_label.config(text="Stint: (C) --; (E) --", fg="#d4d4d4")
        self.status_label.config(text=status_text)
        self._hide_pit_overlay()

    def _set_connection_state(self, connected: bool) -> None:
        if connected == self._connected:
            return
        self._connected = connected
        if not connected:
            self._reset_stint()
            self._last_on_pitroad = None
            self._pit_overlay_until = 0.0
            self._pit_overlay_value = None
            self._set_standby_state("Waiting for iRacing connection...")

    def _safe_float(self, key: str) -> Optional[float]:
        try:
            value = self.ir[key]
        except Exception:
            return None
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _safe_int(self, key: str) -> Optional[int]:
        try:
            value = self.ir[key]
        except Exception:
            return None
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _safe_bool(self, key: str) -> Optional[bool]:
        try:
            value = self.ir[key]
        except Exception:
            return None
        if value is None:
            return None
        return bool(value)

    def _manual_reset(self) -> None:
        self._reset_stint()
        self.status_label.config(text="Manual reset")

    def _reset_stint(self) -> None:
        self._stint = None
        self._last_fuel = None
        self._last_lap = None
        self._lap_start_fuel = None
        self._last_lap_used = None
        self._lap_consumptions.clear()

    def _update_stint(
        self,
        fuel_level: float,
        lap: int,
        lapdist: float,
        session_flags: Optional[int],
    ) -> None:
        now = time.time()
        if self._stint is None:
            self._stint = StintState(
                fuel_start=fuel_level,
                lap_start=lap,
                lapdist_start=lapdist,
                started_at=now,
            )
            self._last_fuel = fuel_level
            self._last_lap = lap
            self._lap_start_fuel = fuel_level
            self.status_label.config(text="Stint tracking")
            return

        if self._last_fuel is not None:
            if fuel_level - self._last_fuel >= self.refuel_threshold_l:
                self._pit_hold_until = now + 4.0
                self._stint = StintState(
                    fuel_start=fuel_level,
                    lap_start=lap,
                    lapdist_start=lapdist,
                    started_at=now,
                )
                self._lap_start_fuel = fuel_level
                self._last_lap_used = None
                self.status_label.config(text="Refuel detected")

        if self._last_lap is not None and lap is not None:
            if lap > self._last_lap and self._lap_start_fuel is not None:
                lap_progress = self._compute_progress(lap, lapdist)
                lap_used = max(0.0, self._lap_start_fuel - fuel_level)
                self._lap_start_fuel = fuel_level
                if lap_progress is None or lap_progress < 1:
                    self._last_lap_used = None
                else:
                    self._last_lap_used = lap_used
                    if (
                        self._last_lap_used > 0
                        and not self._is_yellow_flag(session_flags)
                        and not self._is_anomalous_lap(self._last_lap_used)
                    ):
                        self._lap_consumptions.append(self._last_lap_used)

        self._last_fuel = fuel_level
        self._last_lap = lap

    def _compute_progress(self, lap: int, lapdist: float) -> Optional[float]:
        if self._stint is None:
            return None
        progress = (lap - self._stint.lap_start) + (lapdist - self._stint.lapdist_start)
        if progress < 0:
            return None
        return progress

    def _parse_target(self) -> Optional[float]:
        return self._parse_target_with_units(self._display_units)

    def _parse_decimal_input(self, value: str) -> Optional[float]:
        normalized_value = value.strip().replace(",", ".")
        try:
            return float(normalized_value)
        except ValueError:
            return None

    def _parse_target_with_units(self, display_units: Optional[int]) -> Optional[float]:
        value = self._parse_decimal_input(self.target_var.get())
        if value is None:
            return None
        if display_units == 0:
            return value / self.LITER_TO_GALLON
        return value

    def _toggle_target_lock(self) -> None:
        if self.lock_target_var.get():
            target = self._parse_target()
            if target is not None:
                self._locked_target = target
                self.target_entry.configure(state="disabled")
                self.status_label.config(text="Target locked")
            else:
                self.lock_target_var.set(False)
        else:
            self._locked_target = None
            self.target_entry.configure(state="normal")
            self.status_label.config(text="Target unlocked")

    def _toggle_advanced_info(self) -> None:
        show_advanced = not self.show_advanced_var.get()
        self.show_advanced_var.set(show_advanced)
        if show_advanced:
            self.advanced_frame.pack(fill="x", padx=12, pady=(0, 4))
            self._apply_window_geometry()
        else:
            self.advanced_frame.pack_forget()
            self._apply_window_geometry()
        self.advanced_toggle_text.set("I")

    def _apply_advanced_target(self, mode: str) -> None:
        if mode == "plus":
            target = self._plus_one_target
        else:
            target = self._minus_one_target
        if target is None:
            return
        self.target_var.set(f"{self._from_liters(target):.2f}")
        if self.lock_target_var.get():
            self._locked_target = target
        self.status_label.config(text="Target updated from advanced")

    def _is_yellow_flag(self, session_flags: Optional[int]) -> bool:
        if session_flags is None:
            return False
        flags = getattr(irsdk, "Flags", None)
        if flags is None:
            return False
        for attr in ("YELLOW", "CAUTION", "YELLOW_WAVING", "CAUTION_WAVING"):
            mask = getattr(flags, attr, None)
            if mask is not None and session_flags & mask:
                return True
        return False

    def _is_anomalous_lap(self, lap_used: float) -> bool:
        if lap_used <= 0:
            return True
        if len(self._lap_consumptions) < 3:
            return False
        avg = sum(self._lap_consumptions) / len(self._lap_consumptions)
        if avg <= 0:
            return False
        deviation = abs(lap_used - avg) / avg
        return deviation >= self.anomaly_threshold

    def _filtered_average(self, fallback: Optional[float]) -> Optional[float]:
        if fallback is None:
            if self._lap_consumptions:
                return sum(self._lap_consumptions) / len(self._lap_consumptions)
            return None
        if self._lap_consumptions:
            lap_total = sum(self._lap_consumptions)
            return (lap_total + fallback) / (len(self._lap_consumptions) + 1)
        return fallback

    def _stint_average(self, fallback: Optional[float]) -> Optional[float]:
        if self._lap_consumptions:
            return sum(self._lap_consumptions) / len(self._lap_consumptions)
        return fallback

    def _show_pit_overlay(self, avg_value: Optional[float]) -> None:
        if avg_value is None:
            text = f"Stint avg\n--.-- {self._unit_label}/Lap"
        else:
            text = f"Stint avg\n{self._from_liters(avg_value):.2f} {self._unit_label}/Lap"
        self.pit_overlay_label.config(text=text)
        if not self.pit_overlay_frame.winfo_ismapped():
            for frame in (
                self.top_frame,
                self.bottom_frame,
                self.controls_frame,
                self.advanced_frame,
                self.status_label,
            ):
                frame.pack_forget()
            self.pit_overlay_frame.pack(expand=True, fill="both")

    def _hide_pit_overlay(self) -> None:
        if not self.pit_overlay_frame.winfo_ismapped():
            return
        self.pit_overlay_frame.pack_forget()
        self.top_frame.pack(fill="x", padx=12, pady=(10, 4))
        self.bottom_frame.pack(fill="x", padx=12, pady=(0, 6))
        self.controls_frame.pack(fill="x", padx=12, pady=(0, 4))
        if self.show_advanced_var.get():
            self.advanced_frame.pack(fill="x", padx=12, pady=(0, 4))
        self.status_label.pack(anchor="w", padx=12, pady=(0, 6))

    def _update_loop(self) -> None:
        if not getattr(self.ir, "is_initialized", False):
            if not self.ir.startup():
                self._set_connection_state(False)
                self.root.after(500, self._update_loop)
                return
        self._set_connection_state(True)

        self._set_display_units(self._safe_int("DisplayUnits"))
        fuel_level = self._safe_float("FuelLevel")
        lapdist = self._safe_float("LapDistPct")
        lap = self._safe_int("Lap")
        is_on_track = self._safe_bool("IsOnTrack")
        session_flags = self._safe_int("SessionFlags")
        on_pit_road = self._safe_bool("OnPitRoad")

        if fuel_level is None or lap is None or lapdist is None or not is_on_track:
            self._reset_stint()
            self._set_standby_state("Waiting for telemetry...")
            self.root.after(200, self._update_loop)
            return

        self._update_stint(fuel_level, lap, lapdist, session_flags)

        progress = self._compute_progress(lap, lapdist)
        avg_per_lap: Optional[float] = None
        if progress is not None and progress >= self.avg_min_progress:
            assert self._stint is not None
            fuel_used = max(0.0, self._stint.fuel_start - fuel_level)
            if progress > 0:
                avg_per_lap = fuel_used / progress

        avg_per_lap = self._filtered_average(avg_per_lap)
        target = self._locked_target if self.lock_target_var.get() else self._parse_target()

        if avg_per_lap is None:
            self.avg_label.config(
                text=f"--.-- {self._unit_label}/Lap",
                fg="#c8c8c8",
            )
            self.delta_label.config(text="(--)", fg="#c8c8c8")
        else:
            display_avg = self._from_liters(avg_per_lap)
            display_target = self._from_liters(target) if target is not None else None
            delta = display_avg - display_target if display_target is not None else None
            within_target = target is not None and avg_per_lap <= target
            avg_color = "#6fe38f" if within_target else "#ff6b6b"
            delta_color = avg_color
            self.avg_label.config(
                text=f"{display_avg:.2f} {self._unit_label}/Lap",
                fg=avg_color,
            )
            if delta is None:
                self.delta_label.config(text="(--) ", fg="#c8c8c8")
            else:
                self.delta_label.config(text=f"({delta:+.2f})", fg=delta_color)

        self.fuel_label.config(
            text=f"Fuel: {self._from_liters(fuel_level):.2f} {self._unit_label}"
        )

        if avg_per_lap and avg_per_lap > 0:
            remaining = fuel_level / avg_per_lap
            self.laps_label.config(text=f"Remaining: {remaining:.1f} laps")
        else:
            self.laps_label.config(text="Remaining: --.- laps")

        stint_text = "Stint: (C) --; (E) --"
        stint_color = "#d4d4d4"
        planned_laps = None
        base_laps = None

        if avg_per_lap and avg_per_lap > 0:
            remaining = fuel_level / avg_per_lap
            base_laps = max(0, math.floor(remaining))
            if target is not None and target > 0:
                planned_laps = math.floor(fuel_level / target)
                if base_laps >= planned_laps + 1:
                    stint_color = "#b784ff"
                elif base_laps <= planned_laps - 1:
                    stint_color = "#ff6b6b"
                else:
                    stint_color = "#6fe38f"
            if planned_laps is None:
                stint_text = f"Stint: (C) --; (E) {base_laps}"
            else:
                stint_text = f"Stint: (C) {planned_laps}; (E) {base_laps}"

        self.stint_label.config(text=stint_text, fg=stint_color)

        if self.show_advanced_var.get():
            if avg_per_lap and avg_per_lap > 0:
                self._plus_one_laps = base_laps + 1
                self._minus_one_laps = max(base_laps - 1, 1) if base_laps >= 1 else None
                self._plus_one_target = (
                    fuel_level / self._plus_one_laps if self._plus_one_laps else None
                )
                self._minus_one_target = (
                    fuel_level / self._minus_one_laps
                    if self._minus_one_laps
                    else None
                )
                savings_text = ""
                if planned_laps is not None and planned_laps >= 1 and target is not None:
                    gain_lap_target = fuel_level / (planned_laps + 1)
                    save_per_lap = max(0.0, target - gain_lap_target)
                    loss_lap_text = None
                    if planned_laps >= 2:
                        lose_lap_target = fuel_level / (planned_laps - 1)
                        spend_more = max(0.0, lose_lap_target - target)
                        loss_lap_text = (
                            f"Use {self._from_liters(spend_more):.2f} {self._unit_label}/lap more = -1 lap"
                        )
                    savings_lines = [
                        f"Save {self._from_liters(save_per_lap):.2f} {self._unit_label}/lap = +1 lap"
                    ]
                    if loss_lap_text:
                        savings_lines.append(loss_lap_text)
                    savings_text = "\n".join(savings_lines)
                self.advanced_info_label.config(text=savings_text)
                self.advanced_stint_label.config(text="", fg=stint_color)
                if self._plus_one_target is not None:
                    self.plus_one_button.config(
                        text="+1 lap",
                        state="normal",
                    )
                else:
                    self.plus_one_button.config(text="+1 lap", state="disabled")
                if self._minus_one_target is not None:
                    self.minus_one_button.config(
                        text="-1lap",
                        state="normal",
                    )
                else:
                    self.minus_one_button.config(text="-1lap", state="disabled")
            else:
                self._plus_one_target = None
                self._minus_one_target = None
                self._plus_one_laps = None
                self._minus_one_laps = None
                self.advanced_info_label.config(
                    text="Waiting for valid laps to estimate the stint..."
                )
                self.advanced_stint_label.config(text="", fg="#d4d4d4")
                self.plus_one_button.config(text="+1 lap", state="disabled")
                self.minus_one_button.config(text="-1lap", state="disabled")

        if self._last_lap_used is not None:
            self.lastlap_label.config(
                text=f"Last lap: {self._from_liters(self._last_lap_used):.2f} {self._unit_label}"
            )
        else:
            self.lastlap_label.config(text=f"Last lap: --.- {self._unit_label}")

        now = time.time()
        if on_pit_road and not self._last_on_pitroad:
            self._pit_overlay_value = self._stint_average(avg_per_lap)
            self._pit_overlay_until = now + 10.0

        self._last_on_pitroad = on_pit_road

        if now < self._pit_hold_until:
            self.status_label.config(text="PIT")
        elif self._stint is not None:
            self.status_label.config(text="Stint tracking")

        if now < self._pit_overlay_until:
            self._show_pit_overlay(self._pit_overlay_value)
        else:
            self._hide_pit_overlay()

        self.root.after(100, self._update_loop)

    def _start_move(self, event: tk.Event) -> None:
        if self.lock_target_var.get():
            return
        if not self._is_drag_allowed(event.widget):
            return
        self._is_dragging = True
        self._drag_offset_x = event.x_root - self.root.winfo_x()
        self._drag_offset_y = event.y_root - self.root.winfo_y()

    def _on_move(self, event: tk.Event) -> None:
        if not self._is_dragging:
            return
        x = event.x_root - self._drag_offset_x
        y = event.y_root - self._drag_offset_y
        self.root.geometry(f"+{x}+{y}")

    def _stop_move(self, event: tk.Event) -> None:
        if not self._is_dragging:
            return
        self._is_dragging = False
        self._save_window_position()

    def _on_close(self, event: tk.Event | None = None) -> None:
        self._save_window_position()
        try:
            self.ir.shutdown()
        except Exception:
            pass
        self.root.destroy()

    def _ignore_escape(self, event: tk.Event) -> str:
        return "break"

    def _is_drag_allowed(self, widget: tk.Widget) -> bool:
        if isinstance(widget, (tk.Entry, tk.Button, tk.Checkbutton)):
            return False
        return True

    def _apply_window_geometry(self, default_pos: tuple[int, int] | None = None) -> None:
        width = self._get_window_width()
        height = (
            self.WINDOW_HEIGHT_EXPANDED
            if self.show_advanced_var.get()
            else self.WINDOW_HEIGHT_COLLAPSED
        )
        position = self._load_window_position()
        if position is None and default_pos is not None:
            position = default_pos
        if position is None:
            self.root.geometry(f"{width}x{height}")
            return
        x, y = position
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _get_window_width(self) -> int:
        top_frame = getattr(self, "top_frame", None)
        if top_frame is None:
            return self.WINDOW_WIDTH
        self.root.update_idletasks()
        content_width = top_frame.winfo_reqwidth() + 24
        return max(content_width, 1)

    def _load_window_position(self) -> Optional[tuple[int, int]]:
        try:
            data = json.loads(self._position_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        x = data.get("x")
        y = data.get("y")
        if isinstance(x, int) and isinstance(y, int):
            return x, y
        return None

    def _save_window_position(self) -> None:
        try:
            self._position_path.write_text(
                json.dumps({"x": self.root.winfo_x(), "y": self.root.winfo_y()}),
                encoding="utf-8",
            )
        except OSError:
            pass


def main() -> int:
    try:
        FuelConsumptionMonitor()
        tk.mainloop()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
