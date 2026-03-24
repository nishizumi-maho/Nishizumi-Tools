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
    WINDOW_MIN_WIDTH = 308
    WINDOW_MIN_HEIGHT_COLLAPSED = 190
    WINDOW_MIN_HEIGHT_EXPANDED = 310
    LITER_TO_GALLON = 0.2641720524
    BG = "#0f1115"
    CARD_BG = "#0f1115"
    BUTTON_BG = "#1c2533"
    BUTTON_HOVER_BG = "#2b3950"
    CLOSE_BG = "#171b23"
    CLOSE_HOVER_BG = "#a83c4a"

    def __init__(self) -> None:
        self.ir = irsdk.IRSDK()

        self.root = tk.Tk()
        self.root.title("Fuel Consumption Monitor")
        self.root.configure(bg=self.BG)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.92)
        self._is_dragging = False
        self._close_button_visible = False

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

        self._plus_one_target: Optional[float] = None
        self._minus_one_target: Optional[float] = None
        self._plus_one_laps: Optional[int] = None
        self._minus_one_laps: Optional[int] = None
        self._last_lap_time: Optional[float] = None
        self._lap_times: list[float] = []
        self._estimated_tank_capacity_l: Optional[float] = None
        self._strategy_cache_text = "Race: waiting for session estimate..."
        self._strategy_cache_color = "#9fc7ff"
        self._strategy_cache_key: Optional[tuple[object, ...]] = None
        self._strategy_cache_until: float = 0.0

        self._build_ui()
        self._build_close_button_window()
        self._apply_window_geometry(default_pos=(60, 60))

        self.root.bind("<Escape>", self._ignore_escape)
        self.root.bind("<ButtonPress-1>", self._start_move)
        self.root.bind("<B1-Motion>", self._on_move)
        self.root.bind("<ButtonRelease-1>", self._stop_move)
        self.root.bind("<Enter>", self._on_root_enter)
        self.root.bind("<Leave>", self._on_root_leave)
        self.root.bind("<Configure>", self._on_root_configure)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.root.after(80, self._sync_close_button_position)
        self._update_loop()

    def _build_ui(self) -> None:
        self.card_frame = tk.Frame(self.root, bg=self.CARD_BG, highlightthickness=0, bd=0)
        self.card_frame.pack(fill="both", expand=True)

        self.content_frame = tk.Frame(self.card_frame, bg=self.CARD_BG)
        self.content_frame.pack(fill="both", expand=True)

        top = tk.Frame(self.content_frame, bg=self.CARD_BG)
        self.top_frame = top
        top.pack(fill="x", padx=12, pady=(10, 4))

        self.avg_label = tk.Label(
            top,
            text="--.-- L/Lap",
            font=("Segoe UI", 20, "bold"),
            fg="#d8f8d8",
            bg=self.CARD_BG,
        )
        self.avg_label.pack(side="left")

        self.delta_label = tk.Label(
            top,
            text="(+0.00)",
            font=("Segoe UI", 14, "bold"),
            fg="#a0f0a0",
            bg=self.CARD_BG,
            padx=8,
        )
        self.delta_label.pack(side="left")

        bottom = tk.Frame(self.content_frame, bg=self.CARD_BG)
        self.bottom_frame = bottom
        bottom.pack(fill="x", padx=12, pady=(0, 6))

        self.fuel_label = tk.Label(
            bottom,
            text="Fuel: --.-- L",
            font=("Segoe UI", 12),
            fg="#f2f2f2",
            bg=self.CARD_BG,
        )
        self.fuel_label.pack(anchor="w")

        self.laps_label = tk.Label(
            bottom,
            text="Remaining: --.- laps",
            font=("Segoe UI", 12),
            fg="#f2f2f2",
            bg=self.CARD_BG,
        )
        self.laps_label.pack(anchor="w")

        self.lastlap_label = tk.Label(
            bottom,
            text="Last lap: --.- L",
            font=("Segoe UI", 12),
            fg="#f2f2f2",
            bg=self.CARD_BG,
        )
        self.lastlap_label.pack(anchor="w")

        self.stint_label = tk.Label(
            bottom,
            text="Stint: (C) --; (E) --",
            font=("Segoe UI", 12),
            fg="#d4d4d4",
            bg=self.CARD_BG,
        )
        self.stint_label.pack(anchor="w")

        self.strategy_label = tk.Label(
            bottom,
            text="Race: waiting for session estimate...",
            font=("Segoe UI", 10, "bold"),
            fg="#9fc7ff",
            bg=self.CARD_BG,
            justify="left",
            anchor="w",
            wraplength=self.WINDOW_MIN_WIDTH - 32,
        )
        self.strategy_label.pack(anchor="w", fill="x", pady=(2, 0))

        controls = tk.Frame(self.content_frame, bg=self.CARD_BG)
        self.controls_frame = controls
        controls.pack(fill="x", padx=12, pady=(0, 4))

        button_column = tk.Frame(controls, bg=self.CARD_BG)
        button_column.pack(side="left", anchor="n", padx=(0, 8))

        button_row = tk.Frame(button_column, bg=self.CARD_BG)
        button_row.pack(side="top")

        self.reset_button = tk.Button(
            button_row,
            text="R",
            command=self._manual_reset,
            font=("Segoe UI", 8),
            bg=self.BUTTON_BG,
            fg="#e8e8e8",
            activebackground=self.BUTTON_HOVER_BG,
            activeforeground="#ffffff",
            relief="flat",
            padx=2,
            pady=0,
            width=1,
            takefocus=False,
            bd=0,
            highlightthickness=0,
        )
        self.reset_button.pack(side="left", padx=(0, 4))
        self._bind_hover(self.reset_button, self.BUTTON_BG, self.BUTTON_HOVER_BG)

        self.advanced_toggle_button = tk.Button(
            button_row,
            textvariable=self.advanced_toggle_text,
            command=self._toggle_advanced_info,
            font=("Segoe UI", 9, "bold"),
            bg=self.BUTTON_BG,
            fg="#e8e8e8",
            activebackground=self.BUTTON_HOVER_BG,
            activeforeground="#ffffff",
            relief="flat",
            padx=6,
            pady=2,
            takefocus=False,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.advanced_toggle_button.pack(side="left")
        self._bind_hover(self.advanced_toggle_button, self.BUTTON_BG, self.BUTTON_HOVER_BG)

        controls_body = tk.Frame(controls, bg=self.CARD_BG)
        controls_body.pack(side="left", fill="x", expand=True)

        tk.Label(
            controls_body,
            text="Target:",
            font=("Segoe UI", 10),
            fg="#c4c4c4",
            bg=self.CARD_BG,
        ).pack(side="left")

        self.target_entry = tk.Entry(
            controls_body,
            textvariable=self.target_var,
            width=5,
            font=("Segoe UI", 10),
            justify="center",
            bd=0,
            highlightthickness=1,
            highlightbackground="#293242",
            highlightcolor="#47607a",
            relief="flat",
            bg="#151a22",
            fg="#f3f3f3",
            insertbackground="#f3f3f3",
        )
        self.target_entry.pack(side="left", padx=(6, 12))

        tk.Checkbutton(
            controls_body,
            text="",
            variable=self.lock_target_var,
            command=self._toggle_target_lock,
            font=("Segoe UI", 9),
            fg="#c4c4c4",
            bg=self.CARD_BG,
            activebackground=self.CARD_BG,
            activeforeground="#e8e8e8",
            selectcolor=self.BUTTON_BG,
            relief="flat",
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        ).pack(side="left", padx=(0, 10))

        self.advanced_frame = tk.Frame(self.content_frame, bg=self.CARD_BG)
        self.advanced_info_label = tk.Label(
            self.advanced_frame,
            text="",
            font=("Segoe UI", 11),
            fg="#d4d4d4",
            bg=self.CARD_BG,
            justify="left",
            wraplength=self.WINDOW_MIN_WIDTH - 32,
        )
        self.advanced_info_label.pack(anchor="w", pady=(0, 2))

        advanced_buttons = tk.Frame(self.advanced_frame, bg=self.CARD_BG)
        advanced_buttons.pack(fill="x")

        self.plus_one_button = tk.Button(
            advanced_buttons,
            text="+1 lap",
            command=lambda: self._apply_advanced_target("plus"),
            font=("Segoe UI", 11, "bold"),
            bg=self.BUTTON_BG,
            fg="#e8e8e8",
            activebackground=self.BUTTON_HOVER_BG,
            activeforeground="#ffffff",
            relief="flat",
            padx=14,
            pady=4,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.plus_one_button.pack(side="left", padx=(0, 12))
        self._bind_hover(self.plus_one_button, self.BUTTON_BG, self.BUTTON_HOVER_BG)

        self.minus_one_button = tk.Button(
            advanced_buttons,
            text="-1lap",
            command=lambda: self._apply_advanced_target("minus"),
            font=("Segoe UI", 11, "bold"),
            bg=self.BUTTON_BG,
            fg="#e8e8e8",
            activebackground=self.BUTTON_HOVER_BG,
            activeforeground="#ffffff",
            relief="flat",
            padx=14,
            pady=4,
            bd=0,
            highlightthickness=0,
            cursor="hand2",
        )
        self.minus_one_button.pack(side="left")
        self._bind_hover(self.minus_one_button, self.BUTTON_BG, self.BUTTON_HOVER_BG)

        self.advanced_stint_label = tk.Label(
            self.advanced_frame,
            text="",
            font=("Segoe UI", 9, "bold"),
            fg="#d4d4d4",
            bg=self.CARD_BG,
            justify="left",
        )
        self.advanced_stint_label.pack(anchor="w", pady=(2, 0))

        self.status_label = tk.Label(
            self.content_frame,
            text="Waiting for iRacing...",
            font=("Segoe UI", 9),
            fg="#8c8c8c",
            bg=self.CARD_BG,
        )
        self.status_label.pack(anchor="w", padx=12, pady=(0, 6))

        self.pit_overlay_frame = tk.Frame(self.card_frame, bg=self.CARD_BG)
        self.pit_overlay_label = tk.Label(
            self.pit_overlay_frame,
            text="Stint avg\n--.-- L/Lap",
            font=("Segoe UI", 20, "bold"),
            fg="#6fe38f",
            bg=self.CARD_BG,
            justify="center",
        )
        self.pit_overlay_label.pack(expand=True, fill="both", padx=12, pady=16)

        self._apply_window_geometry()

    def _build_close_button_window(self) -> None:
        self.close_window = tk.Toplevel(self.root)
        self.close_window.overrideredirect(True)
        self.close_window.attributes("-topmost", True)
        self.close_window.configure(bg=self.BG)
        self.close_window.withdraw()

        self.close_button = tk.Label(
            self.close_window,
            text="✕",
            font=("Segoe UI", 11, "bold"),
            bg=self.CLOSE_BG,
            fg="#ffffff",
            width=2,
            padx=0,
            pady=2,
            cursor="hand2",
        )
        self.close_button.pack(fill="both", expand=True)
        self._bind_hover(self.close_button, self.CLOSE_BG, self.CLOSE_HOVER_BG)
        self.close_button.bind("<Button-1>", self._on_close)
        self.close_window.bind("<Enter>", self._on_close_window_enter)
        self.close_window.bind("<Leave>", self._on_close_window_leave)

    def _bind_hover(self, widget: tk.Widget, base_bg: str, hover_bg: str) -> None:
        def _enter(_: tk.Event) -> None:
            try:
                widget.configure(bg=hover_bg)
            except tk.TclError:
                pass

        def _leave(_: tk.Event) -> None:
            try:
                widget.configure(bg=base_bg)
            except tk.TclError:
                pass

        widget.bind("<Enter>", _enter, add="+")
        widget.bind("<Leave>", _leave, add="+")

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
        self.strategy_label.config(text="Race: waiting for session estimate...", fg="#9fc7ff")
        self.status_label.config(text=status_text)
        self._hide_pit_overlay()

    def _set_connection_state(self, connected: bool) -> None:
        if connected == self._connected:
            return
        self._connected = connected
        if not connected:
            self._reset_stint()
            self._estimated_tank_capacity_l = None
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
        self._last_lap_time = None
        self._lap_consumptions.clear()
        self._lap_times.clear()

    def _update_stint(
        self,
        fuel_level: float,
        lap: int,
        lapdist: float,
        session_flags: Optional[int],
        lap_last_time: Optional[float],
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

        if self._last_fuel is not None and fuel_level - self._last_fuel >= self.refuel_threshold_l:
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

        if self._last_lap is not None and lap > self._last_lap and self._lap_start_fuel is not None:
            lap_progress = self._compute_progress(lap, lapdist)
            lap_used = max(0.0, self._lap_start_fuel - fuel_level)
            self._lap_start_fuel = fuel_level
            if lap_progress is None or lap_progress < 1:
                self._last_lap_used = None
                self._last_lap_time = None
            else:
                self._last_lap_used = lap_used
                valid_green_lap = (
                    self._last_lap_used > 0
                    and not self._is_yellow_flag(session_flags)
                    and not self._is_anomalous_lap(self._last_lap_used)
                )
                if valid_green_lap:
                    self._lap_consumptions.append(self._last_lap_used)
                if lap_last_time is not None and lap_last_time > 0:
                    self._last_lap_time = lap_last_time
                    if valid_green_lap and not self._is_anomalous_lap_time(lap_last_time):
                        self._lap_times.append(lap_last_time)
                else:
                    self._last_lap_time = None

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
            self.advanced_frame.pack(fill="x", padx=12, pady=(0, 4), before=self.status_label)
        else:
            self.advanced_frame.pack_forget()
        self.advanced_toggle_text.set("I")
        self._refresh_layout()

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

    def _is_anomalous_lap_time(self, lap_time: float) -> bool:
        if lap_time <= 0:
            return True
        if len(self._lap_times) < 3:
            return False
        avg = sum(self._lap_times) / len(self._lap_times)
        if avg <= 0:
            return False
        deviation = abs(lap_time - avg) / avg
        return deviation >= 0.35

    def _estimated_lap_time(self, lap_last_time: Optional[float], lap_best_time: Optional[float]) -> Optional[float]:
        if self._lap_times:
            return sum(self._lap_times) / len(self._lap_times)
        if self._last_lap_time is not None and self._last_lap_time > 0:
            return self._last_lap_time
        if lap_last_time is not None and lap_last_time > 0:
            return lap_last_time
        if lap_best_time is not None and lap_best_time > 0:
            return lap_best_time
        return None

    def _update_tank_capacity_estimate(self, fuel_level: float, fuel_level_pct: Optional[float]) -> None:
        if fuel_level_pct is None or fuel_level_pct <= 0.02 or fuel_level_pct > 1.02:
            return
        estimated_capacity = fuel_level / fuel_level_pct
        if estimated_capacity <= 0:
            return
        if self._estimated_tank_capacity_l is None:
            self._estimated_tank_capacity_l = estimated_capacity
            return
        self._estimated_tank_capacity_l = max(self._estimated_tank_capacity_l, estimated_capacity)

    def _estimate_session_laps_remaining(
        self,
        session_time_remain: Optional[float],
        session_laps_remain_ex: Optional[int],
        lap_time_estimate: Optional[float],
    ) -> Optional[float]:
        if session_time_remain is not None and session_time_remain > 0 and lap_time_estimate and lap_time_estimate > 0:
            return max(0.0, session_time_remain / lap_time_estimate)
        if session_laps_remain_ex is not None and session_laps_remain_ex >= 0:
            return float(session_laps_remain_ex)
        return None

    def _calculate_required_stops(self, laps_to_go: float, current_laps: float, full_tank_laps: float) -> int:
        if laps_to_go <= current_laps:
            return 0
        if full_tank_laps <= 0:
            return 999
        return max(0, math.ceil((laps_to_go - current_laps) / full_tank_laps))

    def _scenario_average_map(self, avg_per_lap: Optional[float]) -> dict[str, float]:
        if avg_per_lap is None or avg_per_lap <= 0:
            return {}
        samples = sorted(value for value in self._lap_consumptions if value > 0)
        if len(samples) >= 4:
            band_size = max(1, math.ceil(len(samples) * 0.35))
            save_avg = sum(samples[:band_size]) / band_size
            push_avg = sum(samples[-band_size:]) / band_size
        else:
            save_avg = avg_per_lap * 0.97
            push_avg = avg_per_lap * 1.03

        save_avg = min(avg_per_lap * 0.995, max(avg_per_lap * 0.88, save_avg))
        push_avg = max(avg_per_lap * 1.005, min(avg_per_lap * 1.12, push_avg))
        return {
            "save": save_avg,
            "current": avg_per_lap,
            "push": push_avg,
        }

    def _build_scenario_plan(
        self,
        avg_per_lap: float,
        fuel_level: float,
        laps_to_go: float,
    ) -> dict[str, Optional[float]]:
        current_laps = fuel_level / avg_per_lap if avg_per_lap > 0 else 0.0
        plan: dict[str, Optional[float]] = {
            "avg": avg_per_lap,
            "current_laps": current_laps,
            "stops": None,
            "margin_laps": current_laps - laps_to_go,
            "max_avg_same_stop": None,
            "save_to_cut_one": None,
            "push_room_same_stop": None,
        }
        if laps_to_go <= 0:
            plan["stops"] = 0
            return plan
        if self._estimated_tank_capacity_l is None or self._estimated_tank_capacity_l <= 0:
            if current_laps >= laps_to_go:
                plan["stops"] = 0
                max_avg_same_stop = fuel_level / laps_to_go
                plan["max_avg_same_stop"] = max_avg_same_stop
                plan["push_room_same_stop"] = max(0.0, max_avg_same_stop - avg_per_lap)
            return plan

        full_tank_laps = self._estimated_tank_capacity_l / avg_per_lap
        stops = self._calculate_required_stops(laps_to_go, current_laps, full_tank_laps)
        total_available_laps = current_laps + stops * full_tank_laps
        max_avg_same_stop = (fuel_level + stops * self._estimated_tank_capacity_l) / laps_to_go
        plan["stops"] = float(stops)
        plan["margin_laps"] = total_available_laps - laps_to_go
        plan["max_avg_same_stop"] = max_avg_same_stop
        plan["push_room_same_stop"] = max(0.0, max_avg_same_stop - avg_per_lap)

        if stops > 0:
            max_avg_one_less = (fuel_level + (stops - 1) * self._estimated_tank_capacity_l) / laps_to_go
            plan["save_to_cut_one"] = max(0.0, avg_per_lap - max_avg_one_less)
        else:
            plan["save_to_cut_one"] = 0.0
        return plan

    def _format_stops(self, value: Optional[float]) -> str:
        if value is None:
            return "--"
        stops = max(0, int(round(value)))
        return f"{stops} stop" if stops == 1 else f"{stops} stops"

    def _build_race_smart_strategy(
        self,
        avg_per_lap: Optional[float],
        fuel_level: float,
        laps_to_go: Optional[float],
    ) -> tuple[str, str, list[str]]:
        if avg_per_lap is None or avg_per_lap <= 0 or laps_to_go is None or laps_to_go <= 0:
            return "Race: waiting for session estimate...", "#9fc7ff", []

        scenarios = self._scenario_average_map(avg_per_lap)
        if not scenarios:
            return "Race: waiting for session estimate...", "#9fc7ff", []

        plans = {
            name: self._build_scenario_plan(value, fuel_level, laps_to_go)
            for name, value in scenarios.items()
        }
        current_plan = plans["current"]
        save_plan = plans["save"]
        push_plan = plans["push"]

        lines = [
            (
                "P/C/S avg: "
                f"{self._from_liters(scenarios['push']):.2f} / "
                f"{self._from_liters(scenarios['current']):.2f} / "
                f"{self._from_liters(scenarios['save']):.2f} {self._unit_label}/lap"
            )
        ]

        if current_plan["stops"] is not None:
            lines.append(
                "Stops est: "
                f"push {self._format_stops(push_plan['stops'])} | "
                f"cur {self._format_stops(current_plan['stops'])} | "
                f"save {self._format_stops(save_plan['stops'])}"
            )

        if current_plan["stops"] == 0:
            push_room = current_plan["push_room_same_stop"] or 0.0
            margin_laps = current_plan["margin_laps"] or 0.0
            if push_room > 0.02:
                text = (
                    f"Race: ~{laps_to_go:.1f} laps left | no-stop on current, "
                    f"push +{self._from_liters(push_room):.2f} {self._unit_label}/lap safely"
                )
            elif margin_laps > 0.35:
                text = f"Race: ~{laps_to_go:.1f} laps left | no-stop is comfortable"
            else:
                text = f"Race: ~{laps_to_go:.1f} laps left | no-stop is on"
            color = "#6fe38f"
        elif save_plan["stops"] is not None and current_plan["stops"] is not None and save_plan["stops"] < current_plan["stops"]:
            save_to_cut = current_plan["save_to_cut_one"] or 0.0
            if save_to_cut <= 0.01:
                text = f"Race: ~{laps_to_go:.1f} laps left | 1 stop less looks possible"
                color = "#6fe38f"
            else:
                text = (
                    f"Race: ~{laps_to_go:.1f} laps left | save "
                    f"{self._from_liters(save_to_cut):.2f} {self._unit_label}/lap to cut 1 stop"
                )
                color = "#ffb86c"
            lines.append(
                f"Cut 1 stop: save {self._from_liters(max(0.0, save_to_cut)):.2f} {self._unit_label}/lap"
            )
        elif push_plan["stops"] is not None and current_plan["stops"] is not None and push_plan["stops"] > current_plan["stops"]:
            push_room = current_plan["push_room_same_stop"] or 0.0
            text = (
                f"Race: ~{laps_to_go:.1f} laps left | push risks +1 stop, "
                f"room is only {self._from_liters(push_room):.2f} {self._unit_label}/lap"
            )
            color = "#ff6b6b"
            lines.append(
                f"Safe push room: +{self._from_liters(push_room):.2f} {self._unit_label}/lap"
            )
        elif current_plan["stops"] is not None:
            margin_laps = current_plan["margin_laps"] or 0.0
            if margin_laps >= 0.6:
                text = (
                    f"Race: ~{laps_to_go:.1f} laps left | current pace is safe for "
                    f"{self._format_stops(current_plan['stops'])}"
                )
                color = "#6fe38f"
            else:
                text = (
                    f"Race: ~{laps_to_go:.1f} laps left | current pace points to "
                    f"{self._format_stops(current_plan['stops'])}"
                )
                color = "#ffb86c"
        else:
            required_avg = fuel_level / laps_to_go
            save_needed = max(0.0, avg_per_lap - required_avg)
            text = (
                f"Race: ~{laps_to_go:.1f} laps left | save "
                f"{self._from_liters(save_needed):.2f} {self._unit_label}/lap for no-stop"
            )
            color = "#ffb86c"
            lines.append(
                f"No-stop target: {self._from_liters(required_avg):.2f} {self._unit_label}/lap"
            )

        cache_key = (
            round(laps_to_go, 1),
            round(avg_per_lap, 3),
            round(scenarios["save"], 3),
            round(scenarios["push"], 3),
            round(current_plan["stops"] or -1, 0),
            round(save_plan["stops"] or -1, 0),
            round(push_plan["stops"] or -1, 0),
        )
        now = time.time()
        if now < self._strategy_cache_until and self._strategy_cache_key is not None:
            old_laps = self._strategy_cache_key[0]
            old_avg = self._strategy_cache_key[1]
            old_cur_stops = self._strategy_cache_key[4]
            if abs(cache_key[0] - old_laps) < 0.4 and abs(cache_key[1] - old_avg) < 0.04 and cache_key[4] == old_cur_stops:
                return self._strategy_cache_text, self._strategy_cache_color, lines

        self._strategy_cache_text = text
        self._strategy_cache_color = color
        self._strategy_cache_key = cache_key
        self._strategy_cache_until = now + 3.0
        return text, color, lines

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
        self.pit_overlay_frame.place(x=0, y=0, relwidth=1, relheight=1)
        self.pit_overlay_frame.lift()

    def _hide_pit_overlay(self) -> None:
        self.pit_overlay_frame.place_forget()

    def _update_loop(self) -> None:
        if not getattr(self.ir, "is_initialized", False):
            if not self.ir.startup():
                self._set_connection_state(False)
                self.root.after(500, self._update_loop)
                return
        self._set_connection_state(True)

        self._set_display_units(self._safe_int("DisplayUnits"))
        fuel_level = self._safe_float("FuelLevel")
        fuel_level_pct = self._safe_float("FuelLevelPct")
        lapdist = self._safe_float("LapDistPct")
        lap = self._safe_int("Lap")
        is_on_track = self._safe_bool("IsOnTrack")
        session_flags = self._safe_int("SessionFlags")
        on_pit_road = self._safe_bool("OnPitRoad")
        session_time_remain = self._safe_float("SessionTimeRemain")
        session_laps_remain_ex = self._safe_int("SessionLapsRemainEx")
        lap_last_time = self._safe_float("LapLastLapTime")
        lap_best_time = self._safe_float("LapBestLapTime")

        if fuel_level is None or lap is None or lapdist is None or not is_on_track:
            self._reset_stint()
            self._set_standby_state("Waiting for telemetry...")
            self.root.after(200, self._update_loop)
            return

        self._update_tank_capacity_estimate(fuel_level, fuel_level_pct)
        self._update_stint(fuel_level, lap, lapdist, session_flags, lap_last_time)

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
            self.avg_label.config(text=f"--.-- {self._unit_label}/Lap", fg="#c8c8c8")
            self.delta_label.config(text="(--)", fg="#c8c8c8")
        else:
            display_avg = self._from_liters(avg_per_lap)
            display_target = self._from_liters(target) if target is not None else None
            delta = display_avg - display_target if display_target is not None else None
            within_target = target is not None and avg_per_lap <= target
            avg_color = "#6fe38f" if within_target else "#ff6b6b"
            self.avg_label.config(text=f"{display_avg:.2f} {self._unit_label}/Lap", fg=avg_color)
            if delta is None:
                self.delta_label.config(text="(--)", fg="#c8c8c8")
            else:
                self.delta_label.config(text=f"({delta:+.2f})", fg=avg_color)

        self.fuel_label.config(text=f"Fuel: {self._from_liters(fuel_level):.2f} {self._unit_label}")

        remaining_laps = None
        if avg_per_lap and avg_per_lap > 0:
            remaining_laps = fuel_level / avg_per_lap
            self.laps_label.config(text=f"Remaining: {remaining_laps:.1f} laps")
        else:
            self.laps_label.config(text="Remaining: --.- laps")

        stint_text = "Stint: (C) --; (E) --"
        stint_color = "#d4d4d4"
        planned_laps = None
        base_laps = None

        lap_time_estimate = self._estimated_lap_time(lap_last_time, lap_best_time)
        session_laps_estimate = self._estimate_session_laps_remaining(
            session_time_remain,
            session_laps_remain_ex,
            lap_time_estimate,
        )
        strategy_text, strategy_color, strategy_details = self._build_race_smart_strategy(
            avg_per_lap,
            fuel_level,
            session_laps_estimate,
        )
        self.strategy_label.config(text=strategy_text, fg=strategy_color)

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
                self._plus_one_target = fuel_level / self._plus_one_laps if self._plus_one_laps else None
                self._minus_one_target = fuel_level / self._minus_one_laps if self._minus_one_laps else None
                savings_lines: list[str] = []
                if session_laps_estimate is not None:
                    savings_lines.append(f"Race est: {session_laps_estimate:.1f} laps left")
                savings_lines.extend(strategy_details)
                if planned_laps is not None and planned_laps >= 1 and target is not None:
                    gain_lap_target = fuel_level / (planned_laps + 1)
                    save_per_lap = max(0.0, target - gain_lap_target)
                    if save_per_lap > 0.01:
                        savings_lines.append(
                            f"Save {self._from_liters(save_per_lap):.2f} {self._unit_label}/lap = +1 lap"
                        )
                    if planned_laps >= 2:
                        lose_lap_target = fuel_level / (planned_laps - 1)
                        spend_more = max(0.0, lose_lap_target - target)
                        savings_lines.append(
                            f"Use {self._from_liters(spend_more):.2f} {self._unit_label}/lap more = -1 lap"
                        )
                self.advanced_info_label.config(text="\n".join(savings_lines))
                self.advanced_stint_label.config(text="", fg=stint_color)
                self.plus_one_button.config(text="+1 lap", state="normal")
                self.minus_one_button.config(
                    text="-1lap",
                    state="normal" if self._minus_one_target is not None else "disabled",
                )
            else:
                self._plus_one_target = None
                self._minus_one_target = None
                self._plus_one_laps = None
                self._minus_one_laps = None
                self.advanced_info_label.config(text="Waiting for valid laps to estimate the stint...")
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

        self._refresh_layout()
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
        self._sync_close_button_position()

    def _stop_move(self, event: tk.Event) -> None:
        if not self._is_dragging:
            return
        self._is_dragging = False
        self._save_window_position()
        self._sync_close_button_position()

    def _on_close(self, event: tk.Event | None = None) -> None:
        self._save_window_position()
        try:
            self.close_window.destroy()
        except Exception:
            pass
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

    def _on_root_enter(self, event: tk.Event) -> None:
        if event.widget is self.root:
            self._show_close_button()

    def _on_root_leave(self, event: tk.Event) -> None:
        if event.widget is self.root:
            self.root.after(50, self._hide_close_button_if_needed)

    def _on_close_window_enter(self, event: tk.Event) -> None:
        self._show_close_button()

    def _on_close_window_leave(self, event: tk.Event) -> None:
        self.root.after(50, self._hide_close_button_if_needed)

    def _on_root_configure(self, event: tk.Event) -> None:
        self._sync_close_button_position()

    def _show_close_button(self) -> None:
        self._close_button_visible = True
        self._sync_close_button_position()
        try:
            self.close_window.deiconify()
            self.close_window.lift()
        except tk.TclError:
            pass

    def _hide_close_button_if_needed(self) -> None:
        try:
            root_x1 = self.root.winfo_pointerx()
            root_y1 = self.root.winfo_pointery()
        except tk.TclError:
            return
        root_inside = self._point_in_window(self.root, root_x1, root_y1)
        close_inside = self._point_in_window(self.close_window, root_x1, root_y1)
        if root_inside or close_inside:
            return
        self._close_button_visible = False
        try:
            self.close_window.withdraw()
        except tk.TclError:
            pass

    def _point_in_window(self, window: tk.Misc, pointer_x: int, pointer_y: int) -> bool:
        try:
            x = window.winfo_rootx()
            y = window.winfo_rooty()
            w = window.winfo_width()
            h = window.winfo_height()
        except tk.TclError:
            return False
        return x <= pointer_x <= x + w and y <= pointer_y <= y + h

    def _sync_close_button_position(self) -> None:
        if not hasattr(self, "close_window"):
            return
        self.root.update_idletasks()
        width = max(28, self.root.winfo_width())
        x = self.root.winfo_x() + width - 18
        y = self.root.winfo_y() - 12
        try:
            self.close_window.geometry(f"28x28+{x}+{y}")
            self.strategy_label.configure(wraplength=max(self.WINDOW_MIN_WIDTH - 32, width - 24))
            self.advanced_info_label.configure(wraplength=max(self.WINDOW_MIN_WIDTH - 32, width - 24))
            if self._close_button_visible:
                self.close_window.deiconify()
                self.close_window.lift()
        except tk.TclError:
            return

    def _apply_window_geometry(self, default_pos: tuple[int, int] | None = None) -> None:
        self.root.update_idletasks()
        width = self._get_window_width()
        height = self._get_window_height()
        position = self._load_window_position()
        if position is None and default_pos is not None:
            position = default_pos
        if position is None:
            self.root.geometry(f"{width}x{height}")
        else:
            x, y = position
            self.root.geometry(f"{width}x{height}+{x}+{y}")
        self.root.minsize(self.WINDOW_MIN_WIDTH, self.WINDOW_MIN_HEIGHT_COLLAPSED)
        self._sync_close_button_position()

    def _refresh_layout(self) -> None:
        self.root.update_idletasks()
        x = self.root.winfo_x()
        y = self.root.winfo_y()
        width = self._get_window_width()
        height = self._get_window_height()
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        self._sync_close_button_position()

    def _get_window_width(self) -> int:
        self.root.update_idletasks()
        content_width = max(
            self.content_frame.winfo_reqwidth(),
            self.pit_overlay_frame.winfo_reqwidth(),
        )
        return max(self.WINDOW_MIN_WIDTH, content_width)

    def _get_window_height(self) -> int:
        self.root.update_idletasks()
        min_height = (
            self.WINDOW_MIN_HEIGHT_EXPANDED
            if self.show_advanced_var.get()
            else self.WINDOW_MIN_HEIGHT_COLLAPSED
        )
        content_height = max(
            self.content_frame.winfo_reqheight(),
            self.pit_overlay_frame.winfo_reqheight(),
        )
        return max(min_height, content_height)

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
