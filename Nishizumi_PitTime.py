#!/usr/bin/env python3
"""Standalone Tkinter overlay for pit-stop loss and rejoin safety estimation."""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
import tkinter as tk
from typing import Dict, List, Optional, Tuple

import irsdk

UPDATE_MS = 16
DEFAULT_LAP_TIME_S = 90.0
GREEN_GAP_S = 5.0
YELLOW_GAP_S = 1.5


def _get_appdata_dir() -> Path:
    root = Path(os.getenv("APPDATA") or Path.home() / ".config")
    path = root / "NishizumiTools"
    path.mkdir(parents=True, exist_ok=True)
    return path


RATE_FILE = _get_appdata_dir() / "nishizumi_pittime_fuel_rates.json"


class PitStopOverlay:
    BG = "#0f1115"
    PANEL = "#171a21"
    PANEL_ALT = "#1d2230"
    CARD = "#22293a"
    TEXT = "#f2f2f2"
    MUTED = "#9aa4b2"
    ENTRY_BG = "#101826"
    ENTRY_BORDER = "#2b3647"

    def __init__(self) -> None:
        self.ir = irsdk.IRSDK()

        self.root = tk.Tk()
        self.root.title("Pit Window Overlay")
        self.root.geometry("500x440")
        self.root.configure(bg=self.BG)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.95)

        self._drag_offset_x = 0
        self._drag_offset_y = 0

        self.base_loss_var = tk.StringVar(value="20.0")
        self.tire_loss_var = tk.StringVar(value="0.0")
        self.fuel_rate_var = tk.StringVar(value="2.2")
        self.custom_fuel_max_var = tk.StringVar(value="")
        self.use_custom_fuel_max_var = tk.BooleanVar(value=False)
        self.lock_inputs_var = tk.BooleanVar(value=False)
        self.minimal_mode_var = tk.BooleanVar(value=False)

        self.connection_var = tk.StringVar(value="Connecting to iRacing...")
        self.car_var = tk.StringVar(value="Car: --")
        self.fuel_state_var = tk.StringVar(value="Fuel: -- / -- L")
        self.fuel_time_var = tk.StringVar(value="Fuel time: -- s")
        self.total_time_var = tk.StringVar(value="Total pit time loss: -- s")
        self.window_var = tk.StringVar(value="Window: --")
        self.status_var = tk.StringVar(value="Status: --")

        self.profile_data = self._load_profiles()
        self.active_car_id: Optional[str] = None
        self.active_car_name: Optional[str] = None
        self.active_track_id: Optional[str] = None
        self.active_track_name: Optional[str] = None
        self._last_fuel_level: Optional[float] = None
        self._last_tick_s: Optional[float] = None
        self._is_fueling = False
        self._fueling_samples: List[float] = []
        self._fuel_rate_autofilled = False
        self._editable_entries: List[tk.Entry] = []
        self._mousewheel_bound = False

        self._build_ui()

    def _build_ui(self) -> None:
        self.title_bar = tk.Frame(self.root, bg=self.BG)
        self.title_bar.pack(fill="x", padx=10, pady=(8, 4))

        title_stack = tk.Frame(self.title_bar, bg=self.BG)
        title_stack.pack(side="left", fill="x", expand=True)
        tk.Label(
            title_stack,
            text="Pit Stop Overlay",
            font=("Segoe UI", 13, "bold"),
            fg="#d8f8d8",
            bg=self.BG,
        ).pack(anchor="w")
        tk.Label(
            title_stack,
            text="Loss + rejoin safety estimator",
            font=("Segoe UI", 9),
            fg=self.MUTED,
            bg=self.BG,
        ).pack(anchor="w")

        self.scroll_canvas = tk.Canvas(
            self.root,
            bg=self.BG,
            highlightthickness=0,
            bd=0,
        )
        self.scrollbar = tk.Scrollbar(
            self.root,
            orient="vertical",
            command=self.scroll_canvas.yview,
            troughcolor=self.BG,
            bg="#253247",
            activebackground="#334766",
        )
        self.scroll_canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.main_frame = tk.Frame(self.scroll_canvas, bg=self.BG)
        self._canvas_window = self.scroll_canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        self.main_frame.bind("<Configure>", self._on_main_frame_configure)
        self.scroll_canvas.bind("<Configure>", self._on_canvas_configure)
        self.scroll_canvas.bind("<Enter>", self._bind_mousewheel)
        self.scroll_canvas.bind("<Leave>", self._unbind_mousewheel)

        self.inputs_frame = tk.LabelFrame(
            self.main_frame,
            text=" Pit loss setup ",
            font=("Segoe UI", 10, "bold"),
            fg=self.TEXT,
            bg=self.PANEL,
            bd=0,
            padx=10,
            pady=10,
        )
        self.inputs_frame.pack(fill="x")

        self._editable_entries.append(
            self._row_entry(self.inputs_frame, "Base pit loss (in + out, no service) [s]", self.base_loss_var, 0)
        )
        self._editable_entries.append(
            self._row_entry(self.inputs_frame, "Tire-change loss [s]", self.tire_loss_var, 1)
        )
        self._row_entry(self.inputs_frame, "Fuel rate [L/s] (auto-learned)", self.fuel_rate_var, 2, editable=False)

        custom_max_check = tk.Checkbutton(
            self.inputs_frame,
            text="Use custom fuel tank max [L]",
            variable=self.use_custom_fuel_max_var,
            font=("Segoe UI", 10),
            fg=self.TEXT,
            bg=self.PANEL,
            activebackground=self.PANEL,
            activeforeground=self.TEXT,
            selectcolor=self.BG,
        )
        custom_max_check.grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(6, 2))
        custom_max_entry = self._row_entry(self.inputs_frame, "Custom fuel max [L]", self.custom_fuel_max_var, 4)
        self._editable_entries.append(custom_max_entry)

        lock_check = tk.Checkbutton(
            self.inputs_frame,
            text="Lock typed inputs (read-only)",
            variable=self.lock_inputs_var,
            command=self._apply_lock_state,
            font=("Segoe UI", 10),
            fg=self.TEXT,
            bg=self.PANEL,
            activebackground=self.PANEL,
            activeforeground=self.TEXT,
            selectcolor=self.BG,
        )
        lock_check.grid(row=5, column=0, columnspan=2, sticky="w", pady=(8, 2))

        minimal_mode_check = tk.Checkbutton(
            self.inputs_frame,
            text="Race minimal mode (only pit rejoin safety bar)",
            variable=self.minimal_mode_var,
            command=self._apply_minimal_mode,
            font=("Segoe UI", 10),
            fg=self.TEXT,
            bg=self.PANEL,
            activebackground=self.PANEL,
            activeforeground=self.TEXT,
            selectcolor=self.BG,
        )
        minimal_mode_check.grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 2))

        self.restore_button = tk.Button(
            self.root,
            text="Return to full view",
            command=self._exit_minimal_mode,
            font=("Segoe UI", 9, "bold"),
            bg="#1f2533",
            fg=self.TEXT,
            activebackground="#2b3447",
            activeforeground="white",
            relief="flat",
            padx=10,
            pady=4,
            takefocus=False,
            cursor="hand2",
        )

        self.connection_label = tk.Label(
            self.main_frame,
            textvariable=self.connection_var,
            fg=self.MUTED,
            bg=self.BG,
            font=("Segoe UI", 9),
        )
        self.connection_label.pack(anchor="w", pady=(8, 2))
        self.car_label = tk.Label(
            self.main_frame,
            textvariable=self.car_var,
            fg=self.MUTED,
            bg=self.BG,
            font=("Segoe UI", 9),
        )
        self.car_label.pack(anchor="w", pady=(0, 4))
        self.top_separator = tk.Frame(self.main_frame, bg="#2b3447", height=1)
        self.top_separator.pack(fill="x", pady=4)

        self.metrics_frame = tk.LabelFrame(
            self.main_frame,
            text=" Live calculations ",
            font=("Segoe UI", 10, "bold"),
            fg=self.TEXT,
            bg=self.PANEL,
            bd=0,
            padx=10,
            pady=10,
        )
        self.metrics_frame.pack(fill="x", pady=(6, 8))

        tk.Label(self.metrics_frame, textvariable=self.fuel_state_var, font=("Segoe UI", 10), fg=self.TEXT, bg=self.PANEL).pack(anchor="w")
        tk.Label(self.metrics_frame, textvariable=self.fuel_time_var, font=("Segoe UI", 10), fg=self.TEXT, bg=self.PANEL).pack(anchor="w", pady=(2, 0))
        tk.Label(self.metrics_frame, textvariable=self.total_time_var, font=("Segoe UI", 11, "bold"), fg="#d8f8d8", bg=self.PANEL).pack(anchor="w", pady=(5, 0))

        self.status_frame = tk.LabelFrame(
            self.main_frame,
            text=" Pit rejoin safety ",
            font=("Segoe UI", 10, "bold"),
            fg=self.TEXT,
            bg=self.PANEL_ALT,
            bd=0,
            padx=10,
            pady=10,
        )
        self.status_frame.pack(fill="both", expand=True)

        self.score_label = tk.Label(
            self.status_frame,
            textvariable=self.window_var,
            font=("Segoe UI", 28, "bold"),
            bg=self.CARD,
            fg="white",
            padx=12,
            pady=10,
        )
        self.score_label.pack(fill="x")

        self.status_details_label = tk.Label(
            self.status_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 11),
            wraplength=430,
            justify="left",
            fg=self.TEXT,
            bg=self.PANEL_ALT,
        )
        self.status_details_label.pack(anchor="w", pady=(10, 4))

        self.legend_label = tk.Label(
            self.status_frame,
            text=(
                "Legend: GREEN >= 5.0 s free front and rear, "
                "YELLOW >= 1.5 s, RED < 1.5 s."
            ),
            fg=self.MUTED,
            bg=self.PANEL_ALT,
            wraplength=430,
            justify="left",
        )
        self.legend_label.pack(anchor="w")

        self.root.bind("<ButtonPress-1>", self._start_move)
        self.root.bind("<B1-Motion>", self._on_move)

    def _on_main_frame_configure(self, _: tk.Event) -> None:
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.scroll_canvas.itemconfigure(self._canvas_window, width=event.width)

    def _bind_mousewheel(self, _: tk.Event) -> None:
        if self._mousewheel_bound:
            return
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel)
        self.root.bind_all("<Button-5>", self._on_mousewheel)
        self._mousewheel_bound = True

    def _unbind_mousewheel(self, _: tk.Event) -> None:
        if not self._mousewheel_bound:
            return
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")
        self._mousewheel_bound = False

    def _on_mousewheel(self, event: tk.Event) -> None:
        if not self.scroll_canvas.winfo_exists():
            return
        if hasattr(event, "delta") and event.delta:
            direction = -1 if event.delta > 0 else 1
            self.scroll_canvas.yview_scroll(direction, "units")
            return
        if getattr(event, "num", None) == 4:
            self.scroll_canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.scroll_canvas.yview_scroll(1, "units")

    def _exit_minimal_mode(self) -> None:
        self.minimal_mode_var.set(False)
        self._apply_minimal_mode()

    def _apply_minimal_mode(self) -> None:
        if self.minimal_mode_var.get():
            self.title_bar.pack_forget()
            self.inputs_frame.pack_forget()
            self.connection_label.pack_forget()
            self.car_label.pack_forget()
            self.top_separator.pack_forget()
            self.metrics_frame.pack_forget()
            self.status_details_label.pack_forget()
            self.legend_label.pack_forget()
            self.scrollbar.pack_forget()
            self.restore_button.place(relx=1.0, x=-10, y=10, anchor="ne")
            self.status_frame.configure(text="", padx=0, pady=0)
            self.status_frame.pack_configure(fill="both", expand=True, pady=0)
            self.score_label.configure(padx=0, pady=0)
            self.score_label.pack_configure(fill="both", expand=True, pady=(28, 6))
            self.root.geometry("320x96")
            return

        self.title_bar.pack(fill="x", padx=10, pady=(8, 4), before=self.scroll_canvas)
        self.restore_button.place_forget()
        self.scrollbar.pack(side="right", fill="y")
        self.inputs_frame.pack(fill="x")
        self.connection_label.pack(anchor="w", pady=(8, 2))
        self.car_label.pack(anchor="w", pady=(0, 4))
        self.top_separator.pack(fill="x", pady=4)
        self.metrics_frame.pack(fill="x", pady=(6, 8))
        self.status_frame.configure(text=" Pit rejoin safety ", padx=10, pady=10)
        self.status_frame.pack_configure(fill="both", expand=True)
        self.score_label.configure(padx=12, pady=10)
        self.score_label.pack_configure(fill="x", expand=False, pady=0)
        self.status_details_label.pack(anchor="w", pady=(10, 4))
        self.legend_label.pack(anchor="w")
        self.root.geometry("500x440")

    def _row_entry(
        self,
        parent: tk.Widget,
        label: str,
        var: tk.StringVar,
        row: int,
        editable: bool = True,
    ) -> tk.Entry:
        tk.Label(
            parent,
            text=label,
            font=("Segoe UI", 10),
            fg=self.TEXT,
            bg=self.PANEL,
        ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        state = "normal" if editable else "readonly"
        entry = tk.Entry(
            parent,
            textvariable=var,
            width=12,
            state=state,
            font=("Segoe UI", 10),
            justify="center",
            bg=self.ENTRY_BG,
            fg=self.TEXT,
            readonlybackground=self.ENTRY_BG,
            disabledforeground="#7f8ea3",
            insertbackground=self.TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.ENTRY_BORDER,
            highlightcolor="#3c79ff",
        )
        entry.grid(row=row, column=1, sticky="e", pady=4)
        parent.grid_columnconfigure(0, weight=1)
        return entry

    def _start_move(self, event: tk.Event) -> None:
        self._drag_offset_x = event.x_root - self.root.winfo_x()
        self._drag_offset_y = event.y_root - self.root.winfo_y()

    def _on_move(self, event: tk.Event) -> None:
        x = event.x_root - self._drag_offset_x
        y = event.y_root - self._drag_offset_y
        self.root.geometry(f"+{x}+{y}")

    @staticmethod
    def _safe_float(value: str, default: float = 0.0, minimum: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return max(minimum, parsed)

    @staticmethod
    def _safe_str(value: object, default: str = "") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text or default

    def _read_telemetry(self, name: str, default: Optional[float] = None):
        try:
            return self.ir[name]
        except Exception:
            return default

    def _driver_fuel_max_liters(self) -> Optional[float]:
        try:
            driver_info = self.ir["DriverInfo"]
            if isinstance(driver_info, dict):
                value = driver_info.get("DriverCarFuelMaxLtr")
                if isinstance(value, (float, int)) and value > 0:
                    return float(value)
        except Exception:
            return None
        return None

    def _driver_identity(self) -> Tuple[Optional[str], Optional[str]]:
        try:
            driver_info = self.ir["DriverInfo"]
        except Exception:
            return None, None

        if not isinstance(driver_info, dict):
            return None, None

        driver_car_idx = driver_info.get("DriverCarIdx")
        drivers = driver_info.get("Drivers")
        if not isinstance(drivers, list):
            return None, None

        for entry in drivers:
            if not isinstance(entry, dict):
                continue
            if entry.get("CarIdx") != driver_car_idx:
                continue

            car_name = self._safe_str(
                entry.get("CarScreenNameShort")
                or entry.get("CarScreenName")
                or entry.get("CarPath")
                or entry.get("UserName"),
                default="Unknown car",
            )
            car_id = self._safe_str(
                entry.get("CarPath")
                or entry.get("CarClassShortName")
                or str(driver_car_idx),
                default="unknown_car",
            )
            return car_id, car_name

        return None, None

    def _track_identity(self) -> Tuple[Optional[str], Optional[str]]:
        try:
            weekend_info = self.ir["WeekendInfo"]
        except Exception:
            return None, None

        if not isinstance(weekend_info, dict):
            return None, None

        track_name = self._safe_str(
            weekend_info.get("TrackDisplayName")
            or weekend_info.get("TrackName")
            or weekend_info.get("TrackID"),
            default="Unknown track",
        )
        track_id = self._safe_str(
            weekend_info.get("TrackName")
            or weekend_info.get("TrackDisplayShortName")
            or weekend_info.get("TrackID"),
            default="unknown_track",
        )
        return track_id, track_name

    def _profile_key(self, car_id: Optional[str], track_id: Optional[str]) -> Optional[str]:
        if not car_id or not track_id:
            return None
        return f"{car_id}::{track_id}"

    def _load_profiles(self) -> Dict[str, Dict[str, float]]:
        if not RATE_FILE.exists():
            return {}
        try:
            raw = json.loads(RATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        profiles: Dict[str, Dict[str, float]] = {}
        for profile_key, data in raw.items():
            if isinstance(data, dict):
                normalized: Dict[str, float] = {}
                for field in ("fuel_rate", "base_loss", "tire_loss", "custom_fuel_max"):
                    value = data.get(field)
                    if value is None:
                        continue
                    try:
                        parsed = float(value)
                    except (TypeError, ValueError):
                        continue
                    if parsed < 0:
                        continue
                    normalized[field] = parsed
                if normalized:
                    profiles[str(profile_key)] = normalized
            else:
                # Backward compatibility with old {car_id: fuel_rate} format.
                try:
                    parsed = float(data)
                except (TypeError, ValueError):
                    continue
                if parsed > 0:
                    profiles[f"{profile_key}::unknown_track"] = {"fuel_rate": parsed}
        return profiles

    def _save_profiles(self) -> None:
        payload: Dict[str, Dict[str, float]] = {}
        for profile_key, values in sorted(self.profile_data.items()):
            row: Dict[str, float] = {}
            for field in ("fuel_rate", "base_loss", "tire_loss", "custom_fuel_max"):
                value = values.get(field)
                if isinstance(value, (float, int)) and value >= 0:
                    row[field] = round(float(value), 4)
            if row:
                payload[profile_key] = row
        try:
            RATE_FILE.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass

    def _apply_profile_if_known(self, car_id: Optional[str], track_id: Optional[str]) -> None:
        profile_key = self._profile_key(car_id, track_id)
        if not profile_key:
            return
        profile = self.profile_data.get(profile_key)
        if not profile:
            return
        fuel_rate = profile.get("fuel_rate")
        if isinstance(fuel_rate, (float, int)) and fuel_rate > 0:
            self.fuel_rate_var.set(f"{float(fuel_rate):.3f}")
            self._fuel_rate_autofilled = True
        base_loss = profile.get("base_loss")
        if isinstance(base_loss, (float, int)) and base_loss >= 0:
            self.base_loss_var.set(f"{float(base_loss):.2f}")
        tire_loss = profile.get("tire_loss")
        if isinstance(tire_loss, (float, int)) and tire_loss >= 0:
            self.tire_loss_var.set(f"{float(tire_loss):.2f}")
        custom_max = profile.get("custom_fuel_max")
        if isinstance(custom_max, (float, int)) and custom_max > 0:
            self.custom_fuel_max_var.set(f"{float(custom_max):.2f}")
            self.use_custom_fuel_max_var.set(True)
        else:
            self.custom_fuel_max_var.set("")
            self.use_custom_fuel_max_var.set(False)

    def _persist_profile_inputs(self) -> None:
        profile_key = self._profile_key(self.active_car_id, self.active_track_id)
        if not profile_key:
            return
        profile = self.profile_data.setdefault(profile_key, {})
        profile["base_loss"] = self._safe_float(self.base_loss_var.get(), default=20.0)
        profile["tire_loss"] = self._safe_float(self.tire_loss_var.get(), default=0.0)

        if self.use_custom_fuel_max_var.get():
            custom_max = self._safe_float(self.custom_fuel_max_var.get(), default=0.0)
            if custom_max > 0:
                profile["custom_fuel_max"] = custom_max
            else:
                profile.pop("custom_fuel_max", None)
        else:
            profile.pop("custom_fuel_max", None)

        self._save_profiles()

    def _apply_lock_state(self) -> None:
        state = "readonly" if self.lock_inputs_var.get() else "normal"
        for entry in self._editable_entries:
            entry.configure(state=state)

    def _learn_fuel_rate(self) -> None:
        now_s = self._read_telemetry("SessionTime")
        fuel_now = self._read_telemetry("FuelLevel")
        on_pit_road = bool(self._read_telemetry("OnPitRoad", 0))

        if not isinstance(now_s, (float, int)) or not isinstance(fuel_now, (float, int)):
            return

        now_s = float(now_s)
        fuel_now = float(fuel_now)

        if self._last_tick_s is None or self._last_fuel_level is None:
            self._last_tick_s = now_s
            self._last_fuel_level = fuel_now
            return

        dt = now_s - self._last_tick_s
        dfuel = fuel_now - self._last_fuel_level
        self._last_tick_s = now_s
        self._last_fuel_level = fuel_now

        if dt <= 0:
            return

        if on_pit_road and dfuel > 0.01:
            instant_rate = dfuel / dt
            if instant_rate > 0:
                self._fueling_samples.append(instant_rate)
            self._is_fueling = True
            return

        profile_key = self._profile_key(self.active_car_id, self.active_track_id)
        if self._is_fueling and self._fueling_samples and profile_key:
            learned_rate = sum(self._fueling_samples) / len(self._fueling_samples)
            profile = self.profile_data.setdefault(profile_key, {})
            profile["fuel_rate"] = learned_rate
            self.fuel_rate_var.set(f"{learned_rate:.3f}")
            self._fuel_rate_autofilled = True
            self._save_profiles()

        self._is_fueling = False
        self._fueling_samples = []

    def _collect_car_deltas(self, lap_time_s: float) -> List[float]:
        player_idx = self._read_telemetry("PlayerCarIdx")
        car_est = self._read_telemetry("CarIdxEstTime")
        track_surface = self._read_telemetry("CarIdxTrackSurface")

        if player_idx is None or car_est is None:
            return []

        try:
            player_idx = int(player_idx)
            player_est = float(car_est[player_idx])
        except Exception:
            return []

        deltas: List[float] = []
        half_lap = max(10.0, lap_time_s / 2.0)

        for idx, other_est in enumerate(car_est):
            if idx == player_idx:
                continue

            if track_surface is not None:
                try:
                    surf = int(track_surface[idx])
                    if surf < 0:
                        continue
                except Exception:
                    pass

            try:
                delta = float(other_est) - player_est
            except Exception:
                continue

            while delta > half_lap:
                delta -= lap_time_s
            while delta < -half_lap:
                delta += lap_time_s
            deltas.append(delta)

        return deltas

    @staticmethod
    def _status_from_gaps(front_gap: float, rear_gap: float) -> Tuple[str, str, float]:
        min_gap = min(front_gap, rear_gap)
        score = max(0.0, min(100.0, (min_gap / GREEN_GAP_S) * 100.0))

        if min_gap >= GREEN_GAP_S:
            return "GREEN", "#15803d", score
        if min_gap >= YELLOW_GAP_S:
            return "YELLOW", "#ca8a04", score
        return "RED", "#b91c1c", score

    def _update(self) -> None:
        connected = self.ir.startup() if not getattr(self.ir, "is_initialized", False) else True
        if not connected:
            self.connection_var.set("Not connected to iRacing. Open sim + click Drive.")
            self.car_var.set("Car: --")
            self.window_var.set("Window: --")
            self.status_var.set("Status: Waiting for telemetry...")
            self.score_label.config(bg="#4b5563")
            self.root.after(150, self._update)
            return

        self.connection_var.set(f"Connected. Updating every {UPDATE_MS} ms.")

        car_id, car_name = self._driver_identity()
        track_id, track_name = self._track_identity()
        if car_id != self.active_car_id or track_id != self.active_track_id:
            self.active_car_id = car_id
            self.active_car_name = car_name
            self.active_track_id = track_id
            self.active_track_name = track_name
            self._fuel_rate_autofilled = False
            self._apply_profile_if_known(car_id, track_id)
            self._is_fueling = False
            self._fueling_samples = []
        if car_name:
            track_label = track_name or "Unknown track"
            self.car_var.set(f"Car: {car_name}  |  Track: {track_label}")
        else:
            self.car_var.set("Car: Unknown")

        self._learn_fuel_rate()
        self._persist_profile_inputs()

        base_loss = self._safe_float(self.base_loss_var.get(), default=20.0)
        tire_loss = self._safe_float(self.tire_loss_var.get(), default=0.0)
        fuel_rate = self._safe_float(self.fuel_rate_var.get(), default=2.2, minimum=0.001)

        fuel_now = self._read_telemetry("FuelLevel", 0.0)
        fuel_max = self._driver_fuel_max_liters()
        if fuel_max is None:
            fuel_max = self._read_telemetry("FuelLevel", 0.0)

        if self.use_custom_fuel_max_var.get():
            custom_max = self._safe_float(self.custom_fuel_max_var.get(), default=0.0)
            if custom_max > 0:
                fuel_max = custom_max

        fuel_now = float(fuel_now or 0.0)
        fuel_max = float(fuel_max or 0.0)
        fuel_to_add = max(0.0, fuel_max - fuel_now)
        fuel_time = fuel_to_add / fuel_rate if fuel_rate > 0 else 0.0

        total_loss = base_loss + tire_loss + fuel_time

        self.fuel_state_var.set(f"Fuel: {fuel_now:.2f} / {fuel_max:.2f} L (add {fuel_to_add:.2f} L)")
        self.fuel_time_var.set(f"Fuel time: {fuel_time:.2f} s")
        self.total_time_var.set(f"Total pit time loss: {total_loss:.2f} s")

        lap_time_s = self._read_telemetry("DriverCarEstLapTime", DEFAULT_LAP_TIME_S)
        if not isinstance(lap_time_s, (float, int)) or lap_time_s <= 1.0:
            lap_time_s = self._read_telemetry("LapBestLapTime", DEFAULT_LAP_TIME_S)
        if not isinstance(lap_time_s, (float, int)) or lap_time_s <= 1.0:
            lap_time_s = DEFAULT_LAP_TIME_S
        lap_time_s = float(lap_time_s)

        deltas = self._collect_car_deltas(lap_time_s)
        projected = [d + total_loss for d in deltas]

        front_candidates = [d for d in projected if d >= 0]
        rear_candidates = [d for d in projected if d < 0]

        front_gap = min(front_candidates) if front_candidates else 99.0
        rear_gap = abs(max(rear_candidates)) if rear_candidates else 99.0

        status, color, score = self._status_from_gaps(front_gap, rear_gap)
        self.score_label.config(bg=color)
        self.window_var.set(f"{score:.0f}%   {status}")
        self.status_var.set(
            f"Projected rejoin gaps -> front: {front_gap:.2f} s, rear: {rear_gap:.2f} s\n"
            f"(using total pit time loss {total_loss:.2f} s)."
        )

        self.root.after(UPDATE_MS, self._update)

    def run(self) -> None:
        self._update()
        self.root.mainloop()


def main() -> int:
    app = PitStopOverlay()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
