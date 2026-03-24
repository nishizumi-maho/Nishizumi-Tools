
#!/usr/bin/env python3
"""Clean iRacing pit-stop overlay for pit loss and rejoin safety estimation.

Main fixes vs the original:
- cleaner overlay UI with collapsible settings
- drag only on title bar (entries/buttons remain usable)
- robust iRacing connection handling
- freezes telemetry buffer for consistent grouped reads
- uses SDK-backed vars/YAML more carefully
- prefers PitSvFuel when available instead of always forcing a full tank
- respects DriverCarMaxFuelPct fuel restrictions when present
- avoids writing profile JSON every frame
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tkinter as tk
from typing import Dict, List, Optional, Tuple

import irsdk

UPDATE_MS = 100
DEFAULT_LAP_TIME_S = 90.0
GREEN_GAP_S = 5.0
YELLOW_GAP_S = 1.5
MAX_REASONABLE_RATE_LPS = 8.0


def _get_appdata_dir() -> Path:
    root = Path(os.getenv("APPDATA") or Path.home() / ".config")
    path = root / "NishizumiTools"
    path.mkdir(parents=True, exist_ok=True)
    return path


PROFILE_FILE = _get_appdata_dir() / "nishizumi_pittime_profiles.json"


class PitStopOverlay:
    BG = "#0f1115"
    PANEL = "#171a21"
    PANEL_ALT = "#1c2230"
    ACCENT = "#8ff0a4"
    TEXT = "#f3f4f6"
    MUTED = "#9ca3af"
    BORDER = "#374151"
    GOOD = "#15803d"
    WARN = "#ca8a04"
    BAD = "#b91c1c"
    BTN = "#242c3c"
    BTN_ACTIVE = "#344059"
    ENTRY_BG = "#101826"

    def __init__(self) -> None:
        self.ir = irsdk.IRSDK()

        self.root = tk.Tk()
        self.root.title("Pit Window Overlay")
        self.root.configure(bg=self.BG)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.97)
        self.root.geometry("420x224+80+80")
        self.root.minsize(320, 120)

        self._drag_offset_x = 0
        self._drag_offset_y = 0

        self.base_loss_var = tk.StringVar(value="20.0")
        self.tire_loss_var = tk.StringVar(value="0.0")
        self.fuel_rate_var = tk.StringVar(value="2.20")
        self.custom_fuel_max_var = tk.StringVar(value="")
        self.use_custom_fuel_max_var = tk.BooleanVar(value=False)
        self.use_custom_fuel_max_var.trace_add("write", self._on_use_custom_fuel_max_changed)
        self.minimal_mode_var = tk.BooleanVar(value=False)
        self.settings_visible_var = tk.BooleanVar(value=False)
        self.use_pending_pit_fuel_var = tk.BooleanVar(value=True)

        self.connection_var = tk.StringVar(value="Connecting to iRacing…")
        self.context_var = tk.StringVar(value="Car: -- | Track: --")
        self.fuel_line_var = tk.StringVar(value="Fuel: --")
        self.loss_line_var = tk.StringVar(value="Pit loss: --")
        self.window_var = tk.StringVar(value="Awaiting telemetry")
        self.gaps_var = tk.StringVar(value="Front -- | Rear --")
        self.status_var = tk.StringVar(value="Waiting for telemetry…")

        self.profile_data = self._load_profiles()
        self.active_car_id: Optional[str] = None
        self.active_track_id: Optional[str] = None

        self._last_tick_s: Optional[float] = None
        self._last_fuel_level: Optional[float] = None
        self._is_fueling = False
        self._fueling_samples: List[float] = []
        self._last_saved_profile_snapshot: Optional[str] = None
        self._ui_flash_until_ms = 0

        self._build_ui()
        self._apply_compact_mode()

    # ---------------------------- UI ----------------------------

    def _build_ui(self) -> None:
        self.shell = tk.Frame(self.root, bg=self.BG, highlightthickness=1, highlightbackground=self.BORDER)
        self.shell.pack(fill="both", expand=True)

        self.title_bar = tk.Frame(self.shell, bg=self.BG, padx=8, pady=6)
        self.title_bar.pack(fill="x")
        self.title_bar.bind("<ButtonPress-1>", self._start_move)
        self.title_bar.bind("<B1-Motion>", self._on_move)

        self.title_label = tk.Label(
            self.title_bar,
            text="Pit Overlay",
            bg=self.BG,
            fg=self.ACCENT,
            font=("Segoe UI", 11, "bold"),
        )
        self.title_label.pack(side="left")
        self.title_label.bind("<ButtonPress-1>", self._start_move)
        self.title_label.bind("<B1-Motion>", self._on_move)

        self.btn_frame = tk.Frame(self.title_bar, bg=self.BG)
        self.btn_frame.pack(side="right")

        self.settings_btn = self._make_title_button("⚙", self._toggle_settings)
        self.settings_btn.pack(side="left", padx=(0, 4))
        self.minimal_btn = self._make_title_button("—", self._toggle_minimal_mode)
        self.minimal_btn.pack(side="left", padx=(0, 4))
        self.close_btn = self._make_title_button("✕", self.root.destroy)
        self.close_btn.pack(side="left")

        self.content_area = tk.Frame(self.shell, bg=self.BG)
        self.content_area.pack(fill="both", expand=True)

        self.content_canvas = tk.Canvas(
            self.content_area,
            bg=self.BG,
            highlightthickness=0,
            bd=0,
            relief="flat",
        )
        self.content_canvas.pack(side="left", fill="both", expand=True)

        self.content_scroll = tk.Scrollbar(
            self.content_area,
            orient="vertical",
            command=self.content_canvas.yview,
        )
        self.content_canvas.configure(yscrollcommand=self.content_scroll.set)

        self.main = tk.Frame(self.content_canvas, bg=self.BG, padx=8, pady=8)
        self.main_window = self.content_canvas.create_window((0, 0), window=self.main, anchor="nw")
        self.main.bind("<Configure>", self._on_main_configure)
        self.content_canvas.bind("<Configure>", self._on_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.root.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        self.root.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

        self.live_card = tk.Frame(self.main, bg=self.PANEL_ALT, padx=10, pady=10, highlightthickness=1, highlightbackground=self.BORDER)
        self.live_card.pack(fill="x")

        tk.Label(
            self.live_card,
            textvariable=self.window_var,
            bg=self.PANEL_ALT,
            fg="white",
            font=("Segoe UI", 24, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            self.live_card,
            textvariable=self.gaps_var,
            bg=self.PANEL_ALT,
            fg=self.TEXT,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))
        tk.Label(
            self.live_card,
            textvariable=self.status_var,
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            font=("Segoe UI", 9),
            anchor="w",
            justify="left",
            wraplength=380,
        ).pack(fill="x", pady=(4, 0))

        self.info_card = tk.Frame(self.main, bg=self.PANEL, padx=10, pady=8, highlightthickness=1, highlightbackground=self.BORDER)
        self.info_card.pack(fill="x", pady=(8, 0))

        self.connection_label = tk.Label(self.info_card, textvariable=self.connection_var, bg=self.PANEL, fg=self.MUTED, font=("Segoe UI", 9), anchor="w")
        self.connection_label.pack(fill="x")
        self.context_label = tk.Label(self.info_card, textvariable=self.context_var, bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 9), anchor="w")
        self.context_label.pack(fill="x", pady=(2, 0))
        self.fuel_label = tk.Label(self.info_card, textvariable=self.fuel_line_var, bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 9), anchor="w")
        self.fuel_label.pack(fill="x", pady=(2, 0))
        self.loss_label = tk.Label(self.info_card, textvariable=self.loss_line_var, bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 9), anchor="w")
        self.loss_label.pack(fill="x", pady=(2, 0))

        self.settings_card = tk.Frame(self.main, bg=self.PANEL, padx=10, pady=10, highlightthickness=1, highlightbackground=self.BORDER)

        self._row_entry(self.settings_card, "Base pit loss [s]", self.base_loss_var, 0)
        self._row_entry(self.settings_card, "Tire-change loss [s]", self.tire_loss_var, 1)
        self._row_entry(self.settings_card, "Fuel rate [L/s]", self.fuel_rate_var, 2, editable=False)
        self.custom_fuel_max_entry = self._row_entry(self.settings_card, "Custom fuel max [L]", self.custom_fuel_max_var, 3)

        tk.Checkbutton(
            self.settings_card,
            text="Use custom fuel max",
            variable=self.use_custom_fuel_max_var,
            bg=self.PANEL,
            fg=self.TEXT,
            activebackground=self.PANEL,
            activeforeground=self.TEXT,
            selectcolor=self.BG,
            font=("Segoe UI", 9),
            command=self._sync_custom_fuel_max_state,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        tk.Checkbutton(
            self.settings_card,
            text="Prefer pending iRacing pit fuel (PitSvFuel)",
            variable=self.use_pending_pit_fuel_var,
            bg=self.PANEL,
            fg=self.TEXT,
            activebackground=self.PANEL,
            activeforeground=self.TEXT,
            selectcolor=self.BG,
            font=("Segoe UI", 9),
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.footer = tk.Label(
            self.main,
            text="GREEN ≥ 5.0s | YELLOW ≥ 1.5s | calibrated base loss still matters",
            bg=self.BG,
            fg=self.MUTED,
            font=("Segoe UI", 8),
            anchor="w",
        )
        self.footer.pack(fill="x", pady=(8, 0))
        self._sync_custom_fuel_max_state()

    def _make_title_button(self, text: str, command) -> tk.Button:
        return tk.Button(
            self.btn_frame,
            text=text,
            command=command,
            bg=self.BTN,
            fg=self.TEXT,
            activebackground=self.BTN_ACTIVE,
            activeforeground="white",
            relief="flat",
            bd=0,
            takefocus=False,
            cursor="hand2",
            width=3,
            font=("Segoe UI Symbol", 10, "bold"),
            padx=0,
            pady=1,
        )

    def _row_entry(self, parent: tk.Widget, label: str, var: tk.StringVar, row: int, editable: bool = True) -> tk.Entry:
        tk.Label(parent, text=label, bg=self.PANEL, fg=self.TEXT, font=("Segoe UI", 9)).grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=4
        )
        entry = tk.Entry(
            parent,
            textvariable=var,
            width=12,
            justify="center",
            state="normal" if editable else "readonly",
            bg=self.ENTRY_BG,
            fg=self.TEXT,
            readonlybackground=self.ENTRY_BG,
            insertbackground=self.TEXT,
            relief="flat",
            highlightthickness=1,
            highlightbackground=self.BORDER,
            highlightcolor="#4f8cff",
            font=("Segoe UI", 9),
        )
        entry.grid(row=row, column=1, sticky="e", pady=4)
        parent.grid_columnconfigure(0, weight=1)
        return entry

    def _on_use_custom_fuel_max_changed(self, *_args) -> None:
        self._sync_custom_fuel_max_state()

    def _sync_custom_fuel_max_state(self) -> None:
        entry = getattr(self, "custom_fuel_max_entry", None)
        if entry is None:
            return
        if self.use_custom_fuel_max_var.get():
            entry.configure(state="normal")
        else:
            if self.root.focus_get() is entry:
                self.root.focus_set()
            entry.configure(state="disabled")

    def _toggle_settings(self) -> None:
        self.settings_visible_var.set(not self.settings_visible_var.get())
        self._apply_compact_mode()

    def _toggle_minimal_mode(self) -> None:
        self.minimal_mode_var.set(not self.minimal_mode_var.get())
        self._apply_compact_mode()

    def _apply_compact_mode(self) -> None:
        minimal = self.minimal_mode_var.get()
        settings = self.settings_visible_var.get() and not minimal

        if minimal:
            self.info_card.pack_forget()
            self.settings_card.pack_forget()
            self.footer.pack_forget()
            self.root.geometry("320x122")
            self.minimal_btn.configure(text="▢")
            self.content_canvas.yview_moveto(0)
            self._update_scrollbar_visibility()
            return

        self.minimal_btn.configure(text="—")

        self.info_card.pack_forget()
        self.settings_card.pack_forget()
        self.footer.pack_forget()

        self.info_card.pack(fill="x", pady=(8, 0), after=self.live_card)

        if settings:
            self.settings_card.pack(fill="x", pady=(8, 0), after=self.info_card)
            self.footer.pack(fill="x", pady=(8, 0), after=self.settings_card)
            self.root.geometry("420x320")
        else:
            self.footer.pack(fill="x", pady=(8, 0), after=self.info_card)
            self.root.geometry("420x224")
            self.content_canvas.yview_moveto(0)

        self.root.after_idle(self._update_scrollbar_visibility)

    def _on_main_configure(self, _event=None) -> None:
        self.content_canvas.configure(scrollregion=self.content_canvas.bbox("all"))
        self._update_scrollbar_visibility()

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.content_canvas.itemconfigure(self.main_window, width=event.width)
        self._update_scrollbar_visibility()

    def _content_overflows(self) -> bool:
        bbox = self.content_canvas.bbox("all")
        if not bbox:
            return False
        content_height = bbox[3] - bbox[1]
        view_height = self.content_canvas.winfo_height()
        return content_height > (view_height + 1)

    def _update_scrollbar_visibility(self) -> None:
        if self.minimal_mode_var.get():
            if self.content_scroll.winfo_manager():
                self.content_scroll.pack_forget()
            self.content_canvas.yview_moveto(0)
            return

        if self._content_overflows():
            if not self.content_scroll.winfo_manager():
                self.content_scroll.pack(side="right", fill="y")
        else:
            if self.content_scroll.winfo_manager():
                self.content_scroll.pack_forget()
            self.content_canvas.yview_moveto(0)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.minimal_mode_var.get() or not self._content_overflows():
            return
        delta = getattr(event, "delta", 0)
        if not delta:
            return
        steps = -int(delta / 120)
        if steps == 0:
            steps = -1 if delta > 0 else 1
        self.content_canvas.yview_scroll(steps, "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        if self.minimal_mode_var.get() or not self._content_overflows():
            return
        num = getattr(event, "num", None)
        if num == 4:
            self.content_canvas.yview_scroll(-1, "units")
        elif num == 5:
            self.content_canvas.yview_scroll(1, "units")

    def _start_move(self, event: tk.Event) -> None:
        self._drag_offset_x = event.x_root - self.root.winfo_x()
        self._drag_offset_y = event.y_root - self.root.winfo_y()

    def _on_move(self, event: tk.Event) -> None:
        x = event.x_root - self._drag_offset_x
        y = event.y_root - self._drag_offset_y
        self.root.geometry(f"+{x}+{y}")

    # ---------------------------- Storage/helpers ----------------------------

    @staticmethod
    def _safe_float(value, default: float = 0.0, minimum: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if math.isnan(parsed) or math.isinf(parsed):
            return default
        return max(minimum, parsed)

    @staticmethod
    def _safe_str(value, default: str = "") -> str:
        if value is None:
            return default
        text = str(value).strip()
        return text or default

    def _load_profiles(self) -> Dict[str, Dict[str, float]]:
        if not PROFILE_FILE.exists():
            return {}
        try:
            raw = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}

        profiles: Dict[str, Dict[str, float]] = {}
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            cleaned: Dict[str, float] = {}
            for field in ("fuel_rate", "base_loss", "tire_loss", "custom_fuel_max"):
                if field not in value:
                    continue
                parsed = self._safe_float(value[field], default=-1.0, minimum=-1.0)
                if parsed >= 0:
                    cleaned[field] = parsed
            if cleaned:
                profiles[str(key)] = cleaned
        return profiles

    def _save_profiles(self) -> None:
        payload: Dict[str, Dict[str, float]] = {}
        for key, values in self.profile_data.items():
            row = {}
            for field in ("fuel_rate", "base_loss", "tire_loss", "custom_fuel_max"):
                if field in values:
                    row[field] = round(float(values[field]), 4)
            if row:
                payload[key] = row

        serialized = json.dumps(payload, indent=2, sort_keys=True)
        if serialized == self._last_saved_profile_snapshot:
            return

        try:
            PROFILE_FILE.write_text(serialized, encoding="utf-8")
            self._last_saved_profile_snapshot = serialized
        except Exception:
            pass

    def _profile_key(self) -> Optional[str]:
        if not self.active_car_id or not self.active_track_id:
            return None
        return f"{self.active_car_id}::{self.active_track_id}"

    def _apply_profile_if_known(self) -> None:
        key = self._profile_key()
        if not key:
            return
        profile = self.profile_data.get(key)
        if not profile:
            return

        if profile.get("fuel_rate", 0) > 0:
            self.fuel_rate_var.set(f"{profile['fuel_rate']:.3f}")
        if profile.get("base_loss", -1) >= 0:
            self.base_loss_var.set(f"{profile['base_loss']:.2f}")
        if profile.get("tire_loss", -1) >= 0:
            self.tire_loss_var.set(f"{profile['tire_loss']:.2f}")
        if profile.get("custom_fuel_max", 0) > 0:
            self.custom_fuel_max_var.set(f"{profile['custom_fuel_max']:.2f}")
            self.use_custom_fuel_max_var.set(True)
        else:
            self.custom_fuel_max_var.set("")
            self.use_custom_fuel_max_var.set(False)

        self._sync_custom_fuel_max_state()

    def _persist_profile_inputs(self) -> None:
        key = self._profile_key()
        if not key:
            return

        profile = self.profile_data.setdefault(key, {})
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

    # ---------------------------- iRacing access ----------------------------

    def _ensure_connection(self) -> bool:
        if self.ir.is_initialized and self.ir.is_connected:
            return True

        if self.ir.is_initialized and not self.ir.is_connected:
            try:
                self.ir.shutdown()
            except Exception:
                pass

        try:
            ok = self.ir.startup()
        except Exception:
            ok = False

        return bool(ok and self.ir.is_initialized and self.ir.is_connected)

    def _read_var(self, name: str, default=None):
        try:
            return self.ir[name]
        except Exception:
            return default

    def _read_yaml(self, key: str) -> Optional[dict]:
        try:
            value = self.ir[key]
        except Exception:
            return None
        return value if isinstance(value, dict) else None

    def _driver_identity(self) -> Tuple[Optional[str], str]:
        driver_info = self._read_yaml("DriverInfo")
        if not driver_info:
            return None, "Unknown car"

        driver_car_idx = driver_info.get("DriverCarIdx")
        drivers = driver_info.get("Drivers")
        if not isinstance(drivers, list):
            return None, "Unknown car"

        for entry in drivers:
            if not isinstance(entry, dict):
                continue
            if entry.get("CarIdx") != driver_car_idx:
                continue

            car_name = self._safe_str(
                entry.get("CarScreenNameShort") or entry.get("CarScreenName") or entry.get("CarPath"),
                default="Unknown car",
            )
            car_id = self._safe_str(
                entry.get("CarPath") or entry.get("CarClassShortName") or driver_car_idx,
                default="unknown_car",
            )
            return car_id, car_name

        return None, "Unknown car"

    def _track_identity(self) -> Tuple[Optional[str], str]:
        weekend = self._read_yaml("WeekendInfo")
        if not weekend:
            return None, "Unknown track"

        track_name = self._safe_str(
            weekend.get("TrackDisplayName") or weekend.get("TrackName") or weekend.get("TrackID"),
            default="Unknown track",
        )
        track_id = self._safe_str(
            weekend.get("TrackName") or weekend.get("TrackDisplayShortName") or weekend.get("TrackID"),
            default="unknown_track",
        )
        return track_id, track_name

    def _estimate_lap_time(self) -> float:
        # Prefer YAML estimate if present, then class estimate, then live lap timings.
        driver_info = self._read_yaml("DriverInfo") or {}
        for candidate in (
            driver_info.get("DriverCarEstLapTime"),
            self._extract_player_class_est_lap(driver_info),
            self._read_var("LapBestLapTime"),
            self._read_var("LapCurrentLapTime"),
        ):
            if isinstance(candidate, (float, int)) and candidate > 1.0:
                return float(candidate)
        return DEFAULT_LAP_TIME_S

    def _extract_player_class_est_lap(self, driver_info: dict) -> Optional[float]:
        driver_car_idx = driver_info.get("DriverCarIdx")
        drivers = driver_info.get("Drivers")
        if not isinstance(drivers, list):
            return None
        for entry in drivers:
            if not isinstance(entry, dict):
                continue
            if entry.get("CarIdx") != driver_car_idx:
                continue
            value = entry.get("CarClassEstLapTime")
            if isinstance(value, (float, int)) and value > 1.0:
                return float(value)
        return None

    def _driver_fuel_max_liters(self) -> Optional[float]:
        driver_info = self._read_yaml("DriverInfo") or {}
        tank = driver_info.get("DriverCarFuelMaxLtr")
        max_pct = driver_info.get("DriverCarMaxFuelPct")
        if isinstance(tank, (float, int)) and tank > 0:
            if isinstance(max_pct, (float, int)) and max_pct > 0:
                return float(tank) * float(max_pct)
            return float(tank)
        return None

    def _pending_fuel_add_liters(self) -> Optional[float]:
        pending = self._read_var("PitSvFuel")
        if isinstance(pending, (float, int)) and pending >= 0:
            return float(pending)
        return None

    # ---------------------------- Calculation ----------------------------

    def _learn_fuel_rate(self) -> None:
        now_s = self._read_var("SessionTime")
        fuel_now = self._read_var("FuelLevel")
        on_pit_road = bool(self._read_var("OnPitRoad", 0))
        pitstop_active = bool(self._read_var("PitstopActive", 0))
        in_stall = bool(self._read_var("PlayerCarInPitStall", 0))

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

        fueling_now = (on_pit_road and in_stall) or pitstop_active
        if fueling_now and dfuel > 0.01:
            rate = dfuel / dt
            if 0.05 <= rate <= MAX_REASONABLE_RATE_LPS:
                self._fueling_samples.append(rate)
            self._is_fueling = True
            return

        if self._is_fueling and self._fueling_samples:
            learned_rate = sum(self._fueling_samples) / len(self._fueling_samples)
            key = self._profile_key()
            if key and learned_rate > 0:
                self.profile_data.setdefault(key, {})["fuel_rate"] = learned_rate
                self.fuel_rate_var.set(f"{learned_rate:.3f}")
                self._save_profiles()

        self._is_fueling = False
        self._fueling_samples = []

    def _collect_car_deltas(self, lap_time_s: float) -> List[float]:
        player_idx = self._read_var("PlayerCarIdx")
        car_est = self._read_var("CarIdxEstTime")
        track_surface = self._read_var("CarIdxTrackSurface")
        car_on_pit_road = self._read_var("CarIdxOnPitRoad")

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

            try:
                if car_on_pit_road is not None and bool(car_on_pit_road[idx]):
                    continue
            except Exception:
                pass

            try:
                if track_surface is not None and int(track_surface[idx]) < 0:
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
            return "GREEN", PitStopOverlay.GOOD, score
        if min_gap >= YELLOW_GAP_S:
            return "YELLOW", PitStopOverlay.WARN, score
        return "RED", PitStopOverlay.BAD, score

    def _calculate(self) -> None:
        car_id, car_name = self._driver_identity()
        track_id, track_name = self._track_identity()

        changed_profile = (car_id != self.active_car_id) or (track_id != self.active_track_id)
        self.active_car_id = car_id
        self.active_track_id = track_id
        if changed_profile:
            self._apply_profile_if_known()

        self._learn_fuel_rate()
        self._persist_profile_inputs()

        base_loss = self._safe_float(self.base_loss_var.get(), default=20.0)
        tire_loss = self._safe_float(self.tire_loss_var.get(), default=0.0)
        fuel_rate = self._safe_float(self.fuel_rate_var.get(), default=2.2, minimum=0.001)

        fuel_now = self._safe_float(self._read_var("FuelLevel"), default=0.0)
        fuel_max = self._driver_fuel_max_liters()

        if self.use_custom_fuel_max_var.get():
            custom_max = self._safe_float(self.custom_fuel_max_var.get(), default=0.0)
            if custom_max > 0:
                fuel_max = custom_max

        pending_add = self._pending_fuel_add_liters()
        if self.use_pending_pit_fuel_var.get() and isinstance(pending_add, float) and pending_add > 0:
            fuel_to_add = pending_add
            fuel_plan_text = f"pending {fuel_to_add:.1f} L"
        else:
            if fuel_max is None or fuel_max <= 0:
                fuel_to_add = 0.0
                fuel_plan_text = "fuel target unknown"
            else:
                fuel_to_add = max(0.0, fuel_max - fuel_now)
                fuel_plan_text = f"fill {fuel_to_add:.1f} L"

        fuel_time = fuel_to_add / fuel_rate if fuel_rate > 0 else 0.0
        total_loss = base_loss + tire_loss + fuel_time

        lap_time_s = self._estimate_lap_time()
        deltas = self._collect_car_deltas(lap_time_s)
        projected = [delta + total_loss for delta in deltas]

        front_candidates = [delta for delta in projected if delta >= 0]
        rear_candidates = [delta for delta in projected if delta < 0]
        front_gap = min(front_candidates) if front_candidates else 99.0
        rear_gap = abs(max(rear_candidates)) if rear_candidates else 99.0

        status, color, score = self._status_from_gaps(front_gap, rear_gap)

        self.context_var.set(f"Car: {car_name} | Track: {track_name}")
        self.fuel_line_var.set(
            f"Fuel {fuel_now:.1f} L | rate {fuel_rate:.2f} L/s | {fuel_plan_text} | refuel {fuel_time:.2f}s"
        )
        self.loss_line_var.set(
            f"Base {base_loss:.2f}s + tires {tire_loss:.2f}s + fuel {fuel_time:.2f}s = {total_loss:.2f}s"
        )
        self.window_var.set(f"{score:.0f}%   {status}")
        self.gaps_var.set(f"Front {front_gap:.2f}s | Rear {rear_gap:.2f}s")
        self.status_var.set(
            f"Projected rejoin after {total_loss:.2f}s pit loss. "
            f"Lap estimate {lap_time_s:.2f}s."
        )
        self.live_card.configure(highlightbackground=color)
        self.window_var.set(f"{score:.0f}%   {status}")

    # ---------------------------- Loop ----------------------------

    def _show_disconnected(self) -> None:
        self.connection_var.set("Not connected to iRacing. Open the sim and click Drive.")
        self.context_var.set("Car: -- | Track: --")
        self.fuel_line_var.set("Fuel: --")
        self.loss_line_var.set("Pit loss: --")
        self.window_var.set("Awaiting telemetry")
        self.gaps_var.set("Front -- | Rear --")
        self.status_var.set("No live telemetry available.")
        self.live_card.configure(highlightbackground=self.BORDER)

    def _update(self) -> None:
        try:
            if not self._ensure_connection():
                self._show_disconnected()
                self.root.after(500, self._update)
                return

            self.connection_var.set(f"Connected to iRacing | refresh {UPDATE_MS} ms")

            self.ir.freeze_var_buffer_latest()
            try:
                self._calculate()
            finally:
                self.ir.unfreeze_var_buffer_latest()

        except Exception as exc:
            self.status_var.set(f"Runtime error: {type(exc).__name__}: {exc}")
            self.live_card.configure(highlightbackground=self.BAD)

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
