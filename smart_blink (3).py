"""Smart blink controller for iRacing telemetry.

Refactored flasher with config migration, rule-based triggers, and a
rate-limited scheduler. Uses safer telemetry access patterns based on
Release1.py (auto-startup, guarded reads).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import random
import re
import threading
import time
from dataclasses import dataclass
from queue import PriorityQueue, Empty
from typing import Any, Dict, Optional, Tuple

import irsdk
import tkinter as tk
from tkinter import messagebox, ttk
import yaml

# -----------------------
# Paths / Config storage
# -----------------------
APP_NAME = "SmartFlasher"
APP_FOLDER = "SmartFlasher"
BASE_PATH = os.getenv("APPDATA") or os.path.expanduser("~")
CONFIG_DIR = os.path.join(BASE_PATH, APP_FOLDER, "configs")
CONFIG_FILE = os.path.join(CONFIG_DIR, "flasher_config.json")


# -----------------------
# Helpers
# -----------------------

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def ms_to_kmh(ms: float) -> float:
    try:
        return float(ms) * 3.6
    except Exception:
        return 0.0


def parse_track_length_to_m(track_len: Any) -> Optional[float]:
    """
    Converte TrackLength (normalmente string tipo "4.010 km" / "2.500 mi") em metros.
    Retorna None se não conseguir.
    """
    if track_len is None:
        return None
    if isinstance(track_len, (int, float)):
        return float(track_len)

    s = str(track_len).strip().lower()
    m = re.search(r"([-+]?\d*\.?\d+)", s)
    if not m:
        return None
    num = float(m.group(1))

    if "km" in s:
        return num * 1000.0
    if "mi" in s:
        return num * 1609.344
    if "m" in s:
        return num
    return num * 1000.0 if num < 50 else num


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


TRACK_SURFACE_OFF = 0
TRACK_SURFACE_PIT_STALL = 1
TRACK_SURFACE_PIT_ROAD = 2
TRACK_SURFACE_ON = 3


# -----------------------
# INPUT (Windows SendInput)
# -----------------------
if os.name == "nt":
    SendInput = ctypes.windll.user32.SendInput
    PUL = ctypes.POINTER(ctypes.c_ulong)

    class KeyBdInput(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", PUL),
        ]

    class HardwareInput(ctypes.Structure):
        _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_ushort), ("wParamH", ctypes.c_ushort)]

    class MouseInput(ctypes.Structure):
        _fields_ = [
            ("dx", ctypes.c_long),
            ("dy", ctypes.c_long),
            ("mouseData", ctypes.c_ulong),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", PUL),
        ]

    class Input_I(ctypes.Union):
        _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]

    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

    KEYEVENTF_SCANCODE = 0x0008
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_EXTENDEDKEY = 0x0001

    def press_key_scan_code(scan_code: int, hold_s: float = 0.06, extended: bool = False) -> None:
        """Envia keydown + keyup usando scan code (Windows)."""
        if not scan_code:
            return
        flags_down = KEYEVENTF_SCANCODE | (KEYEVENTF_EXTENDEDKEY if extended else 0)
        flags_up = flags_down | KEYEVENTF_KEYUP

        extra = ctypes.c_ulong(0)
        ii_ = Input_I()

        ii_.ki = KeyBdInput(0, int(scan_code), flags_down, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

        time.sleep(max(0.0, float(hold_s)))

        ii_.ki = KeyBdInput(0, int(scan_code), flags_up, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

else:
    def press_key_scan_code(scan_code: int, hold_s: float = 0.06, extended: bool = False) -> None:
        return


# -----------------------
# Flash pattern + scheduler
# -----------------------
@dataclass(frozen=True)
class FlashPattern:
    flashes: int = 1
    min_interval_s: float = 0.2
    max_interval_s: float = 0.4
    hold_s: float = 0.06

    def normalized(self) -> "FlashPattern":
        flashes = max(1, int(self.flashes))
        min_interval = max(0.0, float(self.min_interval_s))
        max_interval = max(min_interval, float(self.max_interval_s))
        hold = clamp(float(self.hold_s), 0.01, 0.25)
        return FlashPattern(flashes=flashes, min_interval_s=min_interval, max_interval_s=max_interval, hold_s=hold)


class FlashScheduler:
    """
    Thread dedicado para executar piscadas sem travar o loop da telemetria.
    Usa uma fila por prioridade: menor número = maior prioridade.
    """

    def __init__(
        self,
        scan_code: int,
        extended: bool = False,
        min_press_spacing_s: float = 0.12,
        max_per_minute: int = 40,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.scan_code = int(scan_code) if scan_code else 0
        self.extended = bool(extended)
        self.min_press_spacing_s = max(0.02, float(min_press_spacing_s))
        self.max_per_minute = max(1, int(max_per_minute))
        self.debug = debug

        self._log = logger or logging.getLogger("SmartFlasher")
        self._q: PriorityQueue[Tuple[int, float, FlashPattern, str]] = PriorityQueue()
        self._running = False
        self._t: Optional[threading.Thread] = None

        self._next_press_allowed = 0.0
        self._press_times: list[float] = []

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._t = threading.Thread(target=self._worker, name="FlashScheduler", daemon=True)
        self._t.start()

    def stop(self) -> None:
        self._running = False

    def update_key(self, scan_code: int, extended: bool = False) -> None:
        self.scan_code = int(scan_code) if scan_code else 0
        self.extended = bool(extended)

    def trigger(self, pattern: FlashPattern, priority: int, reason: str = "") -> None:
        pat = pattern.normalized()
        self._q.put((int(priority), time.time(), pat, str(reason)))

    def _rate_limit_ok(self, now: float) -> bool:
        cutoff = now - 60.0
        self._press_times = [t for t in self._press_times if t >= cutoff]
        return len(self._press_times) < self.max_per_minute

    def _worker(self) -> None:
        while self._running:
            try:
                pri, created, pat, reason = self._q.get(timeout=0.25)
            except Empty:
                continue

            if not self.scan_code:
                continue

            if self.debug and reason:
                self._log.info(f"[FLASH] pri={pri} pattern={pat} reason={reason}")

            for i in range(pat.flashes):
                if not self._running:
                    break

                now = time.time()
                if now < self._next_press_allowed:
                    time.sleep(self._next_press_allowed - now)

                now = time.time()
                if not self._rate_limit_ok(now):
                    if self.debug:
                        self._log.warning("[FLASH] rate limit atingido; padrão cancelado.")
                    break

                press_key_scan_code(self.scan_code, hold_s=pat.hold_s, extended=self.extended)
                self._press_times.append(time.time())
                self._next_press_allowed = time.time() + self.min_press_spacing_s

                if i < pat.flashes - 1:
                    wait = random.uniform(pat.min_interval_s, pat.max_interval_s)
                    time.sleep(max(0.0, wait))


DEFAULT_CONFIG: Dict[str, Any] = {
    "version": 2,
    "key_name": "P",
    "scan_code": 0x19,
    "key_extended": False,
    "global": {
        "enabled": True,
        "tick_hz": 20,
        "ignore_in_pits": True,
        "min_my_speed_kmh": 0.0,
        "global_press_spacing_s": 0.12,
        "max_flashes_per_min": 40,
        "debug": False,
    },
    "rules": {
        "pass_request": {
            "enabled": True,
            "priority": 2,
            "cooldown_s": 3.0,
            "lookahead_s": 6.0,
            "trigger_gap_s": 0.8,
            "require_diff_class": True,
            "min_rel_speed_kmh": 8.0,
            "ignore_target_off_track": True,
            "pattern": {"flashes": 3, "min_interval_s": 0.5, "max_interval_s": 1.2, "hold_s": 0.06},
        },
        "safety_approach": {
            "enabled": True,
            "priority": 1,
            "cooldown_s": 1.5,
            "lookahead_s": 6.0,
            "trigger_gap_s": 1.0,
            "closing_rate_s_per_s": 0.7,
            "pattern": {"flashes": 1, "min_interval_s": 0.0, "max_interval_s": 0.0, "hold_s": 0.06},
        },
        "hazard_ahead": {
            "enabled": True,
            "priority": 0,
            "cooldown_s": 4.0,
            "lookahead_s": 8.0,
            "trigger_gap_s": 2.0,
            "trigger_off_track": True,
            "trigger_slow_target": True,
            "target_speed_kmh_below": 60.0,
            "closing_rate_s_per_s": 1.2,
            "pattern": {"flashes": 1, "min_interval_s": 0.0, "max_interval_s": 0.0, "hold_s": 0.06},
        },
        "slow_car_ahead": {
            "enabled": True,
            "priority": 1,
            "cooldown_s": 4.0,
            "lookahead_s": 8.0,
            "trigger_gap_s": 1.5,
            "target_speed_kmh_below": 80.0,
            "only_if_on_track": True,
            "pattern": {"flashes": 2, "min_interval_s": 0.25, "max_interval_s": 0.45, "hold_s": 0.06},
        },
        "pit_exit_merge": {
            "enabled": True,
            "priority": 1,
            "cooldown_s": 10.0,
            "pattern": {"flashes": 2, "min_interval_s": 0.25, "max_interval_s": 0.45, "hold_s": 0.06},
        },
        "self_off_track_hazard": {
            "enabled": False,
            "priority": 0,
            "cooldown_s": 8.0,
            "min_speed_kmh": 30.0,
            "pattern": {"flashes": 2, "min_interval_s": 0.20, "max_interval_s": 0.35, "hold_s": 0.06},
        },
    },
}


class SmartFlasher:
    def __init__(self) -> None:
        self.ir = irsdk.IRSDK()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        self.last_session_update = -1
        self.car_classes: Dict[int, int] = {}
        self.track_length_m: float = 4000.0

        self._prev_gap: Dict[int, Tuple[float, float]] = {}
        self._cooldowns: Dict[Tuple[str, Optional[int]], float] = {}
        self._prev_on_pit_road: Optional[bool] = None
        self._prev_my_surface: Optional[int] = None

        self.log = logging.getLogger("SmartFlasher")
        if not self.log.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

        self.config: Dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))
        self.load_config()

        g = self.config.get("global", {})
        self.scheduler = FlashScheduler(
            scan_code=self.config.get("scan_code", 0),
            extended=self.config.get("key_extended", False),
            min_press_spacing_s=safe_float(g.get("global_press_spacing_s"), 0.12),
            max_per_minute=safe_int(g.get("max_flashes_per_min"), 40),
            debug=bool(g.get("debug", False)),
            logger=self.log,
        )

    # -----------------------
    # Telemetry utils (Release1-style safety)
    # -----------------------
    def _read_ir_value(self, key: str, default: Any = None) -> Any:
        try:
            if not getattr(self.ir, "is_initialized", False):
                self.ir.startup()
            return self.ir[key]
        except Exception:
            return default

    def _telemetry_connected(self) -> bool:
        if not getattr(self.ir, "is_initialized", False):
            return False
        if hasattr(self.ir, "is_connected"):
            return bool(self.ir.is_connected)
        return True

    # -----------------------
    # Config
    # -----------------------
    def load_config(self) -> None:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        if not os.path.exists(CONFIG_FILE):
            return

        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
        except Exception:
            return

        self.config = self._merge_and_migrate(loaded)

    def save_config(self) -> None:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            self.log.error(f"Falha ao salvar config em {CONFIG_FILE}: {e}")

    def _merge_and_migrate(self, loaded: Dict[str, Any]) -> Dict[str, Any]:
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))

        for k, v in loaded.items():
            if k in ("global", "rules"):
                continue
            cfg[k] = v

        if isinstance(loaded.get("global"), dict):
            cfg["global"].update(loaded["global"])

        if isinstance(loaded.get("rules"), dict):
            for rk, rv in loaded["rules"].items():
                if rk in cfg["rules"] and isinstance(rv, dict):
                    cfg["rules"][rk].update(rv)
                    if isinstance(rv.get("pattern"), dict):
                        cfg["rules"][rk]["pattern"].update(rv["pattern"])

        if "gap_trigger" in loaded and isinstance(cfg["rules"].get("pass_request"), dict):
            cfg["rules"]["pass_request"]["trigger_gap_s"] = safe_float(
                loaded.get("gap_trigger"), cfg["rules"]["pass_request"]["trigger_gap_s"]
            )
        if "max_flashes" in loaded:
            cfg["rules"]["pass_request"]["pattern"]["flashes"] = safe_int(
                loaded.get("max_flashes"), cfg["rules"]["pass_request"]["pattern"]["flashes"]
            )
        if "min_interval" in loaded:
            cfg["rules"]["pass_request"]["pattern"]["min_interval_s"] = safe_float(
                loaded.get("min_interval"), cfg["rules"]["pass_request"]["pattern"]["min_interval_s"]
            )
        if "max_interval" in loaded:
            cfg["rules"]["pass_request"]["pattern"]["max_interval_s"] = safe_float(
                loaded.get("max_interval"), cfg["rules"]["pass_request"]["pattern"]["max_interval_s"]
            )

        if "safety_flash" in loaded:
            cfg["rules"]["safety_approach"]["enabled"] = bool(loaded.get("safety_flash"))
        if "safety_threshold" in loaded:
            th = safe_float(loaded.get("safety_threshold"), 0.04)
            cfg["rules"]["safety_approach"]["closing_rate_s_per_s"] = max(0.1, th * 10.0)

        if "hazard_flash" in loaded:
            cfg["rules"]["hazard_ahead"]["enabled"] = bool(loaded.get("hazard_flash"))

        cfg["version"] = 2
        return cfg

    # -----------------------
    # Public controls
    # -----------------------
    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.scheduler.update_key(self.config.get("scan_code", 0), bool(self.config.get("key_extended", False)))
        self.scheduler.start()
        self.thread = threading.Thread(target=self._loop, name="SmartFlasherLoop", daemon=True)
        self.thread.start()
        self.log.info(f">> Flasher INICIADO (Tecla: {self.config.get('key_name', '?')})")

    def stop(self) -> None:
        self.running = False
        self.scheduler.stop()
        self.log.info(">> Flasher PARADO")

    def test_flash(self, rule_name: str = "pass_request") -> None:
        rules = self.config.get("rules", {})
        rule = rules.get(rule_name, {})
        pat_cfg = rule.get("pattern", {})
        pat = FlashPattern(
            flashes=safe_int(pat_cfg.get("flashes"), 3),
            min_interval_s=safe_float(pat_cfg.get("min_interval_s"), 0.4),
            max_interval_s=safe_float(pat_cfg.get("max_interval_s"), 0.7),
            hold_s=safe_float(pat_cfg.get("hold_s"), 0.06),
        )
        pri = safe_int(rule.get("priority"), 2)
        self.scheduler.trigger(pat, priority=pri, reason=f"TEST:{rule_name}")

    # -----------------------
    # SessionInfo parsing
    # -----------------------
    def _update_session_info(self) -> None:
        if not self._telemetry_connected():
            return

        try:
            siu = self.ir["SessionInfoUpdate"]
        except Exception:
            return

        if siu == self.last_session_update:
            return

        try:
            raw = self.ir.get_session_info()
            if not raw:
                return
            data = yaml.safe_load(raw)
        except Exception:
            return

        try:
            drivers = data.get("DriverInfo", {}).get("Drivers", [])
            classes = {}
            for d in drivers:
                try:
                    classes[int(d["CarIdx"])] = int(d["CarClassID"])
                except Exception:
                    continue
            self.car_classes = classes
        except Exception:
            pass

        try:
            weekend = data.get("WeekendInfo", {})
            tl = weekend.get("TrackLength")
            parsed = parse_track_length_to_m(tl)
            if parsed and parsed > 100.0:
                self.track_length_m = float(parsed)
        except Exception:
            pass

        self.last_session_update = siu

    # -----------------------
    # Gap estimation
    # -----------------------
    def _estimate_gap_s(
        self,
        my_pct: float,
        target_pct: float,
        my_speed_mps: float,
        target_speed_mps: float,
    ) -> Optional[float]:
        try:
            diff = float(target_pct) - float(my_pct)
        except Exception:
            return None

        if diff < -0.5:
            diff += 1.0
        if diff > 0.5:
            diff -= 1.0

        dist_m = diff * float(self.track_length_m)
        if dist_m <= 0.0:
            return None

        speed_ref = max(1.0, (float(my_speed_mps) + float(target_speed_mps)) / 2.0)
        return dist_m / speed_ref

    def _closing_rate(self, target_idx: int, gap_s: float, now: float) -> float:
        prev = self._prev_gap.get(target_idx)
        self._prev_gap[target_idx] = (gap_s, now)
        if not prev:
            return 0.0
        prev_gap, prev_t = prev
        dt = now - prev_t
        if dt <= 0.001:
            return 0.0
        return (prev_gap - gap_s) / dt

    # -----------------------
    # Cooldowns
    # -----------------------
    def _cooldown_ok(self, rule_name: str, target_idx: Optional[int], now: float) -> bool:
        until = self._cooldowns.get((rule_name, target_idx), 0.0)
        return now >= until

    def _set_cooldown(self, rule_name: str, target_idx: Optional[int], now: float, cooldown_s: float) -> None:
        self._cooldowns[(rule_name, target_idx)] = now + max(0.0, float(cooldown_s))

    # -----------------------
    # Main loop
    # -----------------------
    def _ir_get(self, name: str, default: Any = None) -> Any:
        return self._read_ir_value(name, default)

    def _loop(self) -> None:
        while self.running:
            if not self._telemetry_connected():
                self._prev_on_pit_road = None
                self._prev_my_surface = None
                try:
                    self.ir.startup()
                except Exception:
                    pass
                time.sleep(1.0)
                continue

            self._update_session_info()

            try:
                self.ir.freeze_var_buffer_latest()
            except Exception:
                time.sleep(0.2)
                continue

            cfg_global = self.config.get("global", {})
            if not cfg_global.get("enabled", True):
                time.sleep(0.2)
                continue

            tick_hz = max(5, safe_int(cfg_global.get("tick_hz"), 20))
            dt_sleep = 1.0 / float(tick_hz)

            is_replay = bool(self._ir_get("IsReplayPlaying", False))
            if is_replay:
                time.sleep(0.5)
                continue

            my_idx = self._ir_get("PlayerCarIdx", None)
            lap_pcts = self._ir_get("CarIdxLapDistPct", None)
            surfaces = self._ir_get("CarIdxTrackSurface", None)
            car_speeds = self._ir_get("CarIdxSpeed", None)
            my_speed = self._ir_get("Speed", None)
            on_pit_road = bool(self._ir_get("PlayerCarOnPitRoad", False))
            car_classes = self._ir_get("CarIdxClass", None)

            if my_idx is None or lap_pcts is None:
                time.sleep(dt_sleep)
                continue

            try:
                my_pct = lap_pcts[my_idx]
            except Exception:
                time.sleep(dt_sleep)
                continue

            if my_pct is None or my_pct == -1:
                time.sleep(dt_sleep)
                continue

            if my_speed is None:
                try:
                    my_speed = float(car_speeds[my_idx]) if car_speeds is not None else 0.0
                except Exception:
                    my_speed = 0.0

            my_speed_kmh = ms_to_kmh(my_speed)
            min_my_speed_kmh = safe_float(cfg_global.get("min_my_speed_kmh"), 0.0)
            if min_my_speed_kmh > 0 and my_speed_kmh < min_my_speed_kmh:
                time.sleep(dt_sleep)
                continue

            ignore_in_pits = bool(cfg_global.get("ignore_in_pits", True))
            if ignore_in_pits and on_pit_road:
                self._prev_on_pit_road = on_pit_road
                time.sleep(dt_sleep)
                continue

            if self._prev_on_pit_road is None:
                self._prev_on_pit_road = on_pit_road
            else:
                if self._prev_on_pit_road is True and on_pit_road is False:
                    self._eval_pit_exit_merge(time.time())
                self._prev_on_pit_road = on_pit_road

            self._eval_self_off_track(
                now=time.time(),
                my_idx=int(my_idx),
                surfaces=surfaces,
                my_speed_kmh=float(my_speed_kmh),
                player_surface=self._ir_get("PlayerTrackSurface", None),
            )

            if car_speeds is None:
                time.sleep(dt_sleep)
                continue

            my_class = self._resolve_car_class(int(my_idx), car_classes)

            now = time.time()

            lookahead_s = 0.0
            rules: Dict[str, Any] = self.config.get("rules", {})
            for rcfg in rules.values():
                if isinstance(rcfg, dict) and rcfg.get("enabled", False):
                    lookahead_s = max(lookahead_s, safe_float(rcfg.get("lookahead_s"), 0.0))
            if lookahead_s <= 0:
                lookahead_s = 6.0

            candidates: list[tuple[float, int]] = []
            for i, pct in enumerate(lap_pcts):
                if i == my_idx:
                    continue
                if pct is None or pct == -1:
                    continue
                try:
                    ts = float(car_speeds[i])
                except Exception:
                    continue
                if ts < 0:
                    continue
                target_surface = None
                if surfaces is not None:
                    try:
                        target_surface = int(surfaces[i])
                    except Exception:
                        target_surface = None
                if target_surface in (TRACK_SURFACE_PIT_STALL, TRACK_SURFACE_PIT_ROAD):
                    continue

                gap_s = self._estimate_gap_s(my_pct, pct, float(my_speed), ts)
                if gap_s is None:
                    continue
                if gap_s <= 0 or gap_s > lookahead_s:
                    continue
                candidates.append((float(gap_s), int(i)))

            candidates.sort(key=lambda item: item[0])
            for gap_s, target_idx in candidates:
                self._eval_rules_for_target(
                    target_idx=target_idx,
                    gap_s=gap_s,
                    my_idx=int(my_idx),
                    my_class=int(my_class),
                    my_speed_mps=float(my_speed),
                    now=now,
                    surfaces=surfaces,
                    car_speeds=car_speeds,
                )

            time.sleep(dt_sleep)

    def _resolve_car_class(self, car_idx: int, car_classes: Any) -> int:
        if isinstance(car_classes, (list, tuple)):
            try:
                value = car_classes[car_idx]
                if value is not None and value != -1:
                    return int(value)
            except Exception:
                pass
        return self.car_classes.get(car_idx, -1)

    # -----------------------
    # Rule evaluation
    # -----------------------
    def _pattern_from_cfg(self, rule_cfg: Dict[str, Any]) -> FlashPattern:
        p = rule_cfg.get("pattern", {}) if isinstance(rule_cfg.get("pattern"), dict) else {}
        return FlashPattern(
            flashes=safe_int(p.get("flashes"), 1),
            min_interval_s=safe_float(p.get("min_interval_s"), 0.2),
            max_interval_s=safe_float(p.get("max_interval_s"), 0.4),
            hold_s=safe_float(p.get("hold_s"), 0.06),
        ).normalized()

    def _eval_pit_exit_merge(self, now: float) -> None:
        rcfg = self.config.get("rules", {}).get("pit_exit_merge", {})
        if not isinstance(rcfg, dict) or not rcfg.get("enabled", False):
            return
        if not self._cooldown_ok("pit_exit_merge", None, now):
            return
        pat = self._pattern_from_cfg(rcfg)
        pri = safe_int(rcfg.get("priority"), 1)
        self.scheduler.trigger(pat, priority=pri, reason="PIT_EXIT_MERGE")
        self._set_cooldown("pit_exit_merge", None, now, safe_float(rcfg.get("cooldown_s"), 10.0))

    def _eval_self_off_track(
        self,
        now: float,
        my_idx: int,
        surfaces: Any,
        my_speed_kmh: float,
        player_surface: Any,
    ) -> None:
        rcfg = self.config.get("rules", {}).get("self_off_track_hazard", {})
        if not isinstance(rcfg, dict) or not rcfg.get("enabled", False):
            return

        my_surface = None
        if player_surface is not None:
            try:
                my_surface = int(player_surface)
            except Exception:
                my_surface = None

        if my_surface is None and surfaces is not None:
            try:
                my_surface = int(surfaces[my_idx])
            except Exception:
                my_surface = None

        if my_surface is None:
            return

        if self._prev_my_surface is None:
            self._prev_my_surface = my_surface
            return

        transitioned_off = self._prev_my_surface != TRACK_SURFACE_OFF and my_surface == TRACK_SURFACE_OFF
        self._prev_my_surface = my_surface
        if not transitioned_off:
            return

        if my_speed_kmh < safe_float(rcfg.get("min_speed_kmh"), 30.0):
            return

        if not self._cooldown_ok("self_off_track_hazard", None, now):
            return

        pat = self._pattern_from_cfg(rcfg)
        pri = safe_int(rcfg.get("priority"), 0)
        self.scheduler.trigger(pat, priority=pri, reason=f"SELF_OFF_TRACK v={my_speed_kmh:.0f}kmh")
        self._set_cooldown("self_off_track_hazard", None, now, safe_float(rcfg.get("cooldown_s"), 8.0))

    def _eval_rules_for_target(
        self,
        target_idx: int,
        gap_s: float,
        my_idx: int,
        my_class: int,
        my_speed_mps: float,
        now: float,
        surfaces: Any,
        car_speeds: Any,
    ) -> None:
        try:
            target_speed_mps = float(car_speeds[target_idx])
        except Exception:
            target_speed_mps = 0.0
        target_speed_kmh = ms_to_kmh(target_speed_mps)
        rel_speed_kmh = ms_to_kmh(my_speed_mps - target_speed_mps)

        target_surface = None
        if surfaces is not None:
            try:
                target_surface = int(surfaces[target_idx])
            except Exception:
                target_surface = None
        if target_surface in (TRACK_SURFACE_PIT_STALL, TRACK_SURFACE_PIT_ROAD):
            return
        is_off_track = target_surface == TRACK_SURFACE_OFF

        closing_rate = self._closing_rate(target_idx, gap_s, now)

        self._rule_hazard_ahead(target_idx, gap_s, target_speed_kmh, is_off_track, closing_rate, now)
        self._rule_slow_car_ahead(target_idx, gap_s, target_speed_kmh, target_surface, now)
        self._rule_safety_approach(target_idx, gap_s, closing_rate, now)
        self._rule_pass_request(target_idx, gap_s, rel_speed_kmh, my_class, now, is_off_track)

    def _rule_hazard_ahead(
        self,
        target_idx: int,
        gap_s: float,
        target_speed_kmh: float,
        is_off_track: bool,
        closing_rate: float,
        now: float,
    ) -> None:
        rcfg = self.config.get("rules", {}).get("hazard_ahead", {})
        if not isinstance(rcfg, dict) or not rcfg.get("enabled", False):
            return

        if gap_s > safe_float(rcfg.get("trigger_gap_s"), 2.0):
            return
        if not self._cooldown_ok("hazard_ahead", target_idx, now):
            return

        trig_off = bool(rcfg.get("trigger_off_track", True)) and is_off_track
        trig_slow = bool(rcfg.get("trigger_slow_target", True)) and target_speed_kmh < safe_float(
            rcfg.get("target_speed_kmh_below"), 60.0
        )
        trig_closing = closing_rate > safe_float(rcfg.get("closing_rate_s_per_s"), 1.2)

        if trig_off or trig_slow or trig_closing:
            pat = self._pattern_from_cfg(rcfg)
            pri = safe_int(rcfg.get("priority"), 0)
            self.scheduler.trigger(
                pat,
                priority=pri,
                reason=(
                    f"HAZARD idx={target_idx} gap={gap_s:.2f}s v={target_speed_kmh:.0f}kmh close={closing_rate:.2f}"
                ),
            )
            self._set_cooldown("hazard_ahead", target_idx, now, safe_float(rcfg.get("cooldown_s"), 4.0))

    def _rule_safety_approach(self, target_idx: int, gap_s: float, closing_rate: float, now: float) -> None:
        rcfg = self.config.get("rules", {}).get("safety_approach", {})
        if not isinstance(rcfg, dict) or not rcfg.get("enabled", False):
            return

        if gap_s > safe_float(rcfg.get("trigger_gap_s"), 1.0):
            return
        if closing_rate <= safe_float(rcfg.get("closing_rate_s_per_s"), 0.7):
            return
        if not self._cooldown_ok("safety_approach", target_idx, now):
            return

        pat = self._pattern_from_cfg(rcfg)
        pri = safe_int(rcfg.get("priority"), 1)
        self.scheduler.trigger(pat, priority=pri, reason=f"SAFETY idx={target_idx} gap={gap_s:.2f}s close={closing_rate:.2f}")
        self._set_cooldown("safety_approach", target_idx, now, safe_float(rcfg.get("cooldown_s"), 1.5))

    def _rule_pass_request(
        self,
        target_idx: int,
        gap_s: float,
        rel_speed_kmh: float,
        my_class: int,
        now: float,
        is_off_track: bool,
    ) -> None:
        rcfg = self.config.get("rules", {}).get("pass_request", {})
        if not isinstance(rcfg, dict) or not rcfg.get("enabled", False):
            return

        if gap_s > safe_float(rcfg.get("trigger_gap_s"), 0.8):
            return
        if rel_speed_kmh < safe_float(rcfg.get("min_rel_speed_kmh"), 8.0):
            return
        if bool(rcfg.get("ignore_target_off_track", True)) and is_off_track:
            return

        require_diff_class = bool(rcfg.get("require_diff_class", True))
        if require_diff_class:
            tgt_class = self._resolve_car_class(int(target_idx), self._ir_get("CarIdxClass", None))
            if tgt_class is None or my_class == -1 or int(tgt_class) == int(my_class):
                return

        if not self._cooldown_ok("pass_request", target_idx, now):
            return

        pat = self._pattern_from_cfg(rcfg)
        pri = safe_int(rcfg.get("priority"), 2)
        self.scheduler.trigger(pat, priority=pri, reason=f"PASS_REQUEST idx={target_idx} gap={gap_s:.2f}s relV={rel_speed_kmh:.0f}kmh")
        self._set_cooldown("pass_request", target_idx, now, safe_float(rcfg.get("cooldown_s"), 3.0))

    def _rule_slow_car_ahead(
        self,
        target_idx: int,
        gap_s: float,
        target_speed_kmh: float,
        target_surface: Optional[int],
        now: float,
    ) -> None:
        rcfg = self.config.get("rules", {}).get("slow_car_ahead", {})
        if not isinstance(rcfg, dict) or not rcfg.get("enabled", False):
            return

        if gap_s > safe_float(rcfg.get("trigger_gap_s"), 1.5):
            return

        if target_speed_kmh >= safe_float(rcfg.get("target_speed_kmh_below"), 80.0):
            return

        only_on_track = bool(rcfg.get("only_if_on_track", True))
        if only_on_track and target_surface not in (TRACK_SURFACE_ON, None):
            return

        if not self._cooldown_ok("slow_car_ahead", target_idx, now):
            return

        pat = self._pattern_from_cfg(rcfg)
        pri = safe_int(rcfg.get("priority"), 1)
        self.scheduler.trigger(pat, priority=pri, reason=f"SLOW_CAR idx={target_idx} gap={gap_s:.2f}s v={target_speed_kmh:.0f}kmh")
        self._set_cooldown("slow_car_ahead", target_idx, now, safe_float(rcfg.get("cooldown_s"), 4.0))


class SmartFlasherUI:
    def __init__(self, flasher: SmartFlasher) -> None:
        self.flasher = flasher
        self.root = tk.Tk()
        self.root.title("Smart Flasher")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.status_var = tk.StringVar(value="Stopped")

        self.key_name_var = tk.StringVar()
        self.scan_code_var = tk.StringVar()
        self.key_extended_var = tk.BooleanVar()

        self.global_vars: Dict[str, tk.Variable] = {
            "enabled": tk.BooleanVar(),
            "tick_hz": tk.StringVar(),
            "ignore_in_pits": tk.BooleanVar(),
            "min_my_speed_kmh": tk.StringVar(),
            "global_press_spacing_s": tk.StringVar(),
            "max_flashes_per_min": tk.StringVar(),
            "debug": tk.BooleanVar(),
        }

        self.rule_vars: Dict[str, Dict[str, tk.Variable]] = {}

        self._build_ui()
        self._load_from_config()

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        global_tab = ttk.Frame(notebook)
        rules_tab = ttk.Frame(notebook)
        notebook.add(global_tab, text="Global")
        notebook.add(rules_tab, text="Rules")

        self._build_global_tab(global_tab)
        self._build_rules_tab(rules_tab)

        footer = ttk.Frame(self.root)
        footer.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(footer, textvariable=self.status_var).pack(side="left")

        ttk.Button(footer, text="Save", command=self._save_config).pack(side="right", padx=4)
        ttk.Button(footer, text="Stop", command=self._stop_flasher).pack(side="right", padx=4)
        ttk.Button(footer, text="Start", command=self._start_flasher).pack(side="right", padx=4)

    def _build_global_tab(self, parent: ttk.Frame) -> None:
        key_frame = ttk.Labelframe(parent, text="Key Settings")
        key_frame.pack(fill="x", padx=6, pady=6)

        self._add_entry(key_frame, "Key name", self.key_name_var, 0)
        self._add_entry(key_frame, "Scan code (hex or int)", self.scan_code_var, 1)
        self._add_bool(key_frame, "Extended key", self.key_extended_var, 2)

        global_frame = ttk.Labelframe(parent, text="Global Settings")
        global_frame.pack(fill="x", padx=6, pady=6)

        self._add_bool(global_frame, "Enabled", self.global_vars["enabled"], 0)
        self._add_entry(global_frame, "Tick Hz", self.global_vars["tick_hz"], 1)
        self._add_bool(global_frame, "Ignore in pits", self.global_vars["ignore_in_pits"], 2)
        self._add_entry(global_frame, "Min speed km/h", self.global_vars["min_my_speed_kmh"], 3)
        self._add_entry(global_frame, "Press spacing (s)", self.global_vars["global_press_spacing_s"], 4)
        self._add_entry(global_frame, "Max flashes/min", self.global_vars["max_flashes_per_min"], 5)
        self._add_bool(global_frame, "Debug logging", self.global_vars["debug"], 6)

        control_frame = ttk.Labelframe(parent, text="Test Flash")
        control_frame.pack(fill="x", padx=6, pady=6)

        self.test_rule_var = tk.StringVar(value="pass_request")
        rules = list(self.flasher.config.get("rules", {}).keys())
        rule_select = ttk.Combobox(control_frame, textvariable=self.test_rule_var, values=rules, state="readonly", width=24)
        rule_select.grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Button(control_frame, text="Trigger", command=self._test_flash).grid(row=0, column=1, padx=4, pady=4)

    def _build_rules_tab(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, borderwidth=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._build_rule_frame(
            scroll_frame,
            "pass_request",
            "Pass Request",
            fields=[
                ("enabled", "Enabled", "bool"),
                ("priority", "Priority", "int"),
                ("cooldown_s", "Cooldown (s)", "float"),
                ("lookahead_s", "Lookahead (s)", "float"),
                ("trigger_gap_s", "Trigger gap (s)", "float"),
                ("min_rel_speed_kmh", "Min rel speed km/h", "float"),
                ("require_diff_class", "Require diff class", "bool"),
                ("ignore_target_off_track", "Ignore target off track", "bool"),
                ("pattern.flashes", "Flashes", "int"),
                ("pattern.min_interval_s", "Min interval (s)", "float"),
                ("pattern.max_interval_s", "Max interval (s)", "float"),
                ("pattern.hold_s", "Hold (s)", "float"),
            ],
        )

        self._build_rule_frame(
            scroll_frame,
            "safety_approach",
            "Safety Approach",
            fields=[
                ("enabled", "Enabled", "bool"),
                ("priority", "Priority", "int"),
                ("cooldown_s", "Cooldown (s)", "float"),
                ("lookahead_s", "Lookahead (s)", "float"),
                ("trigger_gap_s", "Trigger gap (s)", "float"),
                ("closing_rate_s_per_s", "Closing rate (s/s)", "float"),
                ("pattern.flashes", "Flashes", "int"),
                ("pattern.min_interval_s", "Min interval (s)", "float"),
                ("pattern.max_interval_s", "Max interval (s)", "float"),
                ("pattern.hold_s", "Hold (s)", "float"),
            ],
        )

        self._build_rule_frame(
            scroll_frame,
            "hazard_ahead",
            "Hazard Ahead",
            fields=[
                ("enabled", "Enabled", "bool"),
                ("priority", "Priority", "int"),
                ("cooldown_s", "Cooldown (s)", "float"),
                ("lookahead_s", "Lookahead (s)", "float"),
                ("trigger_gap_s", "Trigger gap (s)", "float"),
                ("trigger_off_track", "Trigger off track", "bool"),
                ("trigger_slow_target", "Trigger slow target", "bool"),
                ("target_speed_kmh_below", "Target speed km/h", "float"),
                ("closing_rate_s_per_s", "Closing rate (s/s)", "float"),
                ("pattern.flashes", "Flashes", "int"),
                ("pattern.min_interval_s", "Min interval (s)", "float"),
                ("pattern.max_interval_s", "Max interval (s)", "float"),
                ("pattern.hold_s", "Hold (s)", "float"),
            ],
        )

        self._build_rule_frame(
            scroll_frame,
            "slow_car_ahead",
            "Slow Car Ahead",
            fields=[
                ("enabled", "Enabled", "bool"),
                ("priority", "Priority", "int"),
                ("cooldown_s", "Cooldown (s)", "float"),
                ("lookahead_s", "Lookahead (s)", "float"),
                ("trigger_gap_s", "Trigger gap (s)", "float"),
                ("target_speed_kmh_below", "Target speed km/h", "float"),
                ("only_if_on_track", "Only if on track", "bool"),
                ("pattern.flashes", "Flashes", "int"),
                ("pattern.min_interval_s", "Min interval (s)", "float"),
                ("pattern.max_interval_s", "Max interval (s)", "float"),
                ("pattern.hold_s", "Hold (s)", "float"),
            ],
        )

        self._build_rule_frame(
            scroll_frame,
            "pit_exit_merge",
            "Pit Exit Merge",
            fields=[
                ("enabled", "Enabled", "bool"),
                ("priority", "Priority", "int"),
                ("cooldown_s", "Cooldown (s)", "float"),
                ("pattern.flashes", "Flashes", "int"),
                ("pattern.min_interval_s", "Min interval (s)", "float"),
                ("pattern.max_interval_s", "Max interval (s)", "float"),
                ("pattern.hold_s", "Hold (s)", "float"),
            ],
        )

        self._build_rule_frame(
            scroll_frame,
            "self_off_track_hazard",
            "Self Off-Track Hazard",
            fields=[
                ("enabled", "Enabled", "bool"),
                ("priority", "Priority", "int"),
                ("cooldown_s", "Cooldown (s)", "float"),
                ("min_speed_kmh", "Min speed km/h", "float"),
                ("pattern.flashes", "Flashes", "int"),
                ("pattern.min_interval_s", "Min interval (s)", "float"),
                ("pattern.max_interval_s", "Max interval (s)", "float"),
                ("pattern.hold_s", "Hold (s)", "float"),
            ],
        )

    def _build_rule_frame(self, parent: ttk.Frame, rule_key: str, title: str, fields: list[tuple[str, str, str]]) -> None:
        frame = ttk.Labelframe(parent, text=title)
        frame.pack(fill="x", padx=6, pady=6)

        self.rule_vars[rule_key] = {}
        row = 0
        for field_key, label, field_type in fields:
            var: tk.Variable
            if field_type == "bool":
                var = tk.BooleanVar()
                self._add_bool(frame, label, var, row)
                row += 1
            else:
                var = tk.StringVar()
                self._add_entry(frame, label, var, row)
                row += 1
            self.rule_vars[rule_key][field_key] = var

    def _add_entry(self, parent: ttk.Frame, label: str, var: tk.Variable, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky="w", padx=4, pady=2)

    def _add_bool(self, parent: ttk.Frame, label: str, var: tk.Variable, row: int) -> None:
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", padx=4, pady=2)

    def _load_from_config(self) -> None:
        cfg = self.flasher.config
        self.key_name_var.set(str(cfg.get("key_name", "P")))
        self.scan_code_var.set(hex(int(cfg.get("scan_code", 0))))
        self.key_extended_var.set(bool(cfg.get("key_extended", False)))

        gcfg = cfg.get("global", {})
        self.global_vars["enabled"].set(bool(gcfg.get("enabled", True)))
        self.global_vars["tick_hz"].set(str(gcfg.get("tick_hz", 20)))
        self.global_vars["ignore_in_pits"].set(bool(gcfg.get("ignore_in_pits", True)))
        self.global_vars["min_my_speed_kmh"].set(str(gcfg.get("min_my_speed_kmh", 0.0)))
        self.global_vars["global_press_spacing_s"].set(str(gcfg.get("global_press_spacing_s", 0.12)))
        self.global_vars["max_flashes_per_min"].set(str(gcfg.get("max_flashes_per_min", 40)))
        self.global_vars["debug"].set(bool(gcfg.get("debug", False)))

        for rule_key, fields in self.rule_vars.items():
            rcfg = cfg.get("rules", {}).get(rule_key, {})
            for field_key, var in fields.items():
                if field_key.startswith("pattern."):
                    value = rcfg.get("pattern", {}).get(field_key.split(".", 1)[1])
                else:
                    value = rcfg.get(field_key)

                if isinstance(var, tk.BooleanVar):
                    var.set(bool(value))
                else:
                    var.set("" if value is None else str(value))

    def _save_config(self, show_message: bool = True) -> None:
        new_cfg = json.loads(json.dumps(self.flasher.config))
        new_cfg["key_name"] = self.key_name_var.get().strip() or new_cfg.get("key_name", "P")
        new_cfg["scan_code"] = self._parse_int(self.scan_code_var.get(), new_cfg.get("scan_code", 0))
        new_cfg["key_extended"] = bool(self.key_extended_var.get())

        gcfg = new_cfg.setdefault("global", {})
        gcfg["enabled"] = bool(self.global_vars["enabled"].get())
        gcfg["tick_hz"] = self._parse_int(self.global_vars["tick_hz"].get(), gcfg.get("tick_hz", 20))
        gcfg["ignore_in_pits"] = bool(self.global_vars["ignore_in_pits"].get())
        gcfg["min_my_speed_kmh"] = safe_float(self.global_vars["min_my_speed_kmh"].get(), gcfg.get("min_my_speed_kmh", 0.0))
        gcfg["global_press_spacing_s"] = safe_float(
            self.global_vars["global_press_spacing_s"].get(), gcfg.get("global_press_spacing_s", 0.12)
        )
        gcfg["max_flashes_per_min"] = self._parse_int(
            self.global_vars["max_flashes_per_min"].get(), gcfg.get("max_flashes_per_min", 40)
        )
        gcfg["debug"] = bool(self.global_vars["debug"].get())

        rules_cfg = new_cfg.setdefault("rules", {})
        for rule_key, fields in self.rule_vars.items():
            rcfg = rules_cfg.setdefault(rule_key, {})
            for field_key, var in fields.items():
                if field_key.startswith("pattern."):
                    pat = rcfg.setdefault("pattern", {})
                    key = field_key.split(".", 1)[1]
                    pat[key] = self._parse_value(var, pat.get(key))
                else:
                    rcfg[field_key] = self._parse_value(var, rcfg.get(field_key))

        self.flasher.config = new_cfg
        self.flasher.save_config()

        gcfg = self.flasher.config.get("global", {})
        self.flasher.scheduler.min_press_spacing_s = safe_float(gcfg.get("global_press_spacing_s"), 0.12)
        self.flasher.scheduler.max_per_minute = safe_int(gcfg.get("max_flashes_per_min"), 40)
        self.flasher.scheduler.debug = bool(gcfg.get("debug", False))
        self.flasher.scheduler.update_key(self.flasher.config.get("scan_code", 0), bool(self.flasher.config.get("key_extended", False)))

        if show_message:
            messagebox.showinfo("Smart Flasher", "Configuration saved.")

    def _parse_value(self, var: tk.Variable, default: Any) -> Any:
        if isinstance(var, tk.BooleanVar):
            return bool(var.get())
        value = str(var.get()).strip()
        if value == "":
            return default
        if isinstance(default, int):
            return self._parse_int(value, default)
        if isinstance(default, float):
            return safe_float(value, default)
        try:
            if "." in value:
                return float(value)
            return int(value)
        except Exception:
            return value

    def _parse_int(self, value: Any, default: int) -> int:
        try:
            return int(str(value), 0)
        except Exception:
            return safe_int(value, default)

    def _start_flasher(self) -> None:
        if not self.flasher.running:
            self._save_config(show_message=False)
            self.flasher.start()
        self.status_var.set("Running")

    def _stop_flasher(self) -> None:
        if self.flasher.running:
            self.flasher.stop()
        self.status_var.set("Stopped")

    def _test_flash(self) -> None:
        rule_name = self.test_rule_var.get().strip() or "pass_request"
        self.flasher.test_flash(rule_name)

    def _on_close(self) -> None:
        self._stop_flasher()
        try:
            self.flasher.ir.shutdown()
        except Exception:
            pass
        self.root.destroy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Flasher controller")
    parser.add_argument("--headless", action="store_true", help="Run without the Tkinter UI")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    flasher = SmartFlasher()
    if not args.headless:
        ui = SmartFlasherUI(flasher)
        ui.run()
        return 0

    flasher.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping SmartFlasher...")
    finally:
        flasher.stop()
        try:
            flasher.ir.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
