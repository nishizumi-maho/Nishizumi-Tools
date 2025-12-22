# coding: utf-8
"""
Fuel Overlay Pro (iRacing)
--------------------------

An always-on-top overlay (CustomTkinter) that provides:

Fuel & Race
- Estimates fuel burn per lap from telemetry (lap samples).
- Ignores yellow-flag laps (optional) to avoid polluting burn estimates.
- Shows current fuel, laps possible, laps remaining, fuel needed to finish (+ margin), and fuel to add.

Pit Timing Advisor (heuristic)
- Continuously evaluates pit options for: NOW / +1 / +2 / +3 laps (configurable).
- Estimates "clean air success %" based on gaps to nearby cars after a pit stop (heuristic, not ML).
- Accounts for variable pit time from fuel amount (fill-rate) and optional tire service time.
- Optional auto-calibration of fuel fill-rate / base loss / tire time from observed pit stops.

Assistants (heuristics, configurable)
- Wetness-aware "line + tire decision" assistant (rain transition brain).
- Driver-craft risk radar (incident risk).

Macros & Hotkeys (optional)
- Increase/decrease safety margin laps.
- Apply pit plan options via chat macro (#fuel ...).

Requirements
- customtkinter
- keyboard (optional, for hotkeys/macros)
- pyirsdk (recommended) -> pip install pyirsdk

Run:
    python fuel_overlay_pro.py

Notes
- This is NOT an ML model. Everything is heuristics using telemetry and configurable parameters.
- Many telemetry variables are build-dependent. The code uses safe access and degrades gracefully.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import customtkinter as ctk

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import keyboard  # type: ignore
except Exception:
    keyboard = None

# pyirsdk exposes module as `irsdk`
try:
    import irsdk  # type: ignore
except Exception:
    irsdk = None

APP_NAME = "FuelOverlayPro"
CONFIG_FILENAME = "config.json"


# ============================================================
# Config paths (portable + user config)
# ============================================================

def _is_windows() -> bool:
    return os.name == "nt"


def _user_config_dir() -> Path:
    """User-writable config directory (good for PyInstaller builds too)."""
    if _is_windows():
        base = os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    # Linux / others
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else (Path.home() / ".config")
    return base / APP_NAME


def _portable_config_dir(script_dir: Path) -> Path:
    return script_dir / "configs"


def _get_config_paths(portable: bool) -> Tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    if portable:
        cfg_dir = _portable_config_dir(script_dir)
    else:
        cfg_dir = _user_config_dir()
    cfg_file = cfg_dir / CONFIG_FILENAME
    return cfg_dir, cfg_file


def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def _deep_merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursive merge (dst <- src). Keeps defaults from dst and overrides with src."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge_dict(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


# ============================================================
# Flags helpers
# ============================================================

# Fallback bits for SessionFlags (if irsdk.Flags doesn't exist)
_YELLOW_MASK_FALLBACK = (
    0x0008  # yellow
    | 0x0100  # yellow_waving
    | 0x4000  # caution
    | 0x8000  # caution_waving
    | 0x2000  # random_waving
)


def is_yellow_flag(session_flags: int) -> bool:
    if session_flags is None:
        return False

    try:
        if irsdk and hasattr(irsdk, "Flags"):
            mask = (
                irsdk.Flags.yellow
                | irsdk.Flags.yellow_waving
                | irsdk.Flags.caution
                | irsdk.Flags.caution_waving
                | irsdk.Flags.random_waving
            )
            return bool(int(session_flags) & int(mask))
    except Exception:
        pass

    return bool(int(session_flags) & _YELLOW_MASK_FALLBACK)


# Pit service bitfield (PitSvFlags) fallback.
# These values are documented in irsdk_defines.h and widely referenced by iRacing telemetry tooling:
PITSVC_LF = 0x0001
PITSVC_RF = 0x0002
PITSVC_LR = 0x0004
PITSVC_RR = 0x0008
PITSVC_FUEL = 0x0010
PITSVC_WINDSHIELD = 0x0020
PITSVC_FAST_REPAIR = 0x0040


def pit_flags_tires_selected(flags: Optional[int]) -> Optional[bool]:
    if flags is None:
        return None
    try:
        f = int(flags)
    except Exception:
        return None
    return bool(f & (PITSVC_LF | PITSVC_RF | PITSVC_LR | PITSVC_RR))


# ============================================================
# Windows helpers (foreground + clipboard)
# ============================================================

def is_iracing_foreground(window_substring: str = "iracing") -> bool:
    """Return True if foreground window title appears to be iRacing.

    If not Windows (or on any failure), returns True (non-blocking).
    """
    if not _is_windows():
        return True
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False

        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        title = (buff.value or "").lower()
        return window_substring.lower() in title
    except Exception:
        return True


def copy_to_clipboard(text: str) -> bool:
    if not _is_windows():
        return False

    # PowerShell (more robust)
    try:
        ps = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Set-Clipboard -Value @'\n" + text + "\n'@",
        ]
        subprocess.run(ps, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return True
    except Exception:
        pass

    # clip fallback
    try:
        p = subprocess.Popen("clip", stdin=subprocess.PIPE, shell=True)
        p.communicate(text.encode("utf-8", errors="ignore"))
        return True
    except Exception:
        return False


# ============================================================
# Chat injector (for macros)
# ============================================================

class ChatInjector:
    def __init__(
        self,
        *,
        chat_key: str = "t",
        open_delay_s: float = 0.02,
        injection: str = "type",  # type | clipboard
        typing_interval_s: float = 0.001,
        require_iracing_foreground: bool = True,
        iracing_window_substring: str = "iracing",
        debounce_ms: int = 200,
    ) -> None:
        self.chat_key = str(chat_key or "t")
        self.open_delay_s = float(open_delay_s)
        self.injection = str(injection or "type").lower()
        self.typing_interval_s = float(typing_interval_s)
        self.require_iracing_foreground = bool(require_iracing_foreground)
        self.iracing_window_substring = str(iracing_window_substring or "iracing")
        self.debounce_ms = int(debounce_ms)
        self._last_send_t = 0.0

    def send(self, text: str) -> bool:
        if not text or keyboard is None:
            return False

        now = time.monotonic()
        if self.debounce_ms > 0 and (now - self._last_send_t) * 1000.0 < self.debounce_ms:
            return False

        if self.require_iracing_foreground and not is_iracing_foreground(self.iracing_window_substring):
            return False

        try:
            keyboard.press_and_release(self.chat_key)
            time.sleep(max(0.0, self.open_delay_s))

            if self.injection == "clipboard":
                if copy_to_clipboard(text):
                    keyboard.press_and_release("ctrl+v")
                else:
                    keyboard.write(text, delay=self.typing_interval_s)
            else:
                keyboard.write(text, delay=self.typing_interval_s)

            time.sleep(0.005)
            keyboard.press_and_release("enter")

            self._last_send_t = now
            return True
        except Exception:
            return False


class _SafeFormatDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"


# ============================================================
# Rolling utilities
# ============================================================

class RollingWindow:
    def __init__(self, maxlen: int):
        from collections import deque

        self.data: Deque[float] = deque(maxlen=maxlen)

    def add(self, v: float) -> None:
        try:
            self.data.append(float(v))
        except Exception:
            pass

    def values(self) -> List[float]:
        return list(self.data)

    def mean(self) -> Optional[float]:
        vals = self.values()
        if not vals:
            return None
        return sum(vals) / len(vals)

    def median(self) -> Optional[float]:
        vals = self.values()
        if not vals:
            return None
        try:
            return float(statistics.median(vals))
        except Exception:
            return None

    def stdev(self) -> Optional[float]:
        vals = self.values()
        if len(vals) < 2:
            return None
        try:
            return float(statistics.pstdev(vals))
        except Exception:
            return None


class RollingEvents:
    """Store event timestamps and return counts over a time window."""
    def __init__(self):
        from collections import deque
        self.ts: Deque[float] = deque()

    def add(self, t: float) -> None:
        self.ts.append(float(t))

    def count_last(self, seconds: float, now: float) -> int:
        cutoff = now - float(seconds)
        while self.ts and self.ts[0] < cutoff:
            self.ts.popleft()
        return len(self.ts)


# ============================================================
# Lap history (fuel)
# ============================================================

@dataclass
class LapSample:
    lap_number: int
    fuel_used: float
    lap_time: float
    timestamp: float


class FuelHistory:
    """Collects per-lap samples and produces burn estimates.

    Adds a 'current lap projection' so the overlay can show a burn estimate
    even before the first clean lap sample exists.
    """

    def __init__(
        self,
        ignore_yellow: bool = True,
        min_lap_time_s: float = 20.0,
        max_reasonable_burn_per_lap: float = 30.0,
        refuel_delta_threshold: float = 0.30,
        proj_min_progress: float = 0.25,
    ):
        self.ignore_yellow = ignore_yellow
        self.min_lap_time_s = float(min_lap_time_s)
        self.max_reasonable_burn_per_lap = float(max_reasonable_burn_per_lap)
        self.refuel_delta_threshold = float(refuel_delta_threshold)
        self.proj_min_progress = float(proj_min_progress)

        self.samples: List[LapSample] = []

        self._last_lap_completed: Optional[int] = None
        self._lap_start_time: Optional[float] = None
        self._lap_start_fuel: Optional[float] = None
        self._lap_start_dist_pct: Optional[float] = None

        self._prev_fuel: Optional[float] = None

        self._lap_has_yellow = False
        self._lap_has_pit = False
        self._lap_has_offtrack = False

        # rolling burn projection within current lap
        self._proj_burn_hist = RollingWindow(50)

    def reset(self) -> None:
        self.samples.clear()
        self._last_lap_completed = None
        self._lap_start_time = None
        self._lap_start_fuel = None
        self._lap_start_dist_pct = None
        self._prev_fuel = None
        self._lap_has_yellow = False
        self._lap_has_pit = False
        self._lap_has_offtrack = False
        self._proj_burn_hist = RollingWindow(50)

    @staticmethod
    def _wrap_pct(dp: float) -> float:
        # bring to [0,1)
        dp = dp - math.floor(dp)
        return max(0.0, min(0.999999, dp))

    def update(
        self,
        *,
        lap_completed: Optional[int],
        session_time: Optional[float],
        fuel_level: Optional[float],
        lap_dist_pct: Optional[float],
        session_flags: Optional[int],
        is_on_track: Optional[bool],
        on_pit_road: Optional[bool],
    ) -> None:
        if lap_completed is None or session_time is None or fuel_level is None:
            return

        try:
            lap_completed_i = int(lap_completed)
            session_time_f = float(session_time)
            fuel_f = float(fuel_level)
        except Exception:
            return

        # flags/pit/offtrack during the lap
        if self.ignore_yellow and session_flags is not None and is_yellow_flag(int(session_flags)):
            self._lap_has_yellow = True

        if on_pit_road:
            self._lap_has_pit = True

        if is_on_track is False:
            self._lap_has_offtrack = True

        # detect refuel: fuel rises suddenly
        if self._prev_fuel is not None and fuel_f - self._prev_fuel > self.refuel_delta_threshold:
            self._lap_has_pit = True
            # consider this as a new "start fuel" baseline for projections
            self._lap_start_fuel = fuel_f

        self._prev_fuel = fuel_f

        # first read
        if self._last_lap_completed is None:
            self._last_lap_completed = lap_completed_i
            self._lap_start_time = session_time_f
            self._lap_start_fuel = fuel_f
            self._lap_start_dist_pct = float(lap_dist_pct) if lap_dist_pct is not None else None
            return

        # update current-lap projection (even before lap completes)
        if self._lap_start_fuel is not None and lap_dist_pct is not None and self._lap_start_dist_pct is not None:
            try:
                cur_pct = float(lap_dist_pct)
                start_pct = float(self._lap_start_dist_pct)
                progress = self._wrap_pct(cur_pct - start_pct)
                fuel_used_so_far = float(self._lap_start_fuel) - fuel_f
                if progress >= self.proj_min_progress and fuel_used_so_far > 0:
                    proj = fuel_used_so_far / max(1e-6, progress)
                    # gate insane values
                    if 0 < proj <= self.max_reasonable_burn_per_lap:
                        self._proj_burn_hist.add(proj)
            except Exception:
                pass

        # lap completed changed => close lap
        if lap_completed_i != self._last_lap_completed:
            if self._lap_start_time is not None and self._lap_start_fuel is not None:
                lap_time = session_time_f - self._lap_start_time
                fuel_used = self._lap_start_fuel - fuel_f

                eligible = True
                if lap_time < self.min_lap_time_s:
                    eligible = False
                if fuel_used <= 0:
                    eligible = False
                if fuel_used > self.max_reasonable_burn_per_lap:
                    eligible = False
                if self._lap_has_pit or self._lap_has_offtrack:
                    eligible = False
                if self.ignore_yellow and self._lap_has_yellow:
                    eligible = False

                if eligible:
                    self.samples.append(
                        LapSample(
                            lap_number=int(self._last_lap_completed),
                            fuel_used=float(fuel_used),
                            lap_time=float(lap_time),
                            timestamp=time.time(),
                        )
                    )

            # start new lap
            self._last_lap_completed = lap_completed_i
            self._lap_start_time = session_time_f
            self._lap_start_fuel = fuel_f
            self._lap_start_dist_pct = float(lap_dist_pct) if lap_dist_pct is not None else 0.0

            self._lap_has_yellow = False
            self._lap_has_pit = False
            self._lap_has_offtrack = False

    # ---------- calculations ----------

    def _select_window(self, method: str, n: int) -> List[LapSample]:
        method = method.lower()
        n = max(1, int(n))
        if not self.samples:
            return []

        if method in {"all", "road_fav", "ema"}:
            return list(self.samples)
        if method in {"last_n", "top_burn", "median_last_n", "max_last_n", "trimmed_last_n"}:
            return self.samples[-n:]
        if method in {"first_n"}:
            return self.samples[:n]
        return self.samples[-n:]

    @staticmethod
    def _mean(values: List[float]) -> Optional[float]:
        if not values:
            return None
        return sum(values) / len(values)

    def burn_per_lap(self, *, method: str, n: int, top_percent: float) -> Optional[float]:
        window = self._select_window(method, n)
        burns = [s.fuel_used for s in window]
        if not burns:
            return None

        method = method.lower()

        if method in {"all", "last_n", "first_n"}:
            return self._mean(burns)

        if method in {"top_burn", "road_fav"}:
            p = float(top_percent)
            p = min(max(p, 1.0), 100.0)
            k = max(1, int(round(len(burns) * (p / 100.0))))
            burns_sorted = sorted(burns, reverse=True)
            return self._mean(burns_sorted[:k])

        if method == "median_last_n":
            try:
                return float(statistics.median(burns))
            except Exception:
                return self._mean(burns)

        if method == "max_last_n":
            return float(max(burns))

        if method == "trimmed_last_n":
            p = float(top_percent)
            p = min(max(p, 0.0), 45.0)
            burns_sorted = sorted(burns)
            trim = int(len(burns_sorted) * (p / 100.0))
            trimmed = burns_sorted[trim : len(burns_sorted) - trim] if len(burns_sorted) - 2 * trim >= 1 else burns_sorted
            return self._mean(trimmed)

        if method == "ema":
            alpha = 0.30
            ema_val = burns[0]
            for b in burns[1:]:
                ema_val = alpha * b + (1 - alpha) * ema_val
            return float(ema_val)

        return self._mean(burns)

    def lap_time_estimate(self, *, method: str, n: int) -> Optional[float]:
        window = self._select_window(method, n)
        times = [s.lap_time for s in window]
        if not times:
            return None
        return self._mean(times)

    def lap_time_stdev(self, *, method: str, n: int) -> Optional[float]:
        window = self._select_window(method, n)
        times = [s.lap_time for s in window]
        if len(times) < 2:
            return None
        try:
            return float(statistics.pstdev(times))
        except Exception:
            return None

    def projected_burn(self) -> Optional[float]:
        """Projected burn from the current lap (useful at race start)."""
        return self._proj_burn_hist.median()

    def burn_estimate(self, *, method: str, n: int, top_percent: float) -> Optional[float]:
        """Best-available burn estimate: use lap samples when available, otherwise projection."""
        burn = self.burn_per_lap(method=method, n=n, top_percent=top_percent)
        if burn is not None and burn > 0:
            return burn
        return self.projected_burn()


# ============================================================
# Wetness-aware assistant (heuristic)
# ============================================================

TRACK_WETNESS_ENUM = {
    0: "UNKNOWN",
    1: "Dry",
    2: "MostlyDry",
    3: "VeryLightlyWet",
    4: "LightlyWet",
    5: "ModeratelyWet",
    6: "VeryWet",
    7: "ExtremelyWet",
}


class WetnessBrain:
    def __init__(
        self,
        *,
        bins: int = 60,
        dry_wetness_threshold: float = 0.10,
        min_speed_mps: float = 22.0,
        min_steer_abs: float = 0.08,
    ) -> None:
        self.bins = int(max(20, min(bins, 200)))
        self.dry_wetness_threshold = float(dry_wetness_threshold)
        self.min_speed_mps = float(min_speed_mps)
        self.min_steer_abs = float(min_steer_abs)

        self._baseline_lataccel: List[Optional[float]] = [None] * self.bins

        self.wetness_hist = RollingWindow(120)
        self.precip_hist = RollingWindow(120)
        self.grip_ratio_hist = RollingWindow(120)
        self.aquaplane_events = RollingEvents()

        self._last_wetness: Optional[float] = None
        self._last_wetness_raw: Optional[float] = None
        self._last_wetness_label: Optional[str] = None
        self._last_update_t: Optional[float] = None

    @staticmethod
    def _norm_wetness(w: Optional[float]) -> Optional[float]:
        if w is None:
            return None
        try:
            x = float(w)
        except Exception:
            return None

        # New builds expose TrackWetness as enum 1..7 (Dry..ExtremelyWet).
        # Convert to 0..1 scale (Dry=0.0, ExtremelyWet=1.0) while still
        # tolerating older 0..1 or 0..100% formats.
        try:
            xi = int(x)
        except Exception:
            xi = None

        if xi is not None and 1 <= xi <= 7 and abs(x - xi) < 1e-3:
            return (xi - 1) / 6.0

        if 0.0 <= x <= 1.0:
            return x

        if x > 1.5:
            return max(0.0, min(1.0, x / 100.0))

        return None

    @staticmethod
    def _wetness_label(w: Optional[float]) -> Optional[str]:
        try:
            x = int(float(w))
        except Exception:
            return None

        if 1 <= x <= 7:
            return TRACK_WETNESS_ENUM.get(x, f"UNKNOWN({x})")
        return None

    @staticmethod
    def _norm_precip(p: Optional[float]) -> Optional[float]:
        if p is None:
            return None
        try:
            x = float(p)
        except Exception:
            return None
        # precip can be 0..1 or mm/h depending on build; normalize aggressively
        if x > 1.5:
            x = min(1.0, x / 10.0)
        return max(0.0, min(1.0, x))

    def update(
        self,
        *,
        now: float,
        lap_dist_pct: Optional[float],
        track_wetness: Optional[float],
        precipitation: Optional[float],
        declared_wet: Optional[bool],
        speed_mps: Optional[float],
        yaw_rate: Optional[float],
        lat_accel: Optional[float],
        steer: Optional[float],
    ) -> None:
        wet = self._norm_wetness(track_wetness)
        prec = self._norm_precip(precipitation)

        self._last_wetness_raw = track_wetness
        self._last_wetness_label = self._wetness_label(track_wetness)

        if wet is not None:
            self.wetness_hist.add(wet)
        if prec is not None:
            self.precip_hist.add(prec)

        # baseline / grip ratio
        if (
            wet is not None
            and lap_dist_pct is not None
            and speed_mps is not None
            and lat_accel is not None
            and steer is not None
        ):
            try:
                pct = float(lap_dist_pct)
                pct = pct - math.floor(pct)  # keep 0..1
            except Exception:
                pct = None

            if pct is not None and speed_mps >= self.min_speed_mps and abs(steer) >= self.min_steer_abs:
                idx = int(pct * self.bins)
                idx = max(0, min(self.bins - 1, idx))

                lat_abs = abs(float(lat_accel))

                if wet <= self.dry_wetness_threshold:
                    # update dry baseline
                    cur = self._baseline_lataccel[idx]
                    if cur is None or lat_abs > cur:
                        self._baseline_lataccel[idx] = lat_abs
                else:
                    base = self._baseline_lataccel[idx]
                    if base and base > 1e-6:
                        ratio = lat_abs / base
                        ratio = max(0.0, min(1.5, ratio))
                        self.grip_ratio_hist.add(ratio)

        # aquaplane heuristic: yaw/accel mismatch
        if speed_mps is not None and yaw_rate is not None and lat_accel is not None and steer is not None:
            try:
                v = float(speed_mps)
                yr = float(yaw_rate)
                la = float(lat_accel)
                expected_lat = abs(v * yr)
                measured_lat = abs(la)

                # Trigger: high speed + steering + expected lateral accel, but not getting it
                if v > 35.0 and abs(steer) > 0.12 and expected_lat > 6.0:
                    if measured_lat < expected_lat * 0.45:
                        self.aquaplane_events.add(now)
            except Exception:
                pass

        self._last_update_t = now
        self._last_wetness = wet

    def wetness_trend(self) -> Optional[float]:
        vals = self.wetness_hist.values()
        if len(vals) < 10:
            return None
        k = max(3, len(vals) // 5)
        start = sum(vals[:k]) / k
        end = sum(vals[-k:]) / k
        return end - start

    def recommend(self, *, now: float, declared_wet: Optional[bool]) -> Tuple[str, int, Dict[str, Any]]:
        wet = self.wetness_hist.median()
        prec = self.precip_hist.mean()
        grip = self.grip_ratio_hist.median()
        aqua = self.aquaplane_events.count_last(35.0, now)
        trend = self.wetness_trend()

        # Heurística para corrigir bug clássico do iRacing (versões antigas):
        # TrackWetness travado em 1.0 mesmo com pista seca. Nas builds atuais
        # TrackWetness é um enum (1=Dry, 7=ExtremelyWet), mas mantemos esta
        # proteção para sessões onde o valor pareça incoerente.
        #
        # Se:
        #  - pista não está declarada molhada
        #  - praticamente sem chuva
        #  - grip alto (~seco)
        #  - nenhum evento de aquaplanagem
        # então tratamos o "wet" efetivo como quase seco.
        wet_eff = wet
        if (
            wet is not None and wet > 0.85
            and (declared_wet is False or declared_wet is None)
            and (prec is None or prec < 0.02)
            and (grip is None or grip > 0.90)
            and aqua == 0
        ):
            wet_eff = 0.05  # ~5% → praticamente seco

        # Se todos os sinais apontam para pista seca, não recomendamos pit para chuva.
        if (
            (prec is None or prec < 0.02)
            and (wet_eff is None or wet_eff < 0.10)
            and (grip is None or grip > 0.88)
            and aqua == 0
            and (declared_wet is False or wet_eff < 0.08)
        ):
            details = {
                "wet": wet,
                "wet_eff": wet_eff,
                "wet_raw": self._last_wetness_raw,
                "wet_label": self._last_wetness_label,
                "prec": prec,
                "grip": grip,
                "aqua": aqua,
                "trend": trend,
            }
            return "STAY SLICKS", 0, details

        score = 0.0
        if declared_wet:
            score += 12.0

        if wet_eff is not None:
            score += max(0.0, min(1.0, (wet_eff - 0.12) / 0.28)) * 45.0

        if prec is not None:
            score += max(0.0, min(1.0, prec / 0.6)) * 10.0

        if grip is not None:
            score += max(0.0, min(1.0, (0.78 - grip) / 0.30)) * 25.0

        score += max(0.0, min(1.0, aqua / 3.0)) * 20.0

        conf = int(max(0, min(100, round(score))))

        action = "STAY SLICKS"
        if conf >= 70:
            action = "PIT WETS"
        elif conf >= 50:
            if trend is not None and trend > 0.03:
                action = "WAIT 1 LAP (trend ↑)"
            else:
                action = "CONSIDER PIT"

        details = {
            "wet": wet,
            "wet_eff": wet_eff,
            "wet_raw": self._last_wetness_raw,
            "wet_label": self._last_wetness_label,
            "prec": prec,
            "grip": grip,
            "aqua": aqua,
            "trend": trend,
        }
        return action, conf, details


# ============================================================
# Risk radar (heuristic)
# ============================================================

class RiskRadar:
    def __init__(self) -> None:
        self.prev_dist_m: Dict[int, Tuple[float, float]] = {}
        self.steer_hist = RollingWindow(60)
        self.brake_hist = RollingWindow(60)
        self.last_beep_t = 0.0

    @staticmethod
    def _wrap_delta_pct(dp: float) -> float:
        return dp - round(dp)

    def update(
        self,
        *,
        now: float,
        player_idx: Optional[int],
        track_length_m: Optional[float],
        player_lapdist_pct: Optional[float],
        car_idx_lapdist_pct: Optional[List[float]],
        brake: Optional[float],
        steer: Optional[float],
        long_accel: Optional[float],
        car_left_right: Optional[int],
        is_on_track: Optional[bool],
    ) -> Tuple[int, str]:
        if steer is not None:
            self.steer_hist.add(float(steer))
        if brake is not None:
            self.brake_hist.add(float(brake))

        if (
            player_idx is None
            or player_lapdist_pct is None
            or not car_idx_lapdist_pct
        ):
            return 0, "(no traffic data)"

        try:
            pidx = int(player_idx)
            ppct = float(player_lapdist_pct)

            if track_length_m is None:
                tlen = 5000.0
            else:
                tlen = float(track_length_m)
                if tlen <= 100.0:
                    tlen = 5000.0
        except Exception:
            return 0, "(no traffic data)"

        braking = False
        try:
            if brake is not None and float(brake) > 0.20:
                braking = True
            if long_accel is not None and float(long_accel) < -3.5:
                braking = True
        except Exception:
            pass

        overlap = False
        if car_left_right is not None:
            try:
                overlap = int(car_left_right) in (1, 2, 3, 4, 5, 6)
            except Exception:
                overlap = False

        steer_var = self.steer_hist.stdev() or 0.0
        # suavizado: menos sensível a micro correções
        instability = max(0.0, min(1.0, steer_var / 0.30))

        best_behind = (0.0, 0.0, 0.0, None)  # score, closing, dist, caridx
        best_ahead = (0.0, 0.0, 0.0, None)

        for i, opct in enumerate(car_idx_lapdist_pct):
            if i == pidx or opct is None:
                continue
            try:
                val = float(opct)
                if not math.isfinite(val):
                    continue

                dp = self._wrap_delta_pct(val - ppct)
                dist_m = dp * tlen
            except Exception:
                continue

            if abs(dist_m) > 300.0:
                continue

            closing = 0.0
            prev = self.prev_dist_m.get(i)
            if prev is not None:
                prev_dist, prev_t = prev
                dt = max(0.05, now - prev_t)
                if dist_m < 0:
                    closing = (dist_m - prev_dist) / dt
                else:
                    closing = (prev_dist - dist_m) / dt

            self.prev_dist_m[i] = (dist_m, now)

            d = abs(dist_m)

            if dist_m < 0:
                # carro atrás
                score = 0.0
                if d < 80.0:
                    score += (80.0 - d) / 80.0 * 15.0
                if closing > 2.0:
                    score += min(22.0, (closing - 2.0) / 6.0 * 22.0)
                if braking:
                    score += 6.0
                if overlap:
                    score += 10.0
                if score > best_behind[0]:
                    best_behind = (score, closing, dist_m, i)
            else:
                # carro à frente
                score = 0.0
                if d < 45.0:
                    score += (45.0 - d) / 45.0 * 12.0
                if closing > 2.5 and braking:
                    score += min(18.0, (closing - 2.5) / 6.0 * 18.0)
                if overlap:
                    score += 8.0
                if score > best_ahead[0]:
                    best_ahead = (score, closing, dist_m, i)

        risk = 0.0
        reasons: List[str] = []

        if best_behind[0] > 0:
            risk += best_behind[0]
            dist = best_behind[2]
            closing = best_behind[1]
            reasons.append(f"behind {abs(dist):.0f}m closing {closing:+.1f}m/s")

        if best_ahead[0] > 0:
            risk += best_ahead[0]
            dist = best_ahead[2]
            closing = best_ahead[1]
            reasons.append(f"ahead {abs(dist):.0f}m closing {closing:+.1f}m/s")

        if overlap and braking:
            risk += 15.0
            reasons.append("overlap+brake")
        elif overlap:
            risk += 8.0
            reasons.append("overlap")

        if is_on_track is False:
            risk += 22.0
            reasons.append("offtrack/rejoin")

        risk += instability * 15.0
        if instability > 0.7:
            reasons.append("instability")

        risk_i = int(max(0, min(100, round(risk))))
        return risk_i, ("OK" if not reasons else "; ".join(reasons[:3]))


# ============================================================
# Pit timing advisor (heuristic)
# ============================================================

@dataclass
class PitOption:
    offset_laps: int
    fuel_add: Optional[float]
    pit_loss_s: Optional[float]
    gap_ahead_s: Optional[float]
    gap_behind_s: Optional[float]
    min_gap_s: Optional[float]
    success_pct: Optional[int]
    pos_est: Optional[int]
    notes: str = ""


class PitWindowAdvisor:
    @staticmethod
    def _wrap_delta_pct(dp: float) -> float:
        return dp - round(dp)

    @staticmethod
    def _sigmoid(x: float) -> float:
        # stable-ish sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def build_options(
        self,
        *,
        avg_lap_time: Optional[float],
        lap_time_sigma: Optional[float],
        pit_base_loss_s: float,
        fuel_fill_rate: float,
        tire_service_time_s: float,
        take_tires: bool,
        burn_per_lap: Optional[float],
        fuel_now: Optional[float],
        laps_possible: Optional[float],
        laps_remain: Optional[float],
        margin_laps: float,
        fuel_plan_mode: str,
        stint_target_laps: float,
        tank_capacity: Optional[float],
        player_idx: Optional[int],
        player_lapdist_pct: Optional[float],
        car_idx_lapdist_pct: Optional[List[float]],
        track_length_m: Optional[float],
        max_offsets: int = 3,
        clean_air_target_s: float = 2.0,
        max_gap_consider_s: float = 12.0,
    ) -> List[PitOption]:
        if avg_lap_time is None or avg_lap_time <= 0:
            return []
        if burn_per_lap is None or burn_per_lap <= 0:
            return []
        if fuel_now is None:
            return []
        if player_idx is None or player_lapdist_pct is None or not car_idx_lapdist_pct:
            return []

        opts: List[PitOption] = []

        pidx = int(player_idx)
        ppct = float(player_lapdist_pct)

        sigma = float(lap_time_sigma) if lap_time_sigma is not None else 0.0
        # Effective uncertainty for gap prediction:
        #  - we don't know others' pace and the delta pct -> time is a proxy.
        #  - keep a floor so early race still produces a meaningful probability curve.
        sigma_eff = max(0.8, sigma * 0.35 + 0.25)

        def _compute_fuel_add(
            *,
            fuel_after_wait: float,
            laps_remaining_after: Optional[float],
        ) -> Tuple[Optional[float], Optional[float], str]:
            """Return (fuel_add, target_total, note)."""
            mode = (fuel_plan_mode or "finish").lower()
            if mode == "finish":
                if laps_remaining_after is None:
                    return None, None, "no race remaining"
                target_total = (laps_remaining_after + float(margin_laps)) * float(burn_per_lap)
                return max(0.0, target_total - fuel_after_wait), target_total, "finish"
            if mode == "stint":
                target_laps = max(0.0, float(stint_target_laps) + float(margin_laps))
                target_total = target_laps * float(burn_per_lap)
                return max(0.0, target_total - fuel_after_wait), target_total, "stint"
            if mode == "full":
                if tank_capacity is None or tank_capacity <= 0:
                    return None, None, "tank cap?"
                target_total = float(tank_capacity)
                return max(0.0, target_total - fuel_after_wait), target_total, "full"
            # fallback to finish
            if laps_remaining_after is None:
                return None, None, "no race remaining"
            target_total = (laps_remaining_after + float(margin_laps)) * float(burn_per_lap)
            return max(0.0, target_total - fuel_after_wait), target_total, "finish"

        # Sempre só avaliamos a opção de parar AGORA (offset 0)
        for offset in (0,):
            # Ensure we can wait offset laps on current fuel (rough guard)
            if laps_possible is not None and laps_possible < float(offset) + 0.15:
                continue

            fuel_after_wait = float(fuel_now) - float(offset) * float(burn_per_lap)
            fuel_after_wait = max(0.0, fuel_after_wait)

            laps_rem_after = None
            if laps_remain is not None:
                laps_rem_after = max(0.0, float(laps_remain) - float(offset))

            fuel_add, target_total, plan_note = _compute_fuel_add(
                fuel_after_wait=fuel_after_wait,
                laps_remaining_after=laps_rem_after,
            )

            # Pit time model: base + fueling + tires
            pit_loss = float(pit_base_loss_s)
            if fuel_add is not None and fuel_fill_rate > 1e-6:
                pit_loss += float(fuel_add) / float(fuel_fill_rate)
            if take_tires:
                pit_loss += float(tire_service_time_s)

            # Predict gaps after pit using LapDistPct deltas (proxy)
            ahead_gap = None
            behind_gap = None

            gaps_after: List[float] = []
            for i, opct in enumerate(car_idx_lapdist_pct):
                if i == pidx or opct is None:
                    continue
                try:
                    dp = self._wrap_delta_pct(float(opct) - ppct)
                    gap_time = dp * float(avg_lap_time)
                    gap_after = gap_time + pit_loss
                    gaps_after.append(gap_after)
                except Exception:
                    continue

            # Use only cars within a sane window; others are effectively "clean air"
            relevant = [g for g in gaps_after if abs(g) <= max_gap_consider_s]
            if relevant:
                ahead_candidates = [g for g in relevant if g > 0]
                behind_candidates = [g for g in relevant if g < 0]
                if ahead_candidates:
                    ahead_gap = min(ahead_candidates)
                if behind_candidates:
                    behind_gap = max(behind_candidates)  # negative closest to 0

            # min gap on both sides
            min_gap = None
            if ahead_gap is not None and behind_gap is not None:
                min_gap = min(ahead_gap, abs(behind_gap))
            elif ahead_gap is not None:
                min_gap = ahead_gap
            elif behind_gap is not None:
                min_gap = abs(behind_gap)
            else:
                # no relevant cars -> treat as very clean air
                min_gap = max_gap_consider_s

            # Convert min_gap -> "success probability" for clean-air target
            # Logistic around the target:
            #   success ~ sigmoid((min_gap - target) / sigma_eff)
            z = (float(min_gap) - float(clean_air_target_s)) / max(0.2, sigma_eff)
            success = self._sigmoid(z)
            success_pct = int(max(0, min(100, round(success * 100.0))))

            # crude position estimate (higher gaps_after >0 means cars ahead after pit)
            pos_est = None
            pos_pool = relevant if relevant else gaps_after
            if pos_pool:
                pos_est = 1 + sum(1 for g in pos_pool if g > 0)

            notes = plan_note
            if fuel_add is None:
                notes = notes + " | fuel=?"
            if take_tires:
                notes = notes + " | tires"

            opts.append(
                PitOption(
                    offset_laps=offset,
                    fuel_add=fuel_add,
                    pit_loss_s=pit_loss,
                    gap_ahead_s=ahead_gap,
                    gap_behind_s=behind_gap,
                    min_gap_s=min_gap,
                    success_pct=success_pct,
                    pos_est=pos_est,
                    notes=notes.strip(),
                )
            )

        # sort by success (desc) then by pit loss (asc)
        opts.sort(key=lambda o: (-(o.success_pct or 0), (o.pit_loss_s or 9999.0)))
        return opts


# ============================================================
# Pit telemetry calibration (best-effort)
# ============================================================

@dataclass
class PitMeasurement:
    pit_lane_time_s: float
    stall_time_s: Optional[float]
    fuel_added: float
    fueling_time_s: float
    fill_rate: Optional[float]
    tires_selected: Optional[bool]
    pit_sv_fuel: Optional[float]
    timestamp: float


class PitStopCalibrator:
    """Observe pit stops and infer fill-rate / base loss / tire time.

    Uses:
      - OnPitRoad (lane)
      - PlayerCarInPitStall (if available, otherwise speed-based stall heuristic)
      - FuelLevel deltas to infer fuel added and fueling duration
      - PitSvFlags to infer whether tires were selected

    This is best-effort. It will not always be able to infer everything.
    """

    def __init__(self):
        self.on_pit_road_prev: Optional[bool] = None
        self.in_stall_prev: Optional[bool] = None

        self.pit_enter_t: Optional[float] = None
        self.pit_exit_t: Optional[float] = None

        self.stall_enter_t: Optional[float] = None
        self.stall_exit_t: Optional[float] = None

        self.prev_fuel: Optional[float] = None
        self.fuel_added: float = 0.0

        self.fueling_active: bool = False
        self.fuel_start_t: Optional[float] = None
        self.fueling_time_s: float = 0.0

        self.pit_sv_flags_seen: Optional[int] = None
        self.pit_sv_fuel_seen: Optional[float] = None

        self.last_measurement: Optional[PitMeasurement] = None

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def _bool(v: Any) -> Optional[bool]:
        if v is None:
            return None
        try:
            return bool(v)
        except Exception:
            return None

    def update(
        self,
        *,
        now: float,
        on_pit_road: Optional[bool],
        in_pit_stall: Optional[bool],
        speed_mps: Optional[float],
        fuel_level: Optional[float],
        pit_sv_flags: Optional[int],
        pit_sv_fuel: Optional[float],
    ) -> Optional[PitMeasurement]:
        # derive stall if not available
        stall = in_pit_stall
        if stall is None:
            if on_pit_road is None:
                stall = None
            else:
                # heuristic: stopped on pit road
                try:
                    stall = bool(on_pit_road) and (speed_mps is not None and float(speed_mps) < 1.0)
                except Exception:
                    stall = bool(on_pit_road)

        on_pit = bool(on_pit_road) if on_pit_road is not None else False
        in_stall = bool(stall) if stall is not None else False

        # track pit service selections (best effort)
        if pit_sv_flags is not None:
            try:
                self.pit_sv_flags_seen = int(pit_sv_flags)
            except Exception:
                pass
        if pit_sv_fuel is not None:
            try:
                self.pit_sv_fuel_seen = float(pit_sv_fuel)
            except Exception:
                pass

        # pit transitions
        if self.on_pit_road_prev is None:
            self.on_pit_road_prev = on_pit
        if self.in_stall_prev is None:
            self.in_stall_prev = in_stall

        # pit enter
        if (self.on_pit_road_prev is False) and on_pit:
            self.pit_enter_t = now
            self.pit_exit_t = None

        # pit exit
        if (self.on_pit_road_prev is True) and (not on_pit):
            self.pit_exit_t = now

        # stall enter
        if (self.in_stall_prev is False) and in_stall:
            self.stall_enter_t = now
            self.stall_exit_t = None
            self.fuel_added = 0.0
            self.fueling_time_s = 0.0
            self.fueling_active = False
            self.fuel_start_t = None
            self.prev_fuel = float(fuel_level) if fuel_level is not None else None

        # fueling tracking (while in stall)
        if in_stall and fuel_level is not None:
            try:
                f = float(fuel_level)
            except Exception:
                f = None

            if f is not None and self.prev_fuel is not None:
                df = f - self.prev_fuel
                # fuel increasing => fueling active
                if df > 0.001:
                    self.fuel_added += df
                    if not self.fueling_active:
                        self.fueling_active = True
                        self.fuel_start_t = now
                else:
                    # no longer fueling
                    if self.fueling_active and self.fuel_start_t is not None:
                        self.fueling_time_s += max(0.0, now - self.fuel_start_t)
                        self.fueling_active = False
                        self.fuel_start_t = None

            self.prev_fuel = f if f is not None else self.prev_fuel

        # stall exit
        measurement: Optional[PitMeasurement] = None
        if (self.in_stall_prev is True) and (not in_stall):
            self.stall_exit_t = now

            # close fueling segment if still active
            if self.fueling_active and self.fuel_start_t is not None:
                self.fueling_time_s += max(0.0, now - self.fuel_start_t)
                self.fueling_active = False
                self.fuel_start_t = None

            stall_time = None
            if self.stall_enter_t is not None:
                stall_time = max(0.0, now - self.stall_enter_t)

            pit_lane_time = None
            if self.pit_enter_t is not None:
                pit_lane_time = max(0.0, now - self.pit_enter_t)

            tires_sel = pit_flags_tires_selected(self.pit_sv_flags_seen)

            fill_rate = None
            if self.fueling_time_s > 0.5 and self.fuel_added > 0:
                fill_rate = self.fuel_added / self.fueling_time_s

            if pit_lane_time is not None and pit_lane_time > 0.1:
                measurement = PitMeasurement(
                    pit_lane_time_s=float(pit_lane_time),
                    stall_time_s=float(stall_time) if stall_time is not None else None,
                    fuel_added=float(self.fuel_added),
                    fueling_time_s=float(self.fueling_time_s),
                    fill_rate=float(fill_rate) if fill_rate is not None else None,
                    tires_selected=tires_sel,
                    pit_sv_fuel=self.pit_sv_fuel_seen,
                    timestamp=time.time(),
                )
                self.last_measurement = measurement

            # reset for next pit
            self.stall_enter_t = None
            self.pit_enter_t = None
            self.pit_sv_flags_seen = None
            self.pit_sv_fuel_seen = None
            self.prev_fuel = None
            self.fuel_added = 0.0
            self.fueling_time_s = 0.0
            self.fueling_active = False
            self.fuel_start_t = None

        self.on_pit_road_prev = on_pit
        self.in_stall_prev = in_stall
        return measurement


# ============================================================
# UI (Overlay + Settings)
# ============================================================

class FuelOverlayApp(ctk.CTk):
    METHOD_LABELS = {
        "road_fav": "★ ROAD (Recommended) – Top Burn% (Last N)",
        "all": "All (Mean) – all green laps",
        "last_n": "Last N (Mean) – last N green laps",
        "first_n": "First N (Mean) – first N green laps",
        "top_burn": "Top Burn % – mean of top % (Last N)",
        "median_last_n": "Median – (Last N)",
        "max_last_n": "Max – worst burn (Last N)",
        "trimmed_last_n": "Trimmed Mean – cut % tails (Last N)",
        "ema": "EMA – exponential moving average (all green)",
    }

    PLAN_LABELS = {
        "safe": "SAFE",
        "attack": "ATTACK",
        "stretch": "STRETCH",
    }

    FUEL_PLAN_LABELS = {
        "finish": "Finish (race end)",
        "stint": "Stint target",
        "full": "Fill to full tank",
    }

    DEFAULT_APPLY_MACRO = "!fuel {fuel_add:.2f}$"
    PIT_CAL_KEYS = ("pit_base_loss_s", "fuel_fill_rate", "tire_service_time_s", "tires_with_fuel")
    PIT_PROFILE_META_KEYS = ("has_calibration",)
    PIT_PROFILE_KEYS = PIT_CAL_KEYS + PIT_PROFILE_META_KEYS

    def __init__(self, *, portable: bool = False):
        super().__init__()

        ctk.set_appearance_mode("dark")

        self.portable = bool(portable)
        self.config_dir, self.config_file = _get_config_paths(self.portable)
        self.config_data = self._load_config()

        self._active_profile_key: Optional[str] = None
        self._pit_profile_has_data: bool = bool(self.config_data.get("pit", {}).get("has_calibration", False))
        self._macro_lock = threading.Lock()
        self._macro_last_send_t = 0.0

        # IRSDK
        self.ir = irsdk.IRSDK() if irsdk else None

        # subsystems
        self.history = FuelHistory(ignore_yellow=bool(self.config_data["fuel"]["ignore_yellow"]))
        self.wet_brain = WetnessBrain()
        self.risk_radar = RiskRadar()
        self.pit_advisor = PitWindowAdvisor()
        self.pit_cal = PitStopCalibrator()

        # macros
        self.injector = self._build_injector_from_config()

        # detached overlay windows (per section)
        self.detached_windows: Dict[str, ctk.CTkToplevel] = {}
        self.section_defs: Dict[str, Dict[str, Any]] = {}

        # hotkey handles
        self._hk_ids: List[int] = []

        # cache
        self._last_calc: Dict[str, Any] = {}
        self._last_pit_options: List[PitOption] = []
        self._prev_in_pit_stall: Optional[bool] = None
        self._session_info_cache: Optional[Dict[str, Any]] = None
        self._auto_tank_capacity_l: Optional[float] = None

        # window
        self.title(APP_NAME)
        self.attributes("-topmost", bool(self.config_data["ui"].get("always_on_top", True)))
        self.overrideredirect(bool(self.config_data["ui"].get("borderless", True)))

        geo = self.config_data["ui"].get("geometry")
        if isinstance(geo, str) and geo:
            self.geometry(geo)
        else:
            self.geometry("560x410+60+60")

        # drag
        self._drag_offset: Optional[Tuple[int, int]] = None
        self._drag_target = None
        self._drag_enabled: bool = bool(self.config_data["ui"].get("drag_enabled", True))

        self.bind("<ButtonPress-1>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_up)

        # UI
        self._build_ui()
        self._wire_drag_bindings()

        # hotkeys
        self._setup_hotkeys()

        # loop
        self.after(int(self.config_data["ui"].get("refresh_ms", 150)), self._tick)

    # ----------------------------
    # Config
    # ----------------------------

    def _default_config(self) -> dict:
        return {
            "ui": {
                "geometry": "560x410+60+60",
                "refresh_ms": 150,
                "always_on_top": True,
                "borderless": True,
                "drag_enabled": True,
                "easy_mode": False,
                "show_sections": {
                    "fuel": True,
                    "race": True,
                    "pit": True,
                    "weather": True,
                    "risk": True,
                    "hotkeys": True,
                },
            },
            "fuel": {
                "method": "road_fav",
                "n": 10,
                "top_percent": 20,
                "ignore_yellow": True,
                "margin_laps": 1.0,
                "margin_step": 0.5,
                "plan_mode": "safe",
                "plan_modes": {
                    "safe": {"margin_override": 1.5, "take_tires": True},
                    "attack": {"margin_override": 0.5, "take_tires": False},
                    "stretch": {"margin_override": 1.0, "take_tires": False},
                },
                "fuel_plan_mode": "finish",  # finish | stint | full
                "stint_target_laps": 20.0,
                "tank_capacity": None,  # optional for "full" mode
            },
            "pit": {
                "pit_base_loss_s": 40.0,
                "fuel_fill_rate": 2.5,
                "tire_service_time_s": 18.0,
                "tires_with_fuel": False,
                "clean_air_target_s": 2.0,
                "max_offsets": 3,
                "has_calibration": False,
                "advanced_editing": False,
                "auto_calibrate": {
                    "enabled": True,
                    "update_base_loss": True,
                    "update_fill_rate": True,
                    "update_tire_time": True,
                    "ema_alpha": 0.25,
                    "min_fuel_added": 0.5,
                    "min_lane_time": 8.0,
                },
            },
            "macro": {
                "enabled": False,
                "chat_key": "t",
                "open_delay": 0.02,
                "injection": "type",
                "typing_interval": 0.001,
                "require_iracing_foreground": True,
                "iracing_window_substring": "iracing",
                "debounce_ms": 250,
                "auto_fuel_on_pit": False,  # <--- NOVO
                "templates": {
                    "apply_plan": self.DEFAULT_APPLY_MACRO,
                    "wet_preset": "",
                    "slick_preset": "",
                },
            },
            "pit_profiles": {},
            "pit_profile_defaults": {
                "pit_base_loss_s": 40.0,
                "fuel_fill_rate": 2.5,
                "tire_service_time_s": 18.0,
                "tires_with_fuel": False,
                "has_calibration": False,
            },
            "hotkeys": {
                "margin_up": "ctrl+alt+up",
                "margin_down": "ctrl+alt+down",
                "cycle_plan": "ctrl+alt+p",
                "apply_opt1": "alt+1",
                "apply_opt2": "alt+2",
                "apply_opt3": "alt+3",
                "wet": "alt+w",
                "slick": "alt+s",
                "toggle_settings": "ctrl+alt+o",
            },
            "audio": {
                "risk_beep": True,
                "beep_cooldown_s": 3.0,
                "risk_beep_threshold": 85,
            },
        }

    def _load_config(self) -> dict:
        cfg = self._default_config()

        # Backward compatibility: if old config exists near script, import/merge it.
        script_dir = Path(__file__).resolve().parent
        legacy = script_dir / "configs" / "fuel_overlay_config.json"
        if legacy.exists():
            try:
                data = json.loads(legacy.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    # map legacy keys into new structure best-effort
                    mapped = self._map_legacy_config(data)
                    _deep_merge_dict(cfg, mapped)
            except Exception:
                pass

        # load main config
        try:
            if self.config_file.exists():
                data = json.loads(self.config_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    _deep_merge_dict(cfg, data)
        except Exception:
            pass

        return cfg

    @staticmethod
    def _sanitize_profile_piece(text: str) -> str:
        safe = text.strip() if text else ""
        return safe.replace("/", "-")

    def _pit_profile_template(self) -> Dict[str, Any]:
        base = self.config_data.get("pit_profile_defaults")
        if not isinstance(base, dict):
            base = {k: self._default_config()["pit_profile_defaults"].get(k) for k in self.PIT_PROFILE_KEYS}
            self.config_data["pit_profile_defaults"] = base
        else:
            for k in self.PIT_PROFILE_KEYS:
                if k not in base:
                    base[k] = self._default_config()["pit_profile_defaults"].get(k)
        return dict(base)

    def _profile_key_from_session(self, session_info: Optional[Dict[str, Any]]) -> str:
        track = "unknown_track"
        car = "unknown_car"

        def _pick_track(data: Optional[Dict[str, Any]]):
            nonlocal track
            if not isinstance(data, dict):
                return

            t_disp = data.get("TrackDisplayName") or data.get("TrackDisplayShortName") or data.get("TrackName")
            t_cfg = data.get("TrackConfigName") or data.get("TrackVariation")
            pieces = [p for p in [t_disp, t_cfg] if p]
            if pieces:
                track = " - ".join(self._sanitize_profile_piece(str(p)) for p in pieces)

        def _pick_car(data: Optional[Dict[str, Any]]):
            nonlocal car
            if not isinstance(data, dict):
                return

            try:
                player_idx = data.get("DriverCarIdx")
                if player_idx is None:
                    player_idx = self._safe_get("DriverCarIdx")
            except Exception:
                player_idx = None

            try:
                drivers = data.get("Drivers") or []
            except Exception:
                drivers = []

            # fall back to high-level fields first
            car_name = data.get("DriverCarScreenName") or data.get("DriverCarModel") or car

            for drv in drivers:
                try:
                    idx = drv.get("CarIdx")
                except Exception:
                    idx = None

                is_player = False
                try:
                    is_player = bool(drv.get("IsPlayerCar"))
                except Exception:
                    pass

                if player_idx is not None and idx is not None and int(idx) == int(player_idx):
                    is_player = True

                if is_player:
                    car_name = drv.get("CarScreenName") or drv.get("CarModel") or car_name
                    break

            car = self._sanitize_profile_piece(str(car_name))

        # Prefer detailed session_info when available
        try:
            _pick_track(session_info.get("WeekendInfo", {}) if session_info else {})
        except Exception:
            pass

        try:
            _pick_car(session_info.get("DriverInfo", {}) if session_info else {})
        except Exception:
            pass

        # Fallback to live telemetry dictionaries if session_info was missing/empty
        if track == "unknown_track":
            try:
                _pick_track(self._safe_get("WeekendInfo"))
            except Exception:
                pass

        if car == "unknown_car":
            try:
                _pick_car(self._safe_get("DriverInfo"))
            except Exception:
                pass

        if track == "unknown_track":
            track = "track_unknown"

        if car == "unknown_car":
            car = "car_unknown"

        return f"{car} @ {track}"

    def _persist_active_profile(self) -> None:
        if not self._active_profile_key:
            return

        profiles = self.config_data.setdefault("pit_profiles", {})
        profile = profiles.get(self._active_profile_key, {})
        for k in self.PIT_PROFILE_KEYS:
            if k in self.config_data.get("pit", {}):
                profile[k] = self.config_data["pit"].get(k)
        profiles[self._active_profile_key] = profile

    def _maybe_switch_pit_profile(self, session_info: Optional[Dict[str, Any]]) -> None:
        profiles = self.config_data.setdefault("pit_profiles", {})
        template = self._pit_profile_template()
        new_key = self._profile_key_from_session(session_info)

        if self._active_profile_key == new_key:
            return

        # save current profile before swapping
        self._persist_active_profile()

        profile = profiles.get(new_key)
        if not isinstance(profile, dict):
            profile = dict(template)
            # seed with current pit values if available
            for k in self.PIT_PROFILE_KEYS:
                if k in self.config_data.get("pit", {}):
                    profile[k] = self.config_data["pit"].get(k, profile.get(k))
            profiles[new_key] = profile

        for k in self.PIT_PROFILE_KEYS:
            self.config_data["pit"][k] = profile.get(k, template.get(k))

        self._pit_profile_has_data = bool(self.config_data.get("pit", {}).get("has_calibration", False))

        self._active_profile_key = new_key
        try:
            self.pit_profile_label.set(f"Active profile: {new_key} (data saved per car/track)")
        except Exception:
            pass

    def _map_legacy_config(self, legacy: Dict[str, Any]) -> Dict[str, Any]:
        """Best-effort mapping from the older single-level config to the new schema."""
        mapped: Dict[str, Any] = {}
        fuel: Dict[str, Any] = {}
        pit: Dict[str, Any] = {}
        macro: Dict[str, Any] = {}
        hotkeys: Dict[str, Any] = {}
        audio: Dict[str, Any] = {}
        ui: Dict[str, Any] = {}

        # fuel fields
        for k in ["method", "n", "top_percent", "margin_laps", "margin_step", "ignore_yellow", "plan_mode", "plan_modes"]:
            if k in legacy:
                fuel[k] = legacy.get(k)

        # pit fields
        if "pit" in legacy and isinstance(legacy["pit"], dict):
            pit.update(legacy["pit"])

        # macro fields
        if "macro" in legacy and isinstance(legacy["macro"], dict):
            macro.update(legacy["macro"])

        # hotkeys fields
        if "hotkeys" in legacy and isinstance(legacy["hotkeys"], dict):
            hotkeys.update(legacy["hotkeys"])

        # audio
        if "audio" in legacy and isinstance(legacy["audio"], dict):
            audio.update(legacy["audio"])

        # geometry
        if "geometry" in legacy:
            ui["geometry"] = legacy.get("geometry")

        if fuel:
            mapped["fuel"] = fuel
        if pit:
            mapped["pit"] = pit
        if macro:
            mapped["macro"] = macro
        if hotkeys:
            mapped["hotkeys"] = hotkeys
        if audio:
            mapped["audio"] = audio
        if ui:
            mapped["ui"] = ui
        return mapped

    def _save_config(self) -> None:
        try:
            _ensure_dir(self.config_dir)
            self._persist_active_profile()
            try:
                self.config_data["ui"]["geometry"] = self.geometry()
            except Exception:
                pass
            self.config_file.write_text(json.dumps(self.config_data, indent=4, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _build_injector_from_config(self) -> ChatInjector:
        m = self.config_data.get("macro", {})
        return ChatInjector(
            chat_key=str(m.get("chat_key", "t")),
            open_delay_s=float(m.get("open_delay", 0.02)),
            injection=str(m.get("injection", "type")),
            typing_interval_s=float(m.get("typing_interval", 0.001)),
            require_iracing_foreground=bool(m.get("require_iracing_foreground", True)),
            iracing_window_substring=str(m.get("iracing_window_substring", "iracing")),
            debounce_ms=int(m.get("debounce_ms", 250)),
        )

    def _apply_plan_template(self) -> str:
        """Return the chat template used to apply fuel plans.

        Always enforces a trailing "$" so the command is executed
        imediatamente ao enviar.
        """

        macro_cfg = self.config_data.get("macro", {})
        tmpl = (
            macro_cfg.get("templates", {}).get("apply_plan")
            if isinstance(macro_cfg.get("templates"), dict)
            else None
        )
        if not tmpl:
            tmpl = self.DEFAULT_APPLY_MACRO

        # Backward compatibility: older configs used "#fuel", which iRacing
        # no longer accepts. Automatically upgrade to the "!fuel" variant
        # while keeping any user customizations intact.
        if str(tmpl).strip().startswith("#fuel"):
            tmpl = str(tmpl).replace("#fuel", "!fuel", 1)
            try:
                macro_cfg.setdefault("templates", {})["apply_plan"] = tmpl
            except Exception:
                pass
        if not str(tmpl).endswith("$"):
            tmpl = str(tmpl) + "$"
        return str(tmpl)

    def _send_pit_fuel_via_sdk(self, fuel_liters: float) -> bool:
        """Try to set pit fuel directly via the iRacing SDK.

        Falls back to False when the SDK isn't available or when the
        current binding doesn't expose a compatible API.
        """

        if irsdk is None or self.ir is None:
            return False

        if not getattr(self.ir, "is_initialized", False) or not getattr(self.ir, "is_connected", False):
            return False

        try:
            for attr in ("pit_command", "pitCommand"):
                if hasattr(self.ir, attr):
                    pit_cmd = getattr(self.ir, attr)
                    try:
                        pit_cmd(fuel=float(fuel_liters))
                        return True
                    except TypeError:
                        try:
                            pit_cmd(float(fuel_liters))
                            return True
                        except Exception:
                            pass
        except Exception:
            return False

        return False

    def _send_chat_macro(self, text: str) -> bool:
        """Enviar comando de chat seguindo as configs do overlay."""

        if not text or keyboard is None:
            return False

        if self.injector is None:
            self.injector = self._build_injector_from_config()

        try:
            with self._macro_lock:
                ok = self.injector.send(text)
            if ok:
                self._macro_last_send_t = time.monotonic()
            return ok
        except Exception:
            return False

    def _dispatch_fuel_macro(
        self,
        fuel_add: float,
        *,
        offset_laps: int = 0,
        pit_loss_s: float = 0.0,
        success_pct: int = 0,
    ) -> None:
        """Apply fuel command via SDK first, falling back to chat macro."""

        fuel_add_val = max(0.0, float(fuel_add))

        if self._send_pit_fuel_via_sdk(fuel_add_val):
            return

        if not self.config_data.get("macro", {}).get("enabled", False):
            return

        tmpl = self._apply_plan_template()
        ctx = _SafeFormatDict(
            {
                "fuel_add": fuel_add_val,
                "offset_laps": int(offset_laps),
                "pit_loss_s": float(pit_loss_s),
                "plan_mode": str(self.config_data["fuel"].get("plan_mode", "safe")),
                "success_pct": int(success_pct),
            }
        )
        try:
            cmd = str(tmpl).format_map(ctx)
        except Exception:
            cmd = f"#fuel {fuel_add_val:.2f}"
        self._send_chat_macro(cmd)

    # ----------------------------
    # UI build
    # ----------------------------

    def _build_ui(self) -> None:
        self.frame = ctk.CTkFrame(self, corner_radius=10)
        self.frame.pack(fill="both", expand=True, padx=8, pady=8)

        self.tabs = ctk.CTkTabview(self.frame, corner_radius=10)
        self.tabs.pack(fill="both", expand=True)

        self.tab_overlay = self.tabs.add("Overlay")
        self.tab_controls = self.tabs.add("Controls")
        self.tab_settings = self.tabs.add("Settings")

        self._build_overlay_tab()
        self._build_controls_tab()
        self._build_settings_tab()

    def _build_overlay_tab(self) -> None:
        # Sections container (scrollable)
        # If the window is short, we still want access to all sections below.
        # Some CustomTkinter versions don't expose scrollbar color kwargs.
        # We fall back gracefully if that's the case.
        try:
            self.section_container = ctk.CTkScrollableFrame(
                self.tab_overlay,
                fg_color="transparent",
                scrollbar_button_color=("#3f3f3f", "#3f3f3f"),
                scrollbar_button_hover_color=("#525252", "#525252"),
            )
        except TypeError:
            self.section_container = ctk.CTkScrollableFrame(
                self.tab_overlay,
                fg_color="transparent",
            )
        self.section_container.pack(fill="both", expand=True, padx=0, pady=0)

        # Fuel/Race
        self.var_fuel = ctk.StringVar(value="Fuel: --")
        self.var_race = ctk.StringVar(value="Race: --")
        self.var_need_callout = ctk.StringVar(value="Need: --")

        self.sec_fuel = self._mk_section(self.section_container, "FUEL & RACE", "fuel")
        self.lbl_fuel = ctk.CTkLabel(self.sec_fuel, textvariable=self.var_fuel, anchor="w", font=("Consolas", 12, "bold"))
        self.lbl_fuel.pack(fill="x", padx=8, pady=(6, 0))
        self._register_section_label("fuel", self.var_fuel, ("Consolas", 12, "bold"), padx=8, pady=(6, 0))

        self.lbl_race = ctk.CTkLabel(self.sec_fuel, textvariable=self.var_race, anchor="w", font=("Consolas", 11))
        self.lbl_race.pack(fill="x", padx=8, pady=(0, 6))
        self._register_section_label("fuel", self.var_race, ("Consolas", 11), padx=8, pady=(0, 6))

        self.lbl_need = ctk.CTkLabel(
            self.sec_fuel,
            textvariable=self.var_need_callout,
            anchor="w",
            font=("Consolas", 13, "bold"),
            text_color="#f5c542",
        )
        self.lbl_need.pack(fill="x", padx=8, pady=(0, 6))
        self._register_section_label("fuel", self.var_need_callout, ("Consolas", 13, "bold"), padx=8, pady=(0, 6))

        # Pit
        self.var_pit = ctk.StringVar(value="Pit: --")

        self.sec_pit = self._mk_section(self.section_container, "PIT WINDOW (heuristic)", "pit")
        self.lbl_pit = ctk.CTkLabel(self.sec_pit, textvariable=self.var_pit, anchor="w", font=("Consolas", 11), justify="left")
        self.lbl_pit.pack(fill="x", padx=8, pady=(6, 6))
        self._register_section_label("pit", self.var_pit, ("Consolas", 11), padx=8, pady=(6, 6), justify="left")
        self._pit_label_default_color = None
        try:
            self._pit_label_default_color = self.lbl_pit.cget("text_color")
        except Exception:
            pass

        # Weather
        self.var_weather = ctk.StringVar(value="Weather: --")

        self.sec_weather = self._mk_section(self.section_container, "WEATHER / TIRES (heuristic)", "weather")
        self.lbl_weather = ctk.CTkLabel(self.sec_weather, textvariable=self.var_weather, anchor="w", font=("Consolas", 11))
        self.lbl_weather.pack(fill="x", padx=8, pady=(6, 6))
        self._register_section_label("weather", self.var_weather, ("Consolas", 11), padx=8, pady=(6, 6))

        # Risk
        self.var_risk = ctk.StringVar(value="Risk: --")

        self.sec_risk = self._mk_section(self.section_container, "RISK RADAR (heuristic)", "risk")
        self.lbl_risk = ctk.CTkLabel(self.sec_risk, textvariable=self.var_risk, anchor="w", font=("Consolas", 11))
        self.lbl_risk.pack(fill="x", padx=8, pady=(6, 6))
        self._register_section_label("risk", self.var_risk, ("Consolas", 11), padx=8, pady=(6, 6))

        # Hotkeys line
        self.var_hotkeys = ctk.StringVar(value="Hotkeys: --")

        self.sec_hotkeys = self._mk_section(self.section_container, "HOTKEYS", "hotkeys")
        self.lbl_hotkeys = ctk.CTkLabel(self.sec_hotkeys, textvariable=self.var_hotkeys, anchor="w", font=("Consolas", 10))
        self.lbl_hotkeys.pack(fill="x", padx=8, pady=(6, 6))
        self._register_section_label("hotkeys", self.var_hotkeys, ("Consolas", 10), padx=8, pady=(6, 6))

        self._apply_section_visibility()

        # Make mouse-wheel scrolling reliable inside the overlay tab
        # (CTkTabview + borderless window can sometimes swallow wheel events).
        self._setup_overlay_scroll_bindings()

    def _build_controls_tab(self) -> None:
        # Top row (status + save)
        top = ctk.CTkFrame(self.tab_controls, fg_color="transparent")
        top.pack(fill="x", padx=6, pady=(6, 8))

        title_box = ctk.CTkFrame(top, fg_color="transparent")
        title_box.pack(side="left", fill="x", expand=True)

        self.lbl_title = ctk.CTkLabel(title_box, text="Fuel Overlay Pro", font=("Segoe UI", 15, "bold"), anchor="w")
        self.lbl_title.pack(anchor="w")
        self.lbl_subtitle = ctk.CTkLabel(
            title_box,
            text="Pit timing, fuel math, and macros in a compact panel",
            font=("Segoe UI", 11),
            anchor="w",
        )
        self.lbl_subtitle.pack(anchor="w")

        btn_box = ctk.CTkFrame(top, fg_color="transparent")
        btn_box.pack(side="right")

        self.btn_cfg = ctk.CTkButton(btn_box, text="⚙ Settings", width=96, command=lambda: self.tabs.set("Settings"))
        self.btn_cfg.pack(side="right")

        self.btn_save = ctk.CTkButton(
            btn_box,
            text="💾 Save layout",
            width=130,
            fg_color=("#1f6aa5", "#1f6aa5"),
            hover_color=("#1b5c8c", "#1b5c8c"),
            command=self._on_save_clicked,
        )
        self.btn_save.pack(side="right", padx=(0, 8))

        # Quick controls row
        ctrl = ctk.CTkFrame(self.tab_controls, fg_color="transparent")
        ctrl.pack(fill="x", padx=8, pady=(0, 8))
        ctrl.columnconfigure(0, weight=1)
        ctrl.columnconfigure(1, weight=1)
        ctrl.columnconfigure(2, weight=0)
        ctrl.columnconfigure(3, weight=0)

        # Method dropdown
        self.var_method = ctk.StringVar(value=self.METHOD_LABELS.get(self.config_data["fuel"]["method"], self.METHOD_LABELS["road_fav"]))
        self.opt_method = ctk.CTkOptionMenu(
            ctrl,
            values=list(self.METHOD_LABELS.values()),
            variable=self.var_method,
            command=self._on_method_change,
            width=340,
        )
        ctk.CTkLabel(ctrl, text="Estimativa de consumo", anchor="w").grid(row=0, column=0, sticky="w")
        self.opt_method.grid(row=1, column=0, sticky="ew", padx=(0, 8))

        # Plan dropdown
        self.var_plan = ctk.StringVar(value=self.PLAN_LABELS.get(self.config_data["fuel"].get("plan_mode", "safe"), self.PLAN_LABELS["safe"]))
        self.opt_plan = ctk.CTkOptionMenu(
            ctrl,
            values=[self.PLAN_LABELS[k] for k in self.PLAN_LABELS],
            variable=self.var_plan,
            command=self._on_plan_change,
            width=110,
        )
        ctk.CTkLabel(ctrl, text="Modo de plano", anchor="w").grid(row=0, column=1, sticky="w")
        self.opt_plan.grid(row=1, column=1, sticky="ew")

        # N
        entry_box = ctk.CTkFrame(ctrl, fg_color="transparent")
        entry_box.grid(row=1, column=2, padx=(12, 6), sticky="e")
        ctk.CTkLabel(entry_box, text="N laps", width=54).pack(anchor="e")
        self.entry_n = ctk.CTkEntry(entry_box, width=60)
        self.entry_n.insert(0, str(self.config_data["fuel"].get("n", 10)))
        self.entry_n.pack(anchor="e", pady=(2, 0))

        # %
        pct_box = ctk.CTkFrame(ctrl, fg_color="transparent")
        pct_box.grid(row=1, column=3, padx=(6, 0), sticky="e")
        ctk.CTkLabel(pct_box, text="Top %", width=54).pack(anchor="e")
        self.entry_pct = ctk.CTkEntry(pct_box, width=60)
        self.entry_pct.insert(0, str(self.config_data["fuel"].get("top_percent", 20)))
        self.entry_pct.pack(anchor="e", pady=(2, 0))

        # Ignore yellow
        self.var_ignore = ctk.BooleanVar(value=bool(self.config_data["fuel"].get("ignore_yellow", True)))
        self.chk_ignore = ctk.CTkCheckBox(ctrl, text="Ignorar voltas em amarelo", variable=self.var_ignore, command=self._on_ignore_toggle)
        self.chk_ignore.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Status line
        self.lbl_status = ctk.CTkLabel(self.tab_controls, text="iRacing: waiting…", anchor="w")
        self.lbl_status.pack(fill="x", padx=8)

    def _setup_overlay_scroll_bindings(self) -> None:
        """Make sure mouse wheel scrolling works reliably for the overlay section list.

        CustomTkinter's CTkScrollableFrame usually handles this, but depending on
        platform / focus / Tabview, the default bindings can be unreliable.
        We enable scrolling only while the cursor is over the scrollable area.
        """

        self._overlay_scroll_enabled = False

        def _enter(_event=None):
            self._overlay_scroll_enabled = True

        def _leave(_event=None):
            self._overlay_scroll_enabled = False

        def _get_canvas():
            try:
                # CustomTkinter uses a Canvas internally. Name can vary by version.
                for name in ("_parent_canvas", "_canvas"):
                    cv = getattr(self.section_container, name, None)
                    if cv is not None:
                        return cv
                return None
            except Exception:
                return None

        def _on_mousewheel(event):
            if not getattr(self, "_overlay_scroll_enabled", False):
                return
            canvas = _get_canvas()
            if canvas is None:
                return
            try:
                # Windows: event.delta is typically +/-120 per notch.
                # macOS / touchpads: delta can be smaller but more frequent.
                delta_raw = int(event.delta) if event.delta else 0
                step = int(-1 * (delta_raw / 120))
                if step == 0:
                    step = -1 if delta_raw > 0 else 1
                canvas.yview_scroll(step * 2, "units")
            except Exception:
                pass

        def _on_button4(_event):
            if not getattr(self, "_overlay_scroll_enabled", False):
                return
            canvas = _get_canvas()
            if canvas is None:
                return
            try:
                canvas.yview_scroll(-2, "units")
            except Exception:
                pass

        def _on_button5(_event):
            if not getattr(self, "_overlay_scroll_enabled", False):
                return
            canvas = _get_canvas()
            if canvas is None:
                return
            try:
                canvas.yview_scroll(2, "units")
            except Exception:
                pass

        # Enable/disable scrolling depending on hover.
        try:
            self.section_container.bind("<Enter>", _enter)
            self.section_container.bind("<Leave>", _leave)
        except Exception:
            pass

        # Also bind to the internal canvas if it exists (more reliable hover).
        try:
            canvas = _get_canvas()
            if canvas is not None:
                canvas.bind("<Enter>", _enter)
                canvas.bind("<Leave>", _leave)
        except Exception:
            pass

        # Global bindings gated by hover flag.
        # (bind_all is OK because we only scroll when hover is enabled)
        try:
            self.bind_all("<MouseWheel>", _on_mousewheel)
        except Exception:
            pass
        try:
            self.bind_all("<Button-4>", _on_button4)
            self.bind_all("<Button-5>", _on_button5)
        except Exception:
            pass

    def _mk_section(self, parent, title: str, key: str):
        sec = ctk.CTkFrame(parent, corner_radius=10)
        sec.pack(fill="x", pady=(0, 10))
        hdr = ctk.CTkFrame(sec, fg_color="transparent")
        hdr.pack(fill="x", padx=8, pady=(6, 0))
        ctk.CTkLabel(hdr, text=title, font=("Segoe UI", 11, "bold"), anchor="w").pack(side="left")
        ctk.CTkButton(hdr, text="Detach", width=64, command=lambda k=key: self._detach_section(k)).pack(side="right")
        self.section_defs[key] = {"title": title, "labels": []}
        return sec

    def _register_section_label(
        self,
        key: str,
        var: ctk.StringVar,
        font: Tuple[str, int, str] | Tuple[str, int],
        *,
        padx: int = 0,
        pady: Tuple[int, int] | int = 0,
        anchor: str = "w",
        justify: str = "center",
    ) -> None:
        try:
            meta = self.section_defs.setdefault(key, {"title": key, "labels": []})
            meta.setdefault("labels", []).append(
                {
                    "var": var,
                    "font": font,
                    "padx": padx,
                    "pady": pady,
                    "anchor": anchor,
                    "justify": justify,
                }
            )
        except Exception:
            pass

    def _apply_detached_window_flags(self, win: ctk.CTkToplevel) -> None:
        try:
            win.attributes("-topmost", bool(self.config_data["ui"].get("always_on_top", True)))
        except Exception:
            pass
        try:
            win.overrideredirect(bool(self.config_data["ui"].get("borderless", True)))
        except Exception:
            pass

    def _detach_section(self, key: str) -> None:
        meta = self.section_defs.get(key)
        if not meta:
            return

        try:
            existing = self.detached_windows.get(key)
            if existing is not None and existing.winfo_exists():
                existing.deiconify()
                existing.lift()
                return
        except Exception:
            pass

        win = ctk.CTkToplevel(self)
        win.title(meta.get("title", key))
        self._apply_detached_window_flags(win)

        frame = ctk.CTkFrame(win, corner_radius=10)
        frame.pack(fill="both", expand=True, padx=8, pady=8)

        for lbl in meta.get("labels", []):
            try:
                ctk.CTkLabel(
                    frame,
                    textvariable=lbl.get("var"),
                    font=lbl.get("font"),
                    anchor=lbl.get("anchor", "w"),
                    justify=lbl.get("justify", "center"),
                ).pack(fill="x", padx=lbl.get("padx", 0), pady=lbl.get("pady", 0))
            except Exception:
                pass

        win.protocol("WM_DELETE_WINDOW", lambda k=key: self._close_detached(k))
        self.detached_windows[key] = win
        self._wire_drag_bindings()

    def _close_detached(self, key: str) -> None:
        try:
            win = self.detached_windows.get(key)
            if win is not None:
                win.destroy()
        except Exception:
            pass
        self.detached_windows.pop(key, None)

    def _build_settings_tab(self) -> None:
        sf = ctk.CTkScrollableFrame(self.tab_settings)
        sf.pack(fill="both", expand=True, padx=8, pady=8)

        # -------- UI settings
        ctk.CTkLabel(sf, text="UI", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", pady=(0, 6))
        self.var_topmost = ctk.BooleanVar(value=bool(self.config_data["ui"].get("always_on_top", True)))
        ctk.CTkCheckBox(sf, text="Always on top", variable=self.var_topmost).pack(anchor="w")

        self.var_borderless = ctk.BooleanVar(value=bool(self.config_data["ui"].get("borderless", True)))
        ctk.CTkCheckBox(sf, text="Borderless window (overrideredirect)", variable=self.var_borderless).pack(anchor="w")

        self.var_drag = ctk.BooleanVar(value=bool(self.config_data["ui"].get("drag_enabled", True)))
        ctk.CTkCheckBox(sf, text="Enable drag to move", variable=self.var_drag).pack(anchor="w")

        self.var_easy_mode = ctk.BooleanVar(value=bool(self.config_data["ui"].get("easy_mode", False)))
        ctk.CTkCheckBox(
            sf,
            text="Easy mode overlay (simplified information)",
            variable=self.var_easy_mode,
        ).pack(anchor="w")

        row_refresh = ctk.CTkFrame(sf, fg_color="transparent")
        row_refresh.pack(fill="x", pady=(6, 2))
        ctk.CTkLabel(row_refresh, text="Refresh (ms):", width=120, anchor="w").pack(side="left")
        self.entry_refresh = ctk.CTkEntry(row_refresh, width=80)
        self.entry_refresh.insert(0, str(self.config_data["ui"].get("refresh_ms", 150)))
        self.entry_refresh.pack(side="left")

        # Sections to show
        ctk.CTkLabel(sf, text="Overlay Sections", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", pady=(12, 6))
        show = self.config_data["ui"].get("show_sections", {})
        self.var_show_fuel = ctk.BooleanVar(value=bool(show.get("fuel", True)))
        self.var_show_pit = ctk.BooleanVar(value=bool(show.get("pit", True)))
        self.var_show_weather = ctk.BooleanVar(value=bool(show.get("weather", True)))
        self.var_show_risk = ctk.BooleanVar(value=bool(show.get("risk", True)))
        self.var_show_hotkeys = ctk.BooleanVar(value=bool(show.get("hotkeys", True)))

        ctk.CTkCheckBox(sf, text="Fuel & Race", variable=self.var_show_fuel).pack(anchor="w")
        ctk.CTkCheckBox(sf, text="Pit Window", variable=self.var_show_pit).pack(anchor="w")
        ctk.CTkCheckBox(sf, text="Weather/Tires", variable=self.var_show_weather).pack(anchor="w")
        ctk.CTkCheckBox(sf, text="Risk Radar", variable=self.var_show_risk).pack(anchor="w")
        ctk.CTkCheckBox(sf, text="Hotkeys line", variable=self.var_show_hotkeys).pack(anchor="w")

        # -------- Fuel settings
        ctk.CTkLabel(sf, text="Fuel Strategy", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", pady=(12, 6))

        # Burn estimation
        ctk.CTkLabel(sf, text="Burn Estimation", font=("Segoe UI", 11, "bold"), anchor="w").pack(fill="x", pady=(6, 4))

        row_method = ctk.CTkFrame(sf, fg_color="transparent")
        row_method.pack(fill="x", pady=2)
        ctk.CTkLabel(row_method, text="Method:", width=120, anchor="w").pack(side="left")
        self.var_method_settings = ctk.StringVar(
            value=self.METHOD_LABELS.get(self.config_data["fuel"].get("method", "road_fav"), self.METHOD_LABELS["road_fav"])
        )
        self.opt_method_settings = ctk.CTkOptionMenu(
            row_method,
            values=list(self.METHOD_LABELS.values()),
            variable=self.var_method_settings,
            width=320,
        )
        self.opt_method_settings.pack(side="left")

        row_npct = ctk.CTkFrame(sf, fg_color="transparent")
        row_npct.pack(fill="x", pady=2)
        ctk.CTkLabel(row_npct, text="N:", width=120, anchor="w").pack(side="left")
        self.entry_n_settings = ctk.CTkEntry(row_npct, width=80)
        self.entry_n_settings.insert(0, str(self.config_data["fuel"].get("n", 10)))
        self.entry_n_settings.pack(side="left")
        ctk.CTkLabel(row_npct, text="Top %:", width=70, anchor="w").pack(side="left", padx=(14, 0))
        self.entry_pct_settings = ctk.CTkEntry(row_npct, width=80)
        self.entry_pct_settings.insert(0, str(self.config_data["fuel"].get("top_percent", 20)))
        self.entry_pct_settings.pack(side="left")

        self.var_ignore_settings = ctk.BooleanVar(value=bool(self.config_data["fuel"].get("ignore_yellow", True)))
        ctk.CTkCheckBox(sf, text="Ignore yellow/caution laps for burn", variable=self.var_ignore_settings).pack(anchor="w", pady=(2, 0))

        row_margin = ctk.CTkFrame(sf, fg_color="transparent")
        row_margin.pack(fill="x", pady=2)
        ctk.CTkLabel(row_margin, text="Base margin laps:", width=120, anchor="w").pack(side="left")
        self.entry_margin = ctk.CTkEntry(row_margin, width=80)
        self.entry_margin.insert(0, str(self.config_data["fuel"].get("margin_laps", 1.0)))
        self.entry_margin.pack(side="left")

        row_margin_step = ctk.CTkFrame(sf, fg_color="transparent")
        row_margin_step.pack(fill="x", pady=2)
        ctk.CTkLabel(row_margin_step, text="Margin step:", width=120, anchor="w").pack(side="left")
        self.entry_margin_step = ctk.CTkEntry(row_margin_step, width=80)
        self.entry_margin_step.insert(0, str(self.config_data["fuel"].get("margin_step", 0.5)))
        self.entry_margin_step.pack(side="left")

        # Fuel plan mode
        row_fuelplan = ctk.CTkFrame(sf, fg_color="transparent")
        row_fuelplan.pack(fill="x", pady=2)
        ctk.CTkLabel(row_fuelplan, text="Fuel plan:", width=120, anchor="w").pack(side="left")
        self.var_fuel_plan = ctk.StringVar(value=self.FUEL_PLAN_LABELS.get(self.config_data["fuel"].get("fuel_plan_mode", "finish"), self.FUEL_PLAN_LABELS["finish"]))
        self.opt_fuel_plan = ctk.CTkOptionMenu(row_fuelplan, values=list(self.FUEL_PLAN_LABELS.values()), variable=self.var_fuel_plan, width=220)
        self.opt_fuel_plan.pack(side="left")

        row_stint = ctk.CTkFrame(sf, fg_color="transparent")
        row_stint.pack(fill="x", pady=2)
        ctk.CTkLabel(row_stint, text="Stint target laps:", width=120, anchor="w").pack(side="left")
        self.entry_stint = ctk.CTkEntry(row_stint, width=80)
        self.entry_stint.insert(0, str(self.config_data["fuel"].get("stint_target_laps", 20.0)))
        self.entry_stint.pack(side="left")

        row_tank = ctk.CTkFrame(sf, fg_color="transparent")
        row_tank.pack(fill="x", pady=2)
        ctk.CTkLabel(row_tank, text="Tank capacity (telemetry):", width=200, anchor="w").pack(side="left")
        self.var_tank_capacity = ctk.StringVar()
        ctk.CTkLabel(row_tank, textvariable=self.var_tank_capacity, anchor="w").pack(side="left")
        self._update_tank_capacity_display(self.config_data["fuel"].get("tank_capacity"))

        # -------- Pit model settings
        ctk.CTkLabel(sf, text="Pit Time Model", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", pady=(12, 6))

        self.var_tires_with_fuel = ctk.BooleanVar(value=bool(self.config_data["pit"].get("tires_with_fuel", False)))
        ctk.CTkCheckBox(
            sf,
            text="Series fuels and changes tires together (no extra tire time)",
            variable=self.var_tires_with_fuel,
            command=self._refresh_pit_entry_state,
        ).pack(anchor="w")
        self.entry_clean_air = self._settings_entry(sf, "Clean air target (s):", self.config_data["pit"].get("clean_air_target_s", 2.0))
        self.entry_offsets = self._settings_entry(sf, "Lookahead laps:", self.config_data["pit"].get("max_offsets", 3))

        self.pit_model_summary = ctk.StringVar()
        ctk.CTkLabel(
            sf,
            textvariable=self.pit_model_summary,
            anchor="w",
            wraplength=520,
            text_color="#b0b0b0",
        ).pack(fill="x", pady=(4, 0))

        self.pit_profile_label = ctk.StringVar()
        ctk.CTkLabel(
            sf,
            textvariable=self.pit_profile_label,
            anchor="w",
            wraplength=520,
            text_color="#9ad6ff",
        ).pack(fill="x", pady=(2, 0))

        self.pit_profile_info = ctk.StringVar()
        ctk.CTkLabel(
            sf,
            textvariable=self.pit_profile_info,
            anchor="w",
            wraplength=520,
            text_color="#a0a0a0",
        ).pack(fill="x", pady=(4, 6))

        self.var_pit_advanced = ctk.BooleanVar(value=bool(self.config_data["pit"].get("advanced_editing", False)))
        ctk.CTkCheckBox(
            sf,
            text="Advanced user: allow editing pit timings",
            variable=self.var_pit_advanced,
            command=self._refresh_pit_entry_state,
        ).pack(anchor="w", pady=(0, 4))

        self.entry_pit_base = self._settings_entry(sf, "Base loss (s):", self.config_data["pit"].get("pit_base_loss_s", ""))
        self.entry_pit_fill = self._settings_entry(sf, "Fill rate (u/s):", self.config_data["pit"].get("fuel_fill_rate", ""))
        self.entry_pit_tires = self._settings_entry(
            sf,
            "Tire time (s):",
            self.config_data["pit"].get("tire_service_time_s", ""),
        )

        # Auto calibration toggles
        ctk.CTkLabel(sf, text="Auto-calibration (from telemetry pit stops)", font=("Segoe UI", 11, "bold"), anchor="w").pack(fill="x", pady=(10, 4))
        ac = self.config_data["pit"].get("auto_calibrate", {})
        self.var_ac_enabled = ctk.BooleanVar(value=bool(ac.get("enabled", True)))
        self.var_ac_base = ctk.BooleanVar(value=bool(ac.get("update_base_loss", True)))
        self.var_ac_fill = ctk.BooleanVar(value=bool(ac.get("update_fill_rate", True)))
        self.var_ac_tire = ctk.BooleanVar(value=bool(ac.get("update_tire_time", True)))

        ctk.CTkCheckBox(sf, text="Enable", variable=self.var_ac_enabled, command=self._refresh_pit_entry_state).pack(anchor="w")
        ctk.CTkCheckBox(sf, text="Update base loss", variable=self.var_ac_base, command=self._refresh_pit_entry_state).pack(anchor="w")
        ctk.CTkCheckBox(sf, text="Update fill rate", variable=self.var_ac_fill, command=self._refresh_pit_entry_state).pack(anchor="w")
        ctk.CTkCheckBox(sf, text="Update tire time", variable=self.var_ac_tire, command=self._refresh_pit_entry_state).pack(anchor="w")

        self._refresh_pit_entry_state()

        # -------- Audio
        ctk.CTkLabel(sf, text="Audio", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", pady=(12, 6))
        self.var_beep = ctk.BooleanVar(value=bool(self.config_data["audio"].get("risk_beep", True)))
        ctk.CTkCheckBox(sf, text="Risk beep", variable=self.var_beep).pack(anchor="w")

        self.entry_beep_thr = self._settings_entry(sf, "Beep threshold:", self.config_data["audio"].get("risk_beep_threshold", 85))
        self.entry_beep_cd = self._settings_entry(sf, "Beep cooldown (s):", self.config_data["audio"].get("beep_cooldown_s", 3.0))

        # -------- Hotkeys
        ctk.CTkLabel(sf, text="Hotkeys", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", pady=(12, 6))
        hk_cfg = self.config_data.get("hotkeys", {})
        self.entry_hk_margin_up = self._settings_entry(sf, "Margin up:", hk_cfg.get("margin_up", ""), width=180)
        self.entry_hk_margin_down = self._settings_entry(sf, "Margin down:", hk_cfg.get("margin_down", ""), width=180)
        self.entry_hk_cycle_plan = self._settings_entry(sf, "Cycle plan:", hk_cfg.get("cycle_plan", ""), width=180)
        self.entry_hk_apply_opt1 = self._settings_entry(sf, "Apply opt #1:", hk_cfg.get("apply_opt1", ""), width=180)
        self.entry_hk_apply_opt2 = self._settings_entry(sf, "Apply opt #2:", hk_cfg.get("apply_opt2", ""), width=180)
        self.entry_hk_apply_opt3 = self._settings_entry(sf, "Apply opt #3:", hk_cfg.get("apply_opt3", ""), width=180)
        self.entry_hk_test_macro = self._settings_entry(
            sf, "Test fuel macro:", hk_cfg.get("test_macro", ""), width=180
        )
        self.entry_hk_toggle_settings = self._settings_entry(
            sf, "Toggle settings:", hk_cfg.get("toggle_settings", ""), width=180
        )

        # -------- Macros
        ctk.CTkLabel(sf, text="Chat Macros", font=("Segoe UI", 12, "bold"), anchor="w").pack(fill="x", pady=(12, 6))
        self.var_macro_enabled = ctk.BooleanVar(value=bool(self.config_data["macro"].get("enabled", False)))
        ctk.CTkCheckBox(sf, text="Enable chat macros", variable=self.var_macro_enabled).pack(anchor="w")

        self.var_macro_auto_fuel = ctk.BooleanVar(
            value=bool(self.config_data["macro"].get("auto_fuel_on_pit", False))
        )
        ctk.CTkCheckBox(
            sf,
            text="Auto fuel on pit entry",
            variable=self.var_macro_auto_fuel,
        ).pack(anchor="w")

        self.entry_chatkey = self._settings_entry(sf, "Chat key:", self.config_data["macro"].get("chat_key", "t"), width=120)

        ctk.CTkButton(
            sf,
            text="Testar macro de fuel",
            command=self._on_test_macro_clicked,
        ).pack(anchor="w", pady=(6, 0))

        # -------- Buttons
        btns = ctk.CTkFrame(sf, fg_color="transparent")
        btns.pack(fill="x", pady=(16, 0))
        ctk.CTkButton(btns, text="Apply", command=self._on_apply_clicked, width=120).pack(side="left")
        ctk.CTkButton(btns, text="Save", command=self._on_save_clicked, width=120).pack(side="left", padx=(10, 0))
        ctk.CTkButton(btns, text="Close", command=lambda: self.tabs.set("Overlay"), width=120).pack(side="right")

    def _settings_entry(self, parent, label: str, value: Any, width: int = 80):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=2)
        ctk.CTkLabel(row, text=label, width=160, anchor="w").pack(side="left")
        ent = ctk.CTkEntry(row, width=width)
        ent.insert(0, str(value))
        ent.pack(side="left")
        return ent

    @staticmethod
    def _entry_disabled(entry: Any) -> bool:
        try:
            return str(entry.cget("state")) == "disabled"
        except Exception:
            return False

    @staticmethod
    def _set_entry_value(entry: Any, value: Any) -> None:
        try:
            state = entry.cget("state")
            entry.configure(state="normal")
            entry.delete(0, ctk.END)
            entry.insert(0, str(value))
            entry.configure(state=state)
        except Exception:
            pass

    @staticmethod
    def _set_entry_enabled(entry: Any, enabled: bool) -> None:
        try:
            entry.configure(state="normal" if enabled else "disabled")
        except Exception:
            pass

    def _refresh_pit_entry_state(self) -> None:
        advanced = bool(self.var_pit_advanced.get())
        for ent in (self.entry_pit_base, self.entry_pit_fill, self.entry_pit_tires):
            self._set_entry_enabled(ent, advanced)

        try:
            self.config_data.setdefault("pit", {})["advanced_editing"] = advanced
        except Exception:
            pass

    def _sync_pit_entries_from_config(self) -> None:
        pit_cfg = self.config_data.get("pit", {})
        self._pit_profile_has_data = bool(pit_cfg.get("has_calibration", False))
        base = pit_cfg.get("pit_base_loss_s")
        fill_rate = pit_cfg.get("fuel_fill_rate")
        tire_time = pit_cfg.get("tire_service_time_s")
        tires_with_fuel = bool(pit_cfg.get("tires_with_fuel", False))

        pieces = []
        if base is not None:
            pieces.append(f"base={base:.1f}s")
        if fill_rate is not None:
            pieces.append(f"fill={fill_rate:.2f}u/s")
        if tire_time is not None:
            pieces.append(f"tires={tire_time:.1f}s")
        if tires_with_fuel:
            pieces.append("tires with fuel")

        if self._pit_profile_has_data:
            summary = "Pit timings (car/track): " + (" | ".join(pieces) if pieces else "--")
            info = "Informational values used for calculations; edit in advanced mode only if necessary."
        else:
            summary = "Pit timings: waiting for the first calibration for this car/track."
            info = "Complete a test pit stop to record timings or enable advanced mode to fill them manually."

        try:
            self.pit_model_summary.set(summary)
            self.var_tires_with_fuel.set(tires_with_fuel)
            self.var_pit_advanced.set(bool(pit_cfg.get("advanced_editing", False)))
            self._set_entry_value(self.entry_pit_base, "" if base is None else base)
            self._set_entry_value(self.entry_pit_fill, "" if fill_rate is None else fill_rate)
            self._set_entry_value(self.entry_pit_tires, "" if tire_time is None else tire_time)
            self.pit_profile_info.set(info)
        except Exception:
            pass

    def _apply_section_visibility(self) -> None:
        """Show/hide overlay sections while keeping a stable order."""
        show = self.config_data["ui"].get("show_sections", {})

        sections = [
            ("fuel", self.sec_fuel),
            ("pit", self.sec_pit),
            ("weather", self.sec_weather),
            ("risk", self.sec_risk),
            ("hotkeys", self.sec_hotkeys),
        ]

        # remove all first (so we can re-pack in a stable order)
        for _, w in sections:
            try:
                w.pack_forget()
            except Exception:
                pass

        # re-pack in order
        for key, w in sections:
            if bool(show.get(key, True)):
                try:
                    w.pack(fill="x", pady=(0, 10))
                except Exception:
                    pass

    def _easy_mode_enabled(self) -> bool:
        try:
            return bool(self.config_data.get("ui", {}).get("easy_mode", False))
        except Exception:
            return False

    # ----------------------------
    # UI callbacks
    # ----------------------------

    def _method_key_from_label(self, label: str) -> str:
        for k, v in self.METHOD_LABELS.items():
            if v == label:
                return k
        return "road_fav"

    def _plan_key_from_label(self, label: str) -> str:
        for k, v in self.PLAN_LABELS.items():
            if v == label:
                return k
        return "safe"

    def _fuelplan_key_from_label(self, label: str) -> str:
        for k, v in self.FUEL_PLAN_LABELS.items():
            if v == label:
                return k
        return "finish"

    def _on_method_change(self, chosen_label: str) -> None:
        self.config_data["fuel"]["method"] = self._method_key_from_label(chosen_label)
        try:
            self.var_method_settings.set(chosen_label)
        except Exception:
            pass

    def _on_plan_change(self, chosen_label: str) -> None:
        self.config_data["fuel"]["plan_mode"] = self._plan_key_from_label(chosen_label)

    def _on_ignore_toggle(self) -> None:
        self.config_data["fuel"]["ignore_yellow"] = bool(self.var_ignore.get())
        self.history.ignore_yellow = bool(self.var_ignore.get())
        try:
            self.var_ignore_settings.set(bool(self.var_ignore.get()))
        except Exception:
            pass

    def _on_apply_clicked(self) -> None:
        self._pull_settings_into_config()
        self._apply_config_runtime()

    def _on_save_clicked(self) -> None:
        self._pull_settings_into_config()
        self._apply_config_runtime()
        self._save_config()

    def _on_test_macro_clicked(self) -> None:
        """Testa o macro de fuel com um valor padrão."""

        self._pull_settings_into_config()
        macro_enabled = bool(self.config_data.get("macro", {}).get("enabled", False))
        if not macro_enabled:
            return

        self._dispatch_fuel_macro(5.0, offset_laps=0, pit_loss_s=0.0, success_pct=0)

    def _pull_settings_into_config(self) -> None:
        # UI
        self.config_data["ui"]["always_on_top"] = bool(self.var_topmost.get())
        self.config_data["ui"]["borderless"] = bool(self.var_borderless.get())
        self.config_data["ui"]["drag_enabled"] = bool(self.var_drag.get())
        self.config_data["ui"]["easy_mode"] = bool(self.var_easy_mode.get())
        try:
            self.config_data["ui"]["refresh_ms"] = int(float(self.entry_refresh.get()))
        except Exception:
            pass

        self.config_data["ui"]["show_sections"] = {
            "fuel": bool(self.var_show_fuel.get()),
            "pit": bool(self.var_show_pit.get()),
            "weather": bool(self.var_show_weather.get()),
            "risk": bool(self.var_show_risk.get()),
            "hotkeys": bool(self.var_show_hotkeys.get()),
        }

        # Fuel

        # Burn estimation (Settings tab)
        try:
            self.config_data["fuel"]["method"] = self._method_key_from_label(str(self.var_method_settings.get()))
        except Exception:
            pass
        try:
            self.config_data["fuel"]["n"] = int(float(self.entry_n_settings.get()))
        except Exception:
            pass
        try:
            self.config_data["fuel"]["top_percent"] = float(self.entry_pct_settings.get())
        except Exception:
            pass
        try:
            self.config_data["fuel"]["ignore_yellow"] = bool(self.var_ignore_settings.get())
        except Exception:
            pass
        try:
            self.config_data["fuel"]["margin_laps"] = float(self.entry_margin.get())
        except Exception:
            pass
        try:
            self.config_data["fuel"]["margin_step"] = float(self.entry_margin_step.get())
        except Exception:
            pass
        self.config_data["fuel"]["fuel_plan_mode"] = self._fuelplan_key_from_label(str(self.var_fuel_plan.get()))
        try:
            self.config_data["fuel"]["stint_target_laps"] = float(self.entry_stint.get())
        except Exception:
            pass

        # Pit
        try:
            self.config_data["pit"]["advanced_editing"] = bool(self.var_pit_advanced.get())
        except Exception:
            pass
        try:
            self.config_data["pit"]["tires_with_fuel"] = bool(self.var_tires_with_fuel.get())
        except Exception:
            pass
        if bool(self.var_pit_advanced.get()):
            try:
                self.config_data["pit"]["pit_base_loss_s"] = float(self.entry_pit_base.get())
            except Exception:
                pass
            try:
                self.config_data["pit"]["fuel_fill_rate"] = float(self.entry_pit_fill.get())
            except Exception:
                pass
            try:
                self.config_data["pit"]["tire_service_time_s"] = float(self.entry_pit_tires.get())
            except Exception:
                pass
            try:
                self.config_data["pit"]["has_calibration"] = True
                self._pit_profile_has_data = True
            except Exception:
                pass
        try:
            self.config_data["pit"]["clean_air_target_s"] = float(self.entry_clean_air.get())
        except Exception:
            pass
        try:
            self.config_data["pit"]["max_offsets"] = int(float(self.entry_offsets.get()))
        except Exception:
            pass

        ac = self.config_data["pit"].get("auto_calibrate", {})
        ac["enabled"] = bool(self.var_ac_enabled.get())
        ac["update_base_loss"] = bool(self.var_ac_base.get())
        ac["update_fill_rate"] = bool(self.var_ac_fill.get())
        ac["update_tire_time"] = bool(self.var_ac_tire.get())
        self.config_data["pit"]["auto_calibrate"] = ac

        # Audio
        self.config_data["audio"]["risk_beep"] = bool(self.var_beep.get())
        try:
            self.config_data["audio"]["risk_beep_threshold"] = int(float(self.entry_beep_thr.get()))
        except Exception:
            pass
        try:
            self.config_data["audio"]["beep_cooldown_s"] = float(self.entry_beep_cd.get())
        except Exception:
            pass

        # Hotkeys
        hk_cfg = self.config_data.setdefault("hotkeys", {})
        hk_entries = {
            "margin_up": self.entry_hk_margin_up,
            "margin_down": self.entry_hk_margin_down,
            "cycle_plan": self.entry_hk_cycle_plan,
            "apply_opt1": self.entry_hk_apply_opt1,
            "apply_opt2": self.entry_hk_apply_opt2,
            "apply_opt3": self.entry_hk_apply_opt3,
            "test_macro": self.entry_hk_test_macro,
            "toggle_settings": self.entry_hk_toggle_settings,
        }
        for key, entry in hk_entries.items():
            try:
                val = str(entry.get()).strip()
                if val:
                    hk_cfg[key] = val
                else:
                    hk_cfg.pop(key, None)
            except Exception:
                pass

        # Macro
        self.config_data["macro"]["enabled"] = bool(self.var_macro_enabled.get())
        self.config_data["macro"]["chat_key"] = str(self.entry_chatkey.get() or "t")
        self.config_data["macro"]["auto_fuel_on_pit"] = bool(self.var_macro_auto_fuel.get())

        self._persist_active_profile()

    def _apply_config_runtime(self) -> None:
        # window flags
        try:
            self.attributes("-topmost", bool(self.config_data["ui"].get("always_on_top", True)))
        except Exception:
            pass

        # borderless: can only be set reliably before mainloop on some platforms.
        # We'll attempt, but if it fails, ignore.
        try:
            self.overrideredirect(bool(self.config_data["ui"].get("borderless", True)))
        except Exception:
            pass

        self._drag_enabled = bool(self.config_data["ui"].get("drag_enabled", True))

        self.history.ignore_yellow = bool(self.config_data["fuel"].get("ignore_yellow", True))

        # rebuild macro injector from latest config
        self.injector = self._build_injector_from_config()

        try:
            self.var_easy_mode.set(bool(self.config_data["ui"].get("easy_mode", False)))
        except Exception:
            pass

        # sync overlay quick controls with config
        try:
            self.var_method.set(self.METHOD_LABELS.get(self.config_data["fuel"].get("method", "road_fav"), self.METHOD_LABELS["road_fav"]))
        except Exception:
            pass
        try:
            self.var_method_settings.set(self.METHOD_LABELS.get(self.config_data["fuel"].get("method", "road_fav"), self.METHOD_LABELS["road_fav"]))
        except Exception:
            pass
        try:
            self.entry_n.delete(0, "end")
            self.entry_n.insert(0, str(self.config_data["fuel"].get("n", 10)))
        except Exception:
            pass
        try:
            self.entry_pct.delete(0, "end")
            self.entry_pct.insert(0, str(self.config_data["fuel"].get("top_percent", 20)))
        except Exception:
            pass
        try:
            self.var_ignore.set(bool(self.config_data["fuel"].get("ignore_yellow", True)))
        except Exception:
            pass
        try:
            self.var_ignore_settings.set(bool(self.config_data["fuel"].get("ignore_yellow", True)))
        except Exception:
            pass

        for win in list(self.detached_windows.values()):
            try:
                self._apply_detached_window_flags(win)
            except Exception:
                pass

        # macros / hotkeys
        try:
            hk_cfg = self.config_data.get("hotkeys", {})
            self._set_entry_value(self.entry_hk_margin_up, hk_cfg.get("margin_up", ""))
            self._set_entry_value(self.entry_hk_margin_down, hk_cfg.get("margin_down", ""))
            self._set_entry_value(self.entry_hk_cycle_plan, hk_cfg.get("cycle_plan", ""))
            self._set_entry_value(self.entry_hk_apply_opt1, hk_cfg.get("apply_opt1", ""))
            self._set_entry_value(self.entry_hk_apply_opt2, hk_cfg.get("apply_opt2", ""))
            self._set_entry_value(self.entry_hk_apply_opt3, hk_cfg.get("apply_opt3", ""))
            self._set_entry_value(self.entry_hk_test_macro, hk_cfg.get("test_macro", ""))
            self._set_entry_value(self.entry_hk_toggle_settings, hk_cfg.get("toggle_settings", ""))
        except Exception:
            pass

        self._remove_hotkeys()
        self._setup_hotkeys()

        # show/hide sections
        self._apply_section_visibility()
        self._sync_pit_entries_from_config()
        self._update_tank_capacity_display(self.config_data["fuel"].get("tank_capacity"))
        self._refresh_pit_entry_state()

    def _wire_drag_bindings(self) -> None:
        """Ensure drag handlers are attached to key containers for easier moves."""

        def _bind_drag(target: Any) -> None:
            if target is None:
                return
            try:
                target.bind("<ButtonPress-1>", self._on_mouse_down, add="+")
                target.bind("<B1-Motion>", self._on_mouse_drag, add="+")
                target.bind("<ButtonRelease-1>", self._on_mouse_up, add="+")
            except Exception:
                pass

        _bind_drag(self.frame if hasattr(self, "frame") else None)
        _bind_drag(self.tabs if hasattr(self, "tabs") else None)

        # also cover detached sections if they exist at binding time
        for win in list(self.detached_windows.values()):
            _bind_drag(win)

    # ----------------------------
    # Drag handlers
    # ----------------------------

    def _on_mouse_down(self, event) -> None:
        if not self._drag_enabled:
            return
        try:
            self._drag_target = event.widget.winfo_toplevel()
        except Exception:
            self._drag_target = None
            return

        try:
            base_x = self._drag_target.winfo_x()
            base_y = self._drag_target.winfo_y()
            self._drag_offset = (event.x_root - base_x, event.y_root - base_y)
        except Exception:
            self._drag_target = None
            self._drag_offset = None

    def _on_mouse_drag(self, event) -> None:
        if not self._drag_enabled or not self._drag_offset or self._drag_target is None:
            return
        x_off, y_off = self._drag_offset
        x = event.x_root - x_off
        y = event.y_root - y_off
        try:
            self._drag_target.geometry(f"+{x}+{y}")
        except Exception:
            pass

    def _on_mouse_up(self, event) -> None:
        if not self._drag_enabled:
            return
        self._drag_offset = None
        if self._drag_target is self:
            self._save_config()
        self._drag_target = None

    # ----------------------------
    # Hotkeys
    # ----------------------------

    def _setup_hotkeys(self) -> None:
        if keyboard is None:
            return

        hk = self.config_data.get("hotkeys", {})

        def margin_add(delta: float):
            step = float(self.config_data["fuel"].get("margin_step", 0.5))
            cur = float(self.config_data["fuel"].get("margin_laps", 1.0))
            self.config_data["fuel"]["margin_laps"] = max(0.0, cur + delta * step)
            self._save_config()

        def cycle_plan():
            keys = list(self.PLAN_LABELS.keys())
            cur = str(self.config_data["fuel"].get("plan_mode", "safe"))
            if cur not in keys:
                cur = "safe"
            idx = keys.index(cur)
            nxt = keys[(idx + 1) % len(keys)]
            self.config_data["fuel"]["plan_mode"] = nxt
            try:
                self.var_plan.set(self.PLAN_LABELS[nxt])
            except Exception:
                pass
            self._save_config()

        def apply_opt(rank: int):
            if not self.config_data.get("macro", {}).get("enabled", False):
                return
            if rank <= 0 or rank > len(self._last_pit_options):
                return
            opt = self._last_pit_options[rank - 1]
            if opt.fuel_add is None:
                return

            self._dispatch_fuel_macro(
                float(opt.fuel_add),
                offset_laps=int(opt.offset_laps),
                pit_loss_s=float(opt.pit_loss_s or 0.0),
                success_pct=int(opt.success_pct or 0),
            )

        def toggle_settings():
            cur = self.tabs.get()
            self.tabs.set("Settings" if cur != "Settings" else "Overlay")

        def trigger_test_macro():
            if not self.config_data.get("macro", {}).get("enabled", False):
                return
            self._dispatch_fuel_macro(5.0, offset_laps=0, pit_loss_s=0.0, success_pct=0)

        try:
            self._hk_ids.append(keyboard.add_hotkey(hk.get("margin_up", "ctrl+alt+up"), lambda: margin_add(+1)))
            self._hk_ids.append(keyboard.add_hotkey(hk.get("margin_down", "ctrl+alt+down"), lambda: margin_add(-1)))
            self._hk_ids.append(keyboard.add_hotkey(hk.get("cycle_plan", "ctrl+alt+p"), cycle_plan))

            self._hk_ids.append(keyboard.add_hotkey(hk.get("apply_opt1", "alt+1"), lambda: apply_opt(1)))
            self._hk_ids.append(keyboard.add_hotkey(hk.get("apply_opt2", "alt+2"), lambda: apply_opt(2)))
            self._hk_ids.append(keyboard.add_hotkey(hk.get("apply_opt3", "alt+3"), lambda: apply_opt(3)))

            self._hk_ids.append(keyboard.add_hotkey(hk.get("toggle_settings", "ctrl+alt+o"), toggle_settings))
            self._hk_ids.append(keyboard.add_hotkey(hk.get("test_macro", "ctrl+alt+m"), trigger_test_macro))
        except Exception:
            pass

    def _remove_hotkeys(self) -> None:
        if keyboard is None:
            return
        for hk_id in self._hk_ids:
            try:
                keyboard.remove_hotkey(hk_id)
            except Exception:
                pass
        self._hk_ids.clear()

    # ----------------------------
    # Telemetry helpers
    # ----------------------------

    def _safe_get(self, key: str) -> Any:
        try:
            return self.ir[key]
        except Exception:
            return None

    def _get_session_info(self) -> Optional[Dict[str, Any]]:
        """Return parsed SessionInfo YAML if available (cached)."""

        # Prefer a cached copy to avoid reparsing on every frame.
        if self._session_info_cache is not None:
            return self._session_info_cache

        # Newer pyirsdk exposes helpers; fall back to manual access.
        raw_info = None
        try:
            if hasattr(self.ir, "get_session_info"):
                data = self.ir.get_session_info()
                if isinstance(data, dict):
                    self._session_info_cache = data
                    return data
        except Exception:
            pass

        try:
            if hasattr(self.ir, "get_session_info_str"):
                raw_info = self.ir.get_session_info_str()
        except Exception:
            raw_info = None

        if raw_info is None:
            raw_info = self._safe_get("SessionInfo")

        if raw_info is None or yaml is None:
            return None

        try:
            info = yaml.safe_load(raw_info)
            if isinstance(info, dict):
                self._session_info_cache = info
                return info
        except Exception:
            return None
        return None

    def _session_track_length_m(self, session_info: Optional[Dict[str, Any]]) -> Optional[float]:
        if not session_info:
            return None
        try:
            return self._track_length_m(session_info.get("WeekendInfo", {}).get("TrackLength"))
        except Exception:
            return None

    def _session_tank_capacity_l(self, session_info: Optional[Dict[str, Any]]) -> Optional[float]:
        if not session_info:
            return None

        try:
            driver = session_info.get("DriverInfo", {})
            tank_l = driver.get("DriverCarFuelMaxLtr")
            max_pct = driver.get("DriverCarMaxFuelPct")
        except Exception:
            return None

        try:
            if tank_l is None:
                return None
            cap = float(tank_l)
            if max_pct is not None:
                pct = float(max_pct)
                if pct > 1.5:
                    pct = pct / 100.0
                pct = max(0.0, min(pct, 1.0))
                cap = cap * pct
            return cap if cap > 0 else None
        except Exception:
            return None

    def _update_tank_capacity_display(self, tank_capacity: Optional[float]) -> None:
        text = "--" if tank_capacity is None else f"{float(tank_capacity):.1f} L"
        try:
            self.var_tank_capacity.set(text)
        except Exception:
            pass

    def _auto_fill_tank_capacity(self, tank_capacity: Optional[float]) -> None:
        """Populate the UI/config tank capacity from telemetry when possible.

        To avoid fighting with manual user edits, only update when the stored
        value matches the previous auto-filled one (or is empty/None). Once the
        user edits the value with a different number, auto updates are suppressed.
        """

        if tank_capacity is None:
            return

        try:
            new_val = float(tank_capacity)
        except Exception:
            return

        cur_val = self.config_data["fuel"].get("tank_capacity")
        should_update = cur_val is None
        if cur_val is not None:
            try:
                should_update = self._auto_tank_capacity_l is not None and abs(float(cur_val) - self._auto_tank_capacity_l) < 1e-3
            except Exception:
                should_update = False

        if not should_update:
            return

        self._auto_tank_capacity_l = new_val
        try:
            self._update_tank_capacity_display(new_val)
        except Exception:
            return

        try:
            self.config_data["fuel"]["tank_capacity"] = new_val
        except Exception:
            pass

    @staticmethod
    def _track_length_m(val: Any) -> Optional[float]:
        """Converte TrackLength do iRacing para metros.

        Aceita:
          - número em km (ex: 4.5 -> 4500m)
          - número em m   (ex: 4200 -> 4200m)
          - string "4.5 km", "2.3 mi", "4200 m", etc.
        """
        if val is None:
            return None

        # 1) Tenta converter direto como número
        try:
            x = float(val)
        except Exception:
            # 2) Trata string com unidades
            try:
                s = str(val).strip().lower()

                # extrai primeiro número da string (com . ou ,)
                num_chars = []
                seen_digit = False
                for ch in s:
                    if ch.isdigit() or ch in ".,":   # parte numérica
                        num_chars.append(ch)
                        seen_digit = True
                    elif seen_digit:
                        # já começou número e chegou em algo não numérico → para
                        break

                if not num_chars:
                    return None

                num = "".join(num_chars).replace(",", ".")
                x = float(num)

                # decide unidade
                if "mi" in s:
                    # milhas → metros
                    return x * 1609.344
                if "km" in s:
                    return x * 1000.0
                if "m" in s:
                    return x

                # fallback: assume km se pista < 80 "unidades"
                if x < 80.0:
                    return x * 1000.0
                return x
            except Exception:
                return None
        else:
            # numérico puro
            if x < 80.0:
                return x * 1000.0  # provavelmente km
            return x

    # ----------------------------
    # Main loop
    # ----------------------------

    def _tick(self) -> None:
        refresh_ms = int(self.config_data["ui"].get("refresh_ms", 150))
        refresh_ms = max(60, min(refresh_ms, 500))
        entered_stall = False

        try:
            # read quick params
            try:
                n = int(float(self.entry_n.get()))
            except Exception:
                n = int(self.config_data["fuel"].get("n", 10))
            n = max(1, min(n, 250))
            self.config_data["fuel"]["n"] = n

            try:
                pct = float(self.entry_pct.get())
            except Exception:
                pct = float(self.config_data["fuel"].get("top_percent", 20))
            pct = max(1.0, min(pct, 100.0))
            self.config_data["fuel"]["top_percent"] = pct

            method = str(self.config_data["fuel"].get("method", "road_fav"))

            plan_mode = str(self.config_data["fuel"].get("plan_mode", "safe"))
            pm = self.config_data["fuel"].get("plan_modes", {}).get(plan_mode, {})
            base_margin = float(self.config_data["fuel"].get("margin_laps", 1.0))
            margin = float(pm.get("margin_override", base_margin))
            take_tires = bool(pm.get("take_tires", False))

            # fuel plan mode
            fuel_plan_mode = str(self.config_data["fuel"].get("fuel_plan_mode", "finish"))
            stint_target_laps = float(self.config_data["fuel"].get("stint_target_laps", 20.0))
            tank_capacity_cfg = self.config_data["fuel"].get("tank_capacity", None)
            try:
                tank_capacity_cfg = float(tank_capacity_cfg) if tank_capacity_cfg is not None else None
            except Exception:
                tank_capacity_cfg = None

            if not self.ir:
                self.lbl_status.configure(text="pyirsdk not found. Install: pip install pyirsdk")
                self.after(500, self._tick)
                return

            # connect
            if not (getattr(self.ir, "is_initialized", False) and getattr(self.ir, "is_connected", False)):
                self._session_info_cache = None
                try:
                    self.ir.startup()
                except Exception:
                    pass

            if not (getattr(self.ir, "is_initialized", False) and getattr(self.ir, "is_connected", False)):
                self.lbl_status.configure(text="iRacing: waiting for connection…")
                self.after(500, self._tick)
                return

            try:
                self.ir.freeze_var_buffer_latest()
            except Exception:
                pass

            session_info = self._get_session_info()
            self._maybe_switch_pit_profile(session_info)

            # telemetry
            fuel_level = self._safe_get("FuelLevel")
            lap_completed = self._safe_get("LapCompleted")
            session_time = self._safe_get("SessionTime")
            session_flags = self._safe_get("SessionFlags")
            is_on_track = self._safe_get("IsOnTrack")
            on_pit_road = self._safe_get("OnPitRoad")

            speed = self._safe_get("Speed")
            yaw_rate = self._safe_get("YawRate")
            lat_accel = self._safe_get("LatAccel")
            long_accel = self._safe_get("LongAccel")
            steer = self._safe_get("SteeringWheelAngle")
            brake = self._safe_get("Brake")

            lap_dist_pct = self._safe_get("LapDistPct")

            track_wetness = self._safe_get("TrackWetness")
            precipitation = self._safe_get("Precipitation")
            declared_wet = self._safe_get("WeatherDeclaredWet")
            track_temp = self._safe_get("TrackTemp")

            player_idx = self._safe_get("PlayerCarIdx")
            car_left_right = self._safe_get("CarLeftRight")

            car_idx_lapdist = self._safe_get("CarIdxLapDistPct")
            if car_idx_lapdist is not None:
                try:
                    car_idx_lapdist = list(car_idx_lapdist)
                except Exception:
                    car_idx_lapdist = None

            track_len_m = self._track_length_m(self._safe_get("TrackLength"))
            if track_len_m is None:
                track_len_m = self._session_track_length_m(session_info)

            # Tank capacity: prefer session info when available (DriverCarFuelMaxLtr/DriverCarMaxFuelPct)
            tank_capacity_session = self._session_tank_capacity_l(session_info)
            if tank_capacity_session is not None:
                self._auto_fill_tank_capacity(tank_capacity_session)
            tank_capacity = tank_capacity_session
            if tank_capacity is None:
                tank_capacity = tank_capacity_cfg

            # Pit window variables (if available)
            pits_open = self._safe_get("PitsOpen")
            in_pit_stall = self._safe_get("PlayerCarInPitStall")

            # Detectar entrada no pit stall (para auto fuel macro)
            try:
                if in_pit_stall is not None:
                    in_stall_flag = bool(in_pit_stall)
                else:
                    # fallback: em pit road e praticamente parado
                    if on_pit_road is not None and bool(on_pit_road) and speed is not None:
                        in_stall_flag = float(speed) < 1.0
                    else:
                        in_stall_flag = False
            except Exception:
                in_stall_flag = False

            if self._prev_in_pit_stall is None:
                entered_stall = False
            else:
                entered_stall = (self._prev_in_pit_stall is False and in_stall_flag is True)

            self._prev_in_pit_stall = in_stall_flag

            # Pit service selections (if available)
            pit_sv_flags = self._safe_get("PitSvFlags")
            pit_sv_fuel = self._safe_get("PitSvFuel")

            yellow_now = is_yellow_flag(int(session_flags or 0))

            # update fuel history
            self.history.update(
                lap_completed=lap_completed,
                session_time=session_time,
                fuel_level=fuel_level,
                lap_dist_pct=float(lap_dist_pct) if lap_dist_pct is not None else None,
                session_flags=session_flags,
                is_on_track=bool(is_on_track) if is_on_track is not None else None,
                on_pit_road=bool(on_pit_road) if on_pit_road is not None else None,
            )

            burn = self.history.burn_estimate(method=method, n=n, top_percent=pct)
            avg_lap_time = self.history.lap_time_estimate(method=method, n=n)
            lap_sigma = self.history.lap_time_stdev(method=method, n=n)

            # laps remaining
            laps_remain = self._estimate_laps_remaining(avg_lap_time)

            # compute fuel figures
            fuel = float(fuel_level or 0.0)
            laps_possible = None
            fuel_need_total = None
            fuel_to_add = None
            finish_leftover = None

            if burn and burn > 0:
                laps_possible = fuel / burn

                # total need depends on fuel plan mode
                mode = (fuel_plan_mode or "finish").lower()
                if mode == "finish":
                    if laps_remain is not None:
                        fuel_need_total = max(0.0, (laps_remain + margin) * burn)
                elif mode == "stint":
                    fuel_need_total = max(0.0, (stint_target_laps + margin) * burn)
                elif mode == "full":
                    if tank_capacity is not None and tank_capacity > 0:
                        fuel_need_total = float(tank_capacity)

                if fuel_need_total is not None:
                    fuel_to_add = max(0.0, fuel_need_total - fuel)
                    finish_leftover = fuel + fuel_to_add - fuel_need_total

            # Auto fuel: ao entrar no pit stall, manda macro com fuel_to_add
            macro_cfg = self.config_data.get("macro", {})
            auto_fuel_enabled = bool(macro_cfg.get("auto_fuel_on_pit", False))
            macro_enabled = bool(macro_cfg.get("enabled", False))
            if entered_stall and auto_fuel_enabled and macro_enabled and fuel_to_add is not None:
                fuel_add_val = max(0.0, float(fuel_to_add))
                self._dispatch_fuel_macro(
                    fuel_add_val,
                    offset_laps=0,
                    pit_loss_s=0.0,
                    success_pct=0,
                )

            # wetness brain
            now = time.time()
            try:
                self.wet_brain.update(
                    now=now,
                    lap_dist_pct=float(lap_dist_pct) if lap_dist_pct is not None else None,
                    track_wetness=float(track_wetness) if track_wetness is not None else None,
                    precipitation=float(precipitation) if precipitation is not None else None,
                    declared_wet=bool(declared_wet) if declared_wet is not None else None,
                    speed_mps=float(speed) if speed is not None else None,
                    yaw_rate=float(yaw_rate) if yaw_rate is not None else None,
                    lat_accel=float(lat_accel) if lat_accel is not None else None,
                    steer=float(steer) if steer is not None else None,
                )
            except Exception:
                pass

            wet_action, wet_conf, wet_details = self.wet_brain.recommend(
                now=now, declared_wet=bool(declared_wet) if declared_wet is not None else None
            )

            # risk radar
            risk_score, risk_reason = self.risk_radar.update(
                now=now,
                player_idx=int(player_idx) if player_idx is not None else None,
                track_length_m=track_len_m,
                player_lapdist_pct=float(lap_dist_pct) if lap_dist_pct is not None else None,
                car_idx_lapdist_pct=car_idx_lapdist,
                brake=float(brake) if brake is not None else None,
                steer=float(steer) if steer is not None else None,
                long_accel=float(long_accel) if long_accel is not None else None,
                car_left_right=int(car_left_right) if car_left_right is not None else None,
                is_on_track=bool(is_on_track) if is_on_track is not None else None,
            )

            self._maybe_beep_risk(risk_score)

            # pit time config (possibly updated by calibration)
            pit_cfg = self.config_data.get("pit", {})
            pit_base = float(pit_cfg.get("pit_base_loss_s", 40.0))
            fill_rate = float(pit_cfg.get("fuel_fill_rate", 2.5))
            tires_with_fuel = bool(pit_cfg.get("tires_with_fuel", False))
            tire_time = 0.0 if tires_with_fuel else float(pit_cfg.get("tire_service_time_s", 18.0))
            clean_air_target = float(pit_cfg.get("clean_air_target_s", 2.0))
            max_offsets = int(pit_cfg.get("max_offsets", 3))

            # pit timing options (continuous)
            opts = self.pit_advisor.build_options(
                avg_lap_time=avg_lap_time,
                lap_time_sigma=lap_sigma,
                pit_base_loss_s=pit_base,
                fuel_fill_rate=fill_rate,
                tire_service_time_s=tire_time,
                take_tires=take_tires,
                burn_per_lap=burn,
                fuel_now=fuel,
                laps_possible=laps_possible,
                laps_remain=laps_remain,
                margin_laps=margin,
                fuel_plan_mode=fuel_plan_mode,
                stint_target_laps=stint_target_laps,
                tank_capacity=tank_capacity,
                player_idx=int(player_idx) if player_idx is not None else None,
                player_lapdist_pct=float(lap_dist_pct) if lap_dist_pct is not None else None,
                car_idx_lapdist_pct=car_idx_lapdist,
                track_length_m=track_len_m,
                max_offsets=max_offsets,
                clean_air_target_s=clean_air_target,
            )
            self._last_pit_options = opts

            # pit telemetry calibration update
            m = self.pit_cal.update(
                now=now,
                on_pit_road=bool(on_pit_road) if on_pit_road is not None else None,
                in_pit_stall=bool(in_pit_stall) if in_pit_stall is not None else None,
                speed_mps=float(speed) if speed is not None else None,
                fuel_level=float(fuel_level) if fuel_level is not None else None,
                pit_sv_flags=int(pit_sv_flags) if pit_sv_flags is not None else None,
                pit_sv_fuel=float(pit_sv_fuel) if pit_sv_fuel is not None else None,
            )
            if m:
                self._apply_pit_calibration(m)

            # --- labels
            status = "YELLOW" if yellow_now else "GREEN"
            status_color = "#f5c542" if yellow_now else "#3ddc84"
            conn = "connected"
            pits_open_s = ""
            try:
                if pits_open is not None:
                    pits_open_s = " | PITS OPEN" if bool(pits_open) else " | PITS CLOSED"
            except Exception:
                pits_open_s = ""

            self.lbl_status.configure(text=f"iRacing: {conn} ({status}){pits_open_s} | samples={len(self.history.samples)}", text_color=status_color)

            easy_mode = self._easy_mode_enabled()
            need_line = "Need: --"
            need_color = "#f5c542"
            if easy_mode:
                burn_s = "collecting" if burn is None or burn <= 0 else f"{burn:.3f}/lap"
                fuel_left_s = f"{fuel:.2f}"
                add_s = "--" if fuel_to_add is None else f"{fuel_to_add:.2f}"
                self.var_fuel.set(f"Fuel: {burn_s} | Left: {fuel_left_s} | Add: {add_s}")

                laps_rem_s = "--" if laps_remain is None else f"{laps_remain:.1f}"
                self.var_race.set(f"Laps remaining: {laps_rem_s}")

                if fuel_need_total is not None and fuel_to_add is not None:
                    if fuel >= fuel_need_total:
                        leftover_s = "" if finish_leftover is None else f" ({finish_leftover:.2f} spare)"
                        need_line = f"Fuel ok to finish{leftover_s}".strip()
                        need_color = "#3ddc84"
                    else:
                        need_line = f"Add {fuel_to_add:.2f} to reach target"
                        need_color = "#ff8c42"
                else:
                    need_line = "Fuel data not ready yet"
            else:
                if burn is None or burn <= 0 or laps_possible is None:
                    proj = self.history.projected_burn()
                    proj_s = "" if proj is None else f" (proj {proj:.3f}/lap)"
                    self.var_fuel.set(f"Burn: collecting{proj_s} | Fuel: {fuel:.2f}")
                    self.var_race.set("Race: RemLaps=-- | Need=-- | Add=--")
                else:
                    self.var_fuel.set(f"Burn: {burn:.3f}/lap | Fuel: {fuel:.2f} | Can: {laps_possible:.1f} laps")
                    if fuel_need_total is not None and fuel_to_add is not None:
                        rem_s = "--" if laps_remain is None else f"{laps_remain:.1f}"
                        left_s = "--" if finish_leftover is None else f"{finish_leftover:.2f}"
                        plan_s = (fuel_plan_mode or "finish").upper()
                        need_line = f"NEED {fuel_need_total:.2f} | Fuel now {fuel:.2f}"
                        if fuel >= fuel_need_total:
                            need_line = need_line + " | NO STOP (fuel > need)"
                            need_color = "#3ddc84"
                        else:
                            need_line = need_line + " | Pit needed"
                            need_color = "#ff8c42"
                        self.var_race.set(
                            f"Race: RemLaps={rem_s} | Plan={plan_s} | Need={fuel_need_total:.2f} | Add={fuel_to_add:.2f} | Left={left_s} | Margin={margin:.1f}L"
                        )
                    else:
                        self.var_race.set("Race: RemLaps=-- | Need=-- | Add=--")

            try:
                self.var_need_callout.set(need_line)
                self.lbl_need.configure(text_color=need_color)
            except Exception:
                pass

            self.var_weather.set(
                self._format_weather_line(track_temp, declared_wet, wet_action, wet_conf, wet_details, easy_mode=easy_mode)
            )
            self.var_risk.set(f"Risk: {risk_score:3d} | {risk_reason}")

            self.var_pit.set(
                self._format_pit_options(opts, pit_base, fill_rate, tire_time, tires_with_fuel, easy_mode=easy_mode)
            )

            hk = self.config_data.get("hotkeys", {})
            self.var_hotkeys.set(
                f"Margin +/-: {hk.get('margin_up','?')} / {hk.get('margin_down','?')} | "
                f"Plan: {hk.get('cycle_plan','?')} | "
                f"Apply: {hk.get('apply_opt1','?')},{hk.get('apply_opt2','?')},{hk.get('apply_opt3','?')} | "
                f"Settings: {hk.get('toggle_settings','?')}"
            )

            # cache for templates
            self._last_calc.update(
                {
                    "burn_per_lap": float(burn) if burn else None,
                    "fuel": float(fuel),
                    "laps_possible": float(laps_possible) if laps_possible is not None else None,
                    "laps_remain": float(laps_remain) if laps_remain is not None else None,
                    "fuel_need_total": float(fuel_need_total) if fuel_need_total is not None else None,
                    "fuel_add": float(fuel_to_add) if fuel_to_add is not None else None,
                    "wet_action": wet_action,
                    "wet_conf": wet_conf,
                    "risk": risk_score,
                    "plan_mode": plan_mode,
                }
            )

            self.after(refresh_ms, self._tick)

        except Exception as e:
            try:
                self.lbl_status.configure(text=f"Error: {e}", text_color="#ff6b6b")
            except Exception:
                pass
            self.after(500, self._tick)

    def _apply_pit_calibration(self, m: PitMeasurement) -> None:
        ac = self.config_data["pit"].get("auto_calibrate", {})
        if not bool(ac.get("enabled", True)):
            return

        alpha = float(ac.get("ema_alpha", 0.25))
        alpha = max(0.05, min(alpha, 0.6))

        min_fuel = float(ac.get("min_fuel_added", 0.5))
        min_lane = float(ac.get("min_lane_time", 8.0))

        pit = self.config_data["pit"]

        # update fill rate
        if bool(ac.get("update_fill_rate", True)) and m.fill_rate is not None and m.fuel_added >= min_fuel:
            cur = float(pit.get("fuel_fill_rate", 2.5))
            pit["fuel_fill_rate"] = (1 - alpha) * cur + alpha * float(m.fill_rate)

        # base/tire separation uses pit_lane_time - fueling_time (we don't know repairs/other services)
        if m.pit_lane_time_s < min_lane:
            return

        residual = max(0.0, float(m.pit_lane_time_s) - float(m.fueling_time_s))

        tires_sel = bool(m.tires_selected) if m.tires_selected is not None else None

        # if tires not selected, residual ~ base loss
        if bool(ac.get("update_base_loss", True)) and tires_sel is False:
            cur = float(pit.get("pit_base_loss_s", 40.0))
            pit["pit_base_loss_s"] = (1 - alpha) * cur + alpha * residual

        # if tires selected, residual ~ base + tire_time
        if bool(ac.get("update_tire_time", True)) and tires_sel is True:
            base = float(pit.get("pit_base_loss_s", 40.0))
            tire_est = residual - base
            if tire_est > 0.0:
                cur = float(pit.get("tire_service_time_s", 18.0))
                pit["tire_service_time_s"] = (1 - alpha) * cur + alpha * tire_est

        pit["has_calibration"] = True
        self._pit_profile_has_data = True

        self._sync_pit_entries_from_config()
        self._persist_active_profile()

    def _estimate_laps_remaining(self, avg_lap_time: Optional[float]) -> Optional[float]:
        # 1) SessionLapsRemainEx
        laps_rem_ex = self._safe_get("SessionLapsRemainEx")
        try:
            if laps_rem_ex is not None:
                val = float(laps_rem_ex)
                if 0 <= val < 10000:
                    return val
        except Exception:
            pass

        # 2) SessionLapsRemain
        laps_rem = self._safe_get("SessionLapsRemain")
        try:
            if laps_rem is not None:
                val = float(laps_rem)
                if 0 <= val < 10000:
                    return val + 1.0
        except Exception:
            pass

        # 3) SessionTimeRemain / avg_lap_time
        time_rem = self._safe_get("SessionTimeRemain")
        try:
            if time_rem is not None and avg_lap_time and avg_lap_time > 0:
                tr = float(time_rem)
                if tr > 0:
                    return tr / float(avg_lap_time)
        except Exception:
            pass

        return None

    @staticmethod
    def _wet_level_from_fraction(wet: Optional[float]) -> str:
        if wet is None:
            return "Unknown"
        try:
            w = float(wet)
        except Exception:
            return "Unknown"

        if w < 0.05:
            return "Dry"
        if w < 0.15:
            return "Mostly dry"
        if w < 0.35:
            return "Damp"
        if w < 0.6:
            return "Mixed"
        if w < 0.8:
            return "Wet"
        return "Very wet"

    def _set_pit_label_color(self, color: Optional[str]) -> None:
        try:
            if color is None:
                if self._pit_label_default_color is None:
                    return
                self.lbl_pit.configure(text_color=self._pit_label_default_color)
            else:
                self.lbl_pit.configure(text_color=color)
        except Exception:
            pass

    def _format_weather_line(
        self,
        track_temp,
        declared_wet,
        action: str,
        conf: int,
        details: Dict[str, Any],
        *,
        easy_mode: bool = False,
    ) -> str:
        # usamos wet_eff (corrigido) se existir, senão o wet bruto
        wet = details.get("wet_eff")
        if wet is None:
            wet = details.get("wet")

        prec = details.get("prec")
        grip = details.get("grip")
        aqua = details.get("aqua")
        trend = details.get("trend")
        wet_label = details.get("wet_label")
        wet_raw = details.get("wet_raw")

        if easy_mode:
            wet_pct = "--%" if wet is None else f"{wet*100:3.0f}%"
            level = wet_label or self._wet_level_from_fraction(wet)
            status = "CHANGE TO WETS" if action == "PIT WETS" else "STAY SLICKS"
            confidence = f" ({conf}% confidence)" if conf is not None else ""
            level_text = level or "Unknown wetness"
            return f"{status} | Track: {level_text} ({wet_pct}){confidence}"

        wet_s = "--" if wet is None else f"{wet*100:3.0f}%"
        if wet_label:
            wet_s += f" ({wet_label})"
        elif wet_raw is not None:
            wet_s += f" (raw={wet_raw})"

        tr_s = "" if trend is None else ("↑" if trend > 0.02 else ("↓" if trend < -0.02 else "→"))
        prec_s = "--" if prec is None else f"{prec:.2f}"
        grip_s = "--" if grip is None else f"{grip:.2f}"
        temp_s = "--" if track_temp is None else f"{float(track_temp):.0f}C"
        dw = "Y" if bool(declared_wet) else "N"

        return f"Wet={wet_s}{tr_s} Precip={prec_s} DeclWet={dw} TrackT={temp_s} | Grip~{grip_s} Aqua={aqua} | {action} ({conf}%)"

    def _format_pit_options(
        self,
        opts: List[PitOption],
        pit_base: float,
        fill_rate: float,
        tire_time: float,
        tires_with_fuel: bool = False,
        *,
        easy_mode: bool = False,
    ) -> str:
        if easy_mode:
            if not opts:
                self._set_pit_label_color(None)
                return "Pit: data unavailable"

            best = opts[0]
            add_s = "" if best.fuel_add is None else f" | Add {best.fuel_add:.1f}"
            if best.offset_laps == 0:
                self._set_pit_label_color("#3ddc84")
                return f"Good time to pit now{add_s}"

            delay_s = f"+{best.offset_laps} laps" if best.offset_laps is not None else "later"
            self._set_pit_label_color("#ff5c5c")
            return f"Do not pit now (better in {delay_s}){add_s}"

        self._set_pit_label_color(None)
        if not opts:
            return "Pit: -- (need avg lap time + traffic data)"

        lines = []
        tires_s = "with fuel" if tires_with_fuel else f"{tire_time:.0f}s"
        lines.append(f"Model: base={pit_base:.0f}s fill={fill_rate:.2f}u/s tires={tires_s}")
        lines.append("Rank | When | Success | pos~ | gap(a/b) | add | pit | notes")
        for rank, o in enumerate(opts[:4], start=1):
            label = "NOW" if o.offset_laps == 0 else f"+{o.offset_laps}L"
            suc = "--" if o.success_pct is None else f"{o.success_pct:3d}%"
            pe = "--" if o.pos_est is None else f"{o.pos_est:3d}"
            ga = "--" if o.gap_ahead_s is None else f"+{o.gap_ahead_s:4.1f}"
            gb = "--" if o.gap_behind_s is None else f"{o.gap_behind_s:4.1f}"
            fa = "--" if o.fuel_add is None else f"{o.fuel_add:5.1f}"
            pl = "--" if o.pit_loss_s is None else f"{o.pit_loss_s:4.0f}s"
            lines.append(f" [{rank}] | {label:>3} | {suc:>7} | {pe:>4} | {ga:>6}/{gb:>6} | {fa:>5} | {pl:>4} | {o.notes}")

        # show last pit calibration (if any)
        lm = self.pit_cal.last_measurement
        if lm is not None:
            tires = "?" if lm.tires_selected is None else ("Y" if lm.tires_selected else "N")
            fr = "--" if lm.fill_rate is None else f"{lm.fill_rate:.2f}"
            lines.append(f"Last pit obs: lane={lm.pit_lane_time_s:.1f}s fuel+={lm.fuel_added:.2f} time={lm.fueling_time_s:.1f}s fill~{fr} tiresSel={tires}")

        return "\n".join(lines)

    def _maybe_beep_risk(self, risk_score: int) -> None:
        audio_cfg = self.config_data.get("audio", {})
        if not bool(audio_cfg.get("risk_beep", True)):
            return

        thr = int(audio_cfg.get("risk_beep_threshold", 85))
        if risk_score < thr:
            return

        cooldown = float(audio_cfg.get("beep_cooldown_s", 3.0))
        now = time.time()
        if now - self.risk_radar.last_beep_t < cooldown:
            return

        self.risk_radar.last_beep_t = now

        if _is_windows():
            try:
                import winsound

                winsound.Beep(1100, 120)
            except Exception:
                pass

    # ----------------------------
    # Close
    # ----------------------------

    def _close(self) -> None:
        self._save_config()
        self._remove_hotkeys()
        for win in list(self.detached_windows.values()):
            try:
                win.destroy()
            except Exception:
                pass
        self.detached_windows.clear()
        try:
            self.destroy()
        except Exception:
            pass


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fuel Overlay Pro for iRacing")
    parser.add_argument("--portable", action="store_true", help="Store config next to the script (./configs)")
    parser.add_argument("--reset-config", action="store_true", help="Delete config file and exit")
    args = parser.parse_args(argv)

    cfg_dir, cfg_file = _get_config_paths(bool(args.portable))
    if args.reset_config:
        try:
            if cfg_file.exists():
                cfg_file.unlink()
            # also remove legacy config if present
            legacy = Path(__file__).resolve().parent / "configs" / "fuel_overlay_config.json"
            if legacy.exists():
                legacy.unlink()
        except Exception:
            pass
        return 0

    app = FuelOverlayApp(portable=bool(args.portable))
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
