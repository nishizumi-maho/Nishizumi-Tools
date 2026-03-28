#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any

import irsdk
import psutil
import win32con
import win32gui
import win32process


# ============================================================
# CONFIG
# ============================================================
IRACING_DOCS = Path(r"C:\Users\user\Documents\iRacing")
PROFILE_ROOT = IRACING_DOCS / "combo_profiles"
APP_SETTINGS_PATH = PROFILE_ROOT / "app_settings.json"
MANIFEST_PATH = PROFILE_ROOT / "index.json"

RENDERER_OPTIONS = {
    "Monitor": "rendererDX11Monitor.ini",
    "OpenXR": "rendererDX11OpenXR.ini",
    "OpenVR": "rendererDX11OpenVR.ini",
}

GROUPING_OPTIONS = {
    "Carro + pista": "car_track",
    "Carro": "car",
    "Pista": "track",
    "SeriesID": "series",
    "SeriesID + pista": "series_track",
}

SIM_PROCESS_NAMES = {
    "iracingsim64dx11.exe",
    "iracingsim64dx12.exe",
    "iracingsim64.exe",
}

POLL_INTERVAL = 0.25
PRE_CLOSE_DELAY = 5.0
POST_CLOSE_SETTLE_TIMEOUT = 25.0
FILE_STABLE_SECONDS = 2.0
LOG_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================
# HELPERS
# ============================================================
def now_str() -> str:
    return time.strftime(LOG_TIMESTAMP_FORMAT)


def norm(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(str(text).strip().lower().split())


def slugify(text: str | None) -> str:
    raw = norm(text)
    if not raw:
        return "unknown"
    chars: list[str] = []
    for ch in raw:
        chars.append(ch if ch.isalnum() else "_")
    out = "".join(chars)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "unknown"


def file_sha1(path: Path) -> str | None:
    try:
        h = hashlib.sha1()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def files_equal(a: Path, b: Path) -> bool:
    try:
        return a.read_bytes() == b.read_bytes()
    except Exception:
        return False


def safe_get(ir: irsdk.IRSDK, key: str):
    try:
        return ir[key]
    except Exception:
        return None


def wait_for_file_stable(path: Path, stable_seconds: float, timeout: float) -> bool:
    start = time.time()
    last_state = None
    stable_since = None

    while time.time() - start < timeout:
        try:
            stat = path.stat()
            state = (stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            state = None

        if state == last_state:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= stable_seconds:
                return True
        else:
            stable_since = None
            last_state = state

        time.sleep(0.25)

    return False


# ============================================================
# DATA MODEL
# ============================================================
@dataclass
class ComboInfo:
    track_internal: str = ""
    track_config: str = ""
    track_display: str = ""
    track_display_short: str = ""
    car_path: str = ""
    car_screen: str = ""
    car_short: str = ""
    series_id: str = ""

    def track_key(self) -> str:
        base = self.track_internal or self.track_display or "unknown_track"
        cfg = self.track_config or "default"
        return f"{slugify(base)}__cfg_{slugify(cfg)}"

    def car_key(self) -> str:
        base = self.car_path or self.car_screen or self.car_short or "unknown_car"
        return slugify(base)

    def series_key(self) -> str:
        return slugify(self.series_id or "unknown_series")

    def combo_key(self, grouping_mode: str) -> str:
        if grouping_mode == "car_track":
            return f"trk_{self.track_key()}__car_{self.car_key()}"
        if grouping_mode == "car":
            return f"car_{self.car_key()}"
        if grouping_mode == "track":
            return f"trk_{self.track_key()}"
        if grouping_mode == "series":
            return f"series_{self.series_key()}"
        if grouping_mode == "series_track":
            return f"series_{self.series_key()}__trk_{self.track_key()}"
        return f"trk_{self.track_key()}__car_{self.car_key()}"

    def base_filename(self, grouping_mode: str, renderer_stem: str) -> str:
        return f"{self.combo_key(grouping_mode)}__mode_{grouping_mode}__{renderer_stem}"

    def label(self) -> str:
        track_name = self.track_display or self.track_internal or "Unknown track"
        cfg = self.track_config or "default"
        car_name = self.car_screen or self.car_path or self.car_short or "Unknown car"
        series_name = self.series_id or "unknown"
        return f"{track_name} [{cfg}] / {car_name} / Series {series_name}"

    def is_complete(self) -> bool:
        track_present = bool(self.track_internal or self.track_display)
        car_present = bool(self.car_path or self.car_screen or self.car_short)
        series_present = bool(self.series_id)
        return track_present or car_present or series_present

    def normalized(self) -> "ComboInfo":
        return ComboInfo(
            track_internal=norm(self.track_internal),
            track_config=norm(self.track_config),
            track_display=self.track_display.strip() if self.track_display else "",
            track_display_short=self.track_display_short.strip() if self.track_display_short else "",
            car_path=norm(self.car_path),
            car_screen=self.car_screen.strip() if self.car_screen else "",
            car_short=self.car_short.strip() if self.car_short else "",
            series_id=norm(self.series_id),
        )

    @staticmethod
    def from_manifest_entry(entry: dict[str, Any]) -> "ComboInfo":
        return ComboInfo(
            track_internal=str(entry.get("track_internal") or ""),
            track_config=str(entry.get("track_config") or ""),
            track_display=str(entry.get("track_display") or ""),
            track_display_short=str(entry.get("track_display_short") or ""),
            car_path=str(entry.get("car_path") or ""),
            car_screen=str(entry.get("car_screen") or ""),
            car_short=str(entry.get("car_short") or ""),
            series_id=str(entry.get("series_id") or ""),
        )


# ============================================================
# PROFILE MANAGER
# ============================================================
class ProfileManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.ensure_dirs()
        self.settings = self.load_settings()
        self.manifest = self.load_manifest()

        self.ensure_renderer_section(self.get_selected_renderer())
        self.ensure_grouping_section(self.get_selected_renderer(), self.get_selected_grouping())
        self.ensure_global_backup_if_missing()

    def ensure_dirs(self) -> None:
        PROFILE_ROOT.mkdir(parents=True, exist_ok=True)

    def load_settings(self) -> dict[str, Any]:
        if APP_SETTINGS_PATH.exists():
            try:
                data = json.loads(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def save_settings(self) -> None:
        APP_SETTINGS_PATH.write_text(
            json.dumps(self.settings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def needs_initial_setup(self) -> bool:
        renderer_ok = self.settings.get("default_renderer") in RENDERER_OPTIONS.values()
        grouping_ok = self.settings.get("default_grouping") in GROUPING_OPTIONS.values()
        return not (renderer_ok and grouping_ok)

    def load_manifest(self) -> dict[str, Any]:
        if MANIFEST_PATH.exists():
            try:
                data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("renderers"), dict):
                    return data
            except Exception:
                pass
        return {
            "updated_at": now_str(),
            "renderers": {},
        }

    def save_manifest(self) -> None:
        with self.lock:
            self.manifest["updated_at"] = now_str()
            MANIFEST_PATH.write_text(
                json.dumps(self.manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def get_selected_renderer(self) -> str:
        selected = str(self.settings.get("default_renderer") or "rendererDX11OpenXR.ini")
        if selected not in RENDERER_OPTIONS.values():
            selected = "rendererDX11OpenXR.ini"
        return selected

    def set_selected_renderer(self, renderer_file: str) -> None:
        if renderer_file not in RENDERER_OPTIONS.values():
            raise ValueError(f"Renderer inválido: {renderer_file}")
        with self.lock:
            self.settings["default_renderer"] = renderer_file
            self.save_settings()
            self.ensure_renderer_section(renderer_file)
            self.ensure_grouping_section(renderer_file, self.get_selected_grouping())
            self.ensure_global_backup_if_missing(renderer_file)

    def get_selected_grouping(self) -> str:
        selected = str(self.settings.get("default_grouping") or "car_track")
        if selected not in GROUPING_OPTIONS.values():
            selected = "car_track"
        return selected

    def set_selected_grouping(self, grouping_mode: str) -> None:
        if grouping_mode not in GROUPING_OPTIONS.values():
            raise ValueError(f"Agrupamento inválido: {grouping_mode}")
        with self.lock:
            self.settings["default_grouping"] = grouping_mode
            self.save_settings()
            self.ensure_renderer_section(self.get_selected_renderer())
            self.ensure_grouping_section(self.get_selected_renderer(), grouping_mode)

    def get_active_ini(self, renderer_file: str | None = None) -> Path:
        renderer_file = renderer_file or self.get_selected_renderer()
        return IRACING_DOCS / renderer_file

    def renderer_stem(self, renderer_file: str | None = None) -> str:
        renderer_file = renderer_file or self.get_selected_renderer()
        return Path(renderer_file).stem

    def get_renderer_dir(self, renderer_file: str | None = None) -> Path:
        renderer_file = renderer_file or self.get_selected_renderer()
        path = PROFILE_ROOT / self.renderer_stem(renderer_file)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_grouping_dir(self, renderer_file: str | None = None, grouping_mode: str | None = None) -> Path:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        path = self.get_renderer_dir(renderer_file) / grouping_mode
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_global_backup_path(self, renderer_file: str | None = None) -> Path:
        renderer_file = renderer_file or self.get_selected_renderer()
        return PROFILE_ROOT / f"{self.renderer_stem(renderer_file)}.global_backup.ini"

    def ensure_renderer_section(self, renderer_file: str) -> None:
        with self.lock:
            renderers = self.manifest.setdefault("renderers", {})
            if renderer_file not in renderers:
                renderers[renderer_file] = {
                    "global_backup": str(self.get_global_backup_path(renderer_file)),
                    "groupings": {},
                }
                self.save_manifest()

    def ensure_grouping_section(self, renderer_file: str, grouping_mode: str) -> None:
        with self.lock:
            self.ensure_renderer_section(renderer_file)
            renderer_section = self.manifest["renderers"][renderer_file]
            groupings = renderer_section.setdefault("groupings", {})
            if grouping_mode not in groupings:
                groupings[grouping_mode] = {"combos": {}}
                self.save_manifest()

    def renderer_section(self, renderer_file: str | None = None) -> dict[str, Any]:
        renderer_file = renderer_file or self.get_selected_renderer()
        self.ensure_renderer_section(renderer_file)
        return self.manifest["renderers"][renderer_file]

    def grouping_section(self, renderer_file: str | None = None, grouping_mode: str | None = None) -> dict[str, Any]:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        self.ensure_grouping_section(renderer_file, grouping_mode)
        return self.manifest["renderers"][renderer_file]["groupings"][grouping_mode]

    def ensure_global_backup_if_missing(self, renderer_file: str | None = None) -> None:
        renderer_file = renderer_file or self.get_selected_renderer()
        backup_path = self.get_global_backup_path(renderer_file)
        active_ini = self.get_active_ini(renderer_file)
        with self.lock:
            if not backup_path.exists() and active_ini.exists():
                shutil.copy2(active_ini, backup_path)
                self.save_manifest()

    def create_or_refresh_global_backup(self, renderer_file: str | None = None) -> None:
        renderer_file = renderer_file or self.get_selected_renderer()
        active_ini = self.get_active_ini(renderer_file)
        backup_path = self.get_global_backup_path(renderer_file)
        if not active_ini.exists():
            raise FileNotFoundError(f"Arquivo ativo não encontrado: {active_ini}")
        shutil.copy2(active_ini, backup_path)
        self.save_manifest()

    def restore_global_backup(self, renderer_file: str | None = None) -> None:
        renderer_file = renderer_file or self.get_selected_renderer()
        active_ini = self.get_active_ini(renderer_file)
        backup_path = self.get_global_backup_path(renderer_file)
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup global não encontrado: {backup_path}")
        if not active_ini.exists():
            raise FileNotFoundError(f"Arquivo ativo não encontrado: {active_ini}")
        shutil.copy2(backup_path, active_ini)

    def get_profile_ini_path(self, combo: ComboInfo, renderer_file: str | None = None, grouping_mode: str | None = None) -> Path:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        return self.get_grouping_dir(renderer_file, grouping_mode) / f"{combo.base_filename(grouping_mode, self.renderer_stem(renderer_file))}.ini"

    def get_profile_meta_path(self, combo: ComboInfo, renderer_file: str | None = None, grouping_mode: str | None = None) -> Path:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        return self.get_grouping_dir(renderer_file, grouping_mode) / f"{combo.base_filename(grouping_mode, self.renderer_stem(renderer_file))}.json"

    def write_profile_meta(self, combo: ComboInfo, ini_path: Path, renderer_file: str, grouping_mode: str) -> None:
        meta = {
            "renderer_file": renderer_file,
            "grouping_mode": grouping_mode,
            "combo_key": combo.combo_key(grouping_mode),
            "track_internal": combo.track_internal,
            "track_config": combo.track_config,
            "track_display": combo.track_display,
            "track_display_short": combo.track_display_short,
            "car_path": combo.car_path,
            "car_screen": combo.car_screen,
            "car_short": combo.car_short,
            "series_id": combo.series_id,
            "profile_ini": str(ini_path),
            "saved_at": now_str(),
            "sha1": file_sha1(ini_path),
        }
        self.get_profile_meta_path(combo, renderer_file, grouping_mode).write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def list_entries(self, renderer_file: str | None = None, grouping_mode: str | None = None) -> list[dict[str, Any]]:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        combos = self.grouping_section(renderer_file, grouping_mode).get("combos", {})
        items = list(combos.values())
        items.sort(
            key=lambda item: (
                str(item.get("track_display") or item.get("track_internal") or ""),
                str(item.get("track_config") or ""),
                str(item.get("car_screen") or item.get("car_path") or ""),
                str(item.get("series_id") or ""),
            )
        )
        return items

    def get_entry(self, combo_key: str, renderer_file: str | None = None, grouping_mode: str | None = None) -> dict[str, Any] | None:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        return self.grouping_section(renderer_file, grouping_mode).get("combos", {}).get(combo_key)

    def register_combo(self, combo: ComboInfo, renderer_file: str | None = None, grouping_mode: str | None = None) -> dict[str, Any]:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        combo = combo.normalized()
        combo_key = combo.combo_key(grouping_mode)

        with self.lock:
            section = self.grouping_section(renderer_file, grouping_mode)
            combos = section.setdefault("combos", {})
            existing = combos.get(combo_key)
            ini_path = self.get_profile_ini_path(combo, renderer_file, grouping_mode)
            meta_path = self.get_profile_meta_path(combo, renderer_file, grouping_mode)

            if existing is None:
                entry = {
                    "combo_key": combo_key,
                    "renderer_file": renderer_file,
                    "grouping_mode": grouping_mode,
                    "track_internal": combo.track_internal,
                    "track_config": combo.track_config,
                    "track_display": combo.track_display,
                    "track_display_short": combo.track_display_short,
                    "car_path": combo.car_path,
                    "car_screen": combo.car_screen,
                    "car_short": combo.car_short,
                    "series_id": combo.series_id,
                    "profile_ini": str(ini_path),
                    "profile_meta": str(meta_path),
                    "profile_sha1": file_sha1(ini_path) if ini_path.exists() else None,
                    "enabled": True,
                    "autosave_on_manual_close": True,
                    "last_used_at": now_str(),
                    "last_saved_at": None,
                }
                combos[combo_key] = entry
            else:
                entry = existing
                entry.update({
                    "renderer_file": renderer_file,
                    "grouping_mode": grouping_mode,
                    "track_internal": combo.track_internal,
                    "track_config": combo.track_config,
                    "track_display": combo.track_display,
                    "track_display_short": combo.track_display_short,
                    "car_path": combo.car_path,
                    "car_screen": combo.car_screen,
                    "car_short": combo.car_short,
                    "series_id": combo.series_id,
                    "profile_ini": str(ini_path),
                    "profile_meta": str(meta_path),
                    "last_used_at": now_str(),
                })

            self.save_manifest()
            return entry

    def update_entry_options(self, combo_key: str, enabled: bool, autosave: bool, renderer_file: str | None = None, grouping_mode: str | None = None) -> None:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        entry = self.get_entry(combo_key, renderer_file, grouping_mode)
        if not entry:
            raise KeyError(f"Combo não encontrado: {combo_key}")
        entry["enabled"] = bool(enabled)
        entry["autosave_on_manual_close"] = bool(autosave)
        self.save_manifest()

    def save_active_ini_as_profile(self, combo: ComboInfo, renderer_file: str | None = None, grouping_mode: str | None = None) -> dict[str, Any]:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        combo = combo.normalized()
        active_ini = self.get_active_ini(renderer_file)

        if not active_ini.exists():
            raise FileNotFoundError(f"Arquivo ativo não encontrado: {active_ini}")

        entry = self.register_combo(combo, renderer_file, grouping_mode)
        ini_path = self.get_profile_ini_path(combo, renderer_file, grouping_mode)
        shutil.copy2(active_ini, ini_path)
        self.write_profile_meta(combo, ini_path, renderer_file, grouping_mode)

        entry["profile_sha1"] = file_sha1(ini_path)
        entry["last_saved_at"] = now_str()
        self.save_manifest()
        return entry

    def apply_profile_to_active_ini(self, combo_key: str, renderer_file: str | None = None, grouping_mode: str | None = None) -> dict[str, Any]:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        entry = self.get_entry(combo_key, renderer_file, grouping_mode)
        if not entry:
            raise KeyError(f"Combo não encontrado: {combo_key}")

        ini_path = Path(str(entry.get("profile_ini") or ""))
        active_ini = self.get_active_ini(renderer_file)

        if not ini_path.exists():
            raise FileNotFoundError(f"Profile não encontrado: {ini_path}")
        if not active_ini.exists():
            raise FileNotFoundError(f"Arquivo ativo não encontrado: {active_ini}")

        shutil.copy2(ini_path, active_ini)
        entry["last_used_at"] = now_str()
        self.save_manifest()
        return entry

    def active_ini_matches_profile(self, combo_key: str, renderer_file: str | None = None, grouping_mode: str | None = None) -> bool:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        entry = self.get_entry(combo_key, renderer_file, grouping_mode)
        if not entry:
            return False
        ini_path = Path(str(entry.get("profile_ini") or ""))
        active_ini = self.get_active_ini(renderer_file)
        if not ini_path.exists() or not active_ini.exists():
            return False
        return files_equal(ini_path, active_ini)

    def get_suggestions(self, renderer_file: str | None = None, grouping_mode: str | None = None) -> dict[str, list[str]]:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()

        tracks = set()
        configs = set()
        cars = set()
        track_displays = set()
        car_screens = set()
        series_ids = set()

        for entry in self.list_entries(renderer_file, grouping_mode):
            if entry.get("track_internal"):
                tracks.add(str(entry["track_internal"]))
            if entry.get("track_config"):
                configs.add(str(entry["track_config"]))
            if entry.get("car_path"):
                cars.add(str(entry["car_path"]))
            if entry.get("track_display"):
                track_displays.add(str(entry["track_display"]))
            if entry.get("car_screen"):
                car_screens.add(str(entry["car_screen"]))
            if entry.get("series_id"):
                series_ids.add(str(entry["series_id"]))

        return {
            "track_internal": sorted(tracks),
            "track_config": sorted(configs),
            "car_path": sorted(cars),
            "track_display": sorted(track_displays),
            "car_screen": sorted(car_screens),
            "series_id": sorted(series_ids),
        }


# ============================================================
# IRACING RUNTIME
# ============================================================
class IRacingRuntime:
    def __init__(self) -> None:
        self.ir = irsdk.IRSDK(parse_yaml_async=False)

    def reset(self) -> None:
        try:
            self.ir.shutdown()
        except Exception:
            pass
        self.ir = irsdk.IRSDK(parse_yaml_async=False)

    def is_sim_running(self) -> bool:
        return bool(self.get_sim_processes())

    def get_sim_processes(self) -> list[psutil.Process]:
        procs: list[psutil.Process] = []
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                name = (proc.info.get("name") or "").lower()
                if name in SIM_PROCESS_NAMES:
                    procs.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return procs

    def get_sim_pids(self) -> set[int]:
        return {proc.pid for proc in self.get_sim_processes()}

    def ensure_started(self) -> None:
        if not self.ir.is_initialized:
            try:
                self.ir.startup()
            except Exception:
                pass

    def is_connected(self) -> bool:
        if not self.ir.is_initialized:
            return False
        try:
            return bool(self.ir.is_connected)
        except Exception:
            return False

    def detect_combo(self) -> ComboInfo | None:
        weekend = safe_get(self.ir, "WeekendInfo") or {}
        driver_info = safe_get(self.ir, "DriverInfo") or {}

        combo = ComboInfo()

        if isinstance(weekend, dict):
            combo.track_internal = str(weekend.get("TrackName") or "")
            combo.track_config = str(
                weekend.get("TrackConfigName")
                or weekend.get("TrackConfig")
                or ""
            )
            combo.track_display = str(
                weekend.get("TrackDisplayName")
                or weekend.get("TrackDisplayShortName")
                or ""
            )
            combo.track_display_short = str(weekend.get("TrackDisplayShortName") or "")
            combo.series_id = str(weekend.get("SeriesID") or "")

        if isinstance(driver_info, dict):
            driver_car_idx = driver_info.get("DriverCarIdx")
            drivers = driver_info.get("Drivers") or []
            if isinstance(drivers, list) and driver_car_idx is not None:
                for drv in drivers:
                    if isinstance(drv, dict) and drv.get("CarIdx") == driver_car_idx:
                        combo.car_path = str(drv.get("CarPath") or "")
                        combo.car_screen = str(drv.get("CarScreenName") or "")
                        combo.car_short = str(drv.get("CarScreenNameShort") or "")
                        break

        combo = combo.normalized()
        return combo if combo.is_complete() else None

    def post_wm_close_to_sim_windows(self) -> int:
        target_pids = self.get_sim_pids()
        if not target_pids:
            return 0

        hwnds: list[int] = []

        def callback(hwnd: int, _: object) -> bool:
            try:
                if not win32gui.IsWindow(hwnd):
                    return True
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid in target_pids:
                    hwnds.append(hwnd)
            except Exception:
                pass
            return True

        win32gui.EnumWindows(callback, None)

        sent = 0
        for hwnd in hwnds:
            try:
                win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                sent += 1
            except Exception:
                pass
        return sent


# ============================================================
# MONITOR THREAD
# ============================================================
class MonitorService:
    def __init__(self, manager: ProfileManager, log_func) -> None:
        self.manager = manager
        self.log_func = log_func
        self.runtime = IRacingRuntime()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.session: dict[str, Any] | None = None
        self.last_sim_running = False

    def log(self, text: str) -> None:
        self.log_func(f"[{now_str()}] {text}")

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, name="IRacingMonitor", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.runtime.reset()

    def _run(self) -> None:
        self.log(
            "Monitor automático iniciado | "
            f"renderer: {self.manager.get_selected_renderer()} | "
            f"grouping: {self.manager.get_selected_grouping()}"
        )

        while not self.stop_event.is_set():
            renderer_file = self.manager.get_selected_renderer()
            grouping_mode = self.manager.get_selected_grouping()
            active_ini = self.manager.get_active_ini(renderer_file)

            sim_running = self.runtime.is_sim_running()

            if sim_running:
                self.runtime.ensure_started()

            connected = self.runtime.is_connected()

            if sim_running and connected:
                combo = self.runtime.detect_combo()
                if combo is not None:
                    entry = self.manager.register_combo(combo, renderer_file, grouping_mode)
                    combo_key = combo.combo_key(grouping_mode)

                    if self.session is None or self.session.get("combo_key") != combo_key or self.session.get("renderer_file") != renderer_file or self.session.get("grouping_mode") != grouping_mode:
                        self.session = {
                            "combo_key": combo_key,
                            "combo": combo,
                            "renderer_file": renderer_file,
                            "grouping_mode": grouping_mode,
                            "close_scheduled_at": None,
                            "wm_close_sent": False,
                            "profile_apply_pending": False,
                        }
                        self.log(
                            f"Combo detectado: {combo.label()} | "
                            f"renderer in app: {renderer_file} | grouping: {grouping_mode}"
                        )

                        profile_exists = Path(str(entry.get("profile_ini") or "")).exists()
                        enabled = bool(entry.get("enabled", True))

                        if profile_exists and enabled:
                            if self.manager.active_ini_matches_profile(combo_key, renderer_file, grouping_mode):
                                self.log("Profile conhecido já está ativo")
                            else:
                                self.session["close_scheduled_at"] = time.time() + PRE_CLOSE_DELAY
                                self.log(
                                    f"Profile conhecido mas diferente do ativo ({active_ini.name}). "
                                    f"WM_CLOSE agendado para {PRE_CLOSE_DELAY:.0f}s"
                                )
                        else:
                            self.log("First time for this grouping, or the profile is disabled. No automatic close will be performed")

                    if self.session is not None:
                        close_due = self.session.get("close_scheduled_at")
                        wm_close_sent = bool(self.session.get("wm_close_sent"))

                        if close_due is not None and not wm_close_sent and time.time() >= float(close_due):
                            sent = self.runtime.post_wm_close_to_sim_windows()
                            if sent > 0:
                                self.session["wm_close_sent"] = True
                                self.session["profile_apply_pending"] = True
                                self.log(f"WM_CLOSE enviado para {sent} janela(s) do sim")
                            else:
                                self.log("No sim windows were found for WM_CLOSE")
                                self.session["close_scheduled_at"] = None

            if self.last_sim_running and not sim_running:
                self.log("Sim closed")

                if self.session is not None:
                    combo = self.session["combo"]
                    session_renderer = str(self.session.get("renderer_file") or renderer_file)
                    session_grouping = str(self.session.get("grouping_mode") or grouping_mode)
                    session_combo_key = str(self.session.get("combo_key"))
                    session_active_ini = self.manager.get_active_ini(session_renderer)

                    self.log(f"Waiting for {session_active_ini.name} to stabilize...")
                    stable = wait_for_file_stable(
                        session_active_ini,
                        stable_seconds=FILE_STABLE_SECONDS,
                        timeout=POST_CLOSE_SETTLE_TIMEOUT,
                    )

                    if not stable:
                        self.log("Warning: the file did not stabilize within the expected time")
                    else:
                        self.log("File stabilized")

                    if bool(self.session.get("profile_apply_pending")):
                        try:
                            self.manager.apply_profile_to_active_ini(session_combo_key, session_renderer, session_grouping)
                            self.log("Profile correto aplicado no INI ativo. Reabra o sim manualmente quando quiser")
                        except Exception as e:
                            self.log(f"Error applying known profile: {e}")
                    else:
                        entry = self.manager.get_entry(session_combo_key, session_renderer, session_grouping)
                        autosave = True if entry is None else bool(entry.get("autosave_on_manual_close", True))
                        if autosave:
                            try:
                                self.manager.save_active_ini_as_profile(combo, session_renderer, session_grouping)
                                self.log("Manual close detected, profile saved/updated")
                            except Exception as e:
                                self.log(f"Error saving profile on manual close: {e}")
                        else:
                            self.log("Autosave is disabled for this combo; the profile was not updated")

                self.session = None
                self.runtime.reset()

            self.last_sim_running = sim_running
            time.sleep(POLL_INTERVAL)

        self.log("Monitor automático encerrado")


# ============================================================
# FIRST RUN DIALOG
# ============================================================
class FirstRunDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, default_renderer_label: str, default_grouping_label: str) -> None:
        super().__init__(parent)
        self.title("First-time setup")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.result: tuple[str, str] | None = None
        self.renderer_var = tk.StringVar(value=default_renderer_label)
        self.grouping_var = tk.StringVar(value=default_grouping_label)

        frame = ttk.Frame(self, padding=14)
        frame.pack(fill="both", expand=True)

        ttk.Label(
            frame,
            text="Escolha o renderer padrão e o modo de agrupamento dos profiles.\nVocê poderá mudar isso depois no app.",
            justify="left",
        ).pack(anchor="w", pady=(0, 12))

        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=4)
        ttk.Label(row1, text="Default renderer", width=22).pack(side="left")
        ttk.Combobox(
            row1,
            textvariable=self.renderer_var,
            values=list(RENDERER_OPTIONS.keys()),
            state="readonly",
            width=20,
        ).pack(side="left")

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=4)
        ttk.Label(row2, text="Default grouping", width=22).pack(side="left")
        ttk.Combobox(
            row2,
            textvariable=self.grouping_var,
            values=list(GROUPING_OPTIONS.keys()),
            state="readonly",
            width=20,
        ).pack(side="left")

        btns = ttk.Frame(frame)
        btns.pack(fill="x", pady=(14, 0))
        ttk.Button(btns, text="Save", command=self.on_ok).pack(side="right")

        self.protocol("WM_DELETE_WINDOW", self.on_ok)
        self.bind("<Return>", lambda _e: self.on_ok())
        self.update_idletasks()
        self.geometry(f"+{parent.winfo_rootx() + 80}+{parent.winfo_rooty() + 80}")

    def on_ok(self) -> None:
        self.result = (self.renderer_var.get(), self.grouping_var.get())
        self.destroy()


# ============================================================
# TKINTER GUI
# ============================================================
class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("iRacing Renderer Combo Profile Manager")
        self.geometry("1280x830")
        self.minsize(1160, 720)

        self.manager = ProfileManager()
        self.log_queue: list[str] = []
        self.log_lock = threading.Lock()

        self.renderer_ui_var = tk.StringVar(value=self.renderer_to_label(self.manager.get_selected_renderer()))
        self.grouping_ui_var = tk.StringVar(value=self.grouping_to_label(self.manager.get_selected_grouping()))
        self.renderer_label_var = tk.StringVar()
        self.grouping_label_var = tk.StringVar()

        self.track_internal_var = tk.StringVar()
        self.track_config_var = tk.StringVar()
        self.track_display_var = tk.StringVar()
        self.track_display_short_var = tk.StringVar()
        self.car_path_var = tk.StringVar()
        self.car_screen_var = tk.StringVar()
        self.car_short_var = tk.StringVar()
        self.series_id_var = tk.StringVar()
        self.enabled_var = tk.BooleanVar(value=True)
        self.autosave_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")

        self.selected_combo_key: str | None = None

        self._build_ui()
        self._run_first_time_setup_if_needed()

        self.monitor = MonitorService(self.manager, self.enqueue_log)
        self.monitor.start()

        self.refresh_all()
        self.after(200, self._drain_log_queue)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def enqueue_log(self, text: str) -> None:
        with self.log_lock:
            self.log_queue.append(text)

    def _drain_log_queue(self) -> None:
        with self.log_lock:
            items = self.log_queue[:]
            self.log_queue.clear()

        for item in items:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", item + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        self.after(200, self._drain_log_queue)

    def renderer_to_label(self, renderer_file: str) -> str:
        for label, file_name in RENDERER_OPTIONS.items():
            if file_name == renderer_file:
                return label
        return "OpenXR"

    def label_to_renderer(self, label: str) -> str:
        return RENDERER_OPTIONS.get(label, self.manager.get_selected_renderer())

    def grouping_to_label(self, grouping_mode: str) -> str:
        for label, mode in GROUPING_OPTIONS.items():
            if mode == grouping_mode:
                return label
        return "Carro + pista"

    def label_to_grouping(self, label: str) -> str:
        return GROUPING_OPTIONS.get(label, self.manager.get_selected_grouping())

    def current_renderer_file(self) -> str:
        return self.label_to_renderer(self.renderer_ui_var.get())

    def current_grouping_mode(self) -> str:
        return self.label_to_grouping(self.grouping_ui_var.get())

    def _run_first_time_setup_if_needed(self) -> None:
        if not self.manager.needs_initial_setup():
            return

        self.update_idletasks()
        dialog = FirstRunDialog(
            self,
            default_renderer_label="OpenXR",
            default_grouping_label="Carro + pista",
        )
        self.wait_window(dialog)

        renderer_label, grouping_label = dialog.result or ("OpenXR", "Carro + pista")
        renderer_file = self.label_to_renderer(renderer_label)
        grouping_mode = self.label_to_grouping(grouping_label)

        self.manager.set_selected_renderer(renderer_file)
        self.manager.set_selected_grouping(grouping_mode)

        self.renderer_ui_var.set(renderer_label)
        self.grouping_ui_var.set(grouping_label)

        self.enqueue_log(
            f"[{now_str()}] First-time setup salva | "
            f"renderer: {renderer_file} | grouping: {grouping_mode}"
        )

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)

        top = ttk.LabelFrame(root, text="Configuração do app", padding=10)
        top.pack(fill="x")

        row0 = ttk.Frame(top)
        row0.pack(fill="x")

        ttk.Label(row0, text="Default renderer").pack(side="left")
        self.renderer_combo = ttk.Combobox(
            row0,
            textvariable=self.renderer_ui_var,
            values=list(RENDERER_OPTIONS.keys()),
            state="readonly",
            width=16,
        )
        self.renderer_combo.pack(side="left", padx=(10, 14))
        self.renderer_combo.bind("<<ComboboxSelected>>", self.on_renderer_changed)

        ttk.Label(row0, text="Default grouping").pack(side="left")
        self.grouping_combo = ttk.Combobox(
            row0,
            textvariable=self.grouping_ui_var,
            values=list(GROUPING_OPTIONS.keys()),
            state="readonly",
            width=20,
        )
        self.grouping_combo.pack(side="left", padx=(10, 14))
        self.grouping_combo.bind("<<ComboboxSelected>>", self.on_grouping_changed)

        ttk.Label(row0, textvariable=self.renderer_label_var).pack(side="left", padx=(8, 18))
        ttk.Label(row0, textvariable=self.grouping_label_var).pack(side="left")

        self.paths_label = ttk.Label(top, text="", justify="left")
        self.paths_label.pack(anchor="w", pady=(8, 0))

        mid = ttk.Frame(root)
        mid.pack(fill="both", expand=True, pady=(10, 10))

        left = ttk.LabelFrame(mid, text="Known combos for selected renderer + grouping", padding=8)
        left.pack(side="left", fill="both", expand=True)

        columns = ("track", "layout", "car", "series", "enabled", "autosave", "saved")
        self.tree = ttk.Treeview(left, columns=columns, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="Combo key")
        self.tree.heading("track", text="Track")
        self.tree.heading("layout", text="Layout")
        self.tree.heading("car", text="Car")
        self.tree.heading("series", text="SeriesID")
        self.tree.heading("enabled", text="Enabled")
        self.tree.heading("autosave", text="Autosave")
        self.tree.heading("saved", text="Last saved")

        self.tree.column("#0", width=300, anchor="w")
        self.tree.column("track", width=180, anchor="w")
        self.tree.column("layout", width=110, anchor="w")
        self.tree.column("car", width=190, anchor="w")
        self.tree.column("series", width=85, anchor="w")
        self.tree.column("enabled", width=70, anchor="center")
        self.tree.column("autosave", width=75, anchor="center")
        self.tree.column("saved", width=145, anchor="w")

        vsb = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        right = ttk.LabelFrame(mid, text="Selected combo / manual editor", padding=8)
        right.pack(side="left", fill="both", expand=False, padx=(10, 0))

        form = ttk.Frame(right)
        form.pack(fill="x")

        self.track_internal_combo = self._add_labeled_combobox(form, 0, "Track internal", self.track_internal_var)
        self.track_config_combo = self._add_labeled_combobox(form, 1, "Track config/layout", self.track_config_var)
        self.track_display_combo = self._add_labeled_combobox(form, 2, "Track display", self.track_display_var)
        self.track_display_short_combo = self._add_labeled_combobox(form, 3, "Track display short", self.track_display_short_var)
        self.car_path_combo = self._add_labeled_combobox(form, 4, "Car path", self.car_path_var)
        self.car_screen_combo = self._add_labeled_combobox(form, 5, "Car screen", self.car_screen_var)
        self.car_short_combo = self._add_labeled_combobox(form, 6, "Car short", self.car_short_var)
        self.series_combo = self._add_labeled_combobox(form, 7, "SeriesID", self.series_id_var)

        opts = ttk.Frame(right)
        opts.pack(fill="x", pady=(10, 8))
        ttk.Checkbutton(opts, text="Enabled", variable=self.enabled_var).pack(side="left")
        ttk.Checkbutton(opts, text="Autosave on manual close", variable=self.autosave_var).pack(side="left", padx=(12, 0))

        btns = ttk.Frame(right)
        btns.pack(fill="x", pady=(4, 0))

        ttk.Button(btns, text="Clear fields", command=self.clear_fields).grid(row=0, column=0, sticky="ew", padx=(0, 6), pady=3)
        ttk.Button(btns, text="Refresh list", command=self.refresh_all).grid(row=0, column=1, sticky="ew", padx=(0, 6), pady=3)
        ttk.Button(btns, text="Load current sim combo", command=self.load_current_sim_combo).grid(row=0, column=2, sticky="ew", pady=3)

        ttk.Button(btns, text="Save/overwrite profile from selected renderer INI", command=self.save_profile_from_active_ini).grid(row=1, column=0, sticky="ew", padx=(0, 6), pady=3)
        ttk.Button(btns, text="Apply selected profile to selected renderer INI", command=self.apply_selected_profile).grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=3)
        ttk.Button(btns, text="Save enabled/autosave flags", command=self.save_entry_options).grid(row=1, column=2, sticky="ew", pady=3)

        ttk.Button(btns, text="Create/refresh global backup", command=self.create_global_backup).grid(row=2, column=0, sticky="ew", padx=(0, 6), pady=3)
        ttk.Button(btns, text="Restore global backup to selected renderer INI", command=self.restore_global_backup).grid(row=2, column=1, sticky="ew", padx=(0, 6), pady=3)
        ttk.Button(btns, text="Open current profiles folder", command=self.open_profiles_folder).grid(row=2, column=2, sticky="ew", pady=3)

        for col in range(3):
            btns.columnconfigure(col, weight=1)

        status_row = ttk.Frame(root)
        status_row.pack(fill="x", pady=(0, 6))
        ttk.Label(status_row, textvariable=self.status_var).pack(anchor="w")

        log_box = ttk.LabelFrame(root, text="Log", padding=8)
        log_box.pack(fill="both", expand=True)
        self.log_text = ScrolledText(log_box, height=14, wrap="word")
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _add_labeled_combobox(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))
        combo = ttk.Combobox(parent, textvariable=var, width=42)
        combo.grid(row=row, column=1, sticky="ew", pady=3)
        parent.columnconfigure(1, weight=1)
        return combo

    def update_labels(self) -> None:
        renderer_file = self.current_renderer_file()
        grouping_mode = self.current_grouping_mode()

        active_ini = self.manager.get_active_ini(renderer_file)
        profiles_dir = self.manager.get_grouping_dir(renderer_file, grouping_mode)
        backup = self.manager.get_global_backup_path(renderer_file)

        self.renderer_label_var.set(f"Arquivo selecionado: {renderer_file}")
        self.grouping_label_var.set(f"Agrupamento selecionado: {grouping_mode}")
        self.paths_label.config(
            text=(
                f"Active INI: {active_ini}\n"
                f"Profiles dir: {profiles_dir}\n"
                f"Manifest: {MANIFEST_PATH}\n"
                f"Global backup: {backup}"
            )
        )

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.enqueue_log(f"[{now_str()}] {text}")

    def refresh_suggestions(self) -> None:
        suggestions = self.manager.get_suggestions(self.current_renderer_file(), self.current_grouping_mode())
        self.track_internal_combo["values"] = suggestions["track_internal"]
        self.track_config_combo["values"] = suggestions["track_config"]
        self.track_display_combo["values"] = suggestions["track_display"]
        self.car_path_combo["values"] = suggestions["car_path"]
        self.car_screen_combo["values"] = suggestions["car_screen"]
        self.series_combo["values"] = suggestions["series_id"]

    def refresh_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for entry in self.manager.list_entries(self.current_renderer_file(), self.current_grouping_mode()):
            combo_key = str(entry.get("combo_key"))
            track = str(entry.get("track_display") or entry.get("track_internal") or "")
            layout = str(entry.get("track_config") or "")
            car = str(entry.get("car_screen") or entry.get("car_path") or "")
            series_id = str(entry.get("series_id") or "")
            enabled = "yes" if bool(entry.get("enabled", True)) else "no"
            autosave = "yes" if bool(entry.get("autosave_on_manual_close", True)) else "no"
            saved = str(entry.get("last_saved_at") or "")
            self.tree.insert(
                "",
                "end",
                iid=combo_key,
                text=combo_key,
                values=(track, layout, car, series_id, enabled, autosave, saved),
            )

    def refresh_all(self) -> None:
        self.manager.settings = self.manager.load_settings()
        self.manager.manifest = self.manager.load_manifest()
        self.manager.ensure_renderer_section(self.current_renderer_file())
        self.manager.ensure_grouping_section(self.current_renderer_file(), self.current_grouping_mode())
        self.update_labels()
        self.refresh_tree()
        self.refresh_suggestions()
        self.set_status("List, paths, and suggestions refreshed")

    def combo_from_form(self) -> ComboInfo:
        return ComboInfo(
            track_internal=self.track_internal_var.get(),
            track_config=self.track_config_var.get(),
            track_display=self.track_display_var.get(),
            track_display_short=self.track_display_short_var.get(),
            car_path=self.car_path_var.get(),
            car_screen=self.car_screen_var.get(),
            car_short=self.car_short_var.get(),
            series_id=self.series_id_var.get(),
        ).normalized()

    def fill_form(self, combo: ComboInfo, enabled: bool | None = None, autosave: bool | None = None) -> None:
        self.track_internal_var.set(combo.track_internal)
        self.track_config_var.set(combo.track_config)
        self.track_display_var.set(combo.track_display)
        self.track_display_short_var.set(combo.track_display_short)
        self.car_path_var.set(combo.car_path)
        self.car_screen_var.set(combo.car_screen)
        self.car_short_var.set(combo.car_short)
        self.series_id_var.set(combo.series_id)
        if enabled is not None:
            self.enabled_var.set(enabled)
        if autosave is not None:
            self.autosave_var.set(autosave)

    def clear_fields(self) -> None:
        self.selected_combo_key = None
        self.fill_form(ComboInfo(), enabled=True, autosave=True)
        self.set_status("Fields cleared")

    def on_renderer_changed(self, _event: object) -> None:
        renderer_file = self.current_renderer_file()
        self.manager.set_selected_renderer(renderer_file)
        self.selected_combo_key = None
        self.refresh_all()
        self.set_status(f"Default renderer alterado para {renderer_file}")

    def on_grouping_changed(self, _event: object) -> None:
        grouping_mode = self.current_grouping_mode()
        self.manager.set_selected_grouping(grouping_mode)
        self.selected_combo_key = None
        self.refresh_all()
        self.set_status(f"Default grouping alterado para {grouping_mode}")

    def on_tree_select(self, _event: object) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        combo_key = selected[0]
        entry = self.manager.get_entry(combo_key, self.current_renderer_file(), self.current_grouping_mode())
        if not entry:
            return

        combo = ComboInfo.from_manifest_entry(entry)
        self.fill_form(
            combo,
            enabled=bool(entry.get("enabled", True)),
            autosave=bool(entry.get("autosave_on_manual_close", True)),
        )
        self.selected_combo_key = combo_key
        self.set_status(f"Selected combo: {combo.label()}")

    def load_current_sim_combo(self) -> None:
        runtime = IRacingRuntime()
        try:
            if not runtime.is_sim_running():
                messagebox.showinfo("Info", "O sim não está rodando agora.")
                return
            runtime.ensure_started()
            if not runtime.is_connected():
                messagebox.showinfo("Info", "SDK ainda não está conectada ao sim.")
                return
            combo = runtime.detect_combo()
            if combo is None:
                messagebox.showinfo("Info", "Não foi possível detectar um combo completo agora.")
                return
            entry = self.manager.register_combo(combo, self.current_renderer_file(), self.current_grouping_mode())
            self.fill_form(
                combo,
                enabled=bool(entry.get("enabled", True)),
                autosave=bool(entry.get("autosave_on_manual_close", True)),
            )
            self.selected_combo_key = combo.combo_key(self.current_grouping_mode())
            self.refresh_all()
            self.set_status(f"Combo loaded from the sim: {combo.label()}")
        finally:
            runtime.reset()

    def save_profile_from_active_ini(self) -> None:
        combo = self.combo_from_form()
        if not combo.is_complete():
            messagebox.showerror("Erro", "Fill in at least the fields relevant to the selected grouping.")
            return

        try:
            entry = self.manager.save_active_ini_as_profile(combo, self.current_renderer_file(), self.current_grouping_mode())
            combo_key = combo.combo_key(self.current_grouping_mode())
            self.manager.update_entry_options(combo_key, self.enabled_var.get(), self.autosave_var.get(), self.current_renderer_file(), self.current_grouping_mode())
            self.selected_combo_key = combo_key
            self.refresh_all()
            self.set_status(f"Profile saved: {Path(str(entry['profile_ini'])).name}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def apply_selected_profile(self) -> None:
        combo = self.combo_from_form()
        if not combo.is_complete():
            messagebox.showerror("Erro", "Select or fill in a valid combo.")
            return

        runtime = IRacingRuntime()
        try:
            if runtime.is_sim_running():
                messagebox.showwarning(
                    "Sim aberto",
                    "Feche o sim antes de aplicar manualmente um profile ao INI ativo.\n"
                    "O monitor automático já cuida disso quando detectar mismatch.",
                )
                return
        finally:
            runtime.reset()

        try:
            self.manager.apply_profile_to_active_ini(combo.combo_key(self.current_grouping_mode()), self.current_renderer_file(), self.current_grouping_mode())
            self.set_status(f"Profile aplicado manualmente ao INI ativo ({self.current_renderer_file()})")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def save_entry_options(self) -> None:
        combo = self.combo_from_form()
        if not combo.is_complete():
            messagebox.showerror("Erro", "Select or fill in a valid combo first.")
            return

        try:
            self.manager.register_combo(combo, self.current_renderer_file(), self.current_grouping_mode())
            self.manager.update_entry_options(
                combo.combo_key(self.current_grouping_mode()),
                self.enabled_var.get(),
                self.autosave_var.get(),
                self.current_renderer_file(),
                self.current_grouping_mode(),
            )
            self.selected_combo_key = combo.combo_key(self.current_grouping_mode())
            self.refresh_all()
            self.set_status("Flags de enabled/autosave salvas")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def create_global_backup(self) -> None:
        runtime = IRacingRuntime()
        try:
            if runtime.is_sim_running():
                messagebox.showwarning("Sim aberto", "Feche o sim antes de atualizar o backup global.")
                return
        finally:
            runtime.reset()

        try:
            self.manager.create_or_refresh_global_backup(self.current_renderer_file())
            self.set_status(f"Backup global criado/atualizado para {self.current_renderer_file()}")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def restore_global_backup(self) -> None:
        runtime = IRacingRuntime()
        try:
            if runtime.is_sim_running():
                messagebox.showwarning("Sim aberto", "Feche o sim antes de restaurar o backup global.")
                return
        finally:
            runtime.reset()

        try:
            self.manager.restore_global_backup(self.current_renderer_file())
            self.set_status(f"Backup global restaurado no INI ativo ({self.current_renderer_file()})")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def open_profiles_folder(self) -> None:
        try:
            path = self.manager.get_grouping_dir(self.current_renderer_file(), self.current_grouping_mode())
            path.mkdir(parents=True, exist_ok=True)
            import os
            os.startfile(path)  # type: ignore[attr-defined]
            self.set_status("Profiles folder opened")
        except Exception as e:
            messagebox.showerror("Erro", str(e))

    def on_close(self) -> None:
        try:
            self.monitor.stop()
        finally:
            self.destroy()


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
