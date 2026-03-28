#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import irsdk
import psutil
import win32con
import win32gui
import win32process
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# CONFIG
# ============================================================
DEFAULT_IRACING_DOCS = Path.home() / "Documents" / "iRacing"
BOOTSTRAP_SETTINGS_PATH = Path.home() / ".iracing_renderer_combo_profile_manager_bootstrap.json"

RENDERER_OPTIONS = {
    "Monitor": "rendererDX11Monitor.ini",
    "OpenXR": "rendererDX11OpenXR.ini",
    "OpenVR": "rendererDX11OpenVR.ini",
}

GROUPING_OPTIONS = {
    "Car + track": "car_track",
    "Car": "car",
    "Track": "track",
    "SeriesID": "series",
    "SeriesID + track": "series_track",
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
    def __init__(self, iracing_docs: Path) -> None:
        self.iracing_docs = iracing_docs
        self.profile_root = self.iracing_docs / "combo_profiles"
        self.app_settings_path = self.profile_root / "app_settings.json"
        self.manifest_path = self.profile_root / "index.json"
        self.lock = threading.RLock()
        self.ensure_dirs()
        self.settings = self.load_settings()
        self.manifest = self.load_manifest()

        self.ensure_renderer_section(self.get_selected_renderer())
        self.ensure_grouping_section(self.get_selected_renderer(), self.get_selected_grouping())
        self.ensure_global_backup_if_missing()

    def ensure_dirs(self) -> None:
        self.profile_root.mkdir(parents=True, exist_ok=True)

    def load_settings(self) -> dict[str, Any]:
        if self.app_settings_path.exists():
            try:
                data = json.loads(self.app_settings_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
        return {}

    def save_settings(self) -> None:
        self.app_settings_path.write_text(
            json.dumps(self.settings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def needs_initial_setup(self) -> bool:
        renderer_ok = self.settings.get("default_renderer") in RENDERER_OPTIONS.values()
        grouping_ok = self.settings.get("default_grouping") in GROUPING_OPTIONS.values()
        return not (renderer_ok and grouping_ok)

    def load_manifest(self) -> dict[str, Any]:
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
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
            self.manifest_path.write_text(
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
            raise ValueError(f"Invalid renderer: {renderer_file}")
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
            raise ValueError(f"Invalid grouping mode: {grouping_mode}")
        with self.lock:
            self.settings["default_grouping"] = grouping_mode
            self.save_settings()
            self.ensure_renderer_section(self.get_selected_renderer())
            self.ensure_grouping_section(self.get_selected_renderer(), grouping_mode)

    def get_active_ini(self, renderer_file: str | None = None) -> Path:
        renderer_file = renderer_file or self.get_selected_renderer()
        return self.iracing_docs / renderer_file

    def renderer_stem(self, renderer_file: str | None = None) -> str:
        renderer_file = renderer_file or self.get_selected_renderer()
        return Path(renderer_file).stem

    def get_renderer_dir(self, renderer_file: str | None = None) -> Path:
        renderer_file = renderer_file or self.get_selected_renderer()
        path = self.profile_root / self.renderer_stem(renderer_file)
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
        return self.profile_root / f"{self.renderer_stem(renderer_file)}.global_backup.ini"

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
            raise FileNotFoundError(f"Active file not found: {active_ini}")
        shutil.copy2(active_ini, backup_path)
        self.save_manifest()

    def restore_global_backup(self, renderer_file: str | None = None) -> None:
        renderer_file = renderer_file or self.get_selected_renderer()
        active_ini = self.get_active_ini(renderer_file)
        backup_path = self.get_global_backup_path(renderer_file)
        if not backup_path.exists():
            raise FileNotFoundError(f"Global backup not found: {backup_path}")
        if not active_ini.exists():
            raise FileNotFoundError(f"Active file not found: {active_ini}")
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
            raise KeyError(f"Combo not found: {combo_key}")
        entry["enabled"] = bool(enabled)
        entry["autosave_on_manual_close"] = bool(autosave)
        self.save_manifest()

    def save_active_ini_as_profile(self, combo: ComboInfo, renderer_file: str | None = None, grouping_mode: str | None = None) -> dict[str, Any]:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()
        combo = combo.normalized()
        active_ini = self.get_active_ini(renderer_file)

        if not active_ini.exists():
            raise FileNotFoundError(f"Active file not found: {active_ini}")

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
            raise KeyError(f"Combo not found: {combo_key}")

        ini_path = Path(str(entry.get("profile_ini") or ""))
        active_ini = self.get_active_ini(renderer_file)

        if not ini_path.exists():
            raise FileNotFoundError(f"Profile not found: {ini_path}")
        if not active_ini.exists():
            raise FileNotFoundError(f"Active file not found: {active_ini}")

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

    def delete_profile(self, combo_key: str, renderer_file: str | None = None, grouping_mode: str | None = None) -> bool:
        renderer_file = renderer_file or self.get_selected_renderer()
        grouping_mode = grouping_mode or self.get_selected_grouping()

        with self.lock:
            combos = self.grouping_section(renderer_file, grouping_mode).setdefault("combos", {})
            entry = combos.pop(combo_key, None)
            if entry is None:
                return False

            ini_path = Path(str(entry.get("profile_ini") or ""))
            meta_path = Path(str(entry.get("profile_meta") or ""))
            for path in (ini_path, meta_path):
                if path and path.exists():
                    try:
                        path.unlink()
                    except Exception:
                        pass

            self.save_manifest()
            return True

    def delete_all_profiles(self) -> int:
        removed = 0
        with self.lock:
            renderers = self.manifest.setdefault("renderers", {})
            for renderer_section in renderers.values():
                groupings = renderer_section.setdefault("groupings", {})
                for grouping_section in groupings.values():
                    combos = grouping_section.setdefault("combos", {})
                    for entry in list(combos.values()):
                        ini_path = Path(str(entry.get("profile_ini") or ""))
                        meta_path = Path(str(entry.get("profile_meta") or ""))
                        for path in (ini_path, meta_path):
                            if path and path.exists():
                                try:
                                    path.unlink()
                                except Exception:
                                    pass
                    removed += len(combos)
                    combos.clear()

            self.save_manifest()
        return removed


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
    def __init__(
        self,
        manager: ProfileManager,
        log_func,
        on_session_closed: Callable[[], None] | None = None,
    ) -> None:
        self.manager = manager
        self.log_func = log_func
        self.on_session_closed = on_session_closed
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
            "Automatic monitor started | "
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
                            f"Combo detected: {combo.label()} | "
                            f"renderer in app: {renderer_file} | grouping: {grouping_mode}"
                        )

                        profile_exists = Path(str(entry.get("profile_ini") or "")).exists()
                        enabled = bool(entry.get("enabled", True))

                        if profile_exists and enabled:
                            if self.manager.active_ini_matches_profile(combo_key, renderer_file, grouping_mode):
                                self.log("Known profile is already active")
                            else:
                                self.session["close_scheduled_at"] = time.time() + PRE_CLOSE_DELAY
                                self.log(
                                    f"Known profile is different from the active file ({active_ini.name}). "
                                    f"WM_CLOSE scheduled for {PRE_CLOSE_DELAY:.0f}s"
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
                                self.log(f"WM_CLOSE sent to {sent} sim window(s)")
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
                            self.log("Correct profile applied to the active INI. Reopen the sim manually whenever you want")
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
                if self.on_session_closed is not None:
                    self.on_session_closed()

            self.last_sim_running = sim_running
            time.sleep(POLL_INTERVAL)

        self.log("Automatic monitor stopped")


# ============================================================
# FIRST RUN DIALOG (PYSIDE6)
# ============================================================
class FirstRunDialog(QDialog):
    def __init__(self, parent: QWidget | None, default_renderer_label: str, default_grouping_label: str) -> None:
        super().__init__(parent)
        self.setWindowTitle("First-time setup")
        self.setModal(True)

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Choose the default renderer and profile grouping mode.\n"
            "You can change this later in the app."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.renderer_combo = QComboBox()
        self.renderer_combo.addItems(list(RENDERER_OPTIONS.keys()))
        self.renderer_combo.setCurrentText(default_renderer_label)

        self.grouping_combo = QComboBox()
        self.grouping_combo.addItems(list(GROUPING_OPTIONS.keys()))
        self.grouping_combo.setCurrentText(default_grouping_label)

        form.addRow("Default renderer", self.renderer_combo)
        form.addRow("Default grouping", self.grouping_combo)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        btn_row.addWidget(save_btn)
        layout.addLayout(btn_row)

    def result_values(self) -> tuple[str, str]:
        return self.renderer_combo.currentText(), self.grouping_combo.currentText()


# ============================================================
# PYSIDE6 GUI
# ============================================================
class App(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("iRacing Renderer Combo Profile Manager")
        self.resize(1280, 830)
        self.setMinimumSize(1160, 720)

        self.iracing_docs = self._resolve_iracing_docs_path()
        self.manager = ProfileManager(self.iracing_docs)
        self.log_queue: list[str] = []
        self.log_lock = threading.Lock()
        self.refresh_requested = threading.Event()

        self.selected_combo_key: str | None = None

        self._build_ui()
        self._run_first_time_setup_if_needed()

        self.monitor = MonitorService(
            self.manager,
            self.enqueue_log,
            on_session_closed=self.request_refresh_after_sim_close,
        )
        self.monitor.start()

        self.refresh_all()

        self.log_timer = QTimer(self)
        self.log_timer.setInterval(200)
        self.log_timer.timeout.connect(self._drain_log_queue)
        self.log_timer.start()

    def _load_bootstrap_path(self) -> Path | None:
        if not BOOTSTRAP_SETTINGS_PATH.exists():
            return None
        try:
            data = json.loads(BOOTSTRAP_SETTINGS_PATH.read_text(encoding="utf-8"))
            saved = str(data.get("iracing_docs") or "").strip() if isinstance(data, dict) else ""
            return Path(saved) if saved else None
        except Exception:
            return None

    def _save_bootstrap_path(self, iracing_docs: Path) -> None:
        payload = {
            "iracing_docs": str(iracing_docs),
            "saved_at": now_str(),
        }
        BOOTSTRAP_SETTINGS_PATH.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _browse_iracing_docs(self, start_path: Path) -> Path | None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select your Documents/iRacing folder",
            str(start_path),
        )
        if not selected:
            return None
        return Path(selected)

    def _resolve_iracing_docs_path(self) -> Path:
        saved = self._load_bootstrap_path()
        if saved:
            return saved

        default_path = DEFAULT_IRACING_DOCS
        found_renderer_files = [
            renderer_file for renderer_file in RENDERER_OPTIONS.values()
            if (default_path / renderer_file).exists()
        ]
        found_text = ", ".join(found_renderer_files) if found_renderer_files else "none of the known renderer INI files"

        answer = QMessageBox.question(
            self,
            "Confirm iRacing folder",
            "Detected iRacing folder:\n"
            f"{default_path}\n\n"
            f"Renderer files found: {found_text}\n\n"
            "Use this folder?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if answer == QMessageBox.StandardButton.Yes:
            chosen = default_path
        else:
            browse_start = default_path.parent if default_path.parent.exists() else Path.home()
            chosen = self._browse_iracing_docs(browse_start) or default_path
            if chosen == default_path:
                QMessageBox.information(
                    self,
                    "Using default folder",
                    f"No folder selected. The app will use:\n{default_path}",
                )

        self._save_bootstrap_path(chosen)
        return chosen

    def enqueue_log(self, text: str) -> None:
        with self.log_lock:
            self.log_queue.append(text)

    def request_refresh_after_sim_close(self) -> None:
        self.refresh_requested.set()

    def _drain_log_queue(self) -> None:
        with self.log_lock:
            items = self.log_queue[:]
            self.log_queue.clear()

        if not items:
            return

        self.log_text.moveCursor(QTextCursor.MoveOperation.End)
        for item in items:
            self.log_text.append(item)

        if self.refresh_requested.is_set():
            self.refresh_requested.clear()
            self.refresh_all()

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
        return "Car + track"

    def label_to_grouping(self, label: str) -> str:
        return GROUPING_OPTIONS.get(label, self.manager.get_selected_grouping())

    def current_renderer_file(self) -> str:
        return self.label_to_renderer(self.renderer_combo.currentText())

    def current_grouping_mode(self) -> str:
        return self.label_to_grouping(self.grouping_combo.currentText())

    def _run_first_time_setup_if_needed(self) -> None:
        if not self.manager.needs_initial_setup():
            return

        dialog = FirstRunDialog(self, "OpenXR", "Car + track")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            renderer_label, grouping_label = dialog.result_values()
        else:
            renderer_label, grouping_label = "OpenXR", "Car + track"

        renderer_file = self.label_to_renderer(renderer_label)
        grouping_mode = self.label_to_grouping(grouping_label)

        self.manager.set_selected_renderer(renderer_file)
        self.manager.set_selected_grouping(grouping_mode)

        self.renderer_combo.setCurrentText(renderer_label)
        self.grouping_combo.setCurrentText(grouping_label)

        self.enqueue_log(
            f"[{now_str()}] First-time setup saved | "
            f"renderer: {renderer_file} | grouping: {grouping_mode}"
        )

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top_box = QGroupBox("App configuration")
        top_layout = QVBoxLayout(top_box)
        row0 = QHBoxLayout()

        row0.addWidget(QLabel("Default renderer"))
        self.renderer_combo = QComboBox()
        self.renderer_combo.addItems(list(RENDERER_OPTIONS.keys()))
        self.renderer_combo.setCurrentText(self.renderer_to_label(self.manager.get_selected_renderer()))
        self.renderer_combo.currentTextChanged.connect(self.on_renderer_changed)
        row0.addWidget(self.renderer_combo)

        row0.addSpacing(12)
        row0.addWidget(QLabel("Default grouping"))
        self.grouping_combo = QComboBox()
        self.grouping_combo.addItems(list(GROUPING_OPTIONS.keys()))
        self.grouping_combo.setCurrentText(self.grouping_to_label(self.manager.get_selected_grouping()))
        self.grouping_combo.currentTextChanged.connect(self.on_grouping_changed)
        row0.addWidget(self.grouping_combo)

        row0.addSpacing(10)
        change_folder_btn = QPushButton("Change iRacing folder")
        change_folder_btn.clicked.connect(self.change_iracing_folder)
        row0.addWidget(change_folder_btn)

        self.renderer_label = QLabel()
        self.grouping_label = QLabel()
        row0.addWidget(self.renderer_label)
        row0.addWidget(self.grouping_label)
        row0.addStretch(1)

        top_layout.addLayout(row0)
        self.paths_label = QLabel()
        self.paths_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        top_layout.addWidget(self.paths_label)
        root.addWidget(top_box)

        mid_layout = QHBoxLayout()

        left_box = QGroupBox("Known combos for selected renderer + grouping")
        left_layout = QVBoxLayout(left_box)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels([
            "Combo key", "Track", "Layout", "Car", "SeriesID", "Enabled", "Autosave", "Last saved"
        ])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_table_select)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.table)
        mid_layout.addWidget(left_box, 3)

        right_box = QGroupBox("Selected combo / manual editor")
        right_layout = QVBoxLayout(right_box)

        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        self.track_internal_edit = self._make_edit(form_layout, "Track internal")
        self.track_config_edit = self._make_edit(form_layout, "Track config/layout")
        self.track_display_edit = self._make_edit(form_layout, "Track display")
        self.track_display_short_edit = self._make_edit(form_layout, "Track display short")
        self.car_path_edit = self._make_edit(form_layout, "Car path")
        self.car_screen_edit = self._make_edit(form_layout, "Car screen")
        self.car_short_edit = self._make_edit(form_layout, "Car short")
        self.series_id_edit = self._make_edit(form_layout, "SeriesID")
        right_layout.addWidget(form_widget)

        opt_row = QHBoxLayout()
        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setChecked(True)
        self.autosave_check = QCheckBox("Autosave on manual close")
        self.autosave_check.setChecked(True)
        opt_row.addWidget(self.enabled_check)
        opt_row.addWidget(self.autosave_check)
        opt_row.addStretch(1)
        right_layout.addLayout(opt_row)

        btn_grid = QGridLayout()
        buttons = [
            ("Clear fields", self.clear_fields),
            ("Refresh list", self.refresh_all),
            ("Load current sim combo", self.load_current_sim_combo),
            ("Save/overwrite profile from selected renderer INI", self.save_profile_from_active_ini),
            ("Apply selected profile to selected renderer INI", self.apply_selected_profile),
            ("Save enabled/autosave flags", self.save_entry_options),
            ("Create/refresh global backup", self.create_global_backup),
            ("Restore global backup to selected renderer INI", self.restore_global_backup),
            ("Open current profiles folder", self.open_profiles_folder),
            ("Delete selected profile", self.delete_selected_profile),
            ("Delete ALL profiles", self.delete_all_profiles),
        ]
        for idx, (label, cb) in enumerate(buttons):
            r, c = divmod(idx, 3)
            b = QPushButton(label)
            b.clicked.connect(cb)
            btn_grid.addWidget(b, r, c)
        right_layout.addLayout(btn_grid)

        mid_layout.addWidget(right_box, 2)
        root.addLayout(mid_layout, 1)

        status_frame = QFrame()
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(0, 0, 0, 0)
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        root.addWidget(status_frame)

        log_box = QGroupBox("Log")
        log_layout = QVBoxLayout(log_box)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        root.addWidget(log_box, 1)

    def _make_edit(self, layout: QFormLayout, label: str) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        combo.setMinimumWidth(340)
        layout.addRow(label, combo)
        return combo

    def update_labels(self) -> None:
        renderer_file = self.current_renderer_file()
        grouping_mode = self.current_grouping_mode()

        active_ini = self.manager.get_active_ini(renderer_file)
        profiles_dir = self.manager.get_grouping_dir(renderer_file, grouping_mode)
        backup = self.manager.get_global_backup_path(renderer_file)

        self.renderer_label.setText(f"Selected file: {renderer_file}")
        self.grouping_label.setText(f"Selected grouping: {grouping_mode}")
        self.paths_label.setText(
            f"Active INI: {active_ini}\n"
            f"Profiles dir: {profiles_dir}\n"
            f"Manifest: {self.manager.manifest_path}\n"
            f"Global backup: {backup}"
        )

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.enqueue_log(f"[{now_str()}] {text}")

    def _set_combo_values(self, box: QComboBox, values: list[str]) -> None:
        current = box.currentText()
        box.blockSignals(True)
        box.clear()
        box.addItems(values)
        box.setEditText(current)
        box.blockSignals(False)

    def refresh_suggestions(self) -> None:
        suggestions = self.manager.get_suggestions(self.current_renderer_file(), self.current_grouping_mode())
        self._set_combo_values(self.track_internal_edit, suggestions["track_internal"])
        self._set_combo_values(self.track_config_edit, suggestions["track_config"])
        self._set_combo_values(self.track_display_edit, suggestions["track_display"])
        self._set_combo_values(self.car_path_edit, suggestions["car_path"])
        self._set_combo_values(self.car_screen_edit, suggestions["car_screen"])
        self._set_combo_values(self.series_id_edit, suggestions["series_id"])

    def refresh_table(self) -> None:
        entries = self.manager.list_entries(self.current_renderer_file(), self.current_grouping_mode())
        self.table.setRowCount(0)

        for row, entry in enumerate(entries):
            self.table.insertRow(row)
            combo_key = str(entry.get("combo_key"))
            track = str(entry.get("track_display") or entry.get("track_internal") or "")
            layout = str(entry.get("track_config") or "")
            car = str(entry.get("car_screen") or entry.get("car_path") or "")
            series_id = str(entry.get("series_id") or "")
            enabled = "yes" if bool(entry.get("enabled", True)) else "no"
            autosave = "yes" if bool(entry.get("autosave_on_manual_close", True)) else "no"
            saved = str(entry.get("last_saved_at") or "")

            for col, value in enumerate([combo_key, track, layout, car, series_id, enabled, autosave, saved]):
                self.table.setItem(row, col, QTableWidgetItem(value))

        self.table.resizeColumnsToContents()

    def refresh_all(self) -> None:
        self.manager.settings = self.manager.load_settings()
        self.manager.manifest = self.manager.load_manifest()
        self.manager.ensure_renderer_section(self.current_renderer_file())
        self.manager.ensure_grouping_section(self.current_renderer_file(), self.current_grouping_mode())
        self.update_labels()
        self.refresh_table()
        self.refresh_suggestions()
        self.set_status("List, paths, and suggestions refreshed")

    def combo_from_form(self) -> ComboInfo:
        return ComboInfo(
            track_internal=self.track_internal_edit.currentText(),
            track_config=self.track_config_edit.currentText(),
            track_display=self.track_display_edit.currentText(),
            track_display_short=self.track_display_short_edit.currentText(),
            car_path=self.car_path_edit.currentText(),
            car_screen=self.car_screen_edit.currentText(),
            car_short=self.car_short_edit.currentText(),
            series_id=self.series_id_edit.currentText(),
        ).normalized()

    def _set_combo_text(self, box: QComboBox, text: str) -> None:
        idx = box.findText(text)
        if idx >= 0:
            box.setCurrentIndex(idx)
        else:
            box.setEditText(text)

    def fill_form(self, combo: ComboInfo, enabled: bool | None = None, autosave: bool | None = None) -> None:
        self._set_combo_text(self.track_internal_edit, combo.track_internal)
        self._set_combo_text(self.track_config_edit, combo.track_config)
        self._set_combo_text(self.track_display_edit, combo.track_display)
        self._set_combo_text(self.track_display_short_edit, combo.track_display_short)
        self._set_combo_text(self.car_path_edit, combo.car_path)
        self._set_combo_text(self.car_screen_edit, combo.car_screen)
        self._set_combo_text(self.car_short_edit, combo.car_short)
        self._set_combo_text(self.series_id_edit, combo.series_id)
        if enabled is not None:
            self.enabled_check.setChecked(enabled)
        if autosave is not None:
            self.autosave_check.setChecked(autosave)

    def clear_fields(self) -> None:
        self.selected_combo_key = None
        self.fill_form(ComboInfo(), enabled=True, autosave=True)
        self.set_status("Fields cleared")

    def on_renderer_changed(self) -> None:
        renderer_file = self.current_renderer_file()
        self.manager.set_selected_renderer(renderer_file)
        self.selected_combo_key = None
        self.refresh_all()
        self.set_status(f"Default renderer changed to {renderer_file}")

    def on_grouping_changed(self) -> None:
        grouping_mode = self.current_grouping_mode()
        self.manager.set_selected_grouping(grouping_mode)
        self.selected_combo_key = None
        self.refresh_all()
        self.set_status(f"Default grouping changed to {grouping_mode}")

    def on_table_select(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            return
        key_item = self.table.item(row, 0)
        if key_item is None:
            return
        combo_key = key_item.text()
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
                QMessageBox.information(self, "Info", "The sim is not running right now.")
                return
            runtime.ensure_started()
            if not runtime.is_connected():
                QMessageBox.information(self, "Info", "The SDK is not connected to the sim yet.")
                return
            combo = runtime.detect_combo()
            if combo is None:
                QMessageBox.information(self, "Info", "Could not detect a complete combo right now.")
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
            QMessageBox.critical(self, "Error", "Fill in at least the fields relevant to the selected grouping.")
            return

        try:
            entry = self.manager.save_active_ini_as_profile(combo, self.current_renderer_file(), self.current_grouping_mode())
            combo_key = combo.combo_key(self.current_grouping_mode())
            self.manager.update_entry_options(
                combo_key,
                self.enabled_check.isChecked(),
                self.autosave_check.isChecked(),
                self.current_renderer_file(),
                self.current_grouping_mode(),
            )
            self.selected_combo_key = combo_key
            self.refresh_all()
            self.set_status(f"Profile saved: {Path(str(entry['profile_ini'])).name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def apply_selected_profile(self) -> None:
        combo = self.combo_from_form()
        if not combo.is_complete():
            QMessageBox.critical(self, "Error", "Select or fill in a valid combo.")
            return

        runtime = IRacingRuntime()
        try:
            if runtime.is_sim_running():
                QMessageBox.warning(
                    self,
                    "Sim open",
                    "Close the sim before manually applying a profile to the active INI.\n"
                    "The automatic monitor already handles this when it detects a mismatch.",
                )
                return
        finally:
            runtime.reset()

        try:
            self.manager.apply_profile_to_active_ini(
                combo.combo_key(self.current_grouping_mode()),
                self.current_renderer_file(),
                self.current_grouping_mode(),
            )
            self.set_status(f"Profile manually applied to the active INI ({self.current_renderer_file()})")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def save_entry_options(self) -> None:
        combo = self.combo_from_form()
        if not combo.is_complete():
            QMessageBox.critical(self, "Error", "Select or fill in a valid combo first.")
            return

        try:
            self.manager.register_combo(combo, self.current_renderer_file(), self.current_grouping_mode())
            self.manager.update_entry_options(
                combo.combo_key(self.current_grouping_mode()),
                self.enabled_check.isChecked(),
                self.autosave_check.isChecked(),
                self.current_renderer_file(),
                self.current_grouping_mode(),
            )
            self.selected_combo_key = combo.combo_key(self.current_grouping_mode())
            self.refresh_all()
            self.set_status("Enabled/autosave flags saved")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def create_global_backup(self) -> None:
        runtime = IRacingRuntime()
        try:
            if runtime.is_sim_running():
                QMessageBox.warning(self, "Sim open", "Close the sim before updating the global backup.")
                return
        finally:
            runtime.reset()

        try:
            self.manager.create_or_refresh_global_backup(self.current_renderer_file())
            self.set_status(f"Global backup created/updated for {self.current_renderer_file()}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def restore_global_backup(self) -> None:
        runtime = IRacingRuntime()
        try:
            if runtime.is_sim_running():
                QMessageBox.warning(self, "Sim open", "Close the sim before restoring the global backup.")
                return
        finally:
            runtime.reset()

        try:
            self.manager.restore_global_backup(self.current_renderer_file())
            self.set_status(f"Global backup restored to the active INI ({self.current_renderer_file()})")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def open_profiles_folder(self) -> None:
        try:
            path = self.manager.get_grouping_dir(self.current_renderer_file(), self.current_grouping_mode())
            path.mkdir(parents=True, exist_ok=True)
            os.startfile(path)  # type: ignore[attr-defined]
            self.set_status("Profiles folder opened")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def change_iracing_folder(self) -> None:
        browse_start = self.iracing_docs if self.iracing_docs.exists() else DEFAULT_IRACING_DOCS
        chosen = self._browse_iracing_docs(browse_start)
        if chosen is None:
            self.set_status("Folder change canceled")
            return

        answer = QMessageBox.question(
            self,
            "Confirm folder change",
            "Switch iRacing folder to:\n"
            f"{chosen}\n\n"
            "This will reload profiles/settings from the new location.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.set_status("Folder change canceled")
            return

        self.iracing_docs = chosen
        self._save_bootstrap_path(chosen)
        self.manager = ProfileManager(chosen)

        self.renderer_combo.blockSignals(True)
        self.grouping_combo.blockSignals(True)
        self.renderer_combo.setCurrentText(self.renderer_to_label(self.manager.get_selected_renderer()))
        self.grouping_combo.setCurrentText(self.grouping_to_label(self.manager.get_selected_grouping()))
        self.renderer_combo.blockSignals(False)
        self.grouping_combo.blockSignals(False)

        self.selected_combo_key = None
        self._run_first_time_setup_if_needed()
        self.refresh_all()
        self.set_status(f"iRacing folder changed to {chosen}")

    def delete_selected_profile(self) -> None:
        combo = self.combo_from_form()
        if not combo.is_complete():
            QMessageBox.information(self, "Info", "Select a profile first.")
            return

        combo_key = combo.combo_key(self.current_grouping_mode())
        entry = self.manager.get_entry(combo_key, self.current_renderer_file(), self.current_grouping_mode())
        if not entry:
            QMessageBox.information(self, "Info", "Profile not found for the current selection.")
            return

        answer = QMessageBox.question(
            self,
            "Delete selected profile",
            "Delete this profile?\n\n"
            f"{combo_key}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.set_status("Delete selected profile canceled")
            return

        deleted = self.manager.delete_profile(combo_key, self.current_renderer_file(), self.current_grouping_mode())
        if not deleted:
            QMessageBox.information(self, "Info", "The selected profile no longer exists.")
            return

        self.selected_combo_key = None
        self.clear_fields()
        self.refresh_all()
        self.set_status(f"Deleted profile: {combo_key}")

    def delete_all_profiles(self) -> None:
        answer = QMessageBox.question(
            self,
            "Delete ALL profiles",
            "Delete ALL saved profiles across all renderers/groupings?\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            self.set_status("Delete all profiles canceled")
            return

        removed = self.manager.delete_all_profiles()
        self.selected_combo_key = None
        self.clear_fields()
        self.refresh_all()
        self.set_status(f"Deleted {removed} profile(s)")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self.monitor.stop()
        finally:
            event.accept()


def main() -> int:
    app = QApplication([])
    window = App()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
