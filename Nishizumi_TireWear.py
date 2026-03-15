#!/usr/bin/env python3
"""
iRacing Tire Wear Learning Overlay

Single-file application that:
- Reads live telemetry from iRacing (irsdk) at 60 Hz
- Detects stints and learns tire wear behavior over time
- Uses driving load (|LatAccel| * Speed) and temperature modeling
- Saves learned data by track+config+car into nishizumi_tirewear_model.json
- Shows a transparent overlay HUD and a settings/info menu (PyQt5)

Install:
    pip install irsdk numpy pyqt5
Run:
    python Nishizumi_TireWear.py
"""
from __future__ import annotations
import json
import os
import queue
import re
import signal
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import irsdk


MODEL_PATH = "nishizumi_tirewear_model.json"
SETTINGS_PATH = "nishizumi_tirewear_settings.json"
TIRE_KEYS = ("lf", "rf", "lr", "rr")
WEAR_FIELDS = {
    "lf": ("LFwearL", "LFwearM", "LFwearR"),
    "rf": ("RFwearL", "RFwearM", "RFwearR"),
    "lr": ("LRwearL", "LRwearM", "LRwearR"),
    "rr": ("RRwearL", "RRwearM", "RRwearR"),
}
# Inner shoulder for each tire (carcass side). For left tires it's "R", for right tires it's "L".
INNER_WEAR_INDEX = {
    "lf": 2,
    "rf": 0,
    "lr": 2,
    "rr": 0,
}


@dataclass
class TelemetrySnapshot:
    """Immutable snapshot of telemetry values consumed by the model worker."""

    session_time: float
    lap: int
    lap_dist_pct: float
    on_pit_road: bool
    speed_mps: float
    lat_accel: float
    long_accel: float
    steering: float
    track_temp: float
    air_temp: float
    humidity: float
    wear: Dict[str, float]
    track_name: str
    track_config: str
    car_path: str


class DataStorage:
    """Thread-safe JSON persistence for learned tire model samples."""

    def __init__(self, path: str = MODEL_PATH):
        self.path = path
        self.lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> Dict[str, dict]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return raw if isinstance(raw, dict) else {}
        except Exception:
            return {}

    def save(self):
        with self.lock:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)

    def get_samples(self, key: str) -> List[dict]:
        with self.lock:
            return list(self.data.get(key, {}).get("samples", []))

    def add_sample(self, key: str, sample: dict):
        with self.lock:
            if key not in self.data:
                self.data[key] = {"samples": []}
            self.data[key]["samples"].append(sample)


_PHI_DIM = 7


def _phi(track_temp: float, air_temp: float, humidity: float, energy_per_lap: float) -> np.ndarray:
    t_track = float(track_temp)
    e_lap = float(energy_per_lap)
    return np.array(
        [
            1.0,
            t_track,
            t_track * t_track,
            float(air_temp),
            float(humidity),
            e_lap,
            t_track * e_lap,
        ],
        dtype=float,
    )


class RLSEstimator:
    """Single-target recursive least squares estimator."""

    def __init__(self, lam: float = 0.98, sigma0: float = 1e4):
        self.lam = float(lam)
        self.theta = np.zeros(_PHI_DIM, dtype=float)
        self.P = float(sigma0) * np.eye(_PHI_DIM, dtype=float)
        self.n_updates = 0
        self.mad_error = 1e-4

    def update(self, x: np.ndarray, y: float) -> float:
        px = self.P @ x
        gain = px / (self.lam + float(x @ px))
        error = float(y) - float(x @ self.theta)
        self.theta += gain * error
        self.P = (self.P - np.outer(gain, px)) / self.lam
        self.n_updates += 1
        self.mad_error = 0.95 * self.mad_error + 0.05 * abs(error)
        return error

    def predict(self, x: np.ndarray) -> float:
        return max(0.0, float(x @ self.theta))

    @property
    def confidence(self) -> float:
        return float(np.tanh(self.n_updates / 13.0))

    @property
    def uncertainty_trace(self) -> float:
        return float(np.trace(self.P))

    def to_dict(self) -> dict:
        return {
            "theta": self.theta.tolist(),
            "P": [r.tolist() for r in self.P],
            "n_updates": int(self.n_updates),
            "lam": float(self.lam),
            "mad_error": float(self.mad_error),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RLSEstimator":
        obj = cls(lam=float(data.get("lam", 0.98)))
        obj.theta = np.array(data.get("theta", np.zeros(_PHI_DIM)), dtype=float)
        obj.P = np.array(data.get("P", np.eye(_PHI_DIM) * 1e4), dtype=float)
        obj.n_updates = int(data.get("n_updates", 0))
        obj.mad_error = max(1e-6, float(data.get("mad_error", 1e-4)))
        return obj


class TireMLModel:
    """Persistent online tire wear model powered by RLS per tire."""

    def __init__(self, storage: DataStorage):
        self.storage = storage
        self._rls: Dict[str, RLSEstimator] = {t: RLSEstimator() for t in TIRE_KEYS}

    @staticmethod
    def _rls_key(dataset_key: str) -> str:
        return f"{dataset_key}::rls"

    def load_rls(self, dataset_key: str):
        with self.storage.lock:
            raw = self.storage.data.get(self._rls_key(dataset_key), {})
        for tire in TIRE_KEYS:
            self._rls[tire] = RLSEstimator.from_dict(raw[tire]) if tire in raw else RLSEstimator()

    def save_rls(self, dataset_key: str):
        with self.storage.lock:
            self.storage.data[self._rls_key(dataset_key)] = {t: self._rls[t].to_dict() for t in TIRE_KEYS}
        self.storage.save()

    def add_stint_sample(self, dataset_key: str, sample: dict):
        self.storage.add_sample(dataset_key, sample)
        x = _phi(
            float(sample["track_temp"]),
            float(sample["air_temp"]),
            float(sample.get("humidity", 50.0)),
            float(sample.get("energy_per_lap", 0.0)),
        )
        for tire in TIRE_KEYS:
            self._rls[tire].update(x, float(sample[tire]))
        self.save_rls(dataset_key)

    def is_outlier(self, dataset_key: str, candidate: dict) -> bool:
        x = _phi(
            float(candidate["track_temp"]),
            float(candidate["air_temp"]),
            float(candidate.get("humidity", 50.0)),
            float(candidate.get("energy_per_lap", 0.0)),
        )
        for tire in TIRE_KEYS:
            rls = self._rls[tire]
            if rls.n_updates < 4:
                continue
            pred = rls.predict(x)
            pred_var = max(1e-10, float(x @ rls.P @ x) + rls.mad_error * rls.mad_error)
            z = abs(float(candidate[tire]) - pred) / pred_var**0.5
            if z > 3.5:
                return True
        return False

    def get_rates(
        self,
        dataset_key: str,
        track_temp: float,
        air_temp: float = 25.0,
        humidity: float = 50.0,
        energy_per_lap: float = 0.0,
    ) -> Tuple[Dict[str, float], float, int]:
        sample_count = self.sample_count(dataset_key)
        if self._rls["lf"].n_updates == 0:
            return {t: 0.0 for t in TIRE_KEYS}, 0.0, sample_count

        x = _phi(track_temp, air_temp, humidity, energy_per_lap)
        samples = self.storage.get_samples(dataset_key)
        rates = {}
        for tire in TIRE_KEYS:
            rls = self._rls[tire]
            conf = rls.confidence
            prior = float(np.median([float(s[tire]) for s in samples if tire in s])) if samples else 0.0
            rates[tire] = max(0.0, conf * rls.predict(x) + (1.0 - conf) * prior)
        return rates, self._rls["lf"].confidence, sample_count

    def sample_count(self, key: str) -> int:
        return len(self.storage.get_samples(key))

    def get_wear_per_lap_baseline(
        self,
        key: str,
        track_temp: float,
        air_temp: float = 25.0,
        humidity: float = 50.0,
    ) -> Dict[str, float]:
        samples = self.storage.get_samples(key)
        if not samples:
            return {t: 0.0 for t in TIRE_KEYS}

        med_epl = float(np.median([float(s.get("energy_per_lap", 0.0)) for s in samples]))
        rates, _, _ = self.get_rates(key, track_temp, air_temp, humidity, med_epl)
        return {t: rates[t] * med_epl for t in TIRE_KEYS}

    def get_coefficients_report(self, dataset_key: str) -> str:
        names = ["bias", "T_track", "T_track²", "T_air", "humidity", "E/lap", "T_track×E/lap"]
        lines = [f"=== Coefficients: {dataset_key} ==="]
        for tire in TIRE_KEYS:
            rls = self._rls[tire]
            lines.append(
                f"\n[{tire.upper()}] n={rls.n_updates}  conf={rls.confidence:.1%}  "
                f"P_trace={rls.uncertainty_trace:.2e}"
            )
            for name, coef in zip(names, rls.theta):
                lines.append(f"  {name:20s}: {coef:+.6e}")
        return "\n".join(lines)


class StintTracker:
    """Detects stint boundaries and computes validated learning samples at pit-in."""

    MIN_LAPS = 2
    LAP_STD_THRESHOLD = 15.0

    def __init__(self):
        self.in_stint = False
        self.prev_on_pit: Optional[bool] = None
        self.start_data: Dict[str, object] = {}

        self.current_energy = 0.0
        self.last_snapshot: Optional[TelemetrySnapshot] = None

        self.last_lap: Optional[int] = None
        self.last_lap_cross_time: Optional[float] = None
        self.lap_times: List[float] = []
        self.min_speed_kmh = float("inf")
        self.stopped_in_pit = False

    @staticmethod
    def make_dataset_key(s: TelemetrySnapshot) -> str:
        def clean(text: str, fallback: str) -> str:
            return (text or fallback).strip().lower().replace(" ", "_")

        track = clean(s.track_name, "unknown_track")
        config = clean(s.track_config, "default")
        car = clean(s.car_path, "unknown_car")
        return f"{track}+{config}+{car}"

    @staticmethod
    def laps_in_stint(current: TelemetrySnapshot, previous: TelemetrySnapshot) -> float:
        """Return lap progress completed between two snapshots.

        Uses lap + lap distance percent so partial laps are accounted for while
        avoiding negative jumps across session resets/rewinds.
        """

        prev_progress = float(previous.lap) + float(previous.lap_dist_pct)
        curr_progress = float(current.lap) + float(current.lap_dist_pct)
        return max(0.0, curr_progress - prev_progress)

    def _start_stint(self, snapshot: TelemetrySnapshot, speed_kmh: float):
        """Initialize a new stint from the current telemetry snapshot."""

        fresh_tires = bool(self.stopped_in_pit)
        start_wear = {t: 100.0 for t in TIRE_KEYS} if fresh_tires else dict(snapshot.wear)
        self.in_stint = True
        self.start_data = {
            "wear": start_wear,
            "session_time": snapshot.session_time,
            "lap": snapshot.lap,
            "lap_progress": float(snapshot.lap) + float(snapshot.lap_dist_pct),
            "energy": self.current_energy,
            "track_temp": snapshot.track_temp,
            "air_temp": snapshot.air_temp,
            "key": self.make_dataset_key(snapshot),
        }
        self.lap_times = []
        self.min_speed_kmh = speed_kmh
        self.stopped_in_pit = False

    def update(self, snapshot: TelemetrySnapshot) -> Optional[dict]:
        speed_kmh = snapshot.speed_mps * 3.6
        self.min_speed_kmh = min(self.min_speed_kmh, speed_kmh)

        if snapshot.on_pit_road and speed_kmh < 1.0:
            self.stopped_in_pit = True

        # Driving load energy integration.
        if self.last_snapshot is not None:
            dt = max(0.0, snapshot.session_time - self.last_snapshot.session_time)
            self.current_energy += abs(snapshot.lat_accel) * snapshot.speed_mps * dt

        # Capture lap times for outlier rejection.
        if self.last_lap is None:
            self.last_lap = snapshot.lap
            self.last_lap_cross_time = snapshot.session_time
        elif snapshot.lap > self.last_lap:
            if self.last_lap_cross_time is not None:
                lap_time = snapshot.session_time - self.last_lap_cross_time
                if 20.0 <= lap_time <= 500.0:
                    self.lap_times.append(lap_time)
            self.last_lap_cross_time = snapshot.session_time
            self.last_lap = snapshot.lap

        result = None
        if self.prev_on_pit is None:
            self.prev_on_pit = snapshot.on_pit_road

            # App can start while already on-track. Bootstrap a stint so live wear
            # and first learning sample work without requiring a full pit cycle first.
            if not snapshot.on_pit_road:
                self._start_stint(snapshot, speed_kmh)

        # Stint start: in pit before, now not in pit.
        if (not snapshot.on_pit_road) and bool(self.prev_on_pit):
            self._start_stint(snapshot, speed_kmh)

        # Stint end: was out of pit, now entered pit.
        if snapshot.on_pit_road and (not bool(self.prev_on_pit)) and self.in_stint:
            self.in_stint = False

            laps = max(0, snapshot.lap - int(self.start_data.get("lap", snapshot.lap)))
            start_energy = float(self.start_data.get("energy", self.current_energy))
            energy_used = max(1e-9, self.current_energy - start_energy)

            wear_start = self.start_data.get("wear", snapshot.wear)
            wear_end = snapshot.wear
            wear_delta = {t: max(0.0, float(wear_start[t]) - float(wear_end[t])) for t in TIRE_KEYS}
            wear_per_lap = {t: wear_delta[t] / laps if laps > 0 else 0.0 for t in TIRE_KEYS}
            wear_per_energy = {t: wear_delta[t] / energy_used for t in TIRE_KEYS}
            energy_per_lap = energy_used / laps if laps > 0 else 0.0

            lap_std = float(np.std(self.lap_times)) if len(self.lap_times) >= 2 else 0.0

            result = {
                "key": self.start_data.get("key"),
                "laps": laps,
                "lap_std": lap_std,
                "lap_times": list(self.lap_times),
                "min_speed_kmh": self.min_speed_kmh,
                "track_temp": float(snapshot.track_temp),
                "air_temp": float(snapshot.air_temp),
                "energy_per_lap": float(energy_per_lap),
                "wear_per_lap": wear_per_lap,
                "wear_per_energy": wear_per_energy,
            }

        self.prev_on_pit = snapshot.on_pit_road
        self.last_snapshot = snapshot
        return result

    def stint_is_valid(self, stint: dict) -> bool:
        if stint["laps"] < self.MIN_LAPS:
            return False

        # Out-lap and in-lap tend to be much slower and can cause false rejects.
        lap_times = list(stint.get("lap_times", []))
        if len(lap_times) > 2:
            lap_times = lap_times[1:-1]

        if len(lap_times) >= 2 and float(np.std(lap_times)) > self.LAP_STD_THRESHOLD:
            return False
        if stint["min_speed_kmh"] < 20.0:
            return False
        return True

    def build_live_estimate(
        self,
        snapshot: TelemetrySnapshot,
        rates_per_energy: Dict[str, float],
        baseline_wear_per_lap: Dict[str, float],
    ) -> Optional[dict]:
        if not self.in_stint or not self.start_data:
            return None

        start_energy = float(self.start_data.get("energy", self.current_energy))
        energy_used = max(0.0, self.current_energy - start_energy)
        initial_wear = self.start_data.get("wear", snapshot.wear)
        start_lap_progress = float(self.start_data.get("lap_progress", float(snapshot.lap) + float(snapshot.lap_dist_pct)))
        current_lap_progress = float(snapshot.lap) + float(snapshot.lap_dist_pct)
        laps_progress = max(0.0, current_lap_progress - start_lap_progress)

        estimated_tread = {}
        for tire in TIRE_KEYS:
            predicted_wear_energy = energy_used * float(rates_per_energy.get(tire, 0.0))
            predicted_wear_laps = laps_progress * float(baseline_wear_per_lap.get(tire, 0.0))
            predicted_wear = max(predicted_wear_energy, predicted_wear_laps)
            estimated_tread[tire] = float(np.clip(float(initial_wear[tire]) - predicted_wear, 0.0, 100.0))

        laps_done = max(0, snapshot.lap - int(self.start_data.get("lap", snapshot.lap)))
        return {
            "estimated_tread": estimated_tread,
            "energy_used": energy_used,
            "laps_done": laps_done,
            "laps_progress": laps_progress,
        }


class TelemetryReader(threading.Thread):
    """60 Hz telemetry reader; reconnects automatically if sim/session restarts."""

    def __init__(self, out_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.ir = irsdk.IRSDK()
        self.last_meta = {"TrackName": "", "TrackConfigName": "", "CarPath": ""}

    @staticmethod
    def _safe_float(v, default=0.0) -> float:
        try:
            return float(v) if v is not None else float(default)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_int(v, default=0) -> int:
        try:
            return int(v) if v is not None else int(default)
        except Exception:
            return int(default)

    def _connected(self) -> bool:
        try:
            if not getattr(self.ir, "is_initialized", False):
                self.ir.startup()
            return bool(getattr(self.ir, "is_connected", False))
        except Exception:
            return False

    def _parse_metadata(self) -> Dict[str, str]:
        def prettify(text: str) -> str:
            cleaned = re.sub(r"[_\-]+", " ", str(text or "")).strip()
            return cleaned.title()

        # Primary source (same approach used in the main app): direct iRacing blocks.
        try:
            weekend = self.ir["WeekendInfo"]
            if isinstance(weekend, dict):
                track_name = str(weekend.get("TrackDisplayName") or weekend.get("TrackName") or "").strip()
                track_config = str(weekend.get("TrackConfigName") or "").strip()
                if track_name:
                    self.last_meta["TrackName"] = prettify(track_name)
                if track_config:
                    self.last_meta["TrackConfigName"] = prettify(track_config)
        except Exception:
            pass

        try:
            driver_info = self.ir["DriverInfo"]
            if isinstance(driver_info, dict):
                drivers = driver_info.get("Drivers") or []
                idx = driver_info.get("DriverCarIdx")
                try:
                    idx = int(idx) if idx is not None else None
                except (TypeError, ValueError):
                    idx = None
                if idx is not None and 0 <= idx < len(drivers) and isinstance(drivers[idx], dict):
                    entry = drivers[idx]
                    car_path = str(
                        entry.get("CarScreenName")
                        or entry.get("CarScreenNameShort")
                        or entry.get("CarPath")
                        or ""
                    ).strip()
                    if car_path:
                        self.last_meta["CarPath"] = prettify(car_path)
        except Exception:
            pass

        if any(self.last_meta.values()):
            return self.last_meta

        # Prefer parsed SessionInfo dict and fallback to session YAML string.
        try:
            session_info = self.ir["SessionInfo"]
            if isinstance(session_info, dict):
                weekend = session_info.get("WeekendInfo") or {}
                driver_info = session_info.get("DriverInfo") or {}
                drivers = driver_info.get("Drivers") or []
                # PlayerCarIdx is the stable identifier for the local car.
                player_car_idx = self._safe_int(self.ir["PlayerCarIdx"], -1)

                track_name = str(weekend.get("TrackDisplayName") or weekend.get("TrackName") or "").strip()
                track_config = str(weekend.get("TrackConfigName") or "").strip()
                car_path = ""
                player_car = None
                for entry in drivers:
                    if not isinstance(entry, dict):
                        continue
                    if self._safe_int(entry.get("CarIdx"), -1) == player_car_idx:
                        player_car = entry
                        break

                # Fallback for older/odd payloads where list index may match the car idx.
                if player_car is None and 0 <= player_car_idx < len(drivers):
                    candidate = drivers[player_car_idx]
                    if isinstance(candidate, dict):
                        player_car = candidate

                if isinstance(player_car, dict):
                    car_path = str(
                        player_car.get("CarScreenName")
                        or player_car.get("CarScreenNameShort")
                        or player_car.get("CarPath")
                        or ""
                    ).strip()

                track_name = prettify(track_name)
                track_config = prettify(track_config)
                car_path = prettify(car_path)

                if track_name:
                    self.last_meta["TrackName"] = track_name
                if track_config:
                    self.last_meta["TrackConfigName"] = track_config
                if car_path:
                    self.last_meta["CarPath"] = car_path

                if track_name or track_config or car_path:
                    return self.last_meta
        except Exception:
            pass

        text = ""
        try:
            text = self.ir.session_info
        except Exception:
            pass

        if not text:
            return self.last_meta

        # YAML fallback parsing.
        m_track = re.search(r"^\s*TrackDisplayName:\s*(.+?)\s*$", text, flags=re.MULTILINE)
        if m_track:
            self.last_meta["TrackName"] = m_track.group(1).strip().strip('"')
        else:
            m_track_name = re.search(r"^\s*TrackName:\s*(.+?)\s*$", text, flags=re.MULTILINE)
            if m_track_name:
                self.last_meta["TrackName"] = m_track_name.group(1).strip().strip('"')

        m_cfg = re.search(r"^\s*TrackConfigName:\s*(.+?)\s*$", text, flags=re.MULTILINE)
        if m_cfg:
            self.last_meta["TrackConfigName"] = m_cfg.group(1).strip().strip('"')

        m_car = re.search(r"^\s*CarScreenName:\s*(.+?)\s*$", text, flags=re.MULTILINE)
        if m_car:
            self.last_meta["CarPath"] = m_car.group(1).strip().strip('"')
        else:
            m_car_path = re.search(r"^\s*CarPath:\s*(.+?)\s*$", text, flags=re.MULTILINE)
            if m_car_path:
                self.last_meta["CarPath"] = m_car_path.group(1).strip().strip('"')

        self.last_meta["TrackName"] = prettify(self.last_meta.get("TrackName", ""))
        self.last_meta["TrackConfigName"] = prettify(self.last_meta.get("TrackConfigName", ""))
        self.last_meta["CarPath"] = prettify(self.last_meta.get("CarPath", ""))
        return self.last_meta

    @staticmethod
    def _normalize_wear_value(value: float) -> float:
        # Depending on sdk/version tire wear can be [0..1] fraction or [0..100] percent.
        v = max(0.0, float(value))
        return v * 100.0 if v <= 1.5 else v

    def run(self):
        tick_s = 1.0 / 60.0
        while not self.stop_event.is_set():
            try:
                if not self._connected():
                    # Push disconnected marker snapshot-less event through queue state.
                    self.out_queue.put_nowait((None, False))
                    time.sleep(1.0)
                    continue

                meta = self._parse_metadata()
                wear = {}
                for tire, fields in WEAR_FIELDS.items():
                    inner_index = INNER_WEAR_INDEX[tire]
                    inner_field = fields[inner_index]
                    wear[tire] = self._normalize_wear_value(self._safe_float(self.ir[inner_field], 100.0))

                snap = TelemetrySnapshot(
                    session_time=self._safe_float(self.ir["SessionTime"], 0.0),
                    lap=self._safe_int(self.ir["Lap"], 0),
                    lap_dist_pct=self._safe_float(self.ir["LapDistPct"], 0.0),
                    on_pit_road=bool(self.ir["OnPitRoad"]),
                    speed_mps=self._safe_float(self.ir["Speed"], 0.0),
                    lat_accel=self._safe_float(self.ir["LatAccel"], 0.0),
                    long_accel=self._safe_float(self.ir["LongAccel"], 0.0),
                    steering=self._safe_float(self.ir["SteeringWheelAngle"], 0.0),
                    track_temp=self._safe_float(self.ir["TrackTemp"], 0.0),
                    air_temp=self._safe_float(self.ir["AirTemp"], 0.0),
                    humidity=self._safe_float(self.ir["RelativeHumidity"], 0.0),
                    wear=wear,
                    track_name=meta.get("TrackName", ""),
                    track_config=meta.get("TrackConfigName", ""),
                    car_path=meta.get("CarPath", ""),
                )
                self.out_queue.put_nowait((snap, True))
                time.sleep(tick_s)
            except queue.Full:
                time.sleep(tick_s)
            except Exception:
                # Telemetry can disappear mid-session; keep looping.
                time.sleep(0.2)


class ModelWorker(threading.Thread):
    """Model thread that performs live estimation and incremental learning."""

    def __init__(self, in_queue: queue.Queue, state: dict, state_lock: threading.Lock, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.in_queue = in_queue
        self.state = state
        self.state_lock = state_lock
        self.stop_event = stop_event
        self.storage = DataStorage(MODEL_PATH)
        self.model = TireMLModel(self.storage)
        self._last_key = ""
        self.stints = StintTracker()
        self.smoothed_wear_per_lap = {t: 0.0 for t in TIRE_KEYS}

    def _update_state(self, **kwargs):
        with self.state_lock:
            self.state.update(kwargs)
            self.state["updated_at"] = time.time()

    def _consume_reset_request(self) -> bool:
        with self.state_lock:
            if not self.state.get("reset_requested", False):
                return False
            self.state["reset_requested"] = False
            return True

    def _reset_runtime_memory(self):
        self.storage.data = {}
        self.storage.save()
        self.stints = StintTracker()
        self.smoothed_wear_per_lap = {t: 0.0 for t in TIRE_KEYS}
        self._update_state(
            tread={t: 100.0 for t in TIRE_KEYS},
            wear_per_lap={t: 0.0 for t in TIRE_KEYS},
            key="",
            model_confidence=0.0,
            sample_count=0,
            estimate_ready=False,
        )

    def run(self):
        while not self.stop_event.is_set():
            try:
                payload = self.in_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            snap, connected = payload
            self._update_state(connected=bool(connected))
            if self._consume_reset_request():
                self._reset_runtime_memory()
            if snap is None:
                continue

            key = StintTracker.make_dataset_key(snap)
            if key != self._last_key:
                self.model.load_rls(key)
                self._last_key = key

            live_laps_done = 0.0
            if self.stints.last_snapshot is not None:
                live_laps_done = self.stints.laps_in_stint(snap, self.stints.last_snapshot)
            live_energy_per_lap = self.stints.current_energy / max(1.0, live_laps_done)

            rates_energy, model_confidence, sample_count = self.model.get_rates(
                key,
                snap.track_temp,
                snap.air_temp,
                snap.humidity,
                live_energy_per_lap,
            )
            baseline_wear_per_lap = self.model.get_wear_per_lap_baseline(
                key,
                snap.track_temp,
                snap.air_temp,
                snap.humidity,
            )
            self._update_state(
                key=key,
                track_temp=snap.track_temp,
                air_temp=snap.air_temp,
                humidity=snap.humidity,
                model_confidence=model_confidence,
                sample_count=sample_count,
                track_name=snap.track_name,
                track_config=snap.track_config,
                car_path=snap.car_path,
            )

            has_base_samples = sample_count >= 1
            self._update_state(estimate_ready=has_base_samples)

            stint_end = self.stints.update(snap)
            live = self.stints.build_live_estimate(snap, rates_energy, baseline_wear_per_lap) if has_base_samples else None
            if live:
                laps_done = max(1e-6, live.get("laps_progress", float(live["laps_done"])))
                energy_per_lap_live = live["energy_used"] / laps_done
                current_wpl = {t: rates_energy[t] * energy_per_lap_live for t in TIRE_KEYS}
                for t in TIRE_KEYS:
                    current_wpl[t] = max(current_wpl[t], baseline_wear_per_lap.get(t, 0.0))

                # Exponential smoothing for stable wear rate estimate.
                for t in TIRE_KEYS:
                    self.smoothed_wear_per_lap[t] = 0.8 * self.smoothed_wear_per_lap[t] + 0.2 * current_wpl[t]

                self._update_state(
                    tread=dict(live["estimated_tread"]),
                    wear_per_lap=dict(self.smoothed_wear_per_lap),
                )
            elif not has_base_samples:
                self._update_state(wear_per_lap={t: 0.0 for t in TIRE_KEYS})

            if not stint_end:
                continue

            if not self.stints.stint_is_valid(stint_end):
                continue

            sample = {
                "track_temp": float(stint_end["track_temp"]),
                "air_temp": float(stint_end["air_temp"]),
                "laps": int(stint_end["laps"]),
                "energy_per_lap": float(stint_end["energy_per_lap"]),
                "lf": float(stint_end["wear_per_energy"]["lf"]),
                "rf": float(stint_end["wear_per_energy"]["rf"]),
                "lr": float(stint_end["wear_per_energy"]["lr"]),
                "rr": float(stint_end["wear_per_energy"]["rr"]),
            }

            sample["humidity"] = float(stint_end.get("humidity", 50.0))

            if self.model.is_outlier(str(stint_end["key"]), sample):
                continue

            self.model.add_stint_sample(str(stint_end["key"]), sample)
            self._update_state(
                sample_count=self.model.sample_count(key),
                model_confidence=self.model._rls["lf"].confidence,
            )


class InfoDialog(QtWidgets.QDialog):
    """Information panel showing model/session status and learned data details."""

    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setWindowTitle("Tire Overlay - Information")
        self.resize(520, 300)

        self.text = QtWidgets.QPlainTextEdit(self)
        self.text.setReadOnly(True)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)

        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.text)
        lay.addWidget(btn_close, alignment=QtCore.Qt.AlignRight)

    def set_info(self, message: str):
        self.text.setPlainText(message)


class SettingsDialog(QtWidgets.QDialog):
    """Simple settings menu for overlay behavior and appearance."""

    def __init__(self, parent: "OverlayUI"):
        super().__init__(parent)
        self.parent_overlay = parent
        self.setWindowTitle("Tire Overlay - Settings")

        self.cb_on_top = QtWidgets.QCheckBox("Always on top")
        self.cb_on_top.setChecked(parent.settings["always_on_top"])

        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_slider.setRange(80, 255)
        self.alpha_slider.setValue(parent.settings["bg_alpha"])

        self.font_spin = QtWidgets.QSpinBox()
        self.font_spin.setRange(10, 36)
        self.font_spin.setValue(parent.settings["font_size"])

        self.w_spin = QtWidgets.QSpinBox()
        self.w_spin.setRange(240, 900)
        self.w_spin.setValue(parent.settings["width"])
        self.h_spin = QtWidgets.QSpinBox()
        self.h_spin.setRange(120, 600)
        self.h_spin.setValue(parent.settings["height"])

        btn_reset_data = QtWidgets.QPushButton("Reset data (clear memory)")
        btn_reset_data.clicked.connect(parent.reset_all_data)

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.clicked.connect(self.apply)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)

        form = QtWidgets.QFormLayout()
        form.addRow(self.cb_on_top)
        form.addRow("Background opacity", self.alpha_slider)
        form.addRow("Font size", self.font_spin)

        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("W"))
        size_row.addWidget(self.w_spin)
        size_row.addSpacing(12)
        size_row.addWidget(QtWidgets.QLabel("H"))
        size_row.addWidget(self.h_spin)
        form.addRow("Overlay size", size_row)

        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(btn_reset_data)
        bottom.addStretch(1)
        bottom.addWidget(btn_apply)
        bottom.addWidget(btn_close)

        root = QtWidgets.QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(bottom)

        # Force light controls for readability regardless of OS theme.
        self.setStyleSheet(
            """
            QDialog {
                background-color: white;
                color: black;
            }
            QLabel, QCheckBox {
                color: black;
                background: transparent;
            }
            QSlider, QSpinBox, QPushButton {
                background-color: white;
                color: black;
            }
            """
        )

    def apply(self):
        self.parent_overlay.settings["always_on_top"] = self.cb_on_top.isChecked()
        self.parent_overlay.settings["bg_alpha"] = int(self.alpha_slider.value())
        self.parent_overlay.settings["font_size"] = int(self.font_spin.value())
        self.parent_overlay.settings["width"] = int(self.w_spin.value())
        self.parent_overlay.settings["height"] = int(self.h_spin.value())
        self.parent_overlay.apply_settings()
        self.parent_overlay.save_settings()


class OverlayUI(QtWidgets.QWidget):
    """Transparent HUD with mini menu (settings/info), updated at 10 Hz."""

    def __init__(self, state: dict, state_lock: threading.Lock):
        super().__init__()
        self.state = state
        self.state_lock = state_lock
        self.drag_origin: Optional[QtCore.QPoint] = None
        self.settings = self.load_settings()
        self.model_ref = TireMLModel(DataStorage(MODEL_PATH))

        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)

        self.btn_info = QtWidgets.QPushButton("ℹ", self)
        self.btn_info.setFixedWidth(28)
        self.btn_info.clicked.connect(self.open_info)

        self.btn_settings = QtWidgets.QPushButton("⚙", self)
        self.btn_settings.setFixedWidth(28)
        self.btn_settings.clicked.connect(self.open_settings)

        self.btn_close_overlay = QtWidgets.QPushButton("✕", self)
        self.btn_close_overlay.setFixedWidth(28)
        self.btn_close_overlay.clicked.connect(QtWidgets.QApplication.quit)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addStretch(1)
        top_row.addWidget(self.btn_info)
        top_row.addWidget(self.btn_settings)
        top_row.addWidget(self.btn_close_overlay)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top_row)
        layout.addWidget(self.label)
        layout.setContentsMargins(6, 6, 6, 6)
        self.setLayout(layout)

        self.info_dialog = InfoDialog(self)
        self.settings_dialog = SettingsDialog(self)

        self.apply_settings()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)  # 10 Hz

    @staticmethod
    def _color_for_value(v: float) -> str:
        if v > 80.0:
            return "#7CFC00"
        if v >= 65.0:
            return "#FFD700"
        return "#FF4C4C"

    @staticmethod
    def _default_settings() -> dict:
        return {
            "always_on_top": True,
            "bg_alpha": 160,
            "font_size": 18,
            "width": 340,
            "height": 220,
            "x": 120,
            "y": 120,
        }

    def load_settings(self) -> dict:
        cfg = self._default_settings()
        if os.path.exists(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    cfg.update(loaded)
            except Exception:
                pass
        return cfg

    def save_settings(self):
        try:
            with open(SETTINGS_PATH + ".tmp", "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2)
            os.replace(SETTINGS_PATH + ".tmp", SETTINGS_PATH)
        except Exception:
            pass

    def apply_settings(self):
        flags = QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool
        if self.settings["always_on_top"]:
            flags |= QtCore.Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent;")

        font_size = int(self.settings["font_size"])
        alpha = int(self.settings["bg_alpha"])
        self.label.setStyleSheet(
            f"""
            QLabel {{
                color: white;
                font-family: Consolas, monospace;
                font-size: {font_size}px;
                font-weight: bold;
                background-color: rgba(0, 0, 0, {alpha});
                border-radius: 10px;
                padding: 10px;
            }}
            """
        )
        self.resize(int(self.settings["width"]), int(self.settings["height"]))
        self.move(int(self.settings["x"]), int(self.settings["y"]))
        self.show()

    def open_info(self):
        with self.state_lock:
            key = self.state.get("key", "unknown")
            connected = self.state.get("connected", False)
            temp = float(self.state.get("track_temp", 0.0))
            air = float(self.state.get("air_temp", 0.0))
            humidity = float(self.state.get("humidity", 0.0))
            model_conf = float(self.state.get("model_confidence", 0.0))
            samples = int(self.state.get("sample_count", 0))

        msg = (
            f"Connection: {'Connected' if connected else 'Disconnected'}\n"
            f"Dataset key: {key}\n"
            f"Track temp: {temp:.1f} °C\n"
            f"Air temp: {air:.1f} °C\n"
            f"Humidity: {humidity:.1f} %\n"
            f"Samples: {samples}\n"
            f"Model confidence: {model_conf:.1%}\n\n"
        )
        self.model_ref.load_rls(str(key))
        msg += self.model_ref.get_coefficients_report(str(key))
        self.info_dialog.set_info(msg)
        self.info_dialog.show()
        self.info_dialog.raise_()

    def open_settings(self):
        self.settings_dialog.show()
        self.settings_dialog.raise_()

    def refresh(self):
        with self.state_lock:
            tread = dict(self.state.get("tread", {t: 100.0 for t in TIRE_KEYS}))
            connected = bool(self.state.get("connected", False))
            track_name = str(self.state.get("track_name", "") or "-")
            track_config = str(self.state.get("track_config", "") or "-")
            car_path = str(self.state.get("car_path", "") or "-")
            estimate_ready = bool(self.state.get("estimate_ready", False))

        lines = []
        if estimate_ready:
            lines.extend(
                [
                    f"<span style='color:{self._color_for_value(tread.get('lf', 100.0))}'>LF {tread.get('lf', 100.0):5.1f}%</span>",
                    f"<span style='color:{self._color_for_value(tread.get('rf', 100.0))}'>RF {tread.get('rf', 100.0):5.1f}%</span>",
                    f"<span style='color:{self._color_for_value(tread.get('lr', 100.0))}'>LR {tread.get('lr', 100.0):5.1f}%</span>",
                    f"<span style='color:{self._color_for_value(tread.get('rr', 100.0))}'>RR {tread.get('rr', 100.0):5.1f}%</span>",
                ]
            )

        lines.extend(
            [
                f"<span style='color:#B8E0FF'>Track: {track_name} ({track_config})</span>",
                f"<span style='color:#B8E0FF'>Car: {car_path}</span>",
                f"<span style='color:#FFD166'>Model confidence: {self.state.get('model_confidence', 0.0):.0%}</span>",
                f"<span style='color:{'#7CFC00' if connected else '#FF4C4C'}'>SDK: {'ONLINE' if connected else 'OFFLINE'}</span>",
            ]
        )
        self.label.setText("<br>".join(lines))

    def reset_all_data(self):
        confirm_box = self._build_light_message_box(
            icon=QtWidgets.QMessageBox.Warning,
            title="Reset data",
            text="This will clear learned tire model data and current session memory. Continue?",
            buttons=QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            default_button=QtWidgets.QMessageBox.No,
        )
        if confirm_box.exec_() != QtWidgets.QMessageBox.Yes:
            return

        # Clear persisted model samples.
        for path in (MODEL_PATH, MODEL_PATH + ".tmp"):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

        # Reset runtime overlay state.
        with self.state_lock:
            self.state.update(
                tread={t: 100.0 for t in TIRE_KEYS},
                wear_per_lap={t: 0.0 for t in TIRE_KEYS},
                key="",
                model_confidence=0.0,
                sample_count=0,
                estimate_ready=False,
                reset_requested=True,
            )

        done_box = self._build_light_message_box(
            icon=QtWidgets.QMessageBox.Information,
            title="Reset complete",
            text="All learned data was cleared.",
            buttons=QtWidgets.QMessageBox.Ok,
            default_button=QtWidgets.QMessageBox.Ok,
        )
        done_box.exec_()

    def _build_light_message_box(
        self,
        icon: QtWidgets.QMessageBox.Icon,
        title: str,
        text: str,
        buttons: QtWidgets.QMessageBox.StandardButtons,
        default_button: QtWidgets.QMessageBox.StandardButton,
    ) -> QtWidgets.QMessageBox:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(buttons)
        box.setDefaultButton(default_button)
        box.setStyleSheet(
            """
            QMessageBox {
                background-color: white;
                color: black;
            }
            QLabel {
                color: black;
                background: transparent;
            }
            QPushButton {
                background-color: white;
                color: black;
                border: 1px solid #BDBDBD;
                padding: 4px 10px;
                min-width: 72px;
            }
            QPushButton:hover {
                background-color: #F2F2F2;
            }
            """
        )
        return box

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton:
            self.drag_origin = e.globalPos() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.drag_origin is not None and (e.buttons() & QtCore.Qt.LeftButton):
            self.move(e.globalPos() - self.drag_origin)
            e.accept()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if e.button() == QtCore.Qt.LeftButton:
            self.drag_origin = None
            self.settings["x"] = int(self.x())
            self.settings["y"] = int(self.y())
            self.save_settings()
            e.accept()


class MainApp:
    """Application coordinator: starts telemetry/model threads and Qt UI."""

    def __init__(self):
        self.stop_event = threading.Event()
        self.telemetry_queue: queue.Queue = queue.Queue(maxsize=1200)
        self.state_lock = threading.Lock()
        self.state = {
            "tread": {t: 100.0 for t in TIRE_KEYS},
            "wear_per_lap": {t: 0.0 for t in TIRE_KEYS},
            "estimate_ready": False,
            "connected": False,
            "key": "",
            "model_confidence": 0.0,
            "sample_count": 0,
            "track_temp": 0.0,
            "air_temp": 0.0,
            "humidity": 0.0,
            "track_name": "",
            "track_config": "",
            "car_path": "",
            "reset_requested": False,
            "updated_at": time.time(),
        }

        self.telemetry_thread = TelemetryReader(self.telemetry_queue, self.stop_event)
        self.model_thread = ModelWorker(self.telemetry_queue, self.state, self.state_lock, self.stop_event)

    def start(self) -> int:
        self.telemetry_thread.start()
        self.model_thread.start()

        app = QtWidgets.QApplication([])
        overlay = OverlayUI(self.state, self.state_lock)
        overlay.show()

        def handle_signal(*_):
            self.shutdown()
            app.quit()

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        code = app.exec_()
        self.shutdown()
        return int(code)

    def shutdown(self):
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        for t in (self.telemetry_thread, self.model_thread):
            if t.is_alive():
                t.join(timeout=2.0)


def main():
    app = MainApp()
    raise SystemExit(app.start())


if __name__ == "__main__":
    main()
