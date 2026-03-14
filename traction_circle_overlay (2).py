#!/usr/bin/env python3
"""Standalone Tkinter traction-circle overlay for iRacing telemetry.

Features
- Real-time traction circle from LongAccel and LatAccel.
- Learns reference grip usage per LapDistPct bin from valid laps only.
- Detects robust underuse opportunities with session coaching insights.
"""

from __future__ import annotations

import math
import os
import statistics
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from tkinter import filedialog, ttk
from typing import Any, Deque, List, Optional, Sequence, Tuple

import irsdk

G_CONSTANT = 9.80665
UPDATE_MS = 60
BINS_PER_LAP = 200
MIN_REFERENCE_G = 0.75
UNDERUSE_MARGIN = 0.12
MIN_SEGMENT_BINS = 3
MIN_SAMPLES_PER_BIN = 5
MAX_LAP_HISTORY = 1000
RECENT_VALID_LAPS = 5
TOGGLE_MODE_KEY = "m"
MIN_LAPS_FOR_FEEDBACK = 5


@dataclass
class LapData:
    lap_number: int
    lap_time: float
    valid: bool
    bins: List[float]
    long_bins: List[float]
    lat_bins: List[float]


@dataclass
class UnderuseSegment:
    start_percent: float
    end_percent: float
    peak_percent: float
    reference_g: float
    achieved_g: float
    delta_g: float
    severity: str
    phase: str
    recommendation: str
    trend: str
    consistency: float
    confidence: bool


class TractionCircleOverlay:
    def __init__(self) -> None:
        self.ir = irsdk.IRSDK()

        self.root = tk.Tk()
        self.root.title("Traction Circle Limiter")
        self.root.geometry("900x540")
        self.root.configure(bg="#101418")
        self.root.attributes("-topmost", True)

        self.status_var = tk.StringVar(value="Connecting to iRacing...")
        self.header_var = tk.StringVar(value="Car: -- | Track: -- | Session: -- | Valid laps: 0")
        self.current_var = tk.StringVar(value="Current: --")
        self.limit_var = tk.StringVar(value="Estimated limit: --")
        self.quality_var = tk.StringVar(value="Valid laps used: 0 | Invalid laps discarded: 0 | Outliers removed: 0")
        self.summary_var = tk.StringVar(value="Collecting lap data...")
        self.mode_var = tk.StringVar(value="Mode: compact")
        self.reference_var = tk.StringVar(value="Reference: live adaptive")

        self.current_lap_num: Optional[int] = None
        self.current_lap_bins: List[float] = [0.0] * BINS_PER_LAP
        self.current_lap_long_bins: List[float] = [0.0] * BINS_PER_LAP
        self.current_lap_lat_bins: List[float] = [0.0] * BINS_PER_LAP
        self.current_lap_valid = True

        self.lap_history: Deque[LapData] = deque(maxlen=MAX_LAP_HISTORY)
        self.invalid_laps_count = 0
        self.outliers_removed_last = 0
        self.bin_confidence: List[bool] = [False] * BINS_PER_LAP

        self.context_key = ""
        self.current_track = "--"
        self.current_car = "--"
        self.current_session = "--"

        self.recent_peaks: Deque[float] = deque(maxlen=300)
        self.estimated_limit_g = 1.8

        self.compact_mode = True
        self.external_reference_bins: Optional[List[float]] = None
        self.external_reference_path: Optional[str] = None

        self._build_ui()
        self.root.bind(f"<{TOGGLE_MODE_KEY}>", self._toggle_mode)
        self.root.bind(f"<{TOGGLE_MODE_KEY.upper()}>", self._toggle_mode)

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill="both", expand=True)

        controls = ttk.Frame(container)
        controls.pack(fill="x", pady=(0, 8))

        ttk.Label(controls, textvariable=self.status_var, foreground="#4b5563").pack(side="left")
        ttk.Label(controls, text=f"  Toggle: '{TOGGLE_MODE_KEY.upper()}'", foreground="#4b5563").pack(side="left")
        ttk.Button(controls, text="Load IBT", command=self._load_ibt_reference).pack(side="right")
        ttk.Button(controls, text="Use Live", command=self._clear_ibt_reference).pack(side="right", padx=(0, 6))

        ttk.Label(container, textvariable=self.header_var, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))

        body = ttk.Frame(container)
        body.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            body,
            width=400,
            height=400,
            bg="#0b0f13",
            highlightthickness=1,
            highlightbackground="#2b3540",
        )
        self.canvas.pack(side="left", padx=(0, 10), fill="both", expand=False)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(right, text="Real-time grip usage", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(right, textvariable=self.current_var, font=("Consolas", 10)).pack(anchor="w", pady=(8, 0))
        ttk.Label(right, textvariable=self.limit_var, font=("Consolas", 10)).pack(anchor="w")
        ttk.Label(right, textvariable=self.quality_var, font=("Consolas", 9), foreground="#4b5563").pack(anchor="w", pady=(2, 0))

        ttk.Separator(right).pack(fill="x", pady=10)

        ttk.Label(right, textvariable=self.mode_var, font=("Segoe UI", 10, "bold")).pack(anchor="w")
        ttk.Label(right, textvariable=self.reference_var, font=("Segoe UI", 9), foreground="#4b5563").pack(anchor="w")
        self.summary_label = ttk.Label(
            right,
            textvariable=self.summary_var,
            justify="left",
            font=("Consolas", 10),
            wraplength=420,
        )
        self.summary_label.pack(anchor="w", pady=(6, 0))

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        if not math.isfinite(parsed):
            return default
        return parsed

    def _read_var(self, name: str, default: object = None) -> object:
        try:
            return self.ir[name]
        except Exception:
            return default

    @staticmethod
    def _bin_index(lap_dist_pct: float) -> int:
        normalized = max(0.0, min(0.999999, lap_dist_pct))
        return int(normalized * BINS_PER_LAP)

    @staticmethod
    def _get_nested(data: object, *path: object) -> object:
        cur = data
        for item in path:
            if isinstance(item, int):
                if not isinstance(cur, list) or item < 0 or item >= len(cur):
                    return None
                cur = cur[item]
                continue
            if not isinstance(cur, dict):
                return None
            cur = cur.get(item)
            if cur is None:
                return None
        return cur

    def _detect_context(self) -> Tuple[str, str, str, str]:
        weekend = self._read_var("WeekendInfo", {})
        driver_info = self._read_var("DriverInfo", {})
        session_info = self._read_var("SessionInfo", {})
        driver_car_idx = int(self._safe_float(self._read_var("DriverCarIdx", 0)))

        track_id = self._get_nested(weekend, "TrackID")
        track_name = self._get_nested(weekend, "TrackName")
        track_cfg = self._get_nested(weekend, "TrackConfigName")
        track_value = track_id if track_id not in (None, "") else (track_name or track_cfg or "unknown")
        track_display = track_name or track_cfg or str(track_value)

        car_id = self._get_nested(driver_info, "Drivers", driver_car_idx, "CarID")
        car_path = self._get_nested(driver_info, "Drivers", driver_car_idx, "CarPath")
        car_screen = self._get_nested(driver_info, "Drivers", driver_car_idx, "CarScreenName")
        car_value = car_id if car_id not in (None, "") else (car_path or "unknown")
        car_display = car_screen or car_path or str(car_value)

        session_num = int(self._safe_float(self._read_var("SessionNum", 0)))
        session_name = self._get_nested(session_info, "Sessions", session_num, "SessionName")
        session_display = str(session_name or "Unknown")

        context_key = f"track:{track_value}|car:{car_value}"
        return context_key, str(track_display), str(car_display), session_display

    def _reset_for_new_context(self) -> None:
        self.current_lap_num = None
        self.current_lap_bins = [0.0] * BINS_PER_LAP
        self.current_lap_long_bins = [0.0] * BINS_PER_LAP
        self.current_lap_lat_bins = [0.0] * BINS_PER_LAP
        self.current_lap_valid = True
        self.lap_history.clear()
        self.invalid_laps_count = 0
        self.outliers_removed_last = 0
        self.bin_confidence = [False] * BINS_PER_LAP

    def _is_offtrack(self, driver_car_idx: int) -> bool:
        surfaces = self._read_var("CarIdxTrackSurface", [])
        if not isinstance(surfaces, list) or driver_car_idx < 0 or driver_car_idx >= len(surfaces):
            return False
        val = surfaces[driver_car_idx]
        if isinstance(val, str):
            return "offtrack" in val.lower()
        try:
            numeric = int(val)
        except (TypeError, ValueError):
            return False
        return numeric == 1

    def _finalize_current_lap(self, next_lap_num: int) -> None:
        if self.current_lap_num is None:
            self.current_lap_num = next_lap_num
            return
        if next_lap_num == self.current_lap_num:
            return

        lap_time = self._safe_float(self._read_var("LapLastLapTime", 0.0), default=0.0)
        lap_data = LapData(
            lap_number=self.current_lap_num,
            lap_time=lap_time,
            valid=self.current_lap_valid,
            bins=self.current_lap_bins.copy(),
            long_bins=self.current_lap_long_bins.copy(),
            lat_bins=self.current_lap_lat_bins.copy(),
        )
        self.lap_history.append(lap_data)
        if not lap_data.valid:
            self.invalid_laps_count += 1

        self.current_lap_bins = [0.0] * BINS_PER_LAP
        self.current_lap_long_bins = [0.0] * BINS_PER_LAP
        self.current_lap_lat_bins = [0.0] * BINS_PER_LAP
        self.current_lap_valid = True
        self.current_lap_num = next_lap_num

    def _update_lap_storage(
        self,
        lap_num: int,
        lap_dist_pct: float,
        g_total: float,
        long_g: float,
        lat_g: float,
        offtrack: bool,
    ) -> None:
        self._finalize_current_lap(lap_num)
        if offtrack:
            self.current_lap_valid = False

        idx = self._bin_index(lap_dist_pct)
        if g_total > self.current_lap_bins[idx]:
            self.current_lap_bins[idx] = g_total
            self.current_lap_long_bins[idx] = long_g
            self.current_lap_lat_bins[idx] = lat_g

    @staticmethod
    def _iqr_filter(values: Sequence[float]) -> Tuple[List[float], int]:
        if len(values) < 4:
            return list(values), 0
        sorted_vals = sorted(values)
        q1, _, q3 = statistics.quantiles(sorted_vals, n=4, method="inclusive")
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        filtered = [v for v in sorted_vals if lower <= v <= upper]
        return filtered, len(sorted_vals) - len(filtered)

    def _compute_reference_by_bin(self, valid_laps: Sequence[LapData]) -> List[float]:
        reference: List[float] = [0.0] * BINS_PER_LAP
        self.outliers_removed_last = 0
        self.bin_confidence = [False] * BINS_PER_LAP
        if not valid_laps:
            return reference

        for i in range(BINS_PER_LAP):
            values = [lap.bins[i] for lap in valid_laps if lap.bins[i] > 0.05]
            if not values:
                continue
            filtered, removed = self._iqr_filter(values)
            self.outliers_removed_last += removed
            if len(filtered) < MIN_SAMPLES_PER_BIN:
                continue
            filtered.sort()
            idx = min(len(filtered) - 1, int(round((len(filtered) - 1) * 0.9)))
            reference[i] = filtered[idx]
            self.bin_confidence[i] = True
        return reference

    @staticmethod
    def _read_ibt_series(ibt: Any, name: str) -> List[float]:
        for method_name in ("get_all", "get"):
            method = getattr(ibt, method_name, None)
            if callable(method):
                try:
                    values = method(name)
                except Exception:
                    continue
                if values is not None:
                    return [TractionCircleOverlay._safe_float(v, default=0.0) for v in list(values)]

        try:
            values = ibt[name]  # type: ignore[index]
            return [TractionCircleOverlay._safe_float(v, default=0.0) for v in list(values)]
        except Exception:
            return []

    def _reference_from_ibt(self, file_path: str) -> Optional[List[float]]:
        ibt_reader = getattr(irsdk, "IBT", None)
        if ibt_reader is None:
            self.status_var.set("Your irsdk build does not expose IBT reader support.")
            return None

        try:
            ibt = ibt_reader()
            opened = ibt.open(file_path)
        except Exception as exc:
            self.status_var.set(f"Failed to open IBT: {exc}")
            return None

        if opened is False:
            self.status_var.set("Failed to open IBT file.")
            return None

        lap_dist = self._read_ibt_series(ibt, "LapDistPct")
        long_accel = self._read_ibt_series(ibt, "LongAccel")
        lat_accel = self._read_ibt_series(ibt, "LatAccel")
        if not lap_dist or not long_accel or not lat_accel:
            self.status_var.set("IBT is missing LapDistPct / LongAccel / LatAccel.")
            return None

        sample_count = min(len(lap_dist), len(long_accel), len(lat_accel))
        bins = [0.0] * BINS_PER_LAP
        for i in range(sample_count):
            idx = self._bin_index(self._safe_float(lap_dist[i], default=0.0))
            long_g = self._safe_float(long_accel[i], default=0.0) / G_CONSTANT
            lat_g = self._safe_float(lat_accel[i], default=0.0) / G_CONSTANT
            total_g = math.hypot(long_g, lat_g)
            if total_g > bins[idx]:
                bins[idx] = total_g

        if max(bins, default=0.0) < MIN_REFERENCE_G:
            self.status_var.set("IBT loaded, but no meaningful grip reference could be built.")
            return None
        return bins

    def _load_ibt_reference(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Load IBT reference",
            filetypes=[("iRacing telemetry", "*.ibt"), ("All files", "*.*")],
        )
        if not file_path:
            return

        bins = self._reference_from_ibt(file_path)
        if bins is None:
            return

        self.external_reference_bins = bins
        self.external_reference_path = file_path
        self.reference_var.set(f"Reference: IBT {os.path.basename(file_path)}")
        self.status_var.set("IBT reference loaded. Feedback starts after 5 valid laps.")

    def _clear_ibt_reference(self) -> None:
        self.external_reference_bins = None
        self.external_reference_path = None
        self.reference_var.set("Reference: live adaptive")
        self.status_var.set("Using live adaptive reference from your valid laps.")

    @staticmethod
    def _phase_and_recommendation(neg_long: float, lat: float, pos_long: float) -> Tuple[str, str]:
        if neg_long > max(lat, pos_long):
            return "Entrada", "Frear 5-10m mais tarde ou reduzir pressão inicial de freio."
        if lat >= max(neg_long, pos_long):
            return "Meio", "Carregar mais velocidade no apex mantendo estabilidade."
        return "Saída", "Aplicar throttle progressivamente mais cedo na saída."

    @staticmethod
    def _severity_label(delta_g: float) -> str:
        if delta_g >= 0.25:
            return "alta"
        if delta_g >= 0.12:
            return "média"
        return "baixa"

    @staticmethod
    def _trend_label(values: Sequence[float]) -> str:
        if len(values) < 3:
            return "estável"
        split = max(1, len(values) // 2)
        first = statistics.mean(values[:split])
        second = statistics.mean(values[split:])
        if second - first > 0.03:
            return "melhora"
        if second - first < -0.03:
            return "piora"
        return "estável"

    @staticmethod
    def _lapdist_hint(start_percent: float, end_percent: float, peak_percent: float) -> str:
        def phase_label(value: float) -> str:
            if value < 1.0 / 3.0:
                return "início"
            if value < 2.0 / 3.0:
                return "meio"
            return "fim"

        start_pct = start_percent * 100.0
        end_pct = end_percent * 100.0
        peak_pct = peak_percent * 100.0
        return (
            f"LapDistPct {start_pct:.1f}%→{end_pct:.1f}% "
            f"(pico em {peak_pct:.1f}% / {phase_label(peak_percent)} da volta)"
        )

    def _detect_underuse_segments(self, valid_laps: Sequence[LapData], reference: Sequence[float]) -> List[UnderuseSegment]:
        if not valid_laps:
            return []

        recent = list(valid_laps[-RECENT_VALID_LAPS:])
        achieved: List[float] = [0.0] * BINS_PER_LAP
        for i in range(BINS_PER_LAP):
            values = [lap.bins[i] for lap in recent if lap.bins[i] > 0.05]
            if values:
                achieved[i] = statistics.median(values)

        segments: List[Tuple[int, int]] = []
        start = -1
        for i in range(BINS_PER_LAP):
            ref = reference[i]
            ach = achieved[i]
            confident = self.bin_confidence[i]
            is_under = confident and ref >= MIN_REFERENCE_G and ach < ref * (1.0 - UNDERUSE_MARGIN)
            if is_under and start < 0:
                start = i
            if (not is_under or i == BINS_PER_LAP - 1) and start >= 0:
                end = i if is_under and i == BINS_PER_LAP - 1 else i - 1
                if end - start + 1 >= MIN_SEGMENT_BINS:
                    segments.append((start, end))
                start = -1

        results: List[UnderuseSegment] = []
        for seg_start, seg_end in segments:
            best_idx = seg_start
            best_gap = -1.0
            for i in range(seg_start, seg_end + 1):
                gap = reference[i] - achieved[i]
                if gap > best_gap:
                    best_gap = gap
                    best_idx = i

            neg_long = statistics.mean([max(0.0, -lap.long_bins[best_idx]) for lap in recent])
            lat_mag = statistics.mean([abs(lap.lat_bins[best_idx]) for lap in recent])
            pos_long = statistics.mean([max(0.0, lap.long_bins[best_idx]) for lap in recent])
            phase, rec = self._phase_and_recommendation(neg_long, lat_mag, pos_long)

            peak_ref = reference[best_idx]
            peak_ach = achieved[best_idx]
            delta = max(0.0, peak_ref - peak_ach)

            under_laps = 0
            used_laps = 0
            for lap in valid_laps:
                val = lap.bins[best_idx]
                if val <= 0.05:
                    continue
                used_laps += 1
                if val < peak_ref * (1.0 - UNDERUSE_MARGIN / 2.0):
                    under_laps += 1
            consistency = (under_laps / used_laps * 100.0) if used_laps else 0.0

            trend_values = [lap.bins[best_idx] for lap in recent if lap.bins[best_idx] > 0.05]
            trend = self._trend_label(trend_values)

            results.append(
                UnderuseSegment(
                    start_percent=seg_start / BINS_PER_LAP,
                    end_percent=(seg_end + 1) / BINS_PER_LAP,
                    peak_percent=(best_idx + 0.5) / BINS_PER_LAP,
                    reference_g=peak_ref,
                    achieved_g=peak_ach,
                    delta_g=delta,
                    severity=self._severity_label(delta),
                    phase=phase,
                    recommendation=rec,
                    trend=trend,
                    consistency=consistency,
                    confidence=self.bin_confidence[best_idx],
                )
            )

        results.sort(key=lambda s: s.delta_g, reverse=True)
        return results

    @staticmethod
    def _format_summary(segments: Sequence[UnderuseSegment], compact_mode: bool) -> str:
        if not segments:
            return "Sem oportunidades robustas ainda. Complete mais voltas válidas."

        top3 = list(segments[:3])
        lines = ["Top 3 oportunidades (ordenadas por Δg):"]
        for seg in top3:
            lines.append(
                f"• {seg.start_percent*100:.1f}-{seg.end_percent*100:.1f}% | Δ{seg.delta_g:.2f}g | "
                f"{seg.phase} | tendência: {seg.trend} | consistência: {seg.consistency:.0f}%"
            )
            lines.append(f"  ↳ {TractionCircleOverlay._lapdist_hint(seg.start_percent, seg.end_percent, seg.peak_percent)}")

        if compact_mode:
            lines.append("\nCompacto: exibindo apenas principais oportunidades.")
            return "\n".join(lines)

        lines.append("\nDetalhado:")
        for seg in segments:
            lines.append(
                f"- {seg.start_percent*100:.1f}-{seg.end_percent*100:.1f}% "
                f"(pico {seg.peak_percent*100:.1f}%): ref {seg.reference_g:.2f}g / atual {seg.achieved_g:.2f}g / "
                f"Δ{seg.delta_g:.2f}g / severidade {seg.severity} / confiança {'alta' if seg.confidence else 'baixa'}"
            )
            lines.append(f"  {TractionCircleOverlay._lapdist_hint(seg.start_percent, seg.end_percent, seg.peak_percent)}")
            lines.append(f"  Fase {seg.phase}: {seg.recommendation}")
        return "\n".join(lines)

    def _draw_circle(self, long_g: float, lat_g: float) -> None:
        self.canvas.delete("all")
        w = int(self.canvas.winfo_width())
        h = int(self.canvas.winfo_height())
        cx, cy = w // 2, h // 2
        radius = min(w, h) * 0.42

        self.canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline="#374151", width=2)
        self.canvas.create_line(cx - radius, cy, cx + radius, cy, fill="#25303b")
        self.canvas.create_line(cx, cy - radius, cx, cy + radius, fill="#25303b")

        for frac in (0.25, 0.5, 0.75):
            rr = radius * frac
            self.canvas.create_oval(cx - rr, cy - rr, cx + rr, cy + rr, outline="#1f2937")

        limit = max(0.8, self.estimated_limit_g)
        scale = radius / limit

        dot_x = cx + lat_g * scale
        dot_y = cy - long_g * scale
        dot_r = 6
        self.canvas.create_oval(dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r, fill="#22d3ee", outline="")

        self.canvas.create_text(cx, 16, text="LongAccel (+brake / -throttle)", fill="#9ca3af", font=("Segoe UI", 9))
        self.canvas.create_text(16, cy, text="Lat", fill="#9ca3af", font=("Segoe UI", 9), angle=90)

    def _toggle_mode(self, _event: object = None) -> None:
        self.compact_mode = not self.compact_mode
        self.mode_var.set(f"Mode: {'compact' if self.compact_mode else 'detailed'}")

    def _update(self) -> None:
        connected = self.ir.startup() if not getattr(self.ir, "is_initialized", False) else True
        if not connected:
            self.status_var.set("Not connected. Open iRacing and click Drive.")
            self.current_var.set("Current: --")
            self.limit_var.set("Estimated limit: --")
            self.summary_var.set("Waiting for telemetry...")
            self._draw_circle(0.0, 0.0)
            self.root.after(400, self._update)
            return

        self.status_var.set("Connected. 60 Hz telemetry / 60 ms UI refresh.")

        context_key, track_name, car_name, session_name = self._detect_context()
        if self.context_key and context_key != self.context_key:
            self._reset_for_new_context()
        self.context_key = context_key
        self.current_track = track_name
        self.current_car = car_name
        self.current_session = session_name

        long_accel = self._safe_float(self._read_var("LongAccel", 0.0))
        lat_accel = self._safe_float(self._read_var("LatAccel", 0.0))
        lap_num = int(self._safe_float(self._read_var("Lap", 0.0)))
        lap_dist_pct = self._safe_float(self._read_var("LapDistPct", 0.0))
        driver_idx = int(self._safe_float(self._read_var("DriverCarIdx", 0)))

        long_g = long_accel / G_CONSTANT
        lat_g = lat_accel / G_CONSTANT
        g_total = math.hypot(long_g, lat_g)

        self.recent_peaks.append(g_total)
        if len(self.recent_peaks) > 20:
            sorted_vals = sorted(self.recent_peaks)
            self.estimated_limit_g = sorted_vals[int(0.95 * (len(sorted_vals) - 1))]

        offtrack = self._is_offtrack(driver_idx)
        self._update_lap_storage(lap_num, lap_dist_pct, g_total, long_g, lat_g, offtrack)

        valid_laps = [lap for lap in self.lap_history if lap.valid]
        if self.external_reference_bins is not None:
            reference = self.external_reference_bins
            self.outliers_removed_last = 0
            self.bin_confidence = [v >= MIN_REFERENCE_G for v in reference]
        else:
            reference = self._compute_reference_by_bin(valid_laps)

        if len(valid_laps) < MIN_LAPS_FOR_FEEDBACK:
            laps_left = MIN_LAPS_FOR_FEEDBACK - len(valid_laps)
            if self.external_reference_bins is not None:
                segments: List[UnderuseSegment] = []
                self.summary_var.set(
                    f"IBT loaded. Complete {laps_left} more valid lap(s) for coaching vs pro baseline."
                )
            else:
                segments = self._detect_underuse_segments(valid_laps, reference)
        else:
            segments = self._detect_underuse_segments(valid_laps, reference)

        usage_pct = (g_total / max(0.5, self.estimated_limit_g)) * 100.0
        self.current_var.set(
            f"Current: Long {long_g:+.2f}g | Lat {lat_g:+.2f}g | Total {g_total:.2f}g ({usage_pct:4.0f}%)"
        )
        self.limit_var.set(f"Estimated limit: {self.estimated_limit_g:.2f}g")
        self.quality_var.set(
            f"Valid laps used: {len(valid_laps)} | Invalid laps discarded: {self.invalid_laps_count} | "
            f"Outliers removed: {self.outliers_removed_last}"
        )
        self.header_var.set(
            f"Car: {self.current_car} | Track: {self.current_track} | Session: {self.current_session} | "
            f"Valid laps: {len(valid_laps)}"
        )
        self.mode_var.set(f"Mode: {'compact' if self.compact_mode else 'detailed'}")
        if len(valid_laps) >= MIN_LAPS_FOR_FEEDBACK or self.external_reference_bins is None:
            self.summary_var.set(self._format_summary(segments, self.compact_mode))

        self._draw_circle(long_g, lat_g)
        self.root.after(UPDATE_MS, self._update)

    def run(self) -> None:
        self._draw_circle(0.0, 0.0)
        self._update()
        self.root.mainloop()


def main() -> int:
    app = TractionCircleOverlay()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
