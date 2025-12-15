"""fuel_overlay.py

Overlay (Tk/CustomTkinter) de calculadora de combustível + "race chief" assist.

✅ Combustível
- Estima burn por volta (fuel/lap) usando telemetria.
- Ignora voltas em bandeira amarela (configurável) para não poluir o burn.
- Mostra:
  - combustível atual
  - voltas possíveis com o combustível atual
  - estimativa de voltas restantes da corrida
  - combustível total necessário pra terminar (+ margem)
  - quanto precisa encher (Add)

✅ Hotkeys
- Aumentar/diminuir margem de voltas de segurança
- Aplicar "pit plan" (3 opções: pit agora / +1 lap / +2 laps)
- Atalhos para presets (ex.: WET vs SLICK) via chat macro (configurável)

✅ Métodos de burn (dropdown)
- All / Last N / First N / Top Burn % + extras
- Vem com ★ ROAD recomendado (Top Burn % dos últimos N)

✅ Assistentes (heurísticos, configuráveis)
6) Wetness-aware "line + tire decision" assistant (rain transition brain)
- Lê TrackWetness / Precipitation / WeatherDeclaredWet / TrackTemp (se disponíveis)
- Constrói proxy de grip a partir de lat accel vs baseline "dry" e detecta
  eventos de aquaplane (yaw/accel mismatch)
- Dá um call único "PIT WETS" ou "STAY SLICKS" / "WAIT 1–2 LAPS" com confiança

4) Driver-craft risk radar (previsão de incidente)
- Usa CarIdxLapDistPct (relativos) + closing rates + estado do piloto (brake/steer)
- Gera score de risco + motivo(s)
- Pode tocar beep e/ou mandar chat macro para o time (opcional, com cooldown)

⚠️ Importante
- Esses assistentes são *heurísticos* (não são ML treinado em dataset) e servem
  como "co-piloto". Você sempre decide.

Dependências:
- customtkinter
- keyboard
- (opcional, recomendado) pyirsdk -> `pip install pyirsdk`

Como rodar:
    python fuel_overlay.py

"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import customtkinter as ctk

try:
    import keyboard  # type: ignore
except Exception:
    keyboard = None

# pyirsdk expõe o módulo como `irsdk`
try:
    import irsdk  # type: ignore
except Exception:
    irsdk = None


# ==========================
# Paths / config
# ==========================


def _guess_base_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        here,
        os.path.dirname(here),
        os.path.dirname(os.path.dirname(here)),
    ]
    for base in candidates:
        if os.path.isdir(os.path.join(base, "configs")):
            return base
    return os.path.dirname(here)


BASE_DIR = _guess_base_dir()
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
CONFIG_FILE = os.path.join(CONFIG_DIR, "fuel_overlay_config.json")


def _ensure_config_dir() -> None:
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
    except Exception:
        pass


def _deep_merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Merge recursivo (dst <- src).

    Mantém defaults em dst e aplica apenas o que vier em src.
    """

    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge_dict(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


# ==========================
# Flag helpers
# ==========================

# Fallback de bits (caso irsdk.Flags não exista)
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


# ==========================
# Windows helpers (foreground + clipboard)
# ==========================


def _is_windows() -> bool:
    return os.name == "nt"


def is_iracing_foreground(window_substring: str = "iracing") -> bool:
    """Retorna True se a janela em foco aparenta ser o iRacing.

    Se não for Windows (ou der erro), retorna True para não bloquear.
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

    # Powershell (mais robusto)
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

    # clip
    try:
        p = subprocess.Popen("clip", stdin=subprocess.PIPE, shell=True)
        p.communicate(text.encode("utf-8", errors="ignore"))
        return True
    except Exception:
        return False


# ==========================
# Chat injector (para macros)
# ==========================


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
        if not text:
            return False
        if keyboard is None:
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


# ==========================
# Rolling utilities
# ==========================


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
    """Armazena timestamps de eventos e retorna contagens em janelas."""

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


# ==========================
# Histórico de voltas (fuel)
# ==========================


@dataclass
class LapSample:
    lap_number: int
    fuel_used: float
    lap_time: float
    timestamp: float


class FuelHistory:
    def __init__(
        self,
        ignore_yellow: bool = True,
        min_lap_time_s: float = 20.0,
        max_reasonable_burn_per_lap: float = 30.0,
        refuel_delta_threshold: float = 0.3,
    ):
        self.ignore_yellow = ignore_yellow
        self.min_lap_time_s = float(min_lap_time_s)
        self.max_reasonable_burn_per_lap = float(max_reasonable_burn_per_lap)
        self.refuel_delta_threshold = float(refuel_delta_threshold)

        self.samples: List[LapSample] = []

        self._last_lap_completed: Optional[int] = None
        self._lap_start_time: Optional[float] = None
        self._lap_start_fuel: Optional[float] = None
        self._prev_fuel: Optional[float] = None

        self._lap_has_yellow = False
        self._lap_has_pit = False
        self._lap_has_offtrack = False

    def reset(self) -> None:
        self.samples.clear()
        self._last_lap_completed = None
        self._lap_start_time = None
        self._lap_start_fuel = None
        self._prev_fuel = None
        self._lap_has_yellow = False
        self._lap_has_pit = False
        self._lap_has_offtrack = False

    def update(
        self,
        *,
        lap_completed: Optional[int],
        session_time: Optional[float],
        fuel_level: Optional[float],
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

        # flags/pit/offtrack durante a volta
        if self.ignore_yellow and session_flags is not None and is_yellow_flag(int(session_flags)):
            self._lap_has_yellow = True

        if on_pit_road:
            self._lap_has_pit = True

        if is_on_track is False:
            self._lap_has_offtrack = True

        # detecta reabastecimento (fuel subindo)
        if self._prev_fuel is not None and fuel_f - self._prev_fuel > self.refuel_delta_threshold:
            self._lap_has_pit = True
            self._lap_start_fuel = fuel_f

        self._prev_fuel = fuel_f

        # primeira leitura
        if self._last_lap_completed is None:
            self._last_lap_completed = lap_completed_i
            self._lap_start_time = session_time_f
            self._lap_start_fuel = fuel_f
            return

        # fechou volta
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

            # inicia nova volta
            self._last_lap_completed = lap_completed_i
            self._lap_start_time = session_time_f
            self._lap_start_fuel = fuel_f

            self._lap_has_yellow = False
            self._lap_has_pit = False
            self._lap_has_offtrack = False

    # ---------- cálculos ----------

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


# ==========================
# Wetness-aware assistant (heurístico)
# ==========================


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
        self._last_update_t: Optional[float] = None

    @staticmethod
    def _norm_wetness(w: Optional[float]) -> Optional[float]:
        if w is None:
            return None
        try:
            x = float(w)
        except Exception:
            return None

        # alguns builds reportam 0..100
        if x > 1.5:
            x = x / 100.0
        return max(0.0, min(1.0, x))

    @staticmethod
    def _norm_precip(p: Optional[float]) -> Optional[float]:
        if p is None:
            return None
        try:
            x = float(p)
        except Exception:
            return None
        # precip pode vir como 0..1 ou mm/h (depende); normaliza agressivamente
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
                pct = pct - math.floor(pct)  # mantém 0..1
            except Exception:
                pct = None

            if pct is not None and speed_mps >= self.min_speed_mps and abs(steer) >= self.min_steer_abs:
                idx = int(pct * self.bins)
                idx = max(0, min(self.bins - 1, idx))

                lat_abs = abs(float(lat_accel))

                if wet <= self.dry_wetness_threshold:
                    # atualiza baseline "dry"
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

                # gatilho: alta velocidade + steering presente + "não gera lat accel" proporcional
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
        # tendência simples: dif entre média do fim e do começo
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

        # Score de "precisa wets"
        score = 0.0
        if declared_wet:
            score += 12.0

        if wet is not None:
            # zona de decisão 0.12..0.40
            score += max(0.0, min(1.0, (wet - 0.12) / 0.28)) * 45.0

        if prec is not None:
            score += max(0.0, min(1.0, prec / 0.6)) * 10.0

        if grip is not None:
            # grip < 0.75 começa a doer
            score += max(0.0, min(1.0, (0.78 - grip) / 0.30)) * 25.0

        score += max(0.0, min(1.0, aqua / 3.0)) * 20.0

        conf = int(max(0, min(100, round(score))))

        # Decide ação
        action = "STAY SLICKS"
        if conf >= 70:
            action = "PIT WETS"
        elif conf >= 50:
            # transitional: usa tendência
            if trend is not None and trend > 0.03:
                action = "WAIT 1 LAP (trend ↑)"
            else:
                action = "CONSIDER PIT"

        details = {
            "wet": wet,
            "prec": prec,
            "grip": grip,
            "aqua": aqua,
            "trend": trend,
        }
        return action, conf, details


# ==========================
# Risk radar (heurístico)
# ==========================


class RiskRadar:
    def __init__(self) -> None:
        self.prev_dist_m: Dict[int, Tuple[float, float]] = {}
        self.steer_hist = RollingWindow(60)
        self.brake_hist = RollingWindow(60)
        self.last_beep_t = 0.0

    @staticmethod
    def _wrap_delta_pct(dp: float) -> float:
        # traz para [-0.5, 0.5]
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
        # alimenta variâncias
        if steer is not None:
            self.steer_hist.add(float(steer))
        if brake is not None:
            self.brake_hist.add(float(brake))

        if (
            player_idx is None
            or track_length_m is None
            or player_lapdist_pct is None
            or not car_idx_lapdist_pct
        ):
            return 0, "(sem dados de tráfego)"

        try:
            pidx = int(player_idx)
            tlen = float(track_length_m)
            ppct = float(player_lapdist_pct)
        except Exception:
            return 0, "(sem dados de tráfego)"

        # braking zone proxy
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
                overlap = int(car_left_right) in (1, 2, 3)
            except Exception:
                overlap = False

        steer_var = self.steer_hist.stdev() or 0.0
        brake_var = self.brake_hist.stdev() or 0.0

        # var de steer alta (carro instável)
        instability = max(0.0, min(1.0, steer_var / 0.25))  # heurístico

        # avalia ameaças principais
        best_behind = (0.0, 0.0, None)  # score, closing, dist, caridx
        best_ahead = (0.0, 0.0, None)

        for i, opct in enumerate(car_idx_lapdist_pct):
            if i == pidx:
                continue
            if opct is None:
                continue
            try:
                dp = self._wrap_delta_pct(float(opct) - ppct)
                dist_m = dp * tlen
            except Exception:
                continue

            # só perto
            if abs(dist_m) > 250.0:
                continue

            # closing rate
            closing = 0.0
            prev = self.prev_dist_m.get(i)
            if prev is not None:
                prev_dist, prev_t = prev
                dt = max(0.02, now - prev_t)
                if dist_m < 0:
                    # carro atrás: dist negativo; se aumenta (vai para 0), ele está chegando
                    closing = (dist_m - prev_dist) / dt
                else:
                    # carro à frente: se dist diminui, você está chegando
                    closing = (prev_dist - dist_m) / dt

            self.prev_dist_m[i] = (dist_m, now)

            if dist_m < 0:
                # behind threat
                d = abs(dist_m)
                score = 0.0
                if d < 60.0:
                    score += (60.0 - d) / 60.0 * 20.0
                if closing > 3.0:
                    score += min(25.0, (closing - 3.0) / 6.0 * 25.0)
                if braking:
                    score += 10.0
                if overlap:
                    score += 10.0

                if score > best_behind[0]:
                    best_behind = (score, closing, dist_m, i)
            else:
                # ahead threat
                d = abs(dist_m)
                score = 0.0
                if d < 35.0:
                    score += (35.0 - d) / 35.0 * 15.0
                if closing > 3.0 and braking:
                    score += min(20.0, (closing - 3.0) / 6.0 * 20.0)
                if overlap:
                    score += 8.0
                if score > best_ahead[0]:
                    best_ahead = (score, closing, dist_m, i)

        risk = 0.0
        reasons = []

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
            risk += 18.0
            reasons.append("overlap+brake")
        elif overlap:
            risk += 10.0
            reasons.append("overlap")

        if is_on_track is False:
            risk += 25.0
            reasons.append("offtrack/rejoin")

        risk += instability * 18.0
        if instability > 0.6:
            reasons.append("instability")

        # clamp
        risk_i = int(max(0, min(100, round(risk))))

        if not reasons:
            return risk_i, "OK"

        return risk_i, "; ".join(reasons[:3])


# ==========================
# Pit timing advisor (heurístico)
# ==========================


@dataclass
class PitOption:
    offset_laps: int
    fuel_add: Optional[float]
    pit_loss_s: Optional[float]
    gap_ahead_s: Optional[float]
    gap_behind_s: Optional[float]
    clean_air_score: Optional[int]
    pos_est: Optional[int]


class PitWindowAdvisor:
    @staticmethod
    def _wrap_delta_pct(dp: float) -> float:
        return dp - round(dp)

    def build_options(
        self,
        *,
        avg_lap_time: Optional[float],
        pit_base_loss_s: float,
        fuel_fill_rate: float,
        tire_service_time_s: float,
        take_tires: bool,
        burn_per_lap: Optional[float],
        fuel_now: Optional[float],
        laps_possible: Optional[float],
        laps_remain: Optional[float],
        margin_laps: float,
        player_idx: Optional[int],
        player_lapdist_pct: Optional[float],
        car_idx_lapdist_pct: Optional[List[float]],
        track_length_m: Optional[float],
        max_offsets: int = 2,
        clean_air_target_s: float = 2.0,
    ) -> List[PitOption]:
        if avg_lap_time is None or avg_lap_time <= 0:
            return []
        if burn_per_lap is None or burn_per_lap <= 0:
            return []
        if fuel_now is None:
            return []
        if player_idx is None or player_lapdist_pct is None or not car_idx_lapdist_pct or track_length_m is None:
            return []

        opts: List[PitOption] = []

        pidx = int(player_idx)
        ppct = float(player_lapdist_pct)
        tlen = float(track_length_m)

        for offset in range(0, int(max_offsets) + 1):
            # valida combustível para esperar
            if laps_possible is not None and laps_possible < float(offset) + 0.15:
                continue

            fuel_after_wait = float(fuel_now) - float(offset) * float(burn_per_lap)
            fuel_after_wait = max(0.0, fuel_after_wait)

            laps_rem_after = None
            if laps_remain is not None:
                laps_rem_after = max(0.0, float(laps_remain) - float(offset))

            fuel_need_total = None
            fuel_add = None
            if laps_rem_after is not None:
                fuel_need_total = (laps_rem_after + float(margin_laps)) * float(burn_per_lap)
                fuel_add = max(0.0, fuel_need_total - fuel_after_wait)

            # Pit loss = base + fuel time + tires
            pit_loss = float(pit_base_loss_s)
            if fuel_add is not None and fuel_fill_rate > 0:
                pit_loss += float(fuel_add) / float(fuel_fill_rate)
            if take_tires:
                pit_loss += float(tire_service_time_s)

            # Estima gaps após pit (simples: gap_time + pit_loss)
            ahead_gap = None
            behind_gap = None

            gaps_after: List[float] = []
            for i, opct in enumerate(car_idx_lapdist_pct):
                if i == pidx:
                    continue
                if opct is None:
                    continue
                try:
                    dp = self._wrap_delta_pct(float(opct) - ppct)
                    gap_time = dp * float(avg_lap_time)
                    gap_after = gap_time + pit_loss
                    gaps_after.append(gap_after)
                except Exception:
                    continue

            if gaps_after:
                # carros à frente depois do pit: gap_after > 0
                ahead_candidates = [g for g in gaps_after if g > 0]
                behind_candidates = [g for g in gaps_after if g < 0]

                if ahead_candidates:
                    ahead_gap = min(ahead_candidates)
                if behind_candidates:
                    behind_gap = max(behind_candidates)  # negativo mais próximo de 0

            # Clean air score
            score = None
            if ahead_gap is not None and behind_gap is not None:
                min_gap = min(ahead_gap, abs(behind_gap))
                score_f = max(0.0, min(1.0, min_gap / max(0.5, clean_air_target_s)))
                score = int(round(score_f * 100))
            elif ahead_gap is not None:
                score_f = max(0.0, min(1.0, ahead_gap / max(0.5, clean_air_target_s)))
                score = int(round(score_f * 80))
            elif behind_gap is not None:
                score_f = max(0.0, min(1.0, abs(behind_gap) / max(0.5, clean_air_target_s)))
                score = int(round(score_f * 80))

            # posição estimada após pit (heurística): quantos ficam à frente
            pos_est = None
            if gaps_after:
                pos_est = 1 + sum(1 for g in gaps_after if g > 0)

            opts.append(
                PitOption(
                    offset_laps=offset,
                    fuel_add=fuel_add,
                    pit_loss_s=pit_loss,
                    gap_ahead_s=ahead_gap,
                    gap_behind_s=behind_gap,
                    clean_air_score=score,
                    pos_est=pos_est,
                )
            )

        # ordena por score de clean air (desc)
        opts.sort(key=lambda o: (o.clean_air_score or 0), reverse=True)
        return opts


# ==========================
# Overlay UI
# ==========================


class FuelOverlayApp(ctk.CTk):
    METHOD_LABELS = {
        "road_fav": "★ ROAD (Recomendado) – Top Burn% (Últimos N)",
        "all": "All (Média) – todas as voltas verdes",
        "last_n": "Last N (Média) – últimas N voltas verdes",
        "first_n": "First N (Média) – primeiras N voltas verdes",
        "top_burn": "Top Burn % – média do topo % (Últimos N)",
        "median_last_n": "Median – mediana (Últimos N)",
        "max_last_n": "Max – pior burn (Últimos N)",
        "trimmed_last_n": "Trimmed Mean – corta % das pontas (Últimos N)",
        "ema": "EMA – média móvel exponencial (todas verdes)",
    }

    PLAN_LABELS = {
        "safe": "SAFE (conservador)",
        "attack": "ATTACK (mais agressivo)",
        "stretch": "STRETCH (economia)",
    }

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")

        self.config_data = self._load_config()

        # IRSDK
        self.ir = irsdk.IRSDK() if irsdk else None

        # subsistemas
        self.history = FuelHistory(ignore_yellow=bool(self.config_data.get("ignore_yellow", True)))
        self.wet_brain = WetnessBrain()
        self.risk_radar = RiskRadar()
        self.pit_advisor = PitWindowAdvisor()

        # macros
        self.injector = self._build_injector_from_config()

        # hotkey handles
        self._hk_ids: List[int] = []

        # cache do último cálculo (para hotkeys)
        self._last_calc: Dict[str, Any] = {}
        self._last_pit_options: List[PitOption] = []

        # window
        self.title("Fuel Overlay")
        self.attributes("-topmost", True)
        self.overrideredirect(True)

        geo = self.config_data.get("geometry")
        if isinstance(geo, str) and geo:
            # se a config antiga era muito pequena, tenta aumentar só a altura
            try:
                if "x160" in geo:
                    geo = geo.replace("x160", "x300")
                self.geometry(geo)
            except Exception:
                self.geometry("520x300+50+50")
        else:
            self.geometry("520x300+50+50")

        # drag
        self._drag_offset: Optional[Tuple[int, int]] = None
        self.bind("<ButtonPress-1>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_up)

        # close
        self.bind("<Escape>", lambda e: self._close())

        # UI
        self._build_ui()

        # hotkeys
        self._setup_hotkeys()

        # loop
        self.after(140, self._tick)

    # ---------- config ----------

    def _default_config(self) -> dict:
        return {
            "method": "road_fav",
            "n": 10,
            "top_percent": 20,
            "margin_laps": 1.0,
            "margin_step": 0.5,
            "ignore_yellow": True,
            "plan_mode": "safe",
            "plan_modes": {
                "safe": {"margin_override": 1.5, "take_tires": True},
                "attack": {"margin_override": 0.5, "take_tires": False},
                "stretch": {"margin_override": 1.0, "take_tires": False},
            },
            "pit": {
                "pit_base_loss_s": 40.0,
                "fuel_fill_rate": 2.5,
                "tire_service_time_s": 18.0,
                "clean_air_target_s": 2.0,
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
                "templates": {
                    "apply_plan": "#fuel {fuel_add:.2f}",
                    "wet_preset": "",
                    "slick_preset": "",
                },
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
            },
            "audio": {
                "risk_beep": True,
                "beep_cooldown_s": 3.0,
                "risk_beep_threshold": 85,
            },
            "geometry": "520x300+50+50",
        }

    def _load_config(self) -> dict:
        cfg = self._default_config()
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # merge recursivo para não perder defaults quando
                    # o usuário sobrescreve apenas 1 chave de um sub-dict
                    _deep_merge_dict(cfg, data)  # type: ignore[arg-type]
        except Exception:
            pass
        return cfg

    def _save_config(self) -> None:
        try:
            _ensure_config_dir()
            try:
                self.config_data["geometry"] = self.geometry()
            except Exception:
                pass
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, indent=4, ensure_ascii=False)
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

    # ---------- UI ----------

    def _build_ui(self) -> None:
        self.frame = ctk.CTkFrame(self, corner_radius=10)
        self.frame.pack(fill="both", expand=True, padx=8, pady=8)

        # Row 1: method + save
        top = ctk.CTkFrame(self.frame, fg_color="transparent")
        top.pack(fill="x", pady=(0, 6))

        self.var_method = ctk.StringVar(value=self.METHOD_LABELS.get(self.config_data["method"], self.METHOD_LABELS["road_fav"]))
        values = list(self.METHOD_LABELS.values())
        self.opt_method = ctk.CTkOptionMenu(
            top,
            values=values,
            variable=self.var_method,
            command=self._on_method_change,
            width=360,
        )
        self.opt_method.pack(side="left")

        self.btn_save = ctk.CTkButton(top, text="💾", width=46, command=self._save_config)
        self.btn_save.pack(side="right")

        # Row 2: params + plan mode
        params = ctk.CTkFrame(self.frame, fg_color="transparent")
        params.pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(params, text="N:", width=18, anchor="e").pack(side="left")
        self.entry_n = ctk.CTkEntry(params, width=55)
        self.entry_n.insert(0, str(self.config_data.get("n", 10)))
        self.entry_n.pack(side="left", padx=(4, 10))

        ctk.CTkLabel(params, text="%:", width=18, anchor="e").pack(side="left")
        self.entry_pct = ctk.CTkEntry(params, width=55)
        self.entry_pct.insert(0, str(self.config_data.get("top_percent", 20)))
        self.entry_pct.pack(side="left", padx=(4, 10))

        self.var_ignore = ctk.BooleanVar(value=bool(self.config_data.get("ignore_yellow", True)))
        self.chk_ignore = ctk.CTkCheckBox(params, text="Ignorar amarelas", variable=self.var_ignore, command=self._on_ignore_toggle)
        self.chk_ignore.pack(side="left", padx=(6, 10))

        # plan mode
        self.var_plan = ctk.StringVar(value=self.PLAN_LABELS.get(self.config_data.get("plan_mode", "safe"), self.PLAN_LABELS["safe"]))
        self.opt_plan = ctk.CTkOptionMenu(
            params,
            values=list(self.PLAN_LABELS.values()),
            variable=self.var_plan,
            command=self._on_plan_change,
            width=170,
        )
        self.opt_plan.pack(side="right")

        # Labels
        self.lbl_status = ctk.CTkLabel(self.frame, text="iRacing: aguardando...", anchor="w")
        self.lbl_status.pack(fill="x")

        self.lbl_fuel = ctk.CTkLabel(self.frame, text="Fuel: --", anchor="w", font=("Consolas", 14, "bold"))
        self.lbl_fuel.pack(fill="x", pady=(2, 0))

        self.lbl_race = ctk.CTkLabel(self.frame, text="Race: --", anchor="w", font=("Consolas", 12))
        self.lbl_race.pack(fill="x")

        self.lbl_weather = ctk.CTkLabel(self.frame, text="Weather: --", anchor="w", font=("Consolas", 12))
        self.lbl_weather.pack(fill="x", pady=(6, 0))

        self.lbl_risk = ctk.CTkLabel(self.frame, text="Risk: --", anchor="w", font=("Consolas", 12))
        self.lbl_risk.pack(fill="x")

        self.lbl_pit = ctk.CTkLabel(self.frame, text="Pit timing: --", anchor="w", font=("Consolas", 12), justify="left")
        self.lbl_pit.pack(fill="x", pady=(6, 0))

        self.lbl_margin = ctk.CTkLabel(self.frame, text="Hotkeys: --", anchor="w", font=("Consolas", 11))
        self.lbl_margin.pack(fill="x", pady=(6, 0))

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

    def _on_method_change(self, chosen_label: str) -> None:
        self.config_data["method"] = self._method_key_from_label(chosen_label)
        self._save_config()

    def _on_plan_change(self, chosen_label: str) -> None:
        self.config_data["plan_mode"] = self._plan_key_from_label(chosen_label)
        self._save_config()

    def _on_ignore_toggle(self) -> None:
        self.config_data["ignore_yellow"] = bool(self.var_ignore.get())
        self.history.ignore_yellow = bool(self.var_ignore.get())
        self._save_config()

    # ---------- drag ----------

    def _on_mouse_down(self, event) -> None:
        self._drag_offset = (event.x_root - self.winfo_x(), event.y_root - self.winfo_y())

    def _on_mouse_drag(self, event) -> None:
        if not self._drag_offset:
            return
        x_off, y_off = self._drag_offset
        x = event.x_root - x_off
        y = event.y_root - y_off
        self.geometry(f"+{x}+{y}")

    def _on_mouse_up(self, event) -> None:
        self._drag_offset = None
        self._save_config()

    # ---------- hotkeys ----------

    def _setup_hotkeys(self) -> None:
        if keyboard is None:
            return

        hk = self.config_data.get("hotkeys", {})

        def margin_add(delta: float):
            step = float(self.config_data.get("margin_step", 0.5))
            self.config_data["margin_laps"] = max(0.0, float(self.config_data.get("margin_laps", 1.0)) + delta * step)
            self._save_config()

        def cycle_plan():
            keys = list(self.PLAN_LABELS.keys())
            cur = str(self.config_data.get("plan_mode", "safe"))
            if cur not in keys:
                cur = "safe"
            idx = keys.index(cur)
            nxt = keys[(idx + 1) % len(keys)]
            self.config_data["plan_mode"] = nxt
            try:
                self.var_plan.set(self.PLAN_LABELS[nxt])
            except Exception:
                pass
            self._save_config()

        def apply_opt(rank: int):
            # rank 1..3
            if not self.config_data.get("macro", {}).get("enabled", False):
                return
            if rank <= 0:
                return
            if rank > len(self._last_pit_options):
                return
            opt = self._last_pit_options[rank - 1]
            if opt.fuel_add is None:
                return

            tmpl = self.config_data.get("macro", {}).get("templates", {}).get("apply_plan", "#fuel {fuel_add:.2f}")
            ctx = _SafeFormatDict({
                "fuel_add": float(opt.fuel_add),
                "offset_laps": int(opt.offset_laps),
                "pit_loss_s": float(opt.pit_loss_s or 0.0),
                "plan_mode": str(self.config_data.get("plan_mode", "safe")),
            })
            try:
                cmd = str(tmpl).format_map(ctx)
            except Exception:
                cmd = f"#fuel {float(opt.fuel_add):.2f}"
            self.injector.send(cmd)

        def apply_preset(which: str):
            if not self.config_data.get("macro", {}).get("enabled", False):
                return
            tmpl = self.config_data.get("macro", {}).get("templates", {}).get(which, "")
            if not tmpl:
                return
            ctx = _SafeFormatDict(self._last_calc)
            try:
                cmd = str(tmpl).format_map(ctx)
            except Exception:
                cmd = str(tmpl)
            self.injector.send(cmd)

        try:
            self._hk_ids.append(keyboard.add_hotkey(hk.get("margin_up", "ctrl+alt+up"), lambda: margin_add(+1)))
            self._hk_ids.append(keyboard.add_hotkey(hk.get("margin_down", "ctrl+alt+down"), lambda: margin_add(-1)))
            self._hk_ids.append(keyboard.add_hotkey(hk.get("cycle_plan", "ctrl+alt+p"), cycle_plan))

            self._hk_ids.append(keyboard.add_hotkey(hk.get("apply_opt1", "alt+1"), lambda: apply_opt(1)))
            self._hk_ids.append(keyboard.add_hotkey(hk.get("apply_opt2", "alt+2"), lambda: apply_opt(2)))
            self._hk_ids.append(keyboard.add_hotkey(hk.get("apply_opt3", "alt+3"), lambda: apply_opt(3)))

            self._hk_ids.append(keyboard.add_hotkey(hk.get("wet", "alt+w"), lambda: apply_preset("wet_preset")))
            self._hk_ids.append(keyboard.add_hotkey(hk.get("slick", "alt+s"), lambda: apply_preset("slick_preset")))
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

    # ---------- telemetry helpers ----------

    def _safe_get(self, key: str) -> Any:
        try:
            return self.ir[key]
        except Exception:
            return None

    @staticmethod
    def _track_length_m(val: Any) -> Optional[float]:
        if val is None:
            return None
        try:
            x = float(val)
        except Exception:
            return None
        # iRacing costuma reportar em km
        if x < 80.0:
            return x * 1000.0
        return x

    # ---------- main loop ----------

    def _tick(self) -> None:
        try:
            # --- user params
            try:
                n = int(float(self.entry_n.get()))
            except Exception:
                n = int(self.config_data.get("n", 10))
            n = max(1, min(n, 250))
            self.config_data["n"] = n

            try:
                pct = float(self.entry_pct.get())
            except Exception:
                pct = float(self.config_data.get("top_percent", 20))
            pct = max(1.0, min(pct, 100.0))
            self.config_data["top_percent"] = pct

            method = str(self.config_data.get("method", "road_fav"))

            plan_mode = str(self.config_data.get("plan_mode", "safe"))
            pm = self.config_data.get("plan_modes", {}).get(plan_mode, {})
            margin = float(pm.get("margin_override", self.config_data.get("margin_laps", 1.0)))

            self._last_calc["margin_laps"] = margin

            if not self.ir:
                self.lbl_status.configure(text="pyirsdk não encontrado. Instale: pip install pyirsdk")
                self.after(500, self._tick)
                return

            # connect
            if not (getattr(self.ir, "is_initialized", False) and getattr(self.ir, "is_connected", False)):
                try:
                    self.ir.startup()
                except Exception:
                    pass

            if not (getattr(self.ir, "is_initialized", False) and getattr(self.ir, "is_connected", False)):
                self.lbl_status.configure(text="iRacing: aguardando conexão...")
                self.after(500, self._tick)
                return

            try:
                self.ir.freeze_var_buffer_latest()
            except Exception:
                pass

            # --- telemetry
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

            yellow_now = is_yellow_flag(int(session_flags or 0))

            # --- update fuel history
            self.history.update(
                lap_completed=lap_completed,
                session_time=session_time,
                fuel_level=fuel_level,
                session_flags=session_flags,
                is_on_track=bool(is_on_track) if is_on_track is not None else None,
                on_pit_road=bool(on_pit_road) if on_pit_road is not None else None,
            )

            burn = self.history.burn_per_lap(method=method, n=n, top_percent=pct)
            avg_lap_time = self.history.lap_time_estimate(method=method, n=n)

            # --- laps remaining race
            laps_remain = self._estimate_laps_remaining(avg_lap_time)

            # --- compute fuel figures
            fuel = float(fuel_level or 0.0)
            laps_possible = None
            fuel_need_total = None
            fuel_to_add = None
            finish_leftover = None

            if burn and burn > 0:
                laps_possible = fuel / burn

                if laps_remain is not None:
                    fuel_need_total = max(0.0, (laps_remain + margin) * burn)
                    fuel_to_add = max(0.0, fuel_need_total - fuel)
                    finish_leftover = fuel + fuel_to_add - fuel_need_total

            # --- wetness brain
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

            wet_action, wet_conf, wet_details = self.wet_brain.recommend(now=now, declared_wet=bool(declared_wet) if declared_wet is not None else None)

            # --- risk radar
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

            # --- pit timing options
            pit_cfg = self.config_data.get("pit", {})
            pit_base = float(pit_cfg.get("pit_base_loss_s", 40.0))
            fill_rate = float(pit_cfg.get("fuel_fill_rate", 2.5))
            tire_time = float(pit_cfg.get("tire_service_time_s", 18.0))
            clean_air_target = float(pit_cfg.get("clean_air_target_s", 2.0))
            take_tires = bool(pm.get("take_tires", False))

            opts = self.pit_advisor.build_options(
                avg_lap_time=avg_lap_time,
                pit_base_loss_s=pit_base,
                fuel_fill_rate=fill_rate,
                tire_service_time_s=tire_time,
                take_tires=take_tires,
                burn_per_lap=burn,
                fuel_now=fuel,
                laps_possible=laps_possible,
                laps_remain=laps_remain,
                margin_laps=margin,
                player_idx=int(player_idx) if player_idx is not None else None,
                player_lapdist_pct=float(lap_dist_pct) if lap_dist_pct is not None else None,
                car_idx_lapdist_pct=car_idx_lapdist,
                track_length_m=track_len_m,
                max_offsets=2,
                clean_air_target_s=clean_air_target,
            )

            self._last_pit_options = opts

            # --- update labels
            status = "YELLOW" if yellow_now else "GREEN"
            status_extra = f" | samples={len(self.history.samples)}"
            self.lbl_status.configure(text=f"iRacing: conectado ({status}){status_extra}")

            if burn is None or burn <= 0 or laps_possible is None:
                self.lbl_fuel.configure(text=f"Burn: coletando... | Fuel: {fuel:.2f}")
                self.lbl_race.configure(text=f"Race: RemLaps=-- | Need=-- | Add=--")
            else:
                self.lbl_fuel.configure(text=f"Burn: {burn:.3f}/lap | Fuel: {fuel:.2f} | Can: {laps_possible:.1f} laps")

                if laps_remain is not None and fuel_need_total is not None and fuel_to_add is not None:
                    self.lbl_race.configure(
                        text=(
                            f"Race: RemLaps={laps_remain:.1f} | Need={fuel_need_total:.2f} | Add={fuel_to_add:.2f} | Left={finish_leftover:.2f}"
                        )
                    )
                else:
                    self.lbl_race.configure(text=f"Race: RemLaps=-- | Need=-- | Add=--")

            wet_s = self._format_weather_line(track_wetness, precipitation, declared_wet, track_temp, wet_action, wet_conf, wet_details)
            self.lbl_weather.configure(text=wet_s)

            self.lbl_risk.configure(text=f"Risk: {risk_score:3d} | {risk_reason}")

            self.lbl_pit.configure(text=self._format_pit_options(opts, burn, fuel))

            hk = self.config_data.get("hotkeys", {})
            self.lbl_margin.configure(
                text=(
                    f"Margin: {margin:.1f} laps (+/-: {hk.get('margin_up','?')}/{hk.get('margin_down','?')}) | "
                    f"Plan: {hk.get('cycle_plan','?')} | Apply: {hk.get('apply_opt1','?')},{hk.get('apply_opt2','?')},{hk.get('apply_opt3','?')} | "
                    f"Wet/Slick: {hk.get('wet','?')}/{hk.get('slick','?')}"
                )
            )

            # cache context p/ templates
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
                }
            )

            self.after(150, self._tick)

        except Exception as e:
            self.lbl_status.configure(text=f"Erro: {e}")
            self.after(500, self._tick)

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

    def _format_weather_line(self, track_wetness, precipitation, declared_wet, track_temp, action: str, conf: int, details: Dict[str, Any]) -> str:
        wet = details.get("wet")
        prec = details.get("prec")
        grip = details.get("grip")
        aqua = details.get("aqua")
        trend = details.get("trend")

        wet_s = "--" if wet is None else f"{wet*100:4.0f}%"
        tr_s = "" if trend is None else ("↑" if trend > 0.02 else ("↓" if trend < -0.02 else "→"))
        prec_s = "--" if prec is None else f"{prec:.2f}"
        grip_s = "--" if grip is None else f"{grip:.2f}"
        temp_s = "--" if track_temp is None else f"{float(track_temp):.0f}C"
        dw = "Y" if bool(declared_wet) else "N"

        return f"Weather: Wet={wet_s}{tr_s} Precip={prec_s} DeclWet={dw} TrackT={temp_s} | Grip~{grip_s} Aqua={aqua} | {action} ({conf}%)"

    def _format_pit_options(self, opts: List[PitOption], burn: Optional[float], fuel: float) -> str:
        if not opts:
            return "Pit timing: -- (precisa de avg lap time + dados de tráfego)"

        lines = ["Pit timing (clean air heuristic):"]
        for rank, o in enumerate(opts[:3], start=1):
            off = o.offset_laps
            fa = "--" if o.fuel_add is None else f"{o.fuel_add:.2f}"
            pl = "--" if o.pit_loss_s is None else f"{o.pit_loss_s:.0f}s"
            ga = "--" if o.gap_ahead_s is None else f"+{o.gap_ahead_s:.1f}s"
            gb = "--" if o.gap_behind_s is None else f"{o.gap_behind_s:.1f}s"
            sc = "--" if o.clean_air_score is None else f"{o.clean_air_score:02d}"
            pe = "--" if o.pos_est is None else str(o.pos_est)

            label = "NOW" if off == 0 else f"+{off}L"
            lines.append(f"  [{rank}] {label:>3} | score={sc} pos~{pe:>3} | gap {ga}/{gb} | add={fa} | pit≈{pl}")

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

    # ---------- close ----------

    def _close(self) -> None:
        self._save_config()
        self._remove_hotkeys()
        try:
            self.destroy()
        except Exception:
            pass


def main() -> None:
    app = FuelOverlayApp()
    app.mainloop()


if __name__ == "__main__":
    main()
