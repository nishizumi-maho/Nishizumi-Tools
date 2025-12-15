
"""
Smart Flasher (iRacing) - versão refatorada

Principais melhorias:
- Config em JSON com versionamento + migração automática do formato antigo
- Criação automática da pasta de configs (não falha silenciosamente)
- Gap mais realista usando TrackLength + velocidade (em vez de "diff * 120")
- Regras (triggers) independentes, cada uma com:
  - enable/disable
  - parâmetros
  - padrão de piscada (qtd / intervalo)
  - cooldown (por alvo ou global)
- Scheduler de piscadas com fila por prioridade (hazard > safety > pass request)
- Limitador global (máx flashes por minuto) + cooldown global
- Vários gatilhos novos rastreados pela telemetria:
  - Pit Exit / Merge warning
  - Carro muito lento à frente (mesmo ON TRACK)
  - Pass request opcional também para mesma classe (configurável)
"""

from __future__ import annotations

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
import yaml

# -----------------------
# Paths / Config storage
# -----------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Mantém o comportamento original: "sobe um nível e entra em configs"
_BASE_DIR = os.path.dirname(_THIS_DIR)
_CONFIG_DIR = os.path.join(_BASE_DIR, "configs")
if not os.path.isdir(_CONFIG_DIR):
    # fallback: configs ao lado do arquivo
    _CONFIG_DIR = os.path.join(_THIS_DIR, "configs")
CONFIG_DIR = _CONFIG_DIR
CONFIG_FILE = os.path.join(CONFIG_DIR, "flasher_config.json")


# -----------------------
# Helpers
# -----------------------
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


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
        # iRacing normalmente não fornece em metros direto, mas aceita.
        val = float(track_len)
        # se vier em km (muito pequeno), o usuário pode ajustar depois; aqui só devolvemos.
        return val

    s = str(track_len).strip().lower()
    # extrai número
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
    # sem unidade: assume km se parecer pequeno
    return num * 1000.0 if num < 50 else num


def safe_float(v: Any, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


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

    # Flags
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
    # Linux/Mac: mantém o módulo importável sem quebrar.
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
        f = max(1, int(self.flashes))
        mi = max(0.0, float(self.min_interval_s))
        ma = max(mi, float(self.max_interval_s))
        hold = clamp(float(self.hold_s), 0.01, 0.25)
        return FlashPattern(flashes=f, min_interval_s=mi, max_interval_s=ma, hold_s=hold)


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
        # Prioridade: 0 (hazard) > 1 (safety) > 2 (pass) ...
        self._q.put((int(priority), time.time(), pat, str(reason)))

    def _rate_limit_ok(self, now: float) -> bool:
        # Remove >60s
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
                # spacing global
                if now < self._next_press_allowed:
                    time.sleep(self._next_press_allowed - now)

                now = time.time()
                if not self._rate_limit_ok(now):
                    # evita spam total (não executa o restante do padrão)
                    if self.debug:
                        self._log.warning("[FLASH] rate limit atingido; padrão cancelado.")
                    break

                press_key_scan_code(self.scan_code, hold_s=pat.hold_s, extended=self.extended)
                self._press_times.append(time.time())
                self._next_press_allowed = time.time() + self.min_press_spacing_s

                # intervalo entre flashes
                if i < pat.flashes - 1:
                    wait = random.uniform(pat.min_interval_s, pat.max_interval_s)
                    time.sleep(max(0.0, wait))


# -----------------------
# SmartFlasher
# -----------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "version": 2,
    "key_name": "P",
    "scan_code": 0x19,
    "key_extended": False,

    "global": {
        "enabled": True,
        "tick_hz": 20,                 # 20Hz = sleep 0.05s (igual ao original)
        "ignore_in_pits": True,
        "min_my_speed_kmh": 0.0,       # 0 = não filtra (use 20~40 se quiser evitar piscada em baixa velocidade)
        "global_press_spacing_s": 0.12,
        "max_flashes_per_min": 40,
        "debug": False,
    },

    "rules": {
        # 1) Pass Request (padrão: multiclass)
        "pass_request": {
            "enabled": True,
            "priority": 2,
            "cooldown_s": 3.0,
            "lookahead_s": 6.0,                # busca carros à frente até X segundos
            "trigger_gap_s": 0.8,              # dispara quando gap < X
            "require_diff_class": True,        # multiclass (como seu original)
            "min_rel_speed_kmh": 8.0,          # evita flash quando não está realmente chegando
            "ignore_target_off_track": True,
            "pattern": {"flashes": 3, "min_interval_s": 0.5, "max_interval_s": 1.2, "hold_s": 0.06},
        },

        # 2) Safety / Fast approach (flash curto)
        "safety_approach": {
            "enabled": True,
            "priority": 1,
            "cooldown_s": 1.5,
            "lookahead_s": 6.0,
            "trigger_gap_s": 1.0,
            # "fechando rápido": gap reduzindo X segundos por segundo (ex: 0.7 = você tira 0.7s de gap a cada 1s)
            "closing_rate_s_per_s": 0.7,
            "pattern": {"flashes": 1, "min_interval_s": 0.0, "max_interval_s": 0.0, "hold_s": 0.06},
        },

        # 3) Hazard (off-track / lento / closing muito agressivo)
        "hazard_ahead": {
            "enabled": True,
            "priority": 0,
            "cooldown_s": 4.0,
            "lookahead_s": 8.0,
            "trigger_gap_s": 2.0,
            "trigger_off_track": True,          # CarIdxTrackSurface == 0
            "trigger_slow_target": True,        # target_speed_kmh abaixo
            "target_speed_kmh_below": 60.0,
            "closing_rate_s_per_s": 1.2,
            "pattern": {"flashes": 1, "min_interval_s": 0.0, "max_interval_s": 0.0, "hold_s": 0.06},
        },

        # 4) Carro MUITO lento à frente (ON TRACK) - útil em estreias / cautions locais
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

        # 5) Pit exit / merge warning
        "pit_exit_merge": {
            "enabled": True,
            "priority": 1,
            "cooldown_s": 10.0,
            "pattern": {"flashes": 2, "min_interval_s": 0.25, "max_interval_s": 0.45, "hold_s": 0.06},
        },

        # 6) Você saiu da pista (útil para alertar quem vem vindo)
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

        self._prev_gap: Dict[int, Tuple[float, float]] = {}  # carIdx -> (gap_s, t)
        self._cooldowns: Dict[Tuple[str, Optional[int]], float] = {}  # (rule, targetIdx|None) -> until
        self._prev_on_pit_road: Optional[bool] = None
        self._prev_my_surface: Optional[int] = None

        # logging
        self.log = logging.getLogger("SmartFlasher")
        if not self.log.handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

        self.config: Dict[str, Any] = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
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
        """
        Migra formato antigo (campos na raiz: gap_trigger, max_flashes, min_interval...)
        para o novo formato baseado em rules, sem quebrar config antiga.
        """
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy

        # 1) mescla keys conhecidas (superficial)
        for k, v in loaded.items():
            if k in ("global", "rules"):
                continue
            cfg[k] = v

        # 2) global
        if isinstance(loaded.get("global"), dict):
            cfg["global"].update(loaded["global"])

        # 3) rules
        if isinstance(loaded.get("rules"), dict):
            for rk, rv in loaded["rules"].items():
                if rk in cfg["rules"] and isinstance(rv, dict):
                    cfg["rules"][rk].update(rv)
                    if isinstance(rv.get("pattern"), dict):
                        cfg["rules"][rk]["pattern"].update(rv["pattern"])

        # 4) migração do formato antigo
        # - gap_trigger -> pass_request.trigger_gap_s
        if "gap_trigger" in loaded and isinstance(cfg["rules"].get("pass_request"), dict):
            cfg["rules"]["pass_request"]["trigger_gap_s"] = safe_float(loaded.get("gap_trigger"), cfg["rules"]["pass_request"]["trigger_gap_s"])
        # - max_flashes/min_interval/max_interval -> pass_request.pattern
        if "max_flashes" in loaded:
            cfg["rules"]["pass_request"]["pattern"]["flashes"] = safe_int(loaded.get("max_flashes"), cfg["rules"]["pass_request"]["pattern"]["flashes"])
        if "min_interval" in loaded:
            cfg["rules"]["pass_request"]["pattern"]["min_interval_s"] = safe_float(loaded.get("min_interval"), cfg["rules"]["pass_request"]["pattern"]["min_interval_s"])
        if "max_interval" in loaded:
            cfg["rules"]["pass_request"]["pattern"]["max_interval_s"] = safe_float(loaded.get("max_interval"), cfg["rules"]["pass_request"]["pattern"]["max_interval_s"])

        # - safety_flash + safety_threshold -> safety_approach.enabled + closing_rate
        if "safety_flash" in loaded:
            cfg["rules"]["safety_approach"]["enabled"] = bool(loaded.get("safety_flash"))
        if "safety_threshold" in loaded:
            # no original era "gap_delta por ciclo"; aqui vira "closing_rate s/s"
            # mapeia de forma conservadora
            th = safe_float(loaded.get("safety_threshold"), 0.04)
            cfg["rules"]["safety_approach"]["closing_rate_s_per_s"] = max(0.1, th * 10.0)

        # - hazard_flash -> hazard_ahead.enabled
        if "hazard_flash" in loaded:
            cfg["rules"]["hazard_ahead"]["enabled"] = bool(loaded.get("hazard_flash"))

        # sane defaults
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
        """
        Testa o padrão de piscada de uma regra (sem telemetria).
        """
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
        if not self.ir.is_connected:
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

        # drivers -> classes
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

        # track length
        try:
            wl = data.get("WeekendInfo", {})
            tl = wl.get("TrackLength")
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
        """
        Retorna gap estimado em segundos (somente para carros À FRENTE).
        Usa TrackLength (m) * deltaPct -> distância (m) e converte para tempo pelo speed médio.
        """
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
        return (prev_gap - gap_s) / dt  # + = fechando

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
        try:
            return self.ir[name]
        except Exception:
            return default

    def _loop(self) -> None:
        while self.running:
            # Conecta no iRacing
            if not self.ir.is_initialized or not self.ir.is_connected:
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

            # Replays / menu
            is_replay = bool(self._ir_get("IsReplayPlaying", False))
            if is_replay:
                time.sleep(0.5)
                continue

            # Vars
            my_idx = self._ir_get("PlayerCarIdx", None)
            lap_pcts = self._ir_get("CarIdxLapDistPct", None)
            surfaces = self._ir_get("CarIdxTrackSurface", None)
            car_speeds = self._ir_get("CarIdxSpeed", None)
            my_speed = self._ir_get("Speed", None)
            on_pit_road = bool(self._ir_get("PlayerCarOnPitRoad", False))

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

            # speed
            if my_speed is None:
                # fallback: array
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
                # ainda assim detecta saída do pit (transição), porque isso é útil
                if self._prev_on_pit_road is True and on_pit_road is False:
                    # não deveria acontecer aqui, mas mantém robusto
                    pass
                self._prev_on_pit_road = on_pit_road
                time.sleep(dt_sleep)
                continue

            # Pit exit rule (transição True -> False)
            if self._prev_on_pit_road is None:
                self._prev_on_pit_road = on_pit_road
            else:
                if self._prev_on_pit_road is True and on_pit_road is False:
                    self._eval_pit_exit_merge(time.time())
                self._prev_on_pit_road = on_pit_road

            # Você saiu da pista (player) -> regra opcional
            self._eval_self_off_track(
                now=time.time(),
                my_idx=int(my_idx),
                surfaces=surfaces,
                my_speed_kmh=float(my_speed_kmh),
            )

            # Se não temos velocidades por carro, ficamos limitados
            if car_speeds is None:
                time.sleep(dt_sleep)
                continue

            # define classe do player (se disponível)
            my_class = self.car_classes.get(int(my_idx), -1)

            # calcula o carro mais próximo à frente (em tempo estimado)
            now = time.time()
            closest_idx: Optional[int] = None
            closest_gap: float = 99999.0

            # lookahead dinâmico: pega o maior lookahead entre regras ativas
            lookahead_s = 0.0
            rules: Dict[str, Any] = self.config.get("rules", {})
            for rname, rcfg in rules.items():
                if isinstance(rcfg, dict) and rcfg.get("enabled", False):
                    lookahead_s = max(lookahead_s, safe_float(rcfg.get("lookahead_s"), 0.0))
            if lookahead_s <= 0:
                lookahead_s = 6.0

            for i, pct in enumerate(lap_pcts):
                if i == my_idx:
                    continue
                if pct is None or pct == -1:
                    continue
                # velocidade alvo
                try:
                    ts = float(car_speeds[i])
                except Exception:
                    continue
                if ts < 0:
                    continue

                gap_s = self._estimate_gap_s(my_pct, pct, float(my_speed), ts)
                if gap_s is None:
                    continue
                if gap_s <= 0 or gap_s > lookahead_s:
                    continue
                if gap_s < closest_gap:
                    closest_gap = gap_s
                    closest_idx = i

            if closest_idx is not None:
                self._eval_rules_for_target(
                    target_idx=int(closest_idx),
                    gap_s=float(closest_gap),
                    my_idx=int(my_idx),
                    my_class=int(my_class),
                    my_speed_mps=float(my_speed),
                    now=now,
                    surfaces=surfaces,
                    car_speeds=car_speeds,
                )

            time.sleep(dt_sleep)

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
        rules = self.config.get("rules", {})
        rcfg = rules.get("pit_exit_merge", {})
        if not isinstance(rcfg, dict) or not rcfg.get("enabled", False):
            return
        if not self._cooldown_ok("pit_exit_merge", None, now):
            return
        pat = self._pattern_from_cfg(rcfg)
        pri = safe_int(rcfg.get("priority"), 1)
        self.scheduler.trigger(pat, priority=pri, reason="PIT_EXIT_MERGE")
        self._set_cooldown("pit_exit_merge", None, now, safe_float(rcfg.get("cooldown_s"), 10.0))

    def _eval_self_off_track(self, now: float, my_idx: int, surfaces: Any, my_speed_kmh: float) -> None:
        """Dispara quando o player sai da pista (transição OnTrack->OffTrack)."""
        rcfg = self.config.get("rules", {}).get("self_off_track_hazard", {})
        if not isinstance(rcfg, dict) or not rcfg.get("enabled", False):
            return
        if surfaces is None:
            return
        try:
            my_surface = int(surfaces[my_idx])
        except Exception:
            return

        if self._prev_my_surface is None:
            self._prev_my_surface = my_surface
            return

        # 0=OffTrack; 3=OnTrack
        transitioned_off = (self._prev_my_surface != 0 and my_surface == 0)
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
        rules: Dict[str, Any] = self.config.get("rules", {})

        # Dados do alvo
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
        is_off_track = (target_surface == 0)

        # closing rate
        closing_rate = self._closing_rate(target_idx, gap_s, now)

        # 0) Hazard ahead (prioridade máxima)
        self._rule_hazard_ahead(target_idx, gap_s, target_speed_kmh, is_off_track, closing_rate, now)

        # 1) Slow car ahead (on track e bem lento)
        self._rule_slow_car_ahead(target_idx, gap_s, target_speed_kmh, target_surface, now)

        # 2) Safety approach (fechando muito rápido)
        self._rule_safety_approach(target_idx, gap_s, closing_rate, now)

        # 3) Pass request
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
            self.scheduler.trigger(pat, priority=pri, reason=f"HAZARD idx={target_idx} gap={gap_s:.2f}s v={target_speed_kmh:.0f}kmh close={closing_rate:.2f}")
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
            tgt_class = self.car_classes.get(int(target_idx), None)
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
        if only_on_track and target_surface not in (3, None):  # 3=OnTrack; None=desconhecido -> deixa passar
            return

        if not self._cooldown_ok("slow_car_ahead", target_idx, now):
            return

        pat = self._pattern_from_cfg(rcfg)
        pri = safe_int(rcfg.get("priority"), 1)
        self.scheduler.trigger(pat, priority=pri, reason=f"SLOW_CAR idx={target_idx} gap={gap_s:.2f}s v={target_speed_kmh:.0f}kmh")
        self._set_cooldown("slow_car_ahead", target_idx, now, safe_float(rcfg.get("cooldown_s"), 4.0))
