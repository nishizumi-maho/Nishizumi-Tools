import irsdk
import time
import pygame
import os
import threading
import json
from collections import deque
from typing import Dict, Optional, Any


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Atualiza dicionários recursivamente (merge profundo)."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


class _LoadedSound:
    """Wrapper para tocar áudio via pygame.

    Tenta carregar como pygame.mixer.Sound (baixa latência). Se falhar, usa mixer.music
    (útil quando MP3 não é suportado como Sound no seu SDL_mixer).

    Observação: para a menor latência possível, prefira .wav ou .ogg.
    """

    def __init__(self, path: str, volume: float):
        self.path = path
        self.volume = float(max(0.0, min(1.0, volume)))
        self._sound: Optional[pygame.mixer.Sound] = None
        self.is_music_fallback = False

        try:
            self._sound = pygame.mixer.Sound(path)
            self._sound.set_volume(self.volume)
        except Exception:
            # Fallback (geralmente MP3)
            self.is_music_fallback = True

    def play(self):
        if self._sound is not None:
            # Não bloqueia
            self._sound.play()
            return

        # Fallback via mixer.music
        pygame.mixer.music.load(self.path)
        pygame.mixer.music.set_volume(self.volume)
        pygame.mixer.music.play()


class RaceEngineer:
    """Engenheiro de rádio simples para iRacing.

    Melhorias vs. versão original:
    - Áudio não bloqueia o loop (não trava a leitura do SDK enquanto toca).
    - Som pré-carregado (quando possível) para reduzir atraso.
    - Consumo por volta suavizado com janela (rolling average) para evitar falso positivo/negativo.
    - Gatilho mais confiável: dispara ao cruzar o ponto (ponto_aviso) e respeita cooldown.
    - Suporte a múltiplos sons configuráveis via JSON e/ou dicionário.

    Campos de telemetria do SDK usados no loop principal (nomes/idênticos ao Appendix A):
    - ``FuelLevel`` (litros, float) e ``Lap``: base para estimar consumo médio por volta.
    - ``CarIdxLapDistPct`` (%, float) e ``PlayerCarIdx``: percentual percorrido na volta do carro do jogador.
    - ``OnPitRoad`` (bool): ignora leituras enquanto o carro está entre os cones.
    Esses nomes/unidades vêm diretamente da documentação oficial, facilitando a
    comparação com a lista de variáveis do SDK quando for necessário depurar
    leituras ou expandir o cálculo.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        # Quando (percentual da volta) o aviso pode tocar.
        "ponto_aviso": 0.85,
        # Margem em voltas (APÓS completar esta volta). Ex.: 1.0 = se após cruzar a linha você tiver <= 1 volta, avisa.
        # Mantive o valor padrão 1.4 da versão anterior por compatibilidade.
        "margem_voltas": 1.4,
        # Evita spam em caso de glitches / reinício / etc.
        "cooldown_s": 8.0,
        # Quantas voltas recentes usar para estimar consumo médio.
        "usage_window": 5,
        # Quantas amostras válidas de consumo por volta precisamos antes de confiar na estimativa.
        "min_laps_for_estimate": 2,
        # Volume global (0.0 a 1.0)
        "volume": 1.0,
        # Arquivos de som (relativos à pasta de áudio)
        "sounds": {
            "box": "box.mp3",
        },
        # Mixer para baixa latência
        "mixer": {
            "frequency": 44100,
            "size": -16,
            "channels": 2,
            "buffer": 512,
        },
        # Filtros simples para ignorar leituras ruins
        "min_fuel_per_lap": 0.05,
        "max_fuel_per_lap": 50.0,
        # Detecta reabastecimento (aumento abrupto no FuelLevel)
        "refuel_delta": 0.25,
        # Intervalo do loop (segundos)
        "tick_s": 0.1,
    }

    def __init__(
        self,
        audio_folder: str,
        config_filename: str = "race_engineer_config.json",
        sounds: Optional[Dict[str, str]] = None,
        debug: bool = False,
    ):
        self.audio_folder = audio_folder
        self.config_path = os.path.join(audio_folder, config_filename) if config_filename else None
        self.debug = debug

        # Runtime
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.ir = irsdk.IRSDK()

        # Locks
        self._audio_lock = threading.Lock()

        # Config e sons
        self.config: Dict[str, Any] = dict(self.DEFAULT_CONFIG)
        self._load_config_file()
        if sounds:
            # permite sobrescrever/estender por parâmetro
            self.config.setdefault("sounds", {})
            self.config["sounds"].update(sounds)

        self._init_audio()
        self.sounds: Dict[str, _LoadedSound] = {}
        self._load_sounds()

        # Estado de telemetria
        self._usage_samples: deque[float] = deque(maxlen=int(self.config["usage_window"]))
        self.avg_usage: float = 0.0
        self.last_lap: Optional[int] = None
        self.fuel_start: Optional[float] = None
        self.last_pct: Optional[float] = None
        self.last_fuel: Optional[float] = None
        self.alerted_lap: Optional[int] = None
        self.last_alert_time: float = 0.0
        self._on_pit_this_lap: bool = False

    # ---------------------------
    # Config
    # ---------------------------
    def _load_config_file(self):
        if not self.config_path:
            return
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                _deep_update(self.config, cfg)
                if self.debug:
                    print(f">> Config carregada: {self.config_path}")
        except Exception as e:
            print(f"AVISO: não consegui ler config '{self.config_path}': {e}")

    def reload_config(self):
        """Recarrega o JSON (se existir) e recarrega os sons."""
        self.config = dict(self.DEFAULT_CONFIG)
        self._load_config_file()
        # Reaplica janela
        self._usage_samples = deque(maxlen=int(self.config["usage_window"]))
        self._load_sounds()
        if self.debug:
            print(">> Config e sons recarregados")

    # ---------------------------
    # Áudio
    # ---------------------------
    def _init_audio(self):
        try:
            mixer_cfg = self.config.get("mixer", {}) or {}
            pygame.mixer.pre_init(
                int(mixer_cfg.get("frequency", 44100)),
                int(mixer_cfg.get("size", -16)),
                int(mixer_cfg.get("channels", 2)),
                int(mixer_cfg.get("buffer", 512)),
            )
            pygame.mixer.init()
        except Exception as e:
            print(f"AVISO: falha ao iniciar áudio (pygame.mixer): {e}")

    def _resolve_sound_path(self, filename: str) -> str:
        # Se o usuário passar caminho absoluto, respeita.
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.audio_folder, filename)

    def _load_sounds(self):
        self.sounds.clear()
        volume = float(self.config.get("volume", 1.0))
        for name, filename in (self.config.get("sounds", {}) or {}).items():
            path = self._resolve_sound_path(str(filename))
            if not os.path.exists(path):
                print(f"AVISO: som '{name}' não encontrado em: {path}")
                continue
            try:
                self.sounds[name] = _LoadedSound(path, volume=volume)
                if self.debug:
                    mode = "Sound" if not self.sounds[name].is_music_fallback else "music(fallback)"
                    print(f">> Som carregado: {name} -> {path} [{mode}]")
            except Exception as e:
                print(f"AVISO: falha ao carregar som '{name}' ({path}): {e}")

    def set_sound(self, name: str, filename: str):
        """Define/atualiza um som e recarrega somente ele."""
        self.config.setdefault("sounds", {})
        self.config["sounds"][name] = filename
        path = self._resolve_sound_path(filename)
        if not os.path.exists(path):
            print(f"AVISO: som '{name}' não encontrado em: {path}")
            return
        self.sounds[name] = _LoadedSound(path, volume=float(self.config.get("volume", 1.0)))

    def play_sound(self, name: str = "box", force: bool = False):
        """Dispara o áudio configurado. Não bloqueia o loop."""

        snd = self.sounds.get(name)
        if snd is None:
            print(f"AVISO: som '{name}' não está configurado/carregado")
            return

        def _runner():
            try:
                with self._audio_lock:
                    if not force and pygame.mixer.get_busy():
                        return
                    snd.play()
            except Exception as e:
                print(f"Erro audio ({name}): {e}")

        # Sempre em thread para garantir que o loop de telemetria não trave (principalmente no fallback music).
        threading.Thread(target=_runner, daemon=True).start()

    def test_audio(self, name: str = "box"):
        """Toca o som imediatamente para teste."""
        print(f">> Testando áudio: {name}")
        self.play_sound(name=name, force=True)

    # ---------------------------
    # Controle
    # ---------------------------
    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(">> Engenheiro de Rádio INICIADO")

    def stop(self):
        self.running = False
        print(">> Engenheiro de Rádio PARADO")

    # ---------------------------
    # Loop principal
    # ---------------------------
    def _reset_state(self):
        self._usage_samples.clear()
        self.avg_usage = 0.0
        self.last_lap = None
        self.fuel_start = None
        self.last_pct = None
        self.last_fuel = None
        self.alerted_lap = None
        self.last_alert_time = 0.0
        self._on_pit_this_lap = False

    def _compute_avg_usage(self) -> float:
        if not self._usage_samples:
            return 0.0
        return float(sum(self._usage_samples) / len(self._usage_samples))

    def _loop(self):
        tick_s = float(self.config.get("tick_s", 0.1))
        ponto_aviso = float(self.config.get("ponto_aviso", 0.85))
        margem_voltas = float(self.config.get("margem_voltas", 1.4))
        cooldown_s = float(self.config.get("cooldown_s", 8.0))
        min_laps_for_estimate = int(self.config.get("min_laps_for_estimate", 2))
        min_fuel_per_lap = float(self.config.get("min_fuel_per_lap", 0.05))
        max_fuel_per_lap = float(self.config.get("max_fuel_per_lap", 50.0))
        refuel_delta = float(self.config.get("refuel_delta", 0.25))

        while self.running:
            if not self.ir.is_initialized or not self.ir.is_connected:
                try:
                    self.ir.startup()
                except Exception:
                    pass
                self._reset_state()
                time.sleep(1)
                continue

            self.ir.freeze_var_buffer_latest()

            try:
                fuel = float(self.ir['FuelLevel'])
                player_idx = int(self.ir['PlayerCarIdx'])
                pct = float(self.ir['CarIdxLapDistPct'][player_idx])
                lap = int(self.ir['Lap'])
                on_pit = bool(self.ir['OnPitRoad'])
            except Exception:
                time.sleep(tick_s)
                continue

            now = time.monotonic()

            # Inicialização na primeira leitura válida
            if self.last_lap is None:
                self.last_lap = lap
                self.fuel_start = fuel
                self.last_fuel = fuel
                self.last_pct = pct
                self.alerted_lap = None
                self.last_alert_time = 0.0
                self._on_pit_this_lap = bool(on_pit)
                time.sleep(tick_s)
                continue

            # Marca que passamos pelo pit nesta volta
            if on_pit:
                self._on_pit_this_lap = True

            # Detecta reabastecimento (saltos positivos no FuelLevel)
            if self.last_fuel is not None and (fuel - self.last_fuel) > refuel_delta:
                if self.debug:
                    print(f">> Reabastecimento detectado (+{fuel - self.last_fuel:.2f}). Resetando média.")
                self._usage_samples.clear()
                self.avg_usage = 0.0
                # reancora início da volta
                self.fuel_start = fuel

            # Atualiza estimativa ao mudar de volta
            if lap != self.last_lap:
                if not self._on_pit_this_lap and self.fuel_start is not None:
                    used = self.fuel_start - fuel
                    if min_fuel_per_lap < used < max_fuel_per_lap:
                        self._usage_samples.append(float(used))
                        self.avg_usage = self._compute_avg_usage()
                        if self.debug:
                            print(
                                f">> Lap {self.last_lap} -> {lap} | usado={used:.3f} | média({len(self._usage_samples)})={self.avg_usage:.3f}"
                            )
                    elif self.debug:
                        print(f">> Lap {self.last_lap} -> {lap} | usado ignorado: {used:.3f}")

                # Novo lap
                self.last_lap = lap
                self.fuel_start = fuel
                self.alerted_lap = None
                self._on_pit_this_lap = bool(on_pit)

            # Decide se deve alertar
            crossed_warning = (
                self.last_pct is not None
                and self.last_pct < ponto_aviso
                and pct >= ponto_aviso
            )

            if (
                crossed_warning
                and (not on_pit)
                and self.avg_usage > 0
                and len(self._usage_samples) >= min_laps_for_estimate
                and (self.alerted_lap != lap)
                and (now - self.last_alert_time) >= cooldown_s
            ):
                # Estima quantas voltas sobram APÓS completar esta volta
                laps_left_total = fuel / self.avg_usage
                laps_after_finish = laps_left_total - (1.0 - pct)

                if self.debug:
                    print(
                        f">> pct={pct:.3f} | fuel={fuel:.2f} | laps_total={laps_left_total:.2f} | laps_after_finish={laps_after_finish:.2f} | margem={margem_voltas:.2f}"
                    )

                if laps_after_finish <= margem_voltas:
                    self.play_sound("box")
                    self.alerted_lap = lap
                    self.last_alert_time = now

            # Atualiza históricos
            self.last_pct = pct
            self.last_fuel = fuel

            time.sleep(tick_s)
