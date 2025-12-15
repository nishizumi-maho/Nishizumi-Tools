"""pit_strategy.py

Turbo pit macro + generic chat command sender for iRacing.

What this module provides
- Hotkey (keyboard) or joystick button (pygame) to run a pit "turbo" command.
- Generic function to send ANY pit/chat scripting command quickly.
- Two injection modes:
    - "type"     (default): keyboard.write() very fast, no clipboard.
    - "clipboard": copies to clipboard and Ctrl+V (sometimes more reliable for very long strings).
- Optional safety guard to avoid typing into the wrong app:
    - require_iracing_foreground = True

Why it's still chat-based
- iRacing *does* have SDK/broadcast ways to change pit settings, but they vary by
  language binding and require extra wiring. This module focuses on the robust
  option that works for everyone: chat macros.

Project structure compatibility
- Many projects use: <root>/modulos/pit_strategy.py and <root>/configs/pit_config.json
- This file auto-detects the root by looking for a "configs" folder.

Config file (JSON)
- Saved in configs/pit_config.json

"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import threading
import time
from typing import Optional

import keyboard

try:
    import pygame
except Exception:
    pygame = None


# ==========================
# Paths / Config
# ==========================


def _find_base_dir(file_path: str) -> str:
    here = os.path.dirname(os.path.abspath(file_path))
    parent = os.path.dirname(here)
    grand = os.path.dirname(parent)

    for base in (grand, parent, here):
        if os.path.isdir(os.path.join(base, "configs")):
            return base
    return parent


BASE_DIR = _find_base_dir(__file__)
CONFIG_DIR = os.path.join(BASE_DIR, "configs")
CONFIG_FILE = os.path.join(CONFIG_DIR, "pit_config.json")


def _ensure_dirs() -> None:
    os.makedirs(CONFIG_DIR, exist_ok=True)


# ==========================
# Foreground window (Windows)
# ==========================


def _get_foreground_window_title() -> str:
    if os.name != "nt":
        return ""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        buff = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buff, length + 1)
        return buff.value or ""
    except Exception:
        return ""


# ==========================
# Clipboard helper
# ==========================


def _set_clipboard_windows(text: str) -> bool:
    """Best-effort clipboard set for Windows.

    Uses the built-in `clip` tool. We send UTF-16LE because that's what clip expects.
    """

    if os.name != "nt":
        return False
    try:
        subprocess.run("clip", input=text.encode("utf-16le"), check=False)
        return True
    except Exception:
        return False


# ==========================
# Pit Strategy Manager
# ==========================


class PitStrategyManager:
    def __init__(self, main_app_ref=None):
        self.running: bool = False
        self.pneus_ativos: bool = True
        self.main_app = main_app_ref

        self._lock = threading.RLock()
        self._last_trigger_ts = 0.0

        # Default config (kept close to your original app)
        self.config = {
            "bind_code": "KEY:F6",
            "chat_key": "t",
            "open_delay": 0.02,
            "typing_interval": 0.0,
            "injection": "type",  # type | clipboard
            "debounce_ms": 150,
            "require_iracing_foreground": True,
            "iracing_window_substring": "iracing",
            # Toggle commands
            "cmd_when_on": "#-cleartires #ws",
            "cmd_when_off": "#cleartires #-ws",
            "enable_pygame": False,
        }

        self.load_config()

        self._joysticks = []

        self._hotkey_handle = None

    def _init_joysticks(self) -> None:
        if pygame is None:
            return

        try:
            pygame.init()
            pygame.joystick.init()
            self._joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
            for joystick in self._joysticks:
                joystick.init()
        except Exception:
            self._joysticks = []

    # ---------- config ----------

    def load_config(self) -> None:
        _ensure_dirs()
        if not os.path.exists(CONFIG_FILE):
            self.save_config()
            return
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self.config.update(data)
        except Exception:
            pass

        if "enable_pygame" not in self.config:
            self.config["enable_pygame"] = False

    def save_config(self) -> None:
        _ensure_dirs()
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    # ---------- safety guard ----------

    def is_iracing_focused(self) -> bool:
        if not bool(self.config.get("require_iracing_foreground", True)):
            return True
        title = _get_foreground_window_title().lower()
        needle = str(self.config.get("iracing_window_substring", "iracing") or "iracing").lower()
        return needle in title

    # ---------- injection ----------

    def refresh_injector(self) -> None:
        """Kept for GUI compatibility. Injection reads from self.config each send."""
        return

    def _send_text_type(self, text: str) -> None:
        interval = float(self.config.get("typing_interval", 0.0) or 0.0)
        keyboard.write(text, delay=max(0.0, interval))

    def _send_text_clipboard(self, text: str) -> None:
        ok = _set_clipboard_windows(text)
        if not ok:
            self._send_text_type(text)
            return
        keyboard.send("ctrl+v")

    def send_chat_command(self, text: str) -> bool:
        """Open iRacing chat, inject text, press Enter.

        Returns True if attempted, False if blocked.
        """

        if not text:
            return False
        if not self.is_iracing_focused():
            return False

        chat_key = str(self.config.get("chat_key", "t") or "t").strip() or "t"
        open_delay = float(self.config.get("open_delay", 0.02) or 0.02)
        injection = str(self.config.get("injection", "type") or "type").strip().lower()

        with self._lock:
            keyboard.send(chat_key)
            time.sleep(max(0.0, open_delay))

            if injection == "clipboard":
                self._send_text_clipboard(text)
                time.sleep(0.01)
            else:
                self._send_text_type(text)

            keyboard.send("enter")

        return True

    # ---------- turbo toggle ----------

    def executar_turbo(self) -> None:
        if not self.running:
            return

        now = time.time()
        debounce_ms = int(self.config.get("debounce_ms", 150) or 150)
        if (now - self._last_trigger_ts) * 1000.0 < debounce_ms:
            return
        self._last_trigger_ts = now

        self.pneus_ativos = not self.pneus_ativos

        cmd_on = str(self.config.get("cmd_when_on", "#-cleartires #ws") or "#-cleartires #ws")
        cmd_off = str(self.config.get("cmd_when_off", "#cleartires #-ws") or "#cleartires #-ws")
        cmd = cmd_on if self.pneus_ativos else cmd_off

        print(f">> PIT COMANDO: {cmd}")
        self.send_chat_command(cmd)

    # ---------- listeners ----------

    def start_listener(self) -> None:
        if self.running:
            return
        self.running = True

        # Keyboard hotkey
        bind = str(self.config.get("bind_code", "KEY:F6") or "KEY:F6")
        if bind.startswith("KEY:"):
            key = bind.split(":", 1)[1].strip().lower()
            try:
                self._hotkey_handle = keyboard.add_hotkey(key, self.executar_turbo)
            except Exception:
                self._hotkey_handle = None

        # Joystick loop (opt-in)
        if bool(self.config.get("enable_pygame", False)) and bind.startswith("JOY:"):
            self._init_joysticks()
            threading.Thread(target=self._joystick_loop, daemon=True).start()

    def stop_listener(self) -> None:
        self.running = False
        try:
            keyboard.unhook_all()
        except Exception:
            pass
        self._hotkey_handle = None

    def _joystick_loop(self) -> None:
        if pygame is None or not bool(self.config.get("enable_pygame", False)):
            return

        while self.running:
            if not bool(self.config.get("enable_pygame", False)):
                break
            bind = str(self.config.get("bind_code", ""))
            if bind.startswith("JOY:"):
                target = bind
                try:
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONDOWN:
                            if f"JOY:{event.joy}:{event.button}" == target:
                                self.executar_turbo()
                except Exception:
                    pass
            time.sleep(0.01)


# ==========================
# GUI helper: capture a key or joystick button
# ==========================


def capture_input_scan(timeout_s: float = 5.0, allow_joystick: bool = True) -> Optional[str]:
    """Capture a KEY:<name> or JOY:<joy>:<button> for binding."""

    found: Optional[str] = None

    def kh(e):
        nonlocal found
        try:
            if e.event_type == "down":
                found = f"KEY:{str(e.name).upper()}"
        except Exception:
            pass

    hk = keyboard.hook(kh)
    start = time.time()

    # Ensure pygame initialized for joystick capture
    if allow_joystick and pygame is not None:
        try:
            pygame.init()
            pygame.joystick.init()
        except Exception:
            pass

    try:
        while time.time() - start < timeout_s:
            if found:
                break
            if allow_joystick and pygame is not None:
                try:
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONDOWN:
                            found = f"JOY:{event.joy}:{event.button}"
                            break
                except Exception:
                    pass
            if found:
                break
            time.sleep(0.01)
    finally:
        try:
            keyboard.unhook(hk)
        except Exception:
            pass

    return found


__all__ = ["PitStrategyManager", "capture_input_scan"]
