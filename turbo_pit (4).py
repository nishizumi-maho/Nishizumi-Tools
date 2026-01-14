"""Turbo Pit helper for iRacing pit command broadcasts."""

from __future__ import annotations

import ctypes
import importlib.util
import json
import logging
import os
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

IRSDKSpec = importlib.util.find_spec("irsdk")
if IRSDKSpec:
    import irsdk
else:
    irsdk = None

KEYBOARD_SPEC = importlib.util.find_spec("keyboard")
if KEYBOARD_SPEC:
    import keyboard
else:
    keyboard = None

PYGAME_SPEC = importlib.util.find_spec("pygame")
if PYGAME_SPEC:
    import pygame
else:
    pygame = None

APP_NAME = "Turbo Pit"
APP_FOLDER = "TurboPit"
BASE_PATH = os.getenv("APPDATA") or os.path.expanduser("~")
CONFIG_FOLDER = os.path.join(BASE_PATH, APP_FOLDER)
CONFIG_FILE = os.path.join(CONFIG_FOLDER, "config.json")


@dataclass(frozen=True)
class PitAction:
    key: str
    label: str
    commands: Sequence[Tuple[str, int]]


COMMAND_CLEAR = "clear"
COMMAND_FUEL = "fuel"
COMMAND_LF = "lf"
COMMAND_RF = "rf"
COMMAND_LR = "lr"
COMMAND_RR = "rr"
COMMAND_WS = "ws"
COMMAND_FR = "fr"
COMMAND_CLEAR_TIRES = "clear_tires"
COMMAND_CLEAR_WS = "clear_ws"

TOGGLE_ACTION_KEY = "toggle_tires_windshield"

DEFAULT_ACTIONS: List[PitAction] = [
    PitAction(
        TOGGLE_ACTION_KEY,
        "Toggle tires + windshield",
        (
            (COMMAND_CLEAR_TIRES, 0),
            (COMMAND_CLEAR_WS, 0),
        ),
    )
]

DEFAULT_HOTKEYS = {
    TOGGLE_ACTION_KEY: "KEY:CTRL+SHIFT+G",
}


class TurboPit:
    def __init__(self) -> None:
        self._ir: Optional["irsdk.IRSDK"] = None
        self._last_attempt = 0.0

    def _sdk_ready(self) -> bool:
        return os.name == "nt" and irsdk is not None

    def connect(self) -> bool:
        if not self._sdk_ready():
            return False
        if self._ir is None:
            self._ir = irsdk.IRSDK()
        if self._ir.is_initialized:
            return True
        now = time.time()
        if now - self._last_attempt < 0.5:
            return False
        self._last_attempt = now
        try:
            return bool(self._ir.startup())
        except Exception:
            return False

    def is_connected(self) -> bool:
        return bool(self._ir and self._ir.is_initialized)

    def send_pit_command(self, command: str, value: int) -> bool:
        if not self.is_connected() and not self.connect():
            logging.error("iRacing is not connected. Start iRacing first.")
            return False
        resolved = self._resolve_pit_command(command)
        if resolved is None:
            logging.warning("Unknown pit command: %s.", command)
            return False
        try:
            assert self._ir is not None
            if hasattr(self._ir, "pit_command"):
                self._ir.pit_command(resolved, value)
            elif hasattr(self._ir, "broadcast_msg"):
                msg_id = self._resolve_broadcast_msg_id("pit_command")
                if msg_id is None:
                    logging.warning("Broadcast pit command is not available.")
                    return False
                self._ir.broadcast_msg(msg_id, resolved, value)
            else:
                logging.warning("Pit command broadcast is not supported by this SDK build.")
                return False
            logging.info("Sent pit command %s (value=%s).", command, value)
            return True
        except Exception:
            logging.exception("Failed to send pit command %s.", command)
            return False

    def read_var(self, name: str) -> Optional[object]:
        if not self.is_connected() and not self.connect():
            return None
        try:
            assert self._ir is not None
            return self._ir[name]
        except Exception:
            return None

    @staticmethod
    def _resolve_broadcast_msg_id(name: str) -> Optional[int]:
        if irsdk is None:
            return None
        broadcast = getattr(irsdk, "BroadcastMsg", None)
        if broadcast is None:
            return None
        if hasattr(broadcast, name):
            return getattr(broadcast, name)
        if isinstance(broadcast, dict):
            return broadcast.get(name)
        return None

    @staticmethod
    def _resolve_pit_command(command: str) -> Optional[int]:
        if irsdk is None:
            return None
        pit_enum = getattr(irsdk, "PitCommandMode", None)
        if pit_enum is None:
            pit_enum = getattr(irsdk, "PitCommand", None)
        if pit_enum is None:
            return None
        candidates = {
            command,
            command.upper(),
            command.lower(),
            command.replace(" ", ""),
            command.replace(" ", "_"),
            command.replace("-", "_"),
        }
        for name in candidates:
            if hasattr(pit_enum, name):
                return getattr(pit_enum, name)
        if isinstance(pit_enum, dict):
            for name in candidates:
                if name in pit_enum:
                    return pit_enum[name]
        return None


class InputManager:
    """
    Manage keyboard and joystick input.

    - Keyboard input uses the optional `keyboard` library.
    - Joystick input uses pygame when available.
    - Safe mode disables joystick polling (keyboard only).
    """

    def __init__(self, *, safe_mode: bool = False) -> None:
        self.safe_mode = safe_mode
        self.joysticks: List["pygame.joystick.Joystick"] = []
        self.listeners: Dict[str, Callable[[], None]] = {}
        self._input_thread: Optional[threading.Thread] = None
        self._running = False

        if self._pygame_available():
            self._init_pygame()
            self._start_input_loop()

    def _pygame_available(self) -> bool:
        return pygame is not None

    def _init_pygame(self) -> None:
        if not self._pygame_available():
            return
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        self.refresh_joysticks()

    def refresh_joysticks(self) -> None:
        self.joysticks.clear()
        if not self._pygame_available() or self.safe_mode:
            return
        try:
            if not pygame.get_init():
                pygame.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            for idx in range(pygame.joystick.get_count()):
                joy = pygame.joystick.Joystick(idx)
                if not joy.get_init():
                    joy.init()
                self.joysticks.append(joy)
        except Exception:
            self.joysticks.clear()

    def set_safe_mode(self, enabled: bool) -> None:
        self.safe_mode = enabled
        if enabled:
            self.joysticks.clear()
        else:
            self._init_pygame()
        if not self._running and self._pygame_available():
            self._start_input_loop()

    def register_listener(self, code: str, callback: Callable[[], None]) -> None:
        self.listeners[code] = callback

    def clear_listeners(self) -> None:
        self.listeners.clear()

    def _start_input_loop(self) -> None:
        if not self._pygame_available():
            return
        if self._input_thread and self._input_thread.is_alive():
            return
        self._running = True
        self._input_thread = threading.Thread(target=self._input_loop, daemon=True)
        self._input_thread.start()

    def _input_loop(self) -> None:
        while self._running:
            if not self.safe_mode and self._pygame_available() and pygame.get_init():
                try:
                    pygame.event.pump()
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONDOWN:
                            code = f"JOY:{event.joy}:{event.button}"
                            if code in self.listeners:
                                threading.Thread(
                                    target=self.listeners[code], daemon=True
                                ).start()
                except Exception:
                    pass
            time.sleep(0.01)

    def capture_any_input(self, timeout: float = 10.0) -> Optional[str]:
        captured_code: Optional[str] = None
        start = time.time()

        def key_hook(event: "keyboard.KeyboardEvent") -> None:
            nonlocal captured_code
            if event.event_type == "down":
                if event.name == "esc":
                    captured_code = "CANCEL"
                elif event.name:
                    captured_code = f"KEY:{event.name.upper()}"

        hook = None
        if keyboard is not None:
            hook = keyboard.hook(key_hook)

        try:
            while time.time() - start < timeout:
                if captured_code:
                    break

                if not self.safe_mode and self._pygame_available() and pygame.get_init():
                    try:
                        pygame.event.pump()
                        for joy in self.joysticks:
                            for b_idx in range(joy.get_numbuttons()):
                                if joy.get_button(b_idx):
                                    captured_code = f"JOY:{joy.get_id()}:{b_idx}"
                                    break
                            if captured_code:
                                break
                    except Exception:
                        pass

                if captured_code:
                    break

                time.sleep(0.02)
        finally:
            if hook is not None:
                keyboard.unhook(hook)

        return captured_code


class TurboPitApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.pit = TurboPit()
        self.input_manager = InputManager()
        self.actions: List[PitAction] = []
        self.hotkeys: Dict[str, str] = {}
        self.safe_mode = False
        self.auto_clear_on_pitroad = False
        self._keyboard_hotkeys: Dict[str, int] = {}
        self._tk_hotkeys: Dict[str, str] = {}
        self._win_hotkeys: Dict[str, int] = {}
        self._win_hotkey_callbacks: Dict[int, Callable[[], None]] = {}
        self._win_hotkey_next_id = 1
        self._win_hotkey_polling = False
        self._last_on_pit_road = False
        self._toggle_clear_next = True

        self._load_config()
        self._build_ui()
        self._apply_hotkeys()
        self._refresh_status()

    def _load_config(self) -> None:
        os.makedirs(CONFIG_FOLDER, exist_ok=True)
        config = self._default_config()
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                config.update({k: v for k, v in loaded.items() if isinstance(loaded, dict)})
            except Exception:
                logging.exception("Failed to load config, using defaults.")

        self.safe_mode = bool(config.get("safe_mode", False))
        self.auto_clear_on_pitroad = bool(config.get("auto_clear_on_pitroad", False))
        raw_actions = config.get("actions") or []
        parsed_actions = [action for action in self._parse_actions(raw_actions) if action.key == TOGGLE_ACTION_KEY]
        self.actions = parsed_actions or DEFAULT_ACTIONS
        self.hotkeys = {
            action.key: config.get("hotkeys", {}).get(action.key, DEFAULT_HOTKEYS.get(action.key, ""))
            for action in self.actions
        }
        self.input_manager.set_safe_mode(self.safe_mode)

    def _default_config(self) -> Dict[str, object]:
        return {
            "safe_mode": False,
            "auto_clear_on_pitroad": False,
            "actions": [
                {
                    "key": action.key,
                    "label": action.label,
                    "commands": [list(cmd) for cmd in action.commands],
                }
                for action in DEFAULT_ACTIONS
            ],
            "hotkeys": DEFAULT_HOTKEYS,
        }

    def _parse_actions(self, raw_actions: Iterable[object]) -> List[PitAction]:
        parsed: List[PitAction] = []
        for entry in raw_actions:
            if not isinstance(entry, dict):
                continue
            key = entry.get("key")
            label = entry.get("label")
            commands = entry.get("commands")
            if not key or not label or not isinstance(commands, list):
                continue
            cmd_pairs: List[Tuple[str, int]] = []
            for cmd in commands:
                if (
                    isinstance(cmd, (list, tuple))
                    and len(cmd) == 2
                    and isinstance(cmd[0], str)
                    and isinstance(cmd[1], int)
                ):
                    cmd_pairs.append((cmd[0], cmd[1]))
            if cmd_pairs:
                parsed.append(PitAction(key=key, label=label, commands=tuple(cmd_pairs)))
        return parsed

    def _save_config(self) -> None:
        payload = {
            "safe_mode": self.safe_mode,
            "auto_clear_on_pitroad": self.auto_clear_on_pitroad,
            "actions": [
                {
                    "key": action.key,
                    "label": action.label,
                    "commands": [list(cmd) for cmd in action.commands],
                }
                for action in self.actions
            ],
            "hotkeys": self.hotkeys,
        }
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            logging.exception("Failed to save config.")

    def _build_ui(self) -> None:
        self.root.geometry("640x420")
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.main_frame = ttk.Frame(notebook)
        self.doc_frame = ttk.Frame(notebook)
        notebook.add(self.main_frame, text="Pit Controls")
        notebook.add(self.doc_frame, text="Documentation")

        status_frame = ttk.Frame(self.main_frame)
        status_frame.pack(fill=tk.X, padx=12, pady=8)

        self.status_var = tk.StringVar(value="Disconnected")
        status_label = ttk.Label(status_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT)

        sdk_label = ttk.Label(
            status_frame,
            text="IRSDK available" if self.pit._sdk_ready() else "IRSDK not detected",
        )
        sdk_label.pack(side=tk.RIGHT)

        action_frame = ttk.LabelFrame(self.main_frame, text="Pit Action")
        action_frame.pack(fill=tk.X, expand=False, padx=12, pady=8)

        self.action_status_var = tk.StringVar(value="Toggle tires + windshield")
        action_label = ttk.Label(action_frame, textvariable=self.action_status_var)
        action_label.pack(anchor=tk.W, padx=8, pady=6)

        self.toggle_button = ttk.Button(
            action_frame,
            text="Toggle Tires + Windshield",
            command=self._send_toggle_action,
        )
        self.toggle_button.pack(anchor=tk.W, padx=8, pady=6)

        button_frame = ttk.Frame(self.main_frame)
        button_frame.pack(fill=tk.X, padx=12, pady=8)

        ttk.Button(button_frame, text="Assign Hotkey", command=self._assign_hotkey).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(button_frame, text="Clear Hotkey", command=self._clear_hotkey).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(button_frame, text="Open Config Folder", command=self._open_config).pack(
            side=tk.LEFT, padx=4
        )

        options_frame = ttk.LabelFrame(self.main_frame, text="Input Options")
        options_frame.pack(fill=tk.X, padx=12, pady=8)

        self.safe_mode_var = tk.BooleanVar(value=self.safe_mode)
        ttk.Checkbutton(
            options_frame,
            text="Keyboard-only mode (disable controllers)",
            variable=self.safe_mode_var,
            command=self._toggle_safe_mode,
        ).pack(side=tk.LEFT, padx=6, pady=4)

        self.auto_clear_var = tk.BooleanVar(value=self.auto_clear_on_pitroad)
        ttk.Checkbutton(
            options_frame,
            text="OnPitRoad (bool 1/1): auto-clear tires + windshield on pit road entry",
            variable=self.auto_clear_var,
            command=self._toggle_auto_clear_on_pitroad,
        ).pack(side=tk.LEFT, padx=6, pady=4)

        self.input_status_var = tk.StringVar(value=self._input_status_text())
        ttk.Label(options_frame, textvariable=self.input_status_var).pack(side=tk.RIGHT, padx=6)

        info_frame = ttk.LabelFrame(self.main_frame, text="Quick Help")
        info_frame.pack(fill=tk.X, padx=12, pady=8)

        info_label = ttk.Label(
            info_frame,
            text=(
                "Use the Toggle button or assign a hotkey. "
                "Hotkey capture supports keyboard and pygame controllers."
            ),
            wraplength=760,
            justify=tk.LEFT,
        )
        info_label.pack(anchor=tk.W, padx=8, pady=6)

        self._build_documentation()

    def _input_status_text(self) -> str:
        keyboard_status = "OK" if keyboard is not None else "Not installed"
        controller_status = (
            f"{len(self.input_manager.joysticks)} controller(s)" if pygame is not None else "pygame missing"
        )
        return f"Keyboard: {keyboard_status} | Controllers: {controller_status}"

    def _build_documentation(self) -> None:
        doc_text = tk.Text(self.doc_frame, wrap=tk.WORD)
        doc_text.pack(fill=tk.BOTH, expand=True)

        doc_content = self._documentation_text()
        doc_text.insert(tk.END, doc_content)
        doc_text.configure(state=tk.DISABLED)

    def _documentation_text(self) -> str:
        return f"""Turbo Pit - User Guide
========================================

Welcome to Turbo Pit. This UI sends pit commands to iRacing using the broadcast API.
Everything is customizable via the config file located here:

{CONFIG_FILE}

HOW TO CUSTOMIZE ACTIONS
------------------------
The app has a single toggle action:
- Press once to clear (uncheck) tires + windshield.
- Press again to re-enable tires + windshield.

SUPPORTED PIT COMMAND CODES
---------------------------
Turbo Pit uses these commands internally:
- {COMMAND_CLEAR}  -> Clear all pit options
- {COMMAND_FUEL}   -> Add fuel (value is liters)
- {COMMAND_LF}     -> Change left front tire (value 0)
- {COMMAND_RF}     -> Change right front tire (value 0)
- {COMMAND_LR}     -> Change left rear tire (value 0)
- {COMMAND_RR}     -> Change right rear tire (value 0)
- {COMMAND_WS}     -> Clean windshield / tearoff (value 0)
- {COMMAND_FR}     -> Fast repair (value 0)
- {COMMAND_CLEAR_TIRES} -> Uncheck tire changes
- {COMMAND_CLEAR_WS} -> Uncheck clean windshield

HOTKEY FORMATS
--------------
Hotkeys live in the "hotkeys" section of the config file or can be assigned in the UI.
Use one of these formats:
- Keyboard: KEY:CTRL+SHIFT+F
- Keyboard: KEY:ALT+1
- Controller: JOY:<device_id>:<button>

The Assign Hotkey button waits for input from:
- Keyboard (via the "keyboard" Python library)
- Controllers (via pygame/SDL)
On Windows, global hotkeys are registered so the trigger works even when the app
does not have focus. If the keyboard library cannot register a hotkey, the app
falls back to the Windows global hotkey API.

KEYBOARD-ONLY MODE
------------------
Toggle "Keyboard-only mode" to stop using controller input. This is useful if
pygame is installed but you do not want the app to capture joystick buttons.

AUTO-CLEAR ON PIT ROAD
----------------------
Enable "OnPitRoad (bool 1/1)" to automatically clear tires + windshield when the
player car enters pit road (between the cones).

TOGGLE TIRES + WINDSHIELD
-------------------------
The "Toggle tires + windshield" action clears tire changes + windshield on the first
press, then re-enables them on the next press.

TIPS
----
- If a hotkey does not trigger, verify the format and that the dependency
  (keyboard or pygame) is installed.
- The built-in Tkinter hotkey binding works while the app is focused.
"""

    def _refresh_status(self) -> None:
        connected = self.pit.is_connected() or self.pit.connect()
        self.status_var.set("Connected" if connected else "Disconnected")
        self.input_status_var.set(self._input_status_text())
        self._maybe_auto_clear_on_pitroad()
        self.root.after(1000, self._refresh_status)

    def _send_action(self, action: PitAction) -> None:
        if action.key == TOGGLE_ACTION_KEY:
            self._toggle_tires_and_windshield()
            return
        for command, value in action.commands:
            self.pit.send_pit_command(command, value)

    def _assign_hotkey(self) -> None:
        if keyboard is None and pygame is None:
            messagebox.showwarning(
                APP_NAME,
                "Neither keyboard nor pygame is installed. Hotkey capture is unavailable.",
            )
            return
        action = self._toggle_action()
        if action is None:
            return

        capture_window = tk.Toplevel(self.root)
        capture_window.title("Assign Hotkey")
        capture_window.geometry("420x140")
        capture_window.transient(self.root)
        ttk.Label(
            capture_window,
            text="Press a key or controller button (ESC to cancel)...",
        ).pack(pady=16)
        status_var = tk.StringVar(value="Waiting for input...")
        ttk.Label(capture_window, textvariable=status_var).pack()

        def capture() -> None:
            code = self.input_manager.capture_any_input(timeout=10.0)
            self.root.after(0, lambda: self._finish_capture(capture_window, status_var, action, code))

        threading.Thread(target=capture, daemon=True).start()

    def _finish_capture(
        self,
        window: tk.Toplevel,
        status_var: tk.StringVar,
        action: PitAction,
        code: Optional[str],
    ) -> None:
        if code in {None, "CANCEL"}:
            status_var.set("Capture canceled or timed out.")
            window.after(800, window.destroy)
            return
        self.hotkeys[action.key] = code
        self._apply_hotkeys()
        self._save_config()
        status_var.set(f"Assigned: {code}")
        window.after(800, window.destroy)

    def _clear_hotkey(self) -> None:
        action = self._toggle_action()
        if action is None:
            return
        self.hotkeys[action.key] = ""
        self._apply_hotkeys()
        self._save_config()

    def _reload_config(self) -> None:
        self._load_config()
        self._apply_hotkeys()
        self.safe_mode_var.set(self.safe_mode)
        self.auto_clear_var.set(self.auto_clear_on_pitroad)
        messagebox.showinfo(APP_NAME, "Configuration reloaded.")

    def _toggle_safe_mode(self) -> None:
        self.safe_mode = bool(self.safe_mode_var.get())
        self.input_manager.set_safe_mode(self.safe_mode)
        self._save_config()
        self.input_status_var.set(self._input_status_text())

    def _toggle_auto_clear_on_pitroad(self) -> None:
        self.auto_clear_on_pitroad = bool(self.auto_clear_var.get())
        current = self._read_on_pit_road()
        if current is not None:
            self._last_on_pit_road = current
        self._save_config()

    def _open_config(self) -> None:
        if os.name == "nt":
            os.startfile(CONFIG_FOLDER)
            return
        messagebox.showinfo(APP_NAME, f"Config folder: {CONFIG_FOLDER}")

    def _find_action(self, key: str) -> Optional[PitAction]:
        for action in self.actions:
            if action.key == key:
                return action
        return None

    def _toggle_action(self) -> Optional[PitAction]:
        return self._find_action(TOGGLE_ACTION_KEY)

    def _apply_hotkeys(self) -> None:
        for handle in self._keyboard_hotkeys.values():
            if keyboard is not None:
                keyboard.remove_hotkey(handle)
        self._keyboard_hotkeys.clear()
        self.input_manager.clear_listeners()
        for seq in self._tk_hotkeys.values():
            self.root.unbind_all(seq)
        self._tk_hotkeys.clear()
        self._clear_windows_hotkeys()

        for action in self.actions:
            binding = self.hotkeys.get(action.key, "")
            if not binding:
                continue
            if binding.startswith("KEY:"):
                key_combo = binding.replace("KEY:", "").strip()
                if not key_combo:
                    continue
                handle = None
                if keyboard is not None:
                    try:
                        handle = keyboard.add_hotkey(
                            key_combo.lower(),
                            lambda action=action: self._send_action(action),
                        )
                    except Exception:
                        logging.exception("Failed to register keyboard hotkey: %s", key_combo)
                if handle is not None:
                    self._keyboard_hotkeys[action.key] = handle
                elif self._register_windows_hotkey(
                    key_combo, lambda action=action: self._send_action(action)
                ):
                    self._win_hotkeys[action.key] = self._win_hotkey_next_id - 1
                tk_binding = self._tk_binding_for_combo(key_combo)
                if tk_binding:
                    self.root.bind_all(
                        tk_binding,
                        lambda _event, action=action: self._send_action(action),
                    )
                    self._tk_hotkeys[action.key] = tk_binding
            elif binding.startswith("JOY:"):
                self.input_manager.register_listener(
                    binding,
                    lambda action=action: self._send_action(action),
                )

    def _register_windows_hotkey(self, combo: str, callback: Callable[[], None]) -> bool:
        if os.name != "nt":
            return False
        parsed = self._parse_windows_hotkey(combo)
        if parsed is None:
            return False
        modifiers, key_code = parsed
        hwnd = self.root.winfo_id()
        hotkey_id = self._win_hotkey_next_id
        self._win_hotkey_next_id += 1
        user32 = ctypes.windll.user32
        if not user32.RegisterHotKey(hwnd, hotkey_id, modifiers, key_code):
            logging.warning("Unable to register Windows hotkey: %s", combo)
            return False
        self._win_hotkey_callbacks[hotkey_id] = callback
        self._start_windows_hotkey_polling()
        return True

    def _start_windows_hotkey_polling(self) -> None:
        if os.name != "nt" or self._win_hotkey_polling:
            return
        self._win_hotkey_polling = True
        self.root.after(50, self._poll_windows_hotkeys)

    def _poll_windows_hotkeys(self) -> None:
        if os.name != "nt":
            return
        user32 = ctypes.windll.user32
        msg = ctypes.wintypes.MSG()
        wm_hotkey = 0x0312
        pm_remove = 0x0001
        hwnd = self.root.winfo_id()
        while user32.PeekMessageW(ctypes.byref(msg), hwnd, wm_hotkey, wm_hotkey, pm_remove):
            callback = self._win_hotkey_callbacks.get(msg.wParam)
            if callback:
                threading.Thread(target=callback, daemon=True).start()
        if self._win_hotkey_callbacks:
            self.root.after(50, self._poll_windows_hotkeys)
        else:
            self._win_hotkey_polling = False

    def _clear_windows_hotkeys(self) -> None:
        if os.name != "nt":
            return
        if not self._win_hotkey_callbacks:
            self._win_hotkey_polling = False
            return
        user32 = ctypes.windll.user32
        hwnd = self.root.winfo_id()
        for hotkey_id in list(self._win_hotkey_callbacks.keys()):
            user32.UnregisterHotKey(hwnd, hotkey_id)
        self._win_hotkey_callbacks.clear()
        self._win_hotkeys.clear()
        self._win_hotkey_polling = False

    def _parse_windows_hotkey(self, combo: str) -> Optional[Tuple[int, int]]:
        parts = [part.strip().upper() for part in combo.split("+") if part.strip()]
        if not parts:
            return None
        modifier_map = {
            "CTRL": 0x0002,
            "CONTROL": 0x0002,
            "SHIFT": 0x0004,
            "ALT": 0x0001,
            "WIN": 0x0008,
            "WINDOWS": 0x0008,
        }
        modifiers = 0
        for mod in parts[:-1]:
            if mod in modifier_map:
                modifiers |= modifier_map[mod]
            else:
                return None
        key = parts[-1]
        special_keys = {
            "SPACE": 0x20,
            "ENTER": 0x0D,
            "RETURN": 0x0D,
            "TAB": 0x09,
            "ESC": 0x1B,
            "ESCAPE": 0x1B,
            "BACKSPACE": 0x08,
            "DELETE": 0x2E,
            "UP": 0x26,
            "DOWN": 0x28,
            "LEFT": 0x25,
            "RIGHT": 0x27,
        }
        if key in special_keys:
            key_code = special_keys[key]
        elif len(key) == 1:
            key_code = ord(key)
        elif key.startswith("F") and key[1:].isdigit():
            f_number = int(key[1:])
            if 1 <= f_number <= 24:
                key_code = 0x70 + (f_number - 1)
            else:
                return None
        else:
            return None
        return modifiers, key_code

    def _tk_binding_for_combo(self, combo: str) -> Optional[str]:
        parts = [part.strip().upper() for part in combo.split("+") if part.strip()]
        if not parts:
            return None
        modifiers_map = {
            "CTRL": "Control",
            "CONTROL": "Control",
            "SHIFT": "Shift",
            "ALT": "Alt",
        }
        modifiers = [modifiers_map[p] for p in parts[:-1] if p in modifiers_map]
        key = parts[-1]
        special_keys = {
            "SPACE": "space",
            "ENTER": "Return",
            "RETURN": "Return",
            "TAB": "Tab",
            "ESC": "Escape",
            "ESCAPE": "Escape",
            "BACKSPACE": "BackSpace",
            "DELETE": "Delete",
            "UP": "Up",
            "DOWN": "Down",
            "LEFT": "Left",
            "RIGHT": "Right",
        }
        if key in special_keys:
            key_name = special_keys[key]
        elif len(key) == 1:
            key_name = key.lower()
        elif key.startswith("F") and key[1:].isdigit():
            key_name = key
        else:
            return None
        pieces = modifiers + [key_name]
        return f"<{'-'.join(pieces)}>"

    def _send_toggle_action(self) -> None:
        action = self._toggle_action()
        if action is None:
            return
        self._send_action(action)

    def _toggle_tires_and_windshield(self) -> None:
        if self._toggle_clear_next:
            self._clear_tires_and_windshield()
        else:
            self._change_tires_and_windshield()

    def _clear_tires_and_windshield(self) -> None:
        self.pit.send_pit_command(COMMAND_CLEAR_TIRES, 0)
        self.pit.send_pit_command(COMMAND_CLEAR_WS, 0)
        self._toggle_clear_next = False

    def _change_tires_and_windshield(self) -> None:
        for command in (COMMAND_LF, COMMAND_RF, COMMAND_LR, COMMAND_RR, COMMAND_WS):
            self.pit.send_pit_command(command, 0)
        self._toggle_clear_next = True

    def _read_on_pit_road(self) -> Optional[bool]:
        value = self.pit.read_var("OnPitRoad")
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    def _maybe_auto_clear_on_pitroad(self) -> None:
        if not self.auto_clear_on_pitroad:
            self._last_on_pit_road = False
            return
        on_pit_road = self._read_on_pit_road()
        if on_pit_road is None:
            return
        if on_pit_road and not self._last_on_pit_road:
            self._clear_tires_and_windshield()
        self._last_on_pit_road = on_pit_road


def run_cli() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pit = TurboPit()

    if not pit._sdk_ready():
        logging.error("IRSDK is unavailable. This script requires Windows + irsdk.")
        return

    while True:
        status = "Connected" if pit.is_connected() or pit.connect() else "Disconnected"
        print(f"\n{APP_NAME} - {status}")
        print("=" * (len(APP_NAME) + len(status) + 3))
        print("Preset pit actions (pit broadcast only):")
        for idx, action in enumerate(DEFAULT_ACTIONS, start=1):
            print(f"{idx:2d}. {action.label}")
        print(" q. Quit")

        raw = input("\nSelect action: ").strip().lower()
        if raw in {"q", "quit", "exit"}:
            break
        if raw.isdigit() and 1 <= int(raw) <= len(DEFAULT_ACTIONS):
            action = DEFAULT_ACTIONS[int(raw) - 1]
            for command, value in action.commands:
                pit.send_pit_command(command, value)
        else:
            print("Invalid selection. Try again.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if "--cli" in sys.argv:
        run_cli()
        return

    root = tk.Tk()
    app = TurboPitApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
