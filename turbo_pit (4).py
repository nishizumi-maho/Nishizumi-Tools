"""Turbo Pit helper for iRacing pit command broadcasts."""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

IRSDKSpec = importlib.util.find_spec("irsdk")
if IRSDKSpec:
    import irsdk
else:
    irsdk = None

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


class TurboPitApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_NAME)
        self.pit = TurboPit()
        self.actions: List[PitAction] = []
        self.auto_clear_on_pitroad = False
        self._last_on_pit_road = False
        self._toggle_clear_next = True

        self._load_config()
        self._build_ui()
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

        self.auto_clear_on_pitroad = bool(config.get("auto_clear_on_pitroad", False))
        raw_actions = config.get("actions") or []
        parsed_actions = [action for action in self._parse_actions(raw_actions) if action.key == TOGGLE_ACTION_KEY]
        self.actions = parsed_actions or DEFAULT_ACTIONS

    def _default_config(self) -> Dict[str, object]:
        return {
            "auto_clear_on_pitroad": False,
            "actions": [
                {
                    "key": action.key,
                    "label": action.label,
                    "commands": [list(cmd) for cmd in action.commands],
                }
                for action in DEFAULT_ACTIONS
            ],
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
            "auto_clear_on_pitroad": self.auto_clear_on_pitroad,
            "actions": [
                {
                    "key": action.key,
                    "label": action.label,
                    "commands": [list(cmd) for cmd in action.commands],
                }
                for action in self.actions
            ],
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

        ttk.Button(button_frame, text="Open Config Folder", command=self._open_config).pack(
            side=tk.LEFT, padx=4
        )

        options_frame = ttk.LabelFrame(self.main_frame, text="Options")
        options_frame.pack(fill=tk.X, padx=12, pady=8)

        self.auto_clear_var = tk.BooleanVar(value=self.auto_clear_on_pitroad)
        ttk.Checkbutton(
            options_frame,
            text="OnPitRoad (bool 1/1): auto-clear tires + windshield on pit road entry",
            variable=self.auto_clear_var,
            command=self._toggle_auto_clear_on_pitroad,
        ).pack(side=tk.LEFT, padx=6, pady=4)

        info_frame = ttk.LabelFrame(self.main_frame, text="Quick Help")
        info_frame.pack(fill=tk.X, padx=12, pady=8)

        info_label = ttk.Label(
            info_frame,
            text=(
                "Use the Toggle button to quickly clear or re-enable tires + windshield."
            ),
            wraplength=760,
            justify=tk.LEFT,
        )
        info_label.pack(anchor=tk.W, padx=8, pady=6)

        self._build_documentation()

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

HOTKEYS
-------
Turbo Pit no longer uses hotkeys or controller input. Use the on-screen Toggle
button to activate tire + windshield changes.

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
- Use the Toggle button to quickly clear or re-enable tire and windshield service.
"""

    def _refresh_status(self) -> None:
        connected = self.pit.is_connected() or self.pit.connect()
        self.status_var.set("Connected" if connected else "Disconnected")
        self._maybe_auto_clear_on_pitroad()
        self.root.after(1000, self._refresh_status)

    def _send_action(self, action: PitAction) -> None:
        if action.key == TOGGLE_ACTION_KEY:
            self._toggle_tires_and_windshield()
            return
        for command, value in action.commands:
            self.pit.send_pit_command(command, value)

    def _reload_config(self) -> None:
        self._load_config()
        self.auto_clear_var.set(self.auto_clear_on_pitroad)
        messagebox.showinfo(APP_NAME, "Configuration reloaded.")

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
