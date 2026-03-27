#!/usr/bin/env python3
"""Nishizumi Tools launcher menu.

Designed to work both as a normal Python script and as a PyInstaller/auto-py-to-exe
binary. The launcher scans an `apps` folder placed next to the launcher executable
(or next to this script), then lets the user start and stop apps from a simple menu.
"""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox
from typing import Dict, List

SUPPORTED_EXTENSIONS = {".py", ".pyw", ".exe", ".bat", ".cmd"}


@dataclass(frozen=True)
class AppEntry:
    name: str
    path: Path


class LauncherApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Nishizumi Tools Launcher")
        self.root.geometry("640x420")
        self.root.minsize(560, 360)

        self.apps_dir = self._resolve_apps_dir()
        self.entries: List[AppEntry] = []
        self.processes: Dict[str, subprocess.Popen] = {}

        self.status_var = tk.StringVar(value="Ready")
        self.apps_dir_var = tk.StringVar(value=f"Apps folder: {self.apps_dir}")

        self._build_ui()
        self.refresh_apps()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    @staticmethod
    def _base_dir() -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent

    def _resolve_apps_dir(self) -> Path:
        return self._base_dir() / "apps"

    @staticmethod
    def _is_runnable(path: Path) -> bool:
        if not path.is_file():
            return False
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=(12, 8))

        tk.Label(top, textvariable=self.apps_dir_var, anchor="w").pack(fill="x")

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=(0, 8))

        tk.Button(controls, text="Refresh", width=12, command=self.refresh_apps).pack(side="left", padx=(0, 6))
        tk.Button(controls, text="Open Selected", width=14, command=self.open_selected).pack(side="left", padx=3)
        tk.Button(controls, text="Close Selected", width=14, command=self.close_selected).pack(side="left", padx=3)
        tk.Button(controls, text="Close All", width=12, command=self.close_all).pack(side="left", padx=(6, 0))

        list_frame = tk.Frame(self.root)
        list_frame.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.listbox = tk.Listbox(list_frame, activestyle="none")
        self.listbox.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)

        bottom = tk.Frame(self.root)
        bottom.pack(fill="x", padx=12, pady=(0, 12))
        tk.Label(bottom, textvariable=self.status_var, anchor="w").pack(fill="x")

    def refresh_apps(self) -> None:
        self.entries.clear()
        self.listbox.delete(0, tk.END)

        if not self.apps_dir.exists():
            self.status_var.set(f"Apps folder does not exist: {self.apps_dir}")
            return

        launcher_stem = (Path(sys.executable).stem if getattr(sys, "frozen", False) else Path(__file__).stem).lower()

        for path in sorted(self.apps_dir.iterdir(), key=lambda p: p.name.lower()):
            if not self._is_runnable(path):
                continue
            if path.stem.lower() == launcher_stem:
                continue

            entry = AppEntry(name=path.name, path=path)
            self.entries.append(entry)
            self.listbox.insert(tk.END, self._display_name(entry.name))

        if not self.entries:
            self.status_var.set("No runnable apps found in apps folder.")
        else:
            self.status_var.set(f"Found {len(self.entries)} app(s).")

    def _display_name(self, name: str) -> str:
        if name in self.processes and self._is_running(self.processes[name]):
            return f"[RUNNING] {name}"
        return name

    @staticmethod
    def _is_running(proc: subprocess.Popen) -> bool:
        return proc.poll() is None

    def _refresh_row(self, index: int) -> None:
        entry = self.entries[index]
        self.listbox.delete(index)
        self.listbox.insert(index, self._display_name(entry.name))

    def _selected_index(self) -> int | None:
        selected = self.listbox.curselection()
        if not selected:
            return None
        return int(selected[0])

    def open_selected(self) -> None:
        index = self._selected_index()
        if index is None:
            self.status_var.set("Select an app first.")
            return

        entry = self.entries[index]
        if entry.name in self.processes and self._is_running(self.processes[entry.name]):
            self.status_var.set(f"{entry.name} is already running.")
            return

        try:
            proc = self._launch(entry.path)
            self.processes[entry.name] = proc
            self.status_var.set(f"Opened {entry.name}.")
            self._refresh_row(index)
        except Exception as exc:
            self.status_var.set(f"Failed to open {entry.name}.")
            messagebox.showerror("Launch error", f"Could not open {entry.name}.\n\n{exc}")

    def _launch(self, app_path: Path) -> subprocess.Popen:
        suffix = app_path.suffix.lower()

        if suffix in {".py", ".pyw"}:
            if getattr(sys, "frozen", False):
                # When launcher is compiled, rely on file associations for .py/.pyw.
                cmd = [str(app_path)]
                return subprocess.Popen(cmd, cwd=str(app_path.parent), shell=True)
            cmd = [sys.executable, str(app_path)]
            return subprocess.Popen(cmd, cwd=str(app_path.parent))

        return subprocess.Popen([str(app_path)], cwd=str(app_path.parent), shell=False)

    def close_selected(self) -> None:
        index = self._selected_index()
        if index is None:
            self.status_var.set("Select an app first.")
            return

        entry = self.entries[index]
        proc = self.processes.get(entry.name)

        if proc is None or not self._is_running(proc):
            self.status_var.set(f"{entry.name} is not running.")
            self.processes.pop(entry.name, None)
            self._refresh_row(index)
            return

        self._stop_process(proc)
        self.processes.pop(entry.name, None)
        self.status_var.set(f"Closed {entry.name}.")
        self._refresh_row(index)

    def _stop_process(self, proc: subprocess.Popen) -> None:
        if not self._is_running(proc):
            return
        proc.terminate()
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            proc.kill()

    def close_all(self) -> None:
        closed = 0
        for name, proc in list(self.processes.items()):
            if self._is_running(proc):
                self._stop_process(proc)
                closed += 1
            self.processes.pop(name, None)

        for idx in range(len(self.entries)):
            self._refresh_row(idx)

        self.status_var.set(f"Closed {closed} app(s).")

    def on_close(self) -> None:
        running = [name for name, proc in self.processes.items() if self._is_running(proc)]
        if running:
            answer = messagebox.askyesno(
                "Close running apps?",
                "There are running apps launched from this menu.\n"
                "Do you want to close all of them before exiting?",
            )
            if answer:
                self.close_all()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    LauncherApp().run()
