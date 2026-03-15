#!/usr/bin/env python3
"""Launcher menu for Nishizumi overlay tools."""

from __future__ import annotations

import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox


@dataclass(frozen=True)
class AppEntry:
    """Represents one launchable Nishizumi app."""

    key: str
    display_name: str
    script_name: str
    description: str


class AppLauncher:
    """Resolves and starts apps from source or frozen builds."""

    def __init__(self) -> None:
        if getattr(sys, "frozen", False):
            self.base_dir = Path(sys.executable).resolve().parent
        else:
            self.base_dir = Path(__file__).resolve().parent

    def launch(self, app: AppEntry) -> subprocess.Popen:
        command = self._build_command(app)
        return subprocess.Popen(command, cwd=self.base_dir)

    def _build_command(self, app: AppEntry) -> list[str]:
        # Frozen build: try sibling executable first (e.g., Nishizumi_Fuel.exe),
        # then fallback to script names in case all scripts are distributed together.
        if getattr(sys, "frozen", False):
            executable = self.base_dir / self._executable_name(app.script_name)
            if executable.exists():
                return [str(executable)]

        script_path = self.base_dir / app.script_name
        if script_path.exists():
            return [sys.executable, str(script_path)]

        raise FileNotFoundError(
            f"Could not find '{app.script_name}' or an executable equivalent in {self.base_dir}."
        )

    @staticmethod
    def _executable_name(script_name: str) -> str:
        return f"{Path(script_name).stem}.exe" if sys.platform.startswith("win") else Path(script_name).stem


class MenuApp:
    APPS: tuple[AppEntry, ...] = (
        AppEntry(
            key="fuel",
            display_name="Nishizumi Fuel",
            script_name="Nishizumi_Fuel.py",
            description="Fuel consumption and remaining-laps overlay.",
        ),
        AppEntry(
            key="pit",
            display_name="Nishizumi PitTime",
            script_name="Nishizumi_PitTime.py",
            description="Pit-stop time loss and rejoin window estimator.",
        ),
        AppEntry(
            key="tirewear",
            display_name="Nishizumi TireWear",
            script_name="Nishizumi_TireWear.py",
            description="Adaptive tire wear model and overlay.",
        ),
        AppEntry(
            key="traction",
            display_name="Nishizumi Traction",
            script_name="Nishizumi_Traction.py",
            description="Traction-circle coaching overlay.",
        ),
    )

    def __init__(self) -> None:
        self.launcher = AppLauncher()
        self.processes: dict[str, subprocess.Popen] = {}

        self.root = tk.Tk()
        self.root.title("Nishizumi Tools Menu")
        self.root.geometry("520x300")
        self.root.minsize(520, 300)
        self.root.configure(bg="#111827")

        self.status_var = tk.StringVar(value="Ready. Pick an app to start.")
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        title = tk.Label(
            self.root,
            text="Nishizumi Tools Launcher",
            font=("Segoe UI", 16, "bold"),
            fg="#f9fafb",
            bg="#111827",
        )
        title.pack(anchor="w", padx=20, pady=(18, 8))

        subtitle = tk.Label(
            self.root,
            text="Start overlays from one place. You can run multiple tools at the same time.",
            font=("Segoe UI", 10),
            fg="#d1d5db",
            bg="#111827",
        )
        subtitle.pack(anchor="w", padx=20, pady=(0, 14))

        app_frame = tk.Frame(self.root, bg="#111827")
        app_frame.pack(fill="both", expand=True, padx=20)

        for app in self.APPS:
            row = tk.Frame(app_frame, bg="#1f2937")
            row.pack(fill="x", pady=5)

            text_frame = tk.Frame(row, bg="#1f2937")
            text_frame.pack(side="left", fill="x", expand=True, padx=12, pady=10)

            tk.Label(
                text_frame,
                text=app.display_name,
                font=("Segoe UI", 11, "bold"),
                fg="#f3f4f6",
                bg="#1f2937",
            ).pack(anchor="w")

            tk.Label(
                text_frame,
                text=app.description,
                font=("Segoe UI", 9),
                fg="#9ca3af",
                bg="#1f2937",
            ).pack(anchor="w")

            tk.Button(
                row,
                text="Start",
                command=lambda selected=app: self._start_app(selected),
                font=("Segoe UI", 10, "bold"),
                bg="#2563eb",
                fg="white",
                activebackground="#1d4ed8",
                activeforeground="white",
                relief="flat",
                padx=14,
                pady=8,
                takefocus=False,
                cursor="hand2",
            ).pack(side="right", padx=12)

        status = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            fg="#e5e7eb",
            bg="#111827",
            anchor="w",
            justify="left",
        )
        status.pack(fill="x", padx=20, pady=(8, 16))

    def _start_app(self, app: AppEntry) -> None:
        existing = self.processes.get(app.key)
        if existing and existing.poll() is None:
            self.status_var.set(f"{app.display_name} is already running.")
            return

        try:
            process = self.launcher.launch(app)
        except Exception as exc:
            messagebox.showerror("Launch failed", f"Could not start {app.display_name}.\n\n{exc}")
            self.status_var.set(f"Failed to start {app.display_name}.")
            return

        self.processes[app.key] = process
        self.status_var.set(f"Started {app.display_name} (PID {process.pid}).")

    def _on_close(self) -> None:
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    app = MenuApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
