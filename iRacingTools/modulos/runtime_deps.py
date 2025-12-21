"""Runtime dependency helpers for the iRacing tools suite."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tkinter.messagebox


def ensure_customtkinter() -> None:
    """Install ``customtkinter`` at runtime if it is missing.

    The packaged launcher sometimes runs on systems where ``customtkinter``
    is not pre-installed. To keep the UI usable without manual setup, this
    helper checks for the dependency and attempts to install it via ``pip``
    using the current Python executable. If installation fails, an error is
    shown so the user knows how to resolve the issue.
    """

    if importlib.util.find_spec("customtkinter") is not None:
        return

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])
    except Exception as exc:  # pragma: no cover - best effort warning UI
        try:
            tkinter.messagebox.showerror(
                "Dependência ausente",
                "Não foi possível instalar automaticamente o customtkinter.\n"
                "Instale manualmente com:\n"
                "pip install customtkinter\n\n"
                f"Detalhes: {exc}",
            )
        except Exception:
            pass
        raise

