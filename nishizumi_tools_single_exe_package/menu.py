#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PySide6 import QtCore, QtGui, QtWidgets, QtNetwork

APP_TITLE = "Nishizumi Tools"
APP_SUBTITLE = "Default launcher for the overlay collection"
APP_VERSION = "v9"
APP_DIR_NAME = "NishizumiTools"
MENU_STATE_FILE = "menu_state.json"
APP_ICON_FILE_PNG = "nishizumi_tools_icon.png"
APP_ICON_FILE_ICO = "nishizumi_tools_icon.ico"
GITHUB_RELEASE_OWNER = "nishizumi-maho"
GITHUB_RELEASE_REPO = "Nishizumi-Tools"
GITHUB_RELEASES_API_LATEST = (
    f"https://api.github.com/repos/{GITHUB_RELEASE_OWNER}/{GITHUB_RELEASE_REPO}/releases/latest"
)
GITHUB_RELEASES_PAGE_LATEST = (
    f"https://github.com/{GITHUB_RELEASE_OWNER}/{GITHUB_RELEASE_REPO}/releases/latest"
)
GITHUB_API_VERSION = "2022-11-28"
GITHUB_UPDATE_CHECK_INTERVAL_SECONDS = 6 * 60 * 60
LAUNCHER_INSTANCE_NAME = "NishizumiToolsLauncher"


@dataclass(frozen=True)
class AppDefinition:
    key: str
    title: str
    subtitle: str
    module: str


APPS = [
    AppDefinition(
        key="fuel",
        title="FuelMonitor",
        subtitle="Fuel usage, stint projection, target delta, and race-smart hints.",
        module="Nishizumi_FuelMonitor",
    ),
    AppDefinition(
        key="pit",
        title="Pit Calibrator",
        subtitle="Live total, service, base, fuel, and manual tire timing during the stop.",
        module="nishizumi_pitcalibrator",
    ),
    AppDefinition(
        key="tire",
        title="TireWear",
        subtitle="Learned tire degradation model with saved data per car and track.",
        module="Nishizumi_TireWear",
    ),
    AppDefinition(
        key="traction",
        title="Traction",
        subtitle="Grip-usage coaching overlay with live circle and optional IBT reference.",
        module="Nishizumi_Traction",
    ),
    AppDefinition(
        key="caution",
        title="Caution Overlay",
        subtitle="Full course cautions and the laps run under them, in a resizable always-on-top window.",
        module="Nishizumi_CautionOverlay",
    ),
]
APP_MAP = {app.key: app for app in APPS}


def launcher_instance_name() -> str:
    return LAUNCHER_INSTANCE_NAME


class SingleInstanceGuard(QtCore.QObject):
    activation_requested = QtCore.Signal()

    def __init__(self, name: str, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.name = name
        self.server: Optional[QtNetwork.QLocalServer] = None

    def try_activate_existing(self) -> bool:
        socket = QtNetwork.QLocalSocket(self)
        socket.connectToServer(self.name)
        if socket.waitForConnected(400):
            try:
                socket.write(b"ACTIVATE\n")
                socket.flush()
                socket.waitForBytesWritten(200)
            finally:
                socket.disconnectFromServer()
            return True
        return False

    def start_listening(self) -> bool:
        QtNetwork.QLocalServer.removeServer(self.name)
        self.server = QtNetwork.QLocalServer(self)
        self.server.newConnection.connect(self._handle_connection)
        return self.server.listen(self.name)

    def _handle_connection(self) -> None:
        if self.server is None:
            return
        while self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            if socket is None:
                continue
            socket.waitForReadyRead(200)
            try:
                _payload = bytes(socket.readAll()).decode("utf-8", errors="ignore")
            except Exception:
                _payload = ""
            socket.disconnectFromServer()
            self.activation_requested.emit()


def appdata_dir() -> Path:
    root = Path(os.getenv("APPDATA") or (Path.home() / ".config"))
    path = root / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_path() -> Path:
    return appdata_dir() / MENU_STATE_FILE


def runtime_resource_dir() -> Path:
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent


def launcher_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def app_icon_path() -> Optional[Path]:
    for name in (APP_ICON_FILE_PNG, APP_ICON_FILE_ICO):
        path = runtime_resource_dir() / name
        if path.exists():
            return path
    return None


def open_in_file_manager(path: Path) -> None:
    path = path.resolve()
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        QtWidgets.QMessageBox.warning(None, APP_TITLE, f"Could not open folder:\n{exc}")


class AppCard(QtWidgets.QFrame):
    def __init__(self, definition: AppDefinition, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.definition = definition
        self.setObjectName("AppCard")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.title_label = QtWidgets.QLabel(definition.title)
        self.title_label.setObjectName("CardTitle")
        self.subtitle_label = QtWidgets.QLabel(definition.subtitle)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setObjectName("CardSubtitle")

        self.status_dot = QtWidgets.QLabel("●")
        self.status_dot.setObjectName("StatusDot")
        self.status_label = QtWidgets.QLabel("Stopped")
        self.status_label.setObjectName("StatusStopped")

        self.path_label = QtWidgets.QLabel("Contained inside this EXE")
        self.path_label.setObjectName("CardPath")
        self.path_label.setWordWrap(True)

        self.start_button = QtWidgets.QPushButton("Open")
        self.stop_button = QtWidgets.QPushButton("Close")
        self.stop_button.setEnabled(False)

        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.status_dot)
        header.addWidget(self.status_label)

        buttons = QtWidgets.QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        layout.addLayout(header)
        layout.addWidget(self.subtitle_label)
        layout.addWidget(self.path_label)
        layout.addLayout(buttons)

    def set_running(self, running: bool, detail: str = "") -> None:
        if running:
            self.status_label.setText(detail or "Running")
            self.status_label.setObjectName("StatusRunning")
            self.status_dot.setStyleSheet("color: #69db7c; font-size: 16px; background: transparent;")
            self.stop_button.setEnabled(True)
            self.start_button.setText("Reopen")
        else:
            self.status_label.setText(detail or "Stopped")
            self.status_label.setObjectName("StatusStopped")
            self.status_dot.setStyleSheet("color: #6b7280; font-size: 16px; background: transparent;")
            self.stop_button.setEnabled(False)
            self.start_button.setText("Open")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)


class ManagedProcess(QtCore.QObject):
    state_changed = QtCore.Signal(str, bool, str)

    def __init__(self, definition: AppDefinition):
        super().__init__()
        self.definition = definition
        self.process = QtCore.QProcess(self)
        self.process.setWorkingDirectory(str(launcher_dir()))
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("NISHIZUMI_TOOLS_DIR", str(launcher_dir()))
        env.insert("NISHIZUMI_DATA_DIR", str(appdata_dir()))
        icon = app_icon_path()
        if icon is not None:
            env.insert("NISHIZUMI_ICON_PATH", str(icon))
        self.process.setProcessEnvironment(env)
        self.process.errorOccurred.connect(self._on_error)
        self.process.finished.connect(self._on_finished)
        self.process.started.connect(self._on_started)

    def is_running(self) -> bool:
        return self.process.state() != QtCore.QProcess.NotRunning

    def start(self) -> None:
        if self.is_running():
            self.state_changed.emit(self.definition.key, True, "Already running")
            return

        if getattr(sys, "frozen", False):
            program = sys.executable
            arguments = ["--app", self.definition.key]
        else:
            program = sys.executable
            arguments = [str(Path(__file__).resolve()), "--app", self.definition.key]

        self.process.setProgram(program)
        self.process.setArguments(arguments)
        self.process.start()
        if not self.process.waitForStarted(3000):
            self.state_changed.emit(self.definition.key, False, "Failed to start")

    def stop(self) -> None:
        if not self.is_running():
            self.state_changed.emit(self.definition.key, False, "Stopped")
            return
        self.process.terminate()
        if not self.process.waitForFinished(2500):
            self.process.kill()
            self.process.waitForFinished(1500)

    def _on_started(self) -> None:
        self.state_changed.emit(self.definition.key, True, "Running")

    def _on_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus) -> None:
        if exit_status == QtCore.QProcess.NormalExit:
            detail = "Stopped"
            if exit_code not in (0, None):
                detail = f"Exit {exit_code}"
        else:
            detail = "Crashed"
        self.state_changed.emit(self.definition.key, False, detail)

    def _on_error(self, _error: QtCore.QProcess.ProcessError) -> None:
        detail = self.process.errorString() or "Process error"
        self.state_changed.emit(self.definition.key, False, detail)


class LauncherWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self._state = self._load_state()
        self.setObjectName("LauncherWindow")
        self.processes: Dict[str, ManagedProcess] = {}
        self.cards: Dict[str, AppCard] = {}
        self._quitting = False
        self._hide_to_tray_notified = False
        self._last_update_popup_tag: Optional[str] = None

        self.setWindowTitle(APP_TITLE)
        self.setMinimumSize(840, 560)
        self.resize(*self._state.get("window_size", [980, 680]))

        icon = load_icon(self.style())
        self.setWindowIcon(icon)

        self.close_apps_on_exit = QtWidgets.QCheckBox("Close running apps when the launcher really exits")
        self.close_apps_on_exit.setChecked(bool(self._state.get("close_apps_on_exit", True)))
        self.minimize_to_tray = QtWidgets.QCheckBox("Hide launcher to tray when the window is closed")
        self.minimize_to_tray.setChecked(bool(self._state.get("minimize_to_tray", True)))

        self.status_overview = QtWidgets.QLabel("")
        self.status_overview.setObjectName("Overview")
        self.status_overview.setWordWrap(True)
        self.update_status = QtWidgets.QLabel("")
        self.update_status.setObjectName("UpdateStatus")
        self.update_status.setWordWrap(True)
        self.update_status.setOpenExternalLinks(True)
        self.update_status.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)

        self.network_manager = QtNetwork.QNetworkAccessManager(self)
        self.network_manager.finished.connect(self._on_update_reply)
        self.update_timer = QtCore.QTimer(self)
        self.update_timer.setInterval(GITHUB_UPDATE_CHECK_INTERVAL_SECONDS * 1000)
        self.update_timer.timeout.connect(self.check_for_updates)

        self.tray = QtWidgets.QSystemTrayIcon(icon, self)
        self.tray.setToolTip(APP_TITLE)
        self.tray.activated.connect(self._on_tray_activated)
        self._build_tray_menu()
        self.tray.show()

        self._build_ui()
        self._apply_style()
        self._setup_processes()
        self._restore_position()
        self._refresh_overview()
        self._set_update_status("Checking for updates…")
        self.check_for_updates()
        self.update_timer.start()

    def _build_tray_menu(self) -> None:
        menu = QtWidgets.QMenu(self)
        action_show = menu.addAction("Show launcher")
        action_hide = menu.addAction("Hide launcher")
        menu.addSeparator()
        action_data = menu.addAction("Open data folder")
        action_stop_all = menu.addAction("Close all apps")
        menu.addSeparator()
        action_quit = menu.addAction("Quit")

        action_show.triggered.connect(self.show_normal)
        action_hide.triggered.connect(self.hide)
        action_data.triggered.connect(lambda: open_in_file_manager(appdata_dir()))
        action_stop_all.triggered.connect(self._stop_all)
        action_quit.triggered.connect(self.quit_from_tray)
        self.tray.setContextMenu(menu)

    def _build_ui(self) -> None:
        title = QtWidgets.QLabel(APP_TITLE)
        title.setObjectName("MainTitle")
        subtitle = QtWidgets.QLabel(APP_SUBTITLE)
        subtitle.setObjectName("MainSubtitle")
        subtitle.setWordWrap(True)

        top_buttons = QtWidgets.QHBoxLayout()
        btn_open_data = QtWidgets.QPushButton("Open data folder")
        btn_open_folder = QtWidgets.QPushButton("Open launcher folder")
        btn_stop_all = QtWidgets.QPushButton("Close all apps")
        btn_open_data.clicked.connect(lambda: open_in_file_manager(appdata_dir()))
        btn_open_folder.clicked.connect(lambda: open_in_file_manager(launcher_dir()))
        btn_stop_all.clicked.connect(self._stop_all)
        top_buttons.addWidget(btn_open_data)
        top_buttons.addWidget(btn_open_folder)
        top_buttons.addWidget(btn_stop_all)
        top_buttons.addStretch(1)

        header = QtWidgets.QVBoxLayout()
        header.setSpacing(8)
        header.addWidget(title)
        header.addWidget(subtitle)
        header.addLayout(top_buttons)
        header.addWidget(self.close_apps_on_exit)
        header.addWidget(self.minimize_to_tray)
        header.addWidget(self.update_status)
        header.addWidget(self.status_overview)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(14)

        for index, definition in enumerate(APPS):
            card = AppCard(definition, self)
            self.cards[definition.key] = card
            row = index // 2
            col = index % 2
            grid.addWidget(card, row, col)

        container = QtWidgets.QWidget()
        container.setObjectName("CardsContainer")
        container.setLayout(grid)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(container)

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(18)
        outer.addLayout(header)
        outer.addWidget(scroll, 1)

    def _setup_processes(self) -> None:
        for definition in APPS:
            process = ManagedProcess(definition)
            process.state_changed.connect(self._on_process_state_changed)
            self.processes[definition.key] = process
            card = self.cards[definition.key]
            card.start_button.clicked.connect(lambda checked=False, key=definition.key: self._start_app(key))
            card.stop_button.clicked.connect(lambda checked=False, key=definition.key: self._stop_app(key))
            card.set_running(False, "Stopped")

    def _apply_style(self) -> None:
        self.setStyleSheet("""
        QWidget {
            color: #e5edf5;
            font-family: 'Segoe UI';
            font-size: 10pt;
        }
        #LauncherWindow, #CardsContainer, QScrollArea, QScrollArea > QWidget > QWidget {
            background: #0b1118;
        }
        QLabel, QCheckBox {
            background: transparent;
        }
        #MainTitle {
            font-size: 22pt;
            font-weight: 700;
            color: #f4f7fb;
        }
        #MainSubtitle {
            color: #9fb0c3;
            font-size: 10pt;
        }
        #Overview {
            background: #0f1722;
            border: 1px solid #1f2a38;
            border-radius: 12px;
            padding: 12px 14px;
            color: #c9d5e3;
        }
        #UpdateStatus {
            background: #0f1722;
            border: 1px solid #1f2a38;
            border-radius: 12px;
            padding: 10px 14px;
            color: #bfd0e0;
        }
        #AppCard {
            background: #101923;
            border: 1px solid #1f2c3a;
            border-radius: 18px;
        }
        #AppCard:hover {
            border: 1px solid #31475f;
        }
        #CardTitle {
            font-size: 15pt;
            font-weight: 700;
            color: #f3f6fa;
        }
        #CardSubtitle {
            color: #a7b6c6;
            line-height: 1.35;
        }
        #CardPath {
            color: #7f93a8;
            font-size: 9pt;
        }
        #StatusRunning {
            color: #77dd88;
            font-weight: 600;
        }
        #StatusStopped {
            color: #94a3b8;
            font-weight: 600;
        }
        QPushButton {
            background: #172230;
            border: 1px solid #263546;
            border-radius: 10px;
            padding: 8px 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: #1d2b3d;
            border: 1px solid #33506d;
        }
        QPushButton:disabled {
            color: #708096;
            background: #111923;
            border: 1px solid #1a2431;
        }
        QCheckBox {
            color: #cbd7e4;
            spacing: 8px;
        }
        """)

    def _load_state(self) -> dict:
        path = state_path()
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self) -> None:
        payload = {
            "window_size": [self.width(), self.height()],
            "window_pos": [self.x(), self.y()],
            "close_apps_on_exit": self.close_apps_on_exit.isChecked(),
            "minimize_to_tray": self.minimize_to_tray.isChecked(),
        }
        try:
            state_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _restore_position(self) -> None:
        pos = self._state.get("window_pos")
        if isinstance(pos, list) and len(pos) == 2:
            x, y = pos
            if isinstance(x, int) and isinstance(y, int):
                self.move(x, y)

    def _start_app(self, key: str) -> None:
        self.processes[key].start()

    def _stop_app(self, key: str) -> None:
        self.processes[key].stop()

    def _stop_all(self) -> None:
        for process in self.processes.values():
            process.stop()

    def _on_process_state_changed(self, key: str, running: bool, detail: str) -> None:
        card = self.cards[key]
        card.set_running(running, detail)
        self._refresh_overview()

    def _refresh_overview(self) -> None:
        running = [self.cards[key].definition.title for key, process in self.processes.items() if process.is_running()]
        if running:
            self.status_overview.setText(
                f"{APP_TITLE} {APP_VERSION}\n"
                "Running now: " + ", ".join(running) + "\nAll apps are being launched from this same EXE."
            )
        else:
            self.status_overview.setText(
                f"{APP_TITLE} {APP_VERSION}\n"
                "No app is running right now. Open any overlay from here.\n"
                f"Shared data folder: {appdata_dir()}\n"
                "This launcher keeps the icon in the system tray too."
            )

    def _set_update_status(self, text: str) -> None:
        self.update_status.setText(text)

    def _show_update_popup(self, latest_tag: str, latest_name: str, latest_url: str) -> None:
        if self._last_update_popup_tag == latest_tag:
            return
        self._last_update_popup_tag = latest_tag

        message = QtWidgets.QMessageBox(self)
        message.setIcon(QtWidgets.QMessageBox.Information)
        message.setWindowTitle("Update available")
        message.setText(f"A new {APP_TITLE} version is available: {latest_tag}")
        message.setInformativeText(
            f"Current version: {APP_VERSION}\n"
            f"Release: {latest_name}\n\n"
            "Open the release page to download the latest build."
        )
        download_button = message.addButton("Open download page", QtWidgets.QMessageBox.AcceptRole)
        message.addButton("Later", QtWidgets.QMessageBox.RejectRole)
        message.exec()
        if message.clickedButton() is download_button:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(latest_url))

    def check_for_updates(self) -> None:
        request = QtNetwork.QNetworkRequest(QtCore.QUrl(GITHUB_RELEASES_API_LATEST))
        request.setHeader(QtNetwork.QNetworkRequest.UserAgentHeader, f"{APP_TITLE}/{APP_VERSION}")
        request.setRawHeader(b"Accept", b"application/vnd.github+json")
        request.setRawHeader(b"X-GitHub-Api-Version", GITHUB_API_VERSION.encode("utf-8"))
        self.network_manager.get(request)

    def _on_update_reply(self, reply: QtNetwork.QNetworkReply) -> None:
        if reply.error() != QtNetwork.QNetworkReply.NoError:
            self._set_update_status(
                f"Update check failed ({reply.errorString()}). "
                f"<a href='{GITHUB_RELEASES_PAGE_LATEST}'>Open latest release page</a>."
            )
            reply.deleteLater()
            return

        try:
            payload = bytes(reply.readAll())
            data = json.loads(payload.decode("utf-8", errors="replace"))
        except Exception:
            self._set_update_status(
                f"Update check returned invalid data. "
                f"<a href='{GITHUB_RELEASES_PAGE_LATEST}'>Open latest release page</a>."
            )
            reply.deleteLater()
            return

        latest_tag = str(data.get("tag_name") or "").strip()
        latest_name = str(data.get("name") or latest_tag or "latest release").strip()
        latest_url = str(data.get("html_url") or GITHUB_RELEASES_PAGE_LATEST).strip()
        if not latest_tag:
            self._set_update_status(
                f"Could not detect latest release tag. "
                f"<a href='{GITHUB_RELEASES_PAGE_LATEST}'>Open latest release page</a>."
            )
            reply.deleteLater()
            return

        if _is_version_newer(latest_tag, APP_VERSION):
            self._set_update_status(
                f"Update available: <b>{latest_tag}</b> ({latest_name}). "
                f"You are on <b>{APP_VERSION}</b>. "
                f"<a href='{latest_url}'>Download</a>."
            )
            self._show_update_popup(latest_tag, latest_name, latest_url)
        else:
            self._set_update_status(
                f"Up to date: <b>{APP_VERSION}</b>. "
                f"Latest release: <b>{latest_tag}</b>. "
                f"<a href='{latest_url}'>View release</a>."
            )
        reply.deleteLater()

    def _on_tray_activated(self, reason: QtWidgets.QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QtWidgets.QSystemTrayIcon.Trigger,
            QtWidgets.QSystemTrayIcon.DoubleClick,
            QtWidgets.QSystemTrayIcon.MiddleClick,
        ):
            if self.isVisible() and not self.isMinimized():
                self.hide()
            else:
                self.show_normal()

    def show_normal(self) -> None:
        self.show()
        self.setWindowState((self.windowState() & ~QtCore.Qt.WindowMinimized) | QtCore.Qt.WindowActive)
        self.raise_()
        self.activateWindow()

    def quit_from_tray(self) -> None:
        self._quitting = True
        self._save_state()
        if self.close_apps_on_exit.isChecked():
            self._stop_all()
        self.tray.hide()
        QtWidgets.QApplication.quit()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._save_state()
        if not self._quitting and self.minimize_to_tray.isChecked():
            event.ignore()
            self.hide()
            if not self._hide_to_tray_notified:
                self._hide_to_tray_notified = True
                self.tray.showMessage(
                    APP_TITLE,
                    "Launcher hidden to tray. Use the tray icon to open it again or quit.",
                    QtWidgets.QSystemTrayIcon.Information,
                    2500,
                )
            return

        if self.close_apps_on_exit.isChecked():
            self._stop_all()
        self.tray.hide()
        super().closeEvent(event)


def load_icon(style: QtWidgets.QStyle) -> QtGui.QIcon:
    icon_path = app_icon_path()
    if icon_path is not None:
        icon = QtGui.QIcon(str(icon_path))
        if not icon.isNull():
            return icon
    return style.standardIcon(QtWidgets.QStyle.SP_ComputerIcon)


def _version_key(text: str) -> tuple[int, ...]:
    cleaned = "".join(ch if ch.isdigit() else " " for ch in text or "")
    parts = [int(part) for part in cleaned.split() if part.isdigit()]
    return tuple(parts) if parts else (0,)


def _is_version_newer(candidate: str, current: str) -> bool:
    return _version_key(candidate) > _version_key(current)


def run_launcher() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setApplicationName(APP_TITLE)
    app.setOrganizationName(APP_DIR_NAME)

    guard = SingleInstanceGuard(launcher_instance_name(), app)
    if guard.try_activate_existing():
        return 0
    if not guard.start_listening():
        if guard.try_activate_existing():
            return 0

    icon = load_icon(app.style())
    app.setWindowIcon(icon)
    win = LauncherWindow()
    guard.activation_requested.connect(win.show_normal)
    app.aboutToQuit.connect(lambda: QtNetwork.QLocalServer.removeServer(launcher_instance_name()))
    win.show()
    return app.exec()


def run_selected_app(app_key: str) -> int:
    if app_key == "fuel":
        import tkinter as tk
        from Nishizumi_FuelMonitor import FuelConsumptionMonitor

        app = FuelConsumptionMonitor()
        _apply_tk_icon(app.root)
        tk.mainloop()
        return 0

    if app_key == "pit":
        from nishizumi_pitcalibrator import PitCalibratorApp

        app = PitCalibratorApp()
        _apply_tk_icon(app.root)
        app.run()
        return 0

    if app_key == "traction":
        from Nishizumi_Traction import TractionCircleOverlay

        app = TractionCircleOverlay()
        _apply_tk_icon(app.root)
        app.run()
        return 0

    if app_key == "tire":
        from Nishizumi_TireWear import main as tirewear_main

        return int(tirewear_main() or 0)

    if app_key == "caution":
        # The overlay builds its own QApplication; run_overlay skips the
        # command line parser because the launcher passes "--app caution".
        from Nishizumi_CautionOverlay import run_overlay

        return int(run_overlay() or 0)

    raise SystemExit(f"Unknown app key: {app_key}")


def _apply_tk_icon(root) -> None:
    icon_path = app_icon_path()
    if icon_path is None:
        return
    try:
        if icon_path.suffix.lower() == ".ico" and sys.platform.startswith("win"):
            root.iconbitmap(str(icon_path))
        else:
            image = tk_icon_image(str(icon_path))
            if image is not None:
                root.iconphoto(True, image)
                root._nishizumi_icon_ref = image
    except Exception:
        pass


def tk_icon_image(path: str):
    try:
        import tkinter as tk
        return tk.PhotoImage(file=path)
    except Exception:
        return None


def run_selftest() -> int:
    """Import every bundled app and report whether they are all there.

    The packaged build runs this against the frozen EXE: PyInstaller only keeps
    the app modules because they are listed as hidden imports, so a missing one
    fails here instead of at the moment a user clicks Open. The EXE is windowed
    and has no console, so the exit code is the result: 0 when every app
    imports, 1 otherwise.
    """
    import importlib

    failures = []
    for definition in APPS:
        try:
            importlib.import_module(definition.module)
        except Exception as exc:  # pragma: no cover - depends on the build
            failures.append(f"{definition.title} ({definition.module}): {exc}")
            print(f"FAIL {definition.title} -> {definition.module}: {exc}")
        else:
            print(f"ok   {definition.title} -> {definition.module}")

    if failures:
        print(f"{len(failures)} app(s) missing from this build")
        return 1
    print(f"all {len(APPS)} apps are present in this build")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--app", choices=sorted(APP_MAP))
    parser.add_argument("--selftest", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args, _unknown = parser.parse_known_args(sys.argv[1:])
    if args.selftest:
        return run_selftest()
    if args.app:
        return run_selected_app(args.app)
    return run_launcher()


if __name__ == "__main__":
    raise SystemExit(main())
