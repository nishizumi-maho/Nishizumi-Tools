import customtkinter as ctk
import os
import sys
import threading
import subprocess
import tkinter.messagebox
from tkinter import messagebox
import json
import winreg # Biblioteca para mexer no Registro do Windows

# --- CONFIGURAÇÃO DE CAMINHOS ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT_DIR, "configs")
LAUNCHER_CONFIG = os.path.join(CONFIG_DIR, "launcher_config.json")
AUDIO_DIR = os.path.join(ROOT_DIR, "sons")

# Garante que a pasta configs existe
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

# Fix para imports
os.chdir(ROOT_DIR)
sys.path.append(ROOT_DIR)

try:
    from modulos.race_engineer import RaceEngineer
    from modulos.smart_flasher import SmartFlasher
    from modulos.pit_strategy import PitStrategyManager
    from modulos.pit_gui import PitConfigWindow
    from modulos.flasher_gui import FlasherConfigWindow
except ImportError as e:
    tkinter.messagebox.showerror("Erro Fatal", f"Falha nos imports: {e}")
    sys.exit()

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class LauncherApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("iRacing Tools Suite V12 (Auto-Start)")
        self.geometry("600x750")
        
        # --- CARREGA CONFIGS SALVAS ---
        self.settings = self.load_settings()

        # --- INICIALIZA BACKEND ---
        self.engine_module = RaceEngineer(AUDIO_DIR)
        self.flash_module = SmartFlasher()
        self.pit_module = PitStrategyManager()

        # --- LAYOUT ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0,1,2,3,4,5), weight=1)

        # Header
        self.header = ctk.CTkFrame(self, height=80, fg_color="transparent")
        self.header.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        ctk.CTkLabel(self.header, text="iRACING TOOLS MANAGER", font=("Arial", 24, "bold")).pack()
        
        # Checkbox Iniciar com Windows
        self.chk_windows = ctk.CTkCheckBox(self.header, text="Iniciar com o Windows", command=self.toggle_windows_registry)
        self.chk_windows.pack(pady=5)
        if self.check_registry_status(): self.chk_windows.select()

        # 1. Active Telemetry (ARB)
        self.frame_arb = ctk.CTkFrame(self, border_width=2, border_color="#1f538d")
        self.frame_arb.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        f_arb_txt = ctk.CTkFrame(self.frame_arb, fg_color="transparent")
        f_arb_txt.pack(side="left", padx=20, pady=15)
        ctk.CTkLabel(f_arb_txt, text="Active Telemetry Controller", font=("Arial", 16, "bold"), anchor="w").pack(fill="x")
        ctk.CTkButton(self.frame_arb, text="ABRIR PAINEL", command=self.launch_arb, width=120).pack(side="right", padx=20)

        # 2. Race Engineer
        self.frame_eng = ctk.CTkFrame(self)
        self.frame_eng.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        ctk.CTkLabel(self.frame_eng, text="🔊 Race Engineer", font=("Arial", 15, "bold")).pack(side="left", padx=20, pady=15)
        
        self.lbl_eng_state = ctk.CTkLabel(self.frame_eng, text="[PARADO]", font=("Arial", 11, "bold"), text_color="gray")
        self.lbl_eng_state.pack(side="left", padx=10)
        
        f_eng_ctrl = ctk.CTkFrame(self.frame_eng, fg_color="transparent")
        f_eng_ctrl.pack(side="right", padx=10)
        ctk.CTkButton(f_eng_ctrl, text="TESTE", width=60, command=self.engine_module.test_audio, fg_color="#333").pack(side="left", padx=5)
        
        self.sw_eng = ctk.CTkSwitch(f_eng_ctrl, text="", command=self.toggle_engineer, onvalue="LIGADO", offvalue="DESLIGADO", width=40)
        self.sw_eng.pack(side="left", padx=5)

        # 3. Smart Flasher
        self.frame_flash = ctk.CTkFrame(self)
        self.frame_flash.grid(row=3, column=0, padx=20, pady=5, sticky="ew")
        ctk.CTkLabel(self.frame_flash, text="💡 Smart Flasher", font=("Arial", 15, "bold")).pack(side="left", padx=20, pady=15)
        
        self.lbl_flash_state = ctk.CTkLabel(self.frame_flash, text="[PARADO]", font=("Arial", 11, "bold"), text_color="gray")
        self.lbl_flash_state.pack(side="left", padx=10)
        
        f_flash_ctrl = ctk.CTkFrame(self.frame_flash, fg_color="transparent")
        f_flash_ctrl.pack(side="right", padx=10)
        ctk.CTkButton(f_flash_ctrl, text="TESTE", width=60, command=self.flash_module.test_flash, fg_color="#333").pack(side="left", padx=5)
        ctk.CTkButton(f_flash_ctrl, text="⚙️", width=30, command=self.open_flash_config, fg_color="#444").pack(side="left", padx=5)
        
        self.sw_flash = ctk.CTkSwitch(f_flash_ctrl, text="", command=self.toggle_flasher, onvalue="LIGADO", offvalue="DESLIGADO", width=40)
        self.sw_flash.pack(side="left", padx=5)

        # 4. Turbo Pit
        self.frame_pit = ctk.CTkFrame(self)
        self.frame_pit.grid(row=4, column=0, padx=20, pady=5, sticky="ew")
        ctk.CTkLabel(self.frame_pit, text="🔧 Turbo Pit", font=("Arial", 15, "bold")).pack(side="left", padx=20, pady=15)
        
        self.lbl_pit_state = ctk.CTkLabel(self.frame_pit, text="[PARADO]", font=("Arial", 11, "bold"), text_color="gray")
        self.lbl_pit_state.pack(side="left", padx=10)
        
        f_pit_ctrl = ctk.CTkFrame(self.frame_pit, fg_color="transparent")
        f_pit_ctrl.pack(side="right", padx=10)
        ctk.CTkButton(f_pit_ctrl, text="⚙️", width=30, command=self.open_pit_config, fg_color="#444").pack(side="left", padx=5)
        
        self.sw_pit = ctk.CTkSwitch(f_pit_ctrl, text="", command=self.toggle_pit, onvalue="LIGADO", offvalue="DESLIGADO", width=40)
        self.sw_pit.pack(side="left", padx=5)

        # Footer
        self.lbl_status = ctk.CTkLabel(self, text="Pronto.", text_color="gray")
        self.lbl_status.grid(row=5, column=0, pady=10)

        # --- AUTO START (RESTAURA ESTADO ANTERIOR) ---
        self.after(500, self.restore_state)

    # --- PERSISTÊNCIA ---
    def load_settings(self):
        if os.path.exists(LAUNCHER_CONFIG):
            try:
                with open(LAUNCHER_CONFIG, 'r') as f: return json.load(f)
            except: pass
        return {"engineer": False, "flasher": False, "pit": False}

    def save_settings(self):
        state = {
            "engineer": self.sw_eng.get() == "LIGADO",
            "flasher": self.sw_flash.get() == "LIGADO",
            "pit": self.sw_pit.get() == "LIGADO"
        }
        try:
            with open(LAUNCHER_CONFIG, 'w') as f: json.dump(state, f, indent=4)
        except: pass

    def restore_state(self):
        """Liga os módulos que estavam ligados na última vez"""
        if self.settings.get("engineer", False):
            self.sw_eng.select()
            self.toggle_engineer()
        
        if self.settings.get("flasher", False):
            self.sw_flash.select()
            self.toggle_flasher()
            
        if self.settings.get("pit", False):
            self.sw_pit.select()
            self.toggle_pit()

    # --- WINDOWS REGISTRY (AUTO START) ---
    def check_registry_status(self):
        """Verifica se já está configurado para iniciar"""
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_READ)
            winreg.QueryValueEx(key, "iRacingTools")
            winreg.CloseKey(key)
            return True
        except WindowsError:
            return False

    def toggle_windows_registry(self):
        app_path = os.path.abspath(sys.argv[0])
        # Se estiver rodando como .py, usa pythonw.exe para não abrir janela preta
        if app_path.endswith(".py") or app_path.endswith(".pyw"):
            executable = sys.executable.replace("python.exe", "pythonw.exe")
            cmd = f'"{executable}" "{app_path}"'
        else:
            # Se for compilado .exe
            cmd = f'"{app_path}"'

        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"

        if self.chk_windows.get() == 1:
            # Adicionar
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
                winreg.SetValueEx(key, "iRacingTools", 0, winreg.REG_SZ, cmd)
                winreg.CloseKey(key)
                messagebox.showinfo("Sucesso", "Configurado para iniciar com o Windows!")
            except Exception as e:
                messagebox.showerror("Erro", f"Não foi possível gravar no registro:\n{e}")
                self.chk_windows.deselect()
        else:
            # Remover
            try:
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE)
                winreg.DeleteValue(key, "iRacingTools")
                winreg.CloseKey(key)
            except: pass

    # --- FUNÇÕES DOS MÓDULOS ---
    def toggle_engineer(self):
        if self.sw_eng.get() == "LIGADO":
            self.engine_module.start()
            self.lbl_eng_state.configure(text="[ATIVADO]", text_color="#00FF00")
        else:
            self.engine_module.stop()
            self.lbl_eng_state.configure(text="[PARADO]", text_color="gray")
        self.save_settings()

    def toggle_flasher(self):
        if self.sw_flash.get() == "LIGADO":
            self.flash_module.start()
            self.lbl_flash_state.configure(text="[ATIVADO]", text_color="#00FF00")
        else:
            self.flash_module.stop()
            self.lbl_flash_state.configure(text="[PARADO]", text_color="gray")
        self.save_settings()

    def toggle_pit(self):
        if self.sw_pit.get() == "LIGADO":
            self.pit_module.start_listener()
            self.lbl_pit_state.configure(text="[ATIVADO]", text_color="#00FF00")
        else:
            self.pit_module.stop_listener()
            self.lbl_pit_state.configure(text="[PARADO]", text_color="gray")
        self.save_settings()

    def open_pit_config(self): PitConfigWindow(self, self.pit_module)
    def open_flash_config(self): FlasherConfigWindow(self, self.flash_module)

    def launch_arb(self):
        script_path = os.path.join(ROOT_DIR, "modulos", "arb_control.py")
        try: subprocess.Popen([sys.executable, script_path])
        except Exception as e: messagebox.showerror("Erro", str(e))

    def on_close(self):
        self.save_settings()
        self.engine_module.stop()
        self.flash_module.stop()
        self.pit_module.stop_listener()
        self.destroy()

if __name__ == "__main__":
    app = LauncherApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()