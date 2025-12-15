import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import threading
import time
import random
import ctypes
import keyboard
import irsdk
import json
import os
import pygame

# ======================================================================
# 0. CONFIGURAÇÃO DE ARQUIVO (CORRIGIDA PARA PASTA CONFIGS)
# ======================================================================
PASTA_ATUAL = os.path.dirname(os.path.abspath(__file__))
PASTA_RAIZ = os.path.dirname(PASTA_ATUAL)
PASTA_CONFIGS = os.path.join(PASTA_RAIZ, "configs")
if not os.path.exists(PASTA_CONFIGS):
    os.makedirs(PASTA_CONFIGS)
ARQUIVO_JSON = os.path.join(PASTA_CONFIGS, "iracing_config_v7.json")
print(f"Salvando configurações em: {ARQUIVO_JSON}")

# ======================================================================
# 1. ENGINE DE INPUT (SIMULAÇÃO DE TECLAS)
# ======================================================================
SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)

class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_ushort),
                ("wParamH", ctypes.c_ushort)]
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]
class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", Input_I)]

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def click_rapido(scan_code, is_float=False):
    """Clique rápido: usa delay reduzido; is_float escolhe delay."""
    if not scan_code: return
    PressKey(scan_code)
    # usado o seu tempo desejado aqui
    time.sleep(0.05 if not is_float else 0.08)
    ReleaseKey(scan_code)

# ======================================================================
# 2. GERENCIADOR DE INPUT (JOYSTICK + TECLADO HÍBRIDO)
# ======================================================================
class InputManager:
    def __init__(self):
        pygame.init()
        try:
            pygame.joystick.init()
            self.joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
            for j in self.joysticks: j.init()
            print(f"Joysticks detectados: {len(self.joysticks)}")
        except Exception as e:
            print(f"Erro ao iniciar Joysticks: {e}")

        self.listeners = {}  # { 'JOY:0:1': funcao_callback }
        self.capture_mode = False
        self.captured_code = None
        threading.Thread(target=self.loop_inputs, daemon=True).start()

    def loop_inputs(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    code = f"JOY:{event.joy}:{event.button}"
                    if self.capture_mode:
                        self.captured_code = code
                        self.capture_mode = False
                    elif code in self.listeners:
                        threading.Thread(target=self.listeners[code]).start()
            time.sleep(0.01)

    def capturar_qualquer_input(self):
        """
        Bloqueia e espera o usuário apertar QUALQUER coisa (Tecla ou Botão).
        Retorna o código detectado (e.g. 'KEY:A' ou 'JOY:0:1').
        """
        self.captured_code = None
        self.capture_mode = True

        def key_hook(e):
            if self.capture_mode and e.event_type == 'down':
                if e.name not in ['unknown', '']:
                    # normalize name -> uppercase
                    self.captured_code = f"KEY:{e.name.upper()}"
                    self.capture_mode = False

        try:
            hook = keyboard.hook(key_hook)
        except Exception as ex:
            messagebox.showerror("Erro Permissão", "Execute o script como ADMINISTRADOR para capturar teclas!")
            return None

        start = time.time()
        while self.capture_mode and (time.time() - start < 15):
            if self.captured_code: break
            time.sleep(0.03)

        try:
            keyboard.unhook(hook)
        except:
            pass
        self.capture_mode = False
        return self.captured_code

    def capturar_scancode_teclado(self):
        """Captura scan_code e nome de uma tecla do teclado (blocking)."""
        while keyboard.is_pressed('enter'): pass
        start = time.time()
        while time.time() - start < 15:
            evt = keyboard.read_event(suppress=True)
            if evt.event_type == 'down':
                return evt.scan_code, evt.name
        return None, None

input_manager = InputManager()

# ======================================================================
# 3. LÓGICA DE CONTROLE (SMART STEP PREDICTIVE)
# ======================================================================
class GenericController:
    def __init__(self, ir_instance, var_name, is_float=False, status_callback=None):
        self.ir = ir_instance
        self.var_name = var_name
        self.is_float = is_float
        self.running_action = False
        self.key_up = None
        self.key_down = None
        self.update_status = status_callback

    def ler_telemetria(self):
        if not getattr(self.ir, "is_initialized", False) or not getattr(self.ir, "is_connected", False):
            try:
                self.ir.startup()
            except: pass
        if getattr(self.ir, "is_connected", False):
            try:
                val = self.ir[self.var_name]
                if val is not None:
                    return float(val) if self.is_float else int(round(val))
            except:
                pass
        return None

    def ir_para(self, valor_alvo):
        if self.running_action: return
        self.running_action = True

        if self.key_up is None or self.key_down is None:
            if self.update_status: self.update_status("Sem Teclas!", "red")
            self.running_action = False
            return

        if self.update_status: self.update_status("Ajustando...", "orange")
        print(f"[{self.var_name}] Alvo: {valor_alvo}")

        timeout = time.time() + 10
        last_step_size = 0.0

        try:
            while time.time() < timeout:
                atual = self.ler_telemetria()
                if atual is None: break

                diff = valor_alvo - atual
                abs_diff = abs(diff)

                if self.is_float and abs_diff < 0.001: break
                if not self.is_float and diff == 0: break

                # Predição Inteligente
                if last_step_size > 0:
                    if abs_diff < (last_step_size * 0.60):
                        break

                if diff > 0:
                    # click_rapido agora aceita is_float
                    click_rapido(self.key_up, self.is_float)
                else:
                    click_rapido(self.key_down, self.is_float)

                time.sleep(0.05 if not self.is_float else 0.08)

                novo = self.ler_telemetria()
                if novo is not None:
                    step = abs(novo - atual)
                    if step > 0.001: last_step_size = step
                    if abs(valor_alvo - novo) > abs_diff: break  # Safety Stop

        except Exception as e:
            print(f"Erro: {e}")
        finally:
            final = self.ler_telemetria()
            txt = f"{final:.2f}" if self.is_float and final is not None else str(final)
            if self.update_status: self.update_status(f"Pronto: {txt}", "green")
            self.running_action = False

# ======================================================================
# 4. INTERFACE GRÁFICA (ABA)
# ======================================================================
class ControlTab(tk.Frame):
    def __init__(self, parent, controller, label_name):
        super().__init__(parent)
        self.ctrl = controller
        self.ctrl.update_status = self.update_status_label
        self.presets_rows = []

        # --- Config Teclas do Jogo ---
        frame_cfg = tk.LabelFrame(self, text=f"Teclas do Jogo ({label_name})", padx=5, pady=5, fg="blue")
        frame_cfg.pack(fill="x", padx=5, pady=5)

        self.btn_up = tk.Button(frame_cfg, text="Definir Aumentar (+)", command=lambda: self.bind_game_key('up'))
        self.btn_up.pack(side="left", padx=5, fill="x", expand=True)

        self.btn_down = tk.Button(frame_cfg, text="Definir Diminuir (-)", command=lambda: self.bind_game_key('down'))
        self.btn_down.pack(side="left", padx=5, fill="x", expand=True)

        # --- Monitor ---
        self.lbl_monitor = tk.Label(self, text="Valor: --", font=("Arial", 16, "bold"), fg="#333")
        self.lbl_monitor.pack(pady=2)
        self.lbl_status = tk.Label(self, text="Ocioso", font=("Arial", 9), fg="gray")
        self.lbl_status.pack(pady=2)

        # --- Reset ---
        frame_reset = tk.LabelFrame(self, text="Reset / Pânico", padx=5, pady=5, fg="red")
        frame_reset.pack(fill="x", padx=5)
        tk.Label(frame_reset, text="Valor:").pack(side="left")
        self.e_reset_val = tk.Entry(frame_reset, width=8)
        self.e_reset_val.insert(0, "1")
        self.e_reset_val.pack(side="left", padx=5)
        self.btn_reset_bind = tk.Button(frame_reset, text="Definir Botão", width=15, bg="#ffcccc")
        self.btn_reset_bind.pack(side="left", padx=5)

        self.reset_data = {'bind': None}
        self.config_btn_logic(self.btn_reset_bind, self.reset_data)

        # --- Presets ---
        frame_p = tk.LabelFrame(self, text="Seus Presets (Alvos)", padx=5, pady=5)
        frame_p.pack(fill="both", expand=True, padx=5)

        for i in range(4):
            f = tk.Frame(frame_p)
            f.pack(fill="x", pady=2)
            tk.Label(f, text=f"Alvo {i+1}:").pack(side="left")
            ev = tk.Entry(f, width=8)
            ev.pack(side="left", padx=5)

            btn_bind = tk.Button(f, text="Clique p/ Configurar", width=20)
            btn_bind.pack(side="left", padx=5)

            row_data = {'entry': ev, 'btn': btn_bind, 'bind': None}
            self.config_btn_logic(btn_bind, row_data)
            self.presets_rows.append(row_data)

        self.running = True
        threading.Thread(target=self.loop_monitor, daemon=True).start()

    def update_status_label(self, text, color):
        try: self.lbl_status.config(text=text, fg=color)
        except: pass

    def _temporarily_disable_entries(self):
        """
        Coloca todas as Entry desta aba como readonly e remove foco delas para evitar
        que a tecla pressionada durante a captura apareça no campo.
        Retorna uma lista com estados antigos para restaurar depois.
        """
        states = []
        # entries: presets + reset_val
        widgets = [row['entry'] for row in self.presets_rows] + [self.e_reset_val]
        for w in widgets:
            try:
                states.append((w, w.get(), w.cget('state')))
                w.config(state='readonly')
            except:
                states.append((w, None, None))
        # tira foco de qualquer entry -> foca no próprio botão temporariamente
        self.focus_set()
        self.update()
        return states

    def _restore_entries(self, states):
        for item in states:
            try:
                w, val, st = item
                if val is not None:
                    # manter conteúdo; readonly->normal
                    w.config(state='normal')
                    w.delete(0, tk.END)
                    w.insert(0, val)
                    if st == 'readonly':
                        w.config(state='readonly')
                else:
                    w.config(state='normal')
            except:
                pass

    def bind_game_key(self, direction):
        # AQUI É SÓ TECLADO (ScanCodes para o jogo)
        btn = self.btn_up if direction == 'up' else self.btn_down
        orig = btn['text']
        btn.config(text="APERTE TECLA...", bg="yellow")
        self.update()

        # Desabilita entradas para evitar que a tecla apareça
        states = self._temporarily_disable_entries()
        try:
            code, name = input_manager.capturar_scancode_teclado()
        finally:
            self._restore_entries(states)

        if code:
            if direction == 'up': self.ctrl.key_up = code
            else: self.ctrl.key_down = code
            btn.config(text=f"OK: {name.upper()}", bg="lightgreen")
        else:
            btn.config(text=orig, bg="#f0f0f0")

    def config_btn_logic(self, btn, data_store):
        # AQUI É HÍBRIDO (Joystick ou Teclado para ativar)
        def on_click():
            btn.config(text="APERTE (Volante/Teclado)...", bg="yellow")
            self.update()

            # desabilita entradas para evitar escrita acidental
            states = self._temporarily_disable_entries()
            try:
                code = input_manager.capturar_qualquer_input()
            finally:
                self._restore_entries(states)

            if code:
                data_store['bind'] = code
                bg_col = "#90ee90" if "JOY" in code else "#ADD8E6"
                btn.config(text=code, bg=bg_col)
            else:
                btn.config(text="TIMEOUT", bg="#ffcccb")
        btn.config(command=on_click)

    def loop_monitor(self):
        while self.running:
            v = self.ctrl.ler_telemetria()
            if v is not None:
                txt = f"{v:.2f}" if self.ctrl.is_float else f"{v}"
                self.lbl_monitor.config(text=f"Valor: {txt}")
            else:
                self.lbl_monitor.config(text="--")
            time.sleep(0.5)

    def get_config_full(self):
        p_list = []
        for row in self.presets_rows:
            p_list.append({'val': row['entry'].get(), 'bind': row['bind']})
        return {
            'meta_var': self.ctrl.var_name,
            'meta_float': self.ctrl.is_float,
            'game_up': self.ctrl.key_up,
            'game_up_txt': self.btn_up['text'],
            'game_down': self.ctrl.key_down,
            'game_down_txt': self.btn_down['text'],
            'reset_val': self.e_reset_val.get(),
            'reset_bind': self.reset_data['bind'],
            'presets': p_list
        }

    def set_config(self, data):
        if not data: return
        self.ctrl.key_up = data.get('game_up')
        self.ctrl.key_down = data.get('game_down')
        if self.ctrl.key_up: self.btn_up.config(text=data.get('game_up_txt', 'OK'), bg="lightgreen")
        if self.ctrl.key_down: self.btn_down.config(text=data.get('game_down_txt', 'OK'), bg="lightgreen")

        self.e_reset_val.delete(0, tk.END); self.e_reset_val.insert(0, data.get('reset_val', '1'))
        if data.get('reset_bind'):
            self.reset_data['bind'] = data.get('reset_bind')
            self.btn_reset_bind.config(text=self.reset_data['bind'], bg="#90ee90")

        saved_ps = data.get('presets', [])
        for i, row in enumerate(self.presets_rows):
            if i < len(saved_ps):
                item = saved_ps[i]
                row['entry'].delete(0, tk.END); row['entry'].insert(0, item.get('val', ''))
                if item.get('bind'):
                    row['bind'] = item.get('bind')
                    bg_col = "#90ee90" if "JOY" in item.get('bind') else "#ADD8E6"
                    row['btn'].config(text=item.get('bind'), bg=bg_col)

# ======================================================================
# 5. APP PRINCIPAL
# ======================================================================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("iRacing Pro V7.0 (Joystick + Keyboard)")
        self.root.geometry("600x780")

        self.ir = irsdk.IRSDK()
        self.tabs = {}
        self.presets_db = self.load_db()
        self.auto_activate = tk.BooleanVar(value=True)  # auto-activate by default

        f_head = tk.LabelFrame(root, text="Seu Veículo / Preset", padx=10, pady=10)
        f_head.pack(fill="x", padx=10, pady=5)

        self.cb_presets = ttk.Combobox(f_head, width=20)
        self.cb_presets.pack(side="left", padx=5)
        self.update_combo()

        tk.Button(f_head, text="Carregar", command=self.load_preset).pack(side="left")
        tk.Button(f_head, text="Salvar", command=self.save_preset, bg="lightgreen").pack(side="left", padx=5)
        tk.Button(f_head, text="Deletar", command=self.delete_preset, fg="red").pack(side="left")

        tk.Button(root, text="🔎 ESCANEAR CARRO NOVO (Adicionar Abas)", command=self.scan_car, bg="lightblue", height=2).pack(fill="x", padx=10, pady=5)

        # Auto-activate checkbox
        chk = tk.Checkbutton(root, text="Auto-ativar binds (scan/load)", variable=self.auto_activate)
        chk.pack(anchor="w", padx=12)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=6)

        self.btn_ativar = tk.Button(root, text="ATIVAR SISTEMA", command=self.ativar_binds, height=2, bg="orange")
        self.btn_ativar.pack(fill="x", padx=10, pady=10)

    def load_db(self):
        if os.path.exists(ARQUIVO_JSON):
            try:
                with open(ARQUIVO_JSON, "r") as f: return json.load(f)
            except: pass
        return {}

    def save_db(self):
        try:
            with open(ARQUIVO_JSON, "w") as f: json.dump(self.presets_db, f, indent=4)
            print(f"Salvo: {ARQUIVO_JSON}")
        except Exception as e: messagebox.showerror("Erro Save", str(e))

    def update_combo(self):
        self.cb_presets['values'] = list(self.presets_db.keys())

    def load_preset(self):
        name = self.cb_presets.get()
        if name not in self.presets_db:
            messagebox.showerror("Erro", "Preset não encontrado.")
            return

        # remove abas atuais
        for tab_id in self.notebook.tabs(): self.notebook.forget(tab_id)
        self.tabs = {}

        saved_data = self.presets_db[name]
        count = 0
        for label, config in saved_data.items():
            var_name = config.get('meta_var')
            is_float = config.get('meta_float', False)
            if var_name:
                ctrl = GenericController(self.ir, var_name, is_float)
                tab = ControlTab(self.notebook, ctrl, label)
                self.notebook.add(tab, text=label)
                self.tabs[label] = tab
                tab.set_config(config)
                count += 1

        if count > 0:
            messagebox.showinfo("Carregado", f"Carregado {count} controles.")
            # auto-activate if chosen
            if self.auto_activate.get():
                self.ativar_binds()

    def scan_car(self):
        if not self.ir.startup():
            messagebox.showerror("Erro", "Abra o iRacing!")
            return

        # lista ampliada de DCs (inclui os que você listou)
        targets = [
            ("dcStarter", "Starter", False),
            ("dcPitSpeedLimiterToggle", "Pit Limiter", False),
            ("dcDRSToggle", "DRS", False),
            ("dcTearOffVisor", "TearOff Visor", False),
            ("dcBrakeBias", "Brake Bias", True),
            ("dcBrakeBiasFine", "Brake Bias Fine", True),
            ("dcPeakBrakeBias", "Peak Brake Bias", True),
            ("dcBrakeMisc", "Brake Misc", True),
            ("dcEngineBraking", "Engine Braking", True),
            ("dcMGUKDeployMode", "MGU-K Deploy", True),
            ("dcMGUKRegenGain", "MGU-K Regen", True),
            ("dcDashPage", "Dash Page", True),
            ("dcDiffEntry", "Diff Entry", True),
            ("dcDiffMiddle", "Diff Middle", True),
            ("dcDiffExit", "Diff Exit", True),
            ("dcAntiRollFront", "ARB Front", False),
            ("dcAntiRollRear", "ARB Rear", False),
            ("dcWeightJackerRight", "W. Jacker", True),
            ("dcFuelMixture", "Fuel Mix", False),
            ("dcTractionControl", "TC 1", False),
            ("dcTractionControl2", "TC 2", False),
            ("dcABS", "ABS", False),
            # adicione mais variáveis aqui se quiser
        ]

        found = 0
        for ir_var, label, is_float in targets:
            try:
                val = self.ir[ir_var]
            except Exception:
                val = None

            # Se a variável existe e não é booleana True/False -> cria controle
            if val is not None:
                # ignora quando o valor é booleano (True/False)
                if isinstance(val, bool):
                    continue

                # se já existe aba com esse label, pula
                if label in self.tabs: continue

                ctrl = GenericController(self.ir, ir_var, is_float)
                tab = ControlTab(self.notebook, ctrl, label)
                self.notebook.add(tab, text=label)
                self.tabs[label] = tab
                found += 1

        if found > 0:
            messagebox.showinfo("Scanner", f"Achados {found} novos controles.")
            # auto-activate binds se opção ligada
            if self.auto_activate.get():
                self.ativar_binds()
        else:
            messagebox.showwarning("Aviso", "Nenhum novo controle detectado.")

    def save_preset(self):
        name = self.cb_presets.get()
        if not name: name = simpledialog.askstring("Salvar", "Nome do Carro:")
        if name:
            data = {}
            for label, tab in self.tabs.items():
                data[label] = tab.get_config_full()
            self.presets_db[name] = data
            self.save_db()
            self.update_combo()
            self.cb_presets.set(name)
            messagebox.showinfo("Salvo", "Salvo com sucesso!")

    def delete_preset(self):
        name = self.cb_presets.get()
        if name in self.presets_db:
            del self.presets_db[name]
            self.save_db()
            self.update_combo()
            self.cb_presets.set("")

    def ativar_binds(self):
        # Limpa listeners e hotkeys antigos
        input_manager.listeners = {}
        try:
            keyboard.unhook_all_hotkeys()
            keyboard.unhook_all()
        except:
            pass

        count = 0
        for label, tab in self.tabs.items():
            # Presets
            for row in tab.presets_rows:
                bind = row['bind']
                val_str = row['entry'].get()
                if bind and val_str:
                    try:
                        val = float(val_str)
                        action = lambda c=tab.ctrl, v=val: threading.Thread(target=c.ir_para, args=(v,)).start()

                        if bind.startswith("KEY:"):
                            keyname = bind.split(":", 1)[1].lower()
                            keyboard.add_hotkey(keyname, action)
                        elif bind.startswith("JOY:"):
                            input_manager.listeners[bind] = action
                        count += 1
                    except Exception:
                        pass

            # Reset
            r_bind = tab.reset_data['bind']
            r_val = tab.e_reset_val.get()
            if r_bind and r_val:
                try:
                    val = float(r_val)
                    action = lambda c=tab.ctrl, v=val: threading.Thread(target=c.ir_para, args=(v,)).start()
                    if r_bind.startswith("KEY:"):
                        keyname = r_bind.split(":", 1)[1].lower()
                        keyboard.add_hotkey(keyname, action)
                    elif r_bind.startswith("JOY:"):
                        input_manager.listeners[r_bind] = action
                    count += 1
                except Exception:
                    pass

        self.btn_ativar.config(text=f"ATIVO ({count} Binds) - Clique p/ Atualizar", bg="#90ee90")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = App(root)
        root.mainloop()
    except Exception as e:
        print(f"Erro Fatal: {e}")
        input()
