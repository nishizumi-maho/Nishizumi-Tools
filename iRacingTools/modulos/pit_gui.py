import customtkinter as ctk

# Compatibilidade: alguns projetos têm o módulo dentro de "modulos/".
try:
    from modulos.pit_strategy import PitStrategyManager, capture_input_scan  # type: ignore
except Exception:
    from pit_strategy import PitStrategyManager, capture_input_scan  # type: ignore


class PitConfigWindow(ctk.CTkToplevel):
    def __init__(self, parent, manager: PitStrategyManager):
        super().__init__(parent)
        self.manager = manager

        self.title("Configurar Turbo Pit")
        self.geometry("480x520")
        self.attributes("-topmost", True)

        ctk.CTkLabel(self, text="Turbo Pit (Pneus/Fuel)", font=("Arial", 20, "bold")).pack(pady=16)

        # ============ Bind ============
        f_bind = ctk.CTkFrame(self)
        f_bind.pack(pady=8, padx=16, fill="x")
        ctk.CTkLabel(f_bind, text="Botão de Ativação:").pack(side="left", padx=10)
        self.btn_bind = ctk.CTkButton(f_bind, text=self.manager.config.get("bind_code", "KEY:F6"), command=self.start_bind, width=160)
        self.btn_bind.pack(side="right", padx=10)

        self.var_enable_pygame = ctk.BooleanVar(value=bool(self.manager.config.get("enable_pygame", False)))
        self.chk_enable_pygame = ctk.CTkCheckBox(f_bind, text="Permitir pygame/joystick", variable=self.var_enable_pygame)
        self.chk_enable_pygame.pack(anchor="w", padx=10, pady=(8, 0))

        # ============ Chat ============
        f_chat = ctk.CTkFrame(self)
        f_chat.pack(pady=8, padx=16, fill="x")

        ctk.CTkLabel(f_chat, text="Tecla do Chat (iRacing):").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.entry_chat = ctk.CTkEntry(f_chat, width=60)
        self.entry_chat.insert(0, str(self.manager.config.get("chat_key", "t")))
        self.entry_chat.grid(row=0, column=1, padx=10, pady=8, sticky="w")

        ctk.CTkLabel(f_chat, text="Modo de envio:").grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.var_injection = ctk.StringVar(value=str(self.manager.config.get("injection", "type")).lower())
        self.opt_injection = ctk.CTkOptionMenu(f_chat, values=["type", "clipboard"], variable=self.var_injection)
        self.opt_injection.grid(row=1, column=1, padx=10, pady=8, sticky="w")

        ctk.CTkLabel(f_chat, text="Delay abrir chat (s):").grid(row=2, column=0, padx=10, pady=8, sticky="w")
        self.entry_open_delay = ctk.CTkEntry(f_chat, width=80)
        self.entry_open_delay.insert(0, str(self.manager.config.get("open_delay", 0.02)))
        self.entry_open_delay.grid(row=2, column=1, padx=10, pady=8, sticky="w")

        ctk.CTkLabel(f_chat, text="Intervalo digitação (s):").grid(row=3, column=0, padx=10, pady=8, sticky="w")
        self.entry_typing = ctk.CTkEntry(f_chat, width=80)
        self.entry_typing.insert(0, str(self.manager.config.get("typing_interval", 0.001)))
        self.entry_typing.grid(row=3, column=1, padx=10, pady=8, sticky="w")

        # ============ Segurança ============
        f_safe = ctk.CTkFrame(self)
        f_safe.pack(pady=8, padx=16, fill="x")

        self.var_foreground = ctk.BooleanVar(value=bool(self.manager.config.get("require_iracing_foreground", True)))
        self.chk_foreground = ctk.CTkCheckBox(f_safe, text="Só executar com iRacing em foco", variable=self.var_foreground)
        self.chk_foreground.pack(anchor="w", padx=10, pady=(10, 6))

        row2 = ctk.CTkFrame(f_safe, fg_color="transparent")
        row2.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(row2, text="Substring da janela:", text_color="gray").pack(side="left")
        self.entry_win_sub = ctk.CTkEntry(row2, width=120)
        self.entry_win_sub.insert(0, str(self.manager.config.get("iracing_window_substring", "iracing")))
        self.entry_win_sub.pack(side="left", padx=(8, 0))

        # ============ Comandos toggle ============
        f_cmd = ctk.CTkFrame(self)
        f_cmd.pack(pady=8, padx=16, fill="x")
        ctk.CTkLabel(f_cmd, text="Comando quando 'ON':", text_color="gray").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.entry_cmd_on = ctk.CTkEntry(f_cmd, width=300)
        self.entry_cmd_on.insert(0, str(self.manager.config.get("cmd_when_on", "#-cleartires #ws")))
        self.entry_cmd_on.grid(row=0, column=1, padx=10, pady=8, sticky="w")

        ctk.CTkLabel(f_cmd, text="Comando quando 'OFF':", text_color="gray").grid(row=1, column=0, padx=10, pady=8, sticky="w")
        self.entry_cmd_off = ctk.CTkEntry(f_cmd, width=300)
        self.entry_cmd_off.insert(0, str(self.manager.config.get("cmd_when_off", "#cleartires #-ws")))
        self.entry_cmd_off.grid(row=1, column=1, padx=10, pady=8, sticky="w")

        # ============ Debounce ============
        f_db = ctk.CTkFrame(self)
        f_db.pack(pady=8, padx=16, fill="x")
        ctk.CTkLabel(f_db, text="Debounce (ms):", text_color="gray").pack(side="left", padx=10, pady=10)
        self.entry_db = ctk.CTkEntry(f_db, width=80)
        self.entry_db.insert(0, str(self.manager.config.get("debounce_ms", 150)))
        self.entry_db.pack(side="left", padx=(0, 10), pady=10)

        # ============ Save ============
        ctk.CTkButton(self, text="SALVAR", command=self.save, fg_color="green").pack(pady=18)

        ctk.CTkLabel(
            self,
            text=(
                "Dica: 'type' é o mais estável (não depende de clipboard).\n"
                "Se marcar 'iRacing em foco', evita digitar em outro app."
            ),
            text_color="gray",
        ).pack(pady=(0, 12))

    def start_bind(self):
        self.btn_bind.configure(text="Aperte...", fg_color="orange")
        self.update()
        code = capture_input_scan(allow_joystick=bool(self.var_enable_pygame.get()))
        if code:
            self.btn_bind.configure(text=code, fg_color="#1f538d")
        else:
            self.btn_bind.configure(text=self.manager.config.get("bind_code", "KEY:F6"), fg_color="#1f538d")

    def save(self):
        was_running = bool(getattr(self.manager, "running", False))

        self.manager.config["bind_code"] = self.btn_bind.cget("text")
        self.manager.config["chat_key"] = self.entry_chat.get().strip() or "t"
        self.manager.config["injection"] = str(self.var_injection.get()).strip().lower() or "type"

        try:
            self.manager.config["open_delay"] = float(self.entry_open_delay.get())
        except Exception:
            pass
        try:
            self.manager.config["typing_interval"] = float(self.entry_typing.get())
        except Exception:
            pass

        self.manager.config["require_iracing_foreground"] = bool(self.var_foreground.get())
        self.manager.config["iracing_window_substring"] = self.entry_win_sub.get().strip() or "iracing"

        self.manager.config["cmd_when_on"] = self.entry_cmd_on.get().strip() or self.manager.config.get("cmd_when_on", "#-cleartires #ws")
        self.manager.config["cmd_when_off"] = self.entry_cmd_off.get().strip() or self.manager.config.get("cmd_when_off", "#cleartires #-ws")

        self.manager.config["enable_pygame"] = bool(self.var_enable_pygame.get())

        try:
            self.manager.config["debounce_ms"] = int(float(self.entry_db.get()))
        except Exception:
            pass

        self.manager.save_config()
        try:
            self.manager.stop_listener()
        except Exception:
            pass

        try:
            self.manager.refresh_injector()
        except Exception:
            pass

        if was_running:
            try:
                self.manager.start_listener()
            except Exception:
                pass

        self.destroy()
