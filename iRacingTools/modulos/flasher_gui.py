
import os
import customtkinter as ctk
import keyboard

# Mantém a mesma lógica de caminho do smart_flasher (sobe 1 nível e entra em configs)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR = os.path.dirname(_THIS_DIR)
_CONFIG_DIR = os.path.join(_BASE_DIR, "configs")
if not os.path.isdir(_CONFIG_DIR):
    _CONFIG_DIR = os.path.join(_THIS_DIR, "configs")
CONFIG_DIR = _CONFIG_DIR
CONFIG_FILE = os.path.join(CONFIG_DIR, "flasher_config.json")


def _cfg_get(cfg: dict, path: tuple, default=None):
    cur = cfg
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _cfg_set(cfg: dict, path: tuple, value):
    cur = cfg
    for p in path[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[path[-1]] = value


class FlasherConfigWindow(ctk.CTkToplevel):
    """
    Janela de configuração para o SmartFlasher (versão refatorada).
    - Não tenta cobrir 100% de "customização total" via GUI (isso explode em complexidade).
    - O essencial está na GUI, e o resto pode ser customizado manualmente no JSON.
    """

    def __init__(self, parent, flasher_instance):
        super().__init__(parent)
        self.flasher = flasher_instance

        self.title("Configurar Smart Flasher")
        self.geometry("520x720")
        self.attributes("-topmost", True)

        self._entries = []   # [(widget, path, caster)]
        self._switches = []  # [(switch, path)]
        self._error_label = None

        ctk.CTkLabel(self, text="Smart Flasher - Configurações", font=("Arial", 20, "bold")).pack(pady=(18, 8))

        # Top row: keybind + open config folder
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=18, pady=(0, 10))

        ctk.CTkLabel(top, text="Tecla do Farol:").pack(side="left", padx=(10, 6))
        self.btn_bind = ctk.CTkButton(
            top,
            text=f"TECLA: {self.flasher.config.get('key_name','?')}",
            command=self.bind_key,
            width=160,
        )
        self.btn_bind.pack(side="left", padx=6, pady=10)

        self.btn_open = ctk.CTkButton(top, text="Abrir pasta configs", command=self.open_config_folder, width=160)
        self.btn_open.pack(side="right", padx=10, pady=10)

        # Tabs
        tabs = ctk.CTkTabview(self)
        tabs.pack(fill="both", expand=True, padx=18, pady=10)

        tab_global = tabs.add("Geral")
        tab_pass = tabs.add("Pedir Passagem")
        tab_safety = tabs.add("Aproximação")
        tab_hazard = tabs.add("Perigo")
        tab_pit = tabs.add("Pit Exit")
        tab_adv = tabs.add("Avançado")

        self._build_global(tab_global)
        self._build_pass(tab_pass)
        self._build_safety(tab_safety)
        self._build_hazard(tab_hazard)
        self._build_pit(tab_pit)
        self._build_adv(tab_adv)

        # Footer
        footer = ctk.CTkFrame(self)
        footer.pack(fill="x", padx=18, pady=(0, 16))

        self.btn_test = ctk.CTkButton(footer, text="TESTAR (Pass Request)", command=self.test_pass_request)
        self.btn_test.pack(side="left", padx=10, pady=10)

        self.btn_save = ctk.CTkButton(footer, text="SALVAR E FECHAR", command=self.save, fg_color="green", height=40)
        self.btn_save.pack(side="right", padx=10, pady=10, fill="x")

        self._error_label = ctk.CTkLabel(self, text="", text_color="red")
        self._error_label.pack(pady=(0, 10))

    # ---------------- UI building helpers ----------------
    def _section(self, parent, title: str):
        ctk.CTkLabel(parent, text=title, font=("Arial", 14, "bold")).pack(anchor="w", pady=(10, 6), padx=12)

    def _row(self, parent):
        f = ctk.CTkFrame(parent, fg_color="transparent")
        f.pack(fill="x", padx=12, pady=4)
        return f

    def _add_entry(self, parent, label, path, default, caster=float, width=90):
        f = self._row(parent)
        ctk.CTkLabel(f, text=label).pack(side="left", padx=6)
        e = ctk.CTkEntry(f, width=width)
        e.pack(side="right", padx=6)
        e.insert(0, str(_cfg_get(self.flasher.config, path, default)))
        self._entries.append((e, path, caster))
        return e

    def _add_switch(self, parent, label, path, default=False):
        f = self._row(parent)
        sw = ctk.CTkSwitch(f, text=label, onvalue=True, offvalue=False)
        sw.pack(side="left", padx=6)
        if bool(_cfg_get(self.flasher.config, path, default)):
            sw.select()
        self._switches.append((sw, path))
        return sw

    # ---------------- Tabs content ----------------
    def _build_global(self, tab):
        self._section(tab, "Geral / Segurança")
        self._add_switch(tab, "Ativo (habilita/desabilita o app)", ("global", "enabled"), True)
        self._add_switch(tab, "Ignorar enquanto estiver no pit", ("global", "ignore_in_pits"), True)
        self._add_entry(tab, "Tick (Hz):", ("global", "tick_hz"), 20, int)
        self._add_entry(tab, "Vel. mínima do seu carro (km/h):", ("global", "min_my_speed_kmh"), 0.0, float)
        self._add_entry(tab, "Espaçamento mínimo entre piscadas (s):", ("global", "global_press_spacing_s"), 0.12, float)
        self._add_entry(tab, "Máx. piscadas por minuto:", ("global", "max_flashes_per_min"), 40, int)

    def _build_pass(self, tab):
        self._section(tab, "Pedir Passagem (carro à frente)")
        self._add_switch(tab, "Ativar regra", ("rules", "pass_request", "enabled"), True)
        self._add_switch(tab, "Apenas classe diferente (multiclass)", ("rules", "pass_request", "require_diff_class"), True)
        self._add_entry(tab, "Gap p/ disparar (s):", ("rules", "pass_request", "trigger_gap_s"), 0.8, float)
        self._add_entry(tab, "Lookahead (s):", ("rules", "pass_request", "lookahead_s"), 6.0, float)
        self._add_entry(tab, "Velocidade relativa mínima (km/h):", ("rules", "pass_request", "min_rel_speed_kmh"), 8.0, float)
        self._add_entry(tab, "Cooldown por alvo (s):", ("rules", "pass_request", "cooldown_s"), 3.0, float)

        self._section(tab, "Padrão de piscada")
        self._add_entry(tab, "Qtd. piscadas:", ("rules", "pass_request", "pattern", "flashes"), 3, int)
        self._add_entry(tab, "Intervalo mín (s):", ("rules", "pass_request", "pattern", "min_interval_s"), 0.5, float)
        self._add_entry(tab, "Intervalo máx (s):", ("rules", "pass_request", "pattern", "max_interval_s"), 1.2, float)

    def _build_safety(self, tab):
        self._section(tab, "Aproximação rápida (carro à frente)")
        self._add_switch(tab, "Ativar regra", ("rules", "safety_approach", "enabled"), True)
        self._add_entry(tab, "Gap p/ disparar (s):", ("rules", "safety_approach", "trigger_gap_s"), 1.0, float)
        self._add_entry(tab, "Closing rate (s/s):", ("rules", "safety_approach", "closing_rate_s_per_s"), 0.7, float)
        self._add_entry(tab, "Cooldown por alvo (s):", ("rules", "safety_approach", "cooldown_s"), 1.5, float)

        self._section(tab, "Padrão de piscada")
        self._add_entry(tab, "Qtd. piscadas:", ("rules", "safety_approach", "pattern", "flashes"), 1, int)

    def _build_hazard(self, tab):
        self._section(tab, "Perigo à frente (off-track / lento / fechando muito)")
        self._add_switch(tab, "Ativar regra", ("rules", "hazard_ahead", "enabled"), True)
        self._add_entry(tab, "Gap p/ disparar (s):", ("rules", "hazard_ahead", "trigger_gap_s"), 2.0, float)
        self._add_switch(tab, "Disparar se Off-Track", ("rules", "hazard_ahead", "trigger_off_track"), True)
        self._add_switch(tab, "Disparar se alvo estiver lento", ("rules", "hazard_ahead", "trigger_slow_target"), True)
        self._add_entry(tab, "Alvo lento abaixo de (km/h):", ("rules", "hazard_ahead", "target_speed_kmh_below"), 60.0, float)
        self._add_entry(tab, "Closing rate (s/s):", ("rules", "hazard_ahead", "closing_rate_s_per_s"), 1.2, float)
        self._add_entry(tab, "Cooldown por alvo (s):", ("rules", "hazard_ahead", "cooldown_s"), 4.0, float)

        self._section(tab, "Padrão de piscada")
        self._add_entry(tab, "Qtd. piscadas:", ("rules", "hazard_ahead", "pattern", "flashes"), 1, int)

        self._section(tab, "Extra: Você saiu da pista (opcional)")
        self._add_switch(tab, "Ativar regra (self_off_track_hazard)", ("rules", "self_off_track_hazard", "enabled"), False)
        self._add_entry(tab, "Vel. mínima p/ disparar (km/h):", ("rules", "self_off_track_hazard", "min_speed_kmh"), 30.0, float)
        self._add_entry(tab, "Qtd. piscadas:", ("rules", "self_off_track_hazard", "pattern", "flashes"), 2, int)

        self._section(tab, "Extra: Carro muito lento (on track)")
        self._add_switch(tab, "Ativar regra (slow_car_ahead)", ("rules", "slow_car_ahead", "enabled"), True)
        self._add_entry(tab, "Gap p/ disparar (s):", ("rules", "slow_car_ahead", "trigger_gap_s"), 1.5, float)
        self._add_entry(tab, "Alvo lento abaixo de (km/h):", ("rules", "slow_car_ahead", "target_speed_kmh_below"), 80.0, float)
        self._add_entry(tab, "Qtd. piscadas:", ("rules", "slow_car_ahead", "pattern", "flashes"), 2, int)

    def _build_pit(self, tab):
        self._section(tab, "Pit Exit / Merge warning (quando sai do pit)")
        self._add_switch(tab, "Ativar regra", ("rules", "pit_exit_merge", "enabled"), True)
        self._add_entry(tab, "Cooldown global (s):", ("rules", "pit_exit_merge", "cooldown_s"), 10.0, float)

        self._section(tab, "Padrão de piscada")
        self._add_entry(tab, "Qtd. piscadas:", ("rules", "pit_exit_merge", "pattern", "flashes"), 2, int)
        self._add_entry(tab, "Intervalo mín (s):", ("rules", "pit_exit_merge", "pattern", "min_interval_s"), 0.25, float)
        self._add_entry(tab, "Intervalo máx (s):", ("rules", "pit_exit_merge", "pattern", "max_interval_s"), 0.45, float)

    def _build_adv(self, tab):
        self._section(tab, "Avançado / Debug")
        self._add_switch(tab, "Debug (log no console)", ("global", "debug"), False)
        self._add_switch(tab, "Tecla extended (raramente necessário)", ("key_extended",), False)

        f = self._row(tab)
        ctk.CTkLabel(
            f,
            text=f"Arquivo: {CONFIG_FILE}",
            justify="left",
            wraplength=460,
        ).pack(side="left", padx=6)

    # ---------------- Actions ----------------
    def bind_key(self):
        self.btn_bind.configure(text="APERTE A TECLA (ESC cancela)...", fg_color="orange")
        self.update()

        evt = keyboard.read_event(suppress=True)
        while evt.event_type != "down":
            evt = keyboard.read_event(suppress=True)

        if (evt.name or "").lower() == "esc":
            self.btn_bind.configure(text=f"TECLA: {self.flasher.config.get('key_name','?')}", fg_color="#1f538d")
            return

        self.flasher.config["scan_code"] = int(evt.scan_code)
        self.flasher.config["key_name"] = (evt.name or "?").upper()

        self.btn_bind.configure(text=f"TECLA: {self.flasher.config['key_name']}", fg_color="#1f538d")

    def open_config_folder(self):
        try:
            os.makedirs(CONFIG_DIR, exist_ok=True)
            # Windows
            if hasattr(os, "startfile"):
                os.startfile(CONFIG_DIR)  # noqa
        except Exception:
            pass

    def test_pass_request(self):
        try:
            self.flasher.test_flash("pass_request")
        except Exception:
            pass

    def save(self):
        self._error_label.configure(text="")

        # parse entries
        try:
            for e, path, caster in self._entries:
                raw = e.get().strip()
                val = caster(raw) if raw != "" else caster(0)
                _cfg_set(self.flasher.config, path, val)

            for sw, path in self._switches:
                _cfg_set(self.flasher.config, path, bool(sw.get()))

            # garante versionamento
            if "version" not in self.flasher.config:
                self.flasher.config["version"] = 2

            self.flasher.save_config()

            was_running = bool(getattr(self.flasher, "running", False))
            self.flasher.stop()
            if was_running:
                self.flasher.start()

            self.destroy()
        except Exception as e:
            self._error_label.configure(text=f"Erro ao salvar: {e}")
