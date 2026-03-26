# Nishizumi Tools

> 🚀 **Release mais recente:** **[Baixar aqui](https://github.com/Nishizumi-Tools/Nishizumi-Tools/releases/latest)**

Coleção de overlays/assistentes para iRacing em Python. Cada app resolve um problema diferente durante treino, quali e corrida.

## Apps atuais do repositório

- **Nishizumi FuelMonitor** (`apps/Nishizumi_FuelMonitor.py`)  
  Monitor de consumo por volta, projeção de stint e delta para meta de economia.
- **Nishizumi Pit Calibrator** (`apps/nishizumi_pitcalibrator.py`)  
  App novo para **calibrar pit stop real** (tempo total, tempo de serviço, tempo base, combustível e taxa de abastecimento).
- **Nishizumi TireWear** (`apps/Nishizumi_TireWear.py`)  
  Modelo de desgaste de pneus por combinação carro/pista.
- **Nishizumi Traction** (`apps/Nishizumi_Traction.py`)  
  Overlay de coaching de aderência com referência por volta.

---

## Requisitos gerais

- Python **3.10+**
- iRacing aberto com telemetria ativa (entrar em sessão e clicar em **Drive**)
- Dependências:

```bash
pip install irsdk numpy pyqt5
```

Notas:
- `tkinter` já vem com a maioria das instalações padrão de Python.
- Em Windows, os apps que persistem dados usam `%APPDATA%/NishizumiTools`.
- Em Linux/macOS, fallback em `~/.config/NishizumiTools`.

---

## Novo app: Nishizumi Pit Calibrator (detalhado)

**Arquivo:** `apps/nishizumi_pitcalibrator.py`  
**Execução:**

```bash
python apps/nishizumi_pitcalibrator.py
```

### Objetivo

O Pit Calibrator foi feito para uma tarefa específica: **medir e congelar números reais do seu pit stop** para você anotar e usar na estratégia.

Ele não tenta prever janela, não salva perfis automaticamente, e não aplica “mágica” de estratégia. É um calibrador manual, rápido e direto.

### O que ele mede

Quando você arma uma parada:

- **Live total**: tempo total em pit road durante a parada armada.
- **Live service**: tempo de serviço ativo (abastecimento/troca efetiva).
- **Live base**: diferença entre total e serviço (perda base de entrada/saída + tempos sem serviço ativo).
- **Live fuel added**: combustível realmente adicionado na parada.
- **Live fuel rate**: taxa média de abastecimento (L/s) calculada por amostras válidas.
- **Live manual tire time**: tempo manual de pneus quando você marca o botão **TIRE**.
- **Pending pit fuel**: combustível programado no pit service (`PitSvFuel`) para referência.

Ao sair do pit road, os dados viram:

- **Saved total**
- **Saved service**
- **Saved base**
- **Saved fuel added**
- **Saved fuel rate**
- **Saved tire time**

E ficam congelados na tela até a próxima medição.

### Fluxo correto de uso (passo a passo)

1. Abra iRacing e entre no carro.
2. Rode o app.
3. Clique **ARM** antes da volta em que você vai parar.
4. Entre no pit lane normalmente.
5. Durante a parada, acompanhe os campos **Live** atualizando em tempo real.
6. Quando a troca de pneus terminar, clique **TIRE** (opcional, manual).
7. Saia do pit road.
8. O app finaliza automaticamente a medição e congela os valores em **Saved**.
9. Anote os números para alimentar seu setup/planilha/estratégia.

### Botões e controles

- **ARM**: arma/desarma a próxima parada válida.
- **TIRE**: grava manualmente o tempo de pneus no instante do clique (só funciona durante a parada armada ativa).
- **✕**: fecha o app.
- Janela é **arrastável** pela barra superior.

### Comportamentos importantes

- Ele mede **somente a parada armada** (não tenta capturar histórico inteiro).
- Quando desconectado do iRacing, mostra estado de espera.
- Atualiza em alta frequência (refresh rápido) para capturar melhor transições de serviço.
- A taxa de combustível ignora valores absurdos (filtro de faixa razoável).
- Não cria perfil por carro/pista e não persiste histórico por design.

### Quando usar

- Pré-corrida para calibrar pit loss real da combinação.
- Treino de endurance para separar claramente:
  - perda base de pit lane,
  - tempo de serviço,
  - impacto de combustível,
  - tempo de pneus (manual).

---

## FuelMonitor

**Arquivo:** `apps/Nishizumi_FuelMonitor.py`  
**Execução:**

```bash
python apps/Nishizumi_FuelMonitor.py
```

Resumo: média de consumo por volta, delta para meta, combustível restante, voltas estimadas, último giro e comparação de stint planejado vs real.

---

## TireWear

**Arquivo:** `apps/Nishizumi_TireWear.py`  
**Execução:**

```bash
python apps/Nishizumi_TireWear.py
```

Resumo: aprende desgaste por combo carro/pista e estima vida útil de LF/RF/LR/RR com confiança do modelo.

---

## Traction

**Arquivo:** `apps/Nishizumi_Traction.py`  
**Execução:**

```bash
python apps/Nishizumi_Traction.py
```

Resumo: círculo de tração, referência adaptativa por setor e coaching para apontar onde há perda de aderência/tempo.

---

## Estrutura do repositório (atualizada)

- `README.md`
- `requirements.txt`
- `LICENSE`
- `apps/Nishizumi_FuelMonitor.py`
- `apps/nishizumi_pitcalibrator.py`
- `apps/Nishizumi_TireWear.py`
- `apps/Nishizumi_Traction.py`
- `docs/fuel-monitor.md`

---

## Dica rápida de ordem de uso

1. **FuelMonitor** para decisões imediatas de consumo.
2. **Pit Calibrator** para obter números reais de parada.
3. **Traction** para evolução de pilotagem.
4. **TireWear** para stints longos/endurance.
