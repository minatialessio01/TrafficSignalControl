# Piano di Implementazione: Pipeline Sperimentale per la Tesi

## Obiettivo
Costruire una pipeline completa e riproducibile per confrontare il modello **MetaSTGAT avanzato** (sviluppato per la tesi) contro il modello **MetaSTGAT originale del paper**, una serie di **varianti ablate**, e i **baseline algoritmici** (FixedTime, MaxPressure), generando automaticamente tutti i grafici e le tabelle necessari per la tesi.

---

## Modelli da confrontare

| ID | Nome | Modifiche Disattivate rispetto al Pro | Descrizione |
|----|------|----------------------------------------|-------------|
| `metastgat_pro` | MetaSTGAT Pro | — | Modello avanzato completo (stato attuale del codice) |
| `metastgat_paper` | MetaSTGAT Paper | Tutte (M1..M10) | Modello originale del paper: reward `-P_i`, stato 20-dim, no mask, DQN base, buffer flat 10k, single-step, no warmup |
| `ablation_environment` | Ablation 1 – Environment & MDP | M1, M2, M3, M4 | Come Pro ma: reward originale `-P_i`, visibilità globale (no cutoff campo visivo), stato 20-dim (no `wait_vec`), no action mask |
| `ablation_temporal` | Ablation 2 – Temporal LSTM | M5 | Come Pro ma: training single-step (no BPTT, no burn-in, $L=1$) |
| `ablation_rl_core` | Ablation 3 – RL Core | M6 | Come Pro ma: DQN standard invece di Double DQN |
| `ablation_replay_stability` | Ablation 4 – Replay & Stability | M7, M8, M9, M10 | Come Pro ma: buffer uniforme flat 10k (no PER, no IS), MSE loss, hard target update, no grad clip, no warmup, no esplorazione ciclica, no Tanh |
| `fixedtime` | FixedTime | — | Ciclo fisso (nessun training) |
| `maxpressure` | MaxPressure | — | Algoritmo a pressione (nessun training) |

> [!NOTE]
> Ablation basate sulla **Proposta 1 (Sottosistemi Funzionali)** del file `proposte_ablation.md`. CoLight rimosso. Le ablation usano lo stesso launcher (`train.py`/`run_experiment.py`) con flag specifici che disabilitano i blocchi.

---

## Configurazioni di Training e Test

### Training (2 config, stessa griglia 4x4_200m, densità diversa)
- `config_4x4_200m_2k_flat` — traffico leggero, flat
- `config_4x4_200m_2k_peak` — traffico leggero, con picco

Ogni modello viene addestrato sequenzialmente sulle due config per N episodi ciascuna (N configurabile da CLI, default `5` per dev test, produzione `30-50`). Il cambio di config produce la **linea verticale** sui grafici.

### Test (tutte le rimanenti 8 config)
| Config | Veicoli effettivi | Scopo |
|--------|-------------------|-------|
| `config_4x4_200m_6k_flat` | 6320 | Stress stesso layout |
| `config_4x4_200m_6k_peaks` | 6522 | Stress stesso layout, double peak |
| `config_4x4_300m_6k_flat` | 6320 | Stesso traffico, strade più lunghe |
| `config_4x4_300m_6k_peak` | 6686 | Stesso traffico + picco, strade lunghe |
| `config_5x5_200m_8k_flat` | 8481 | Griglia più grande, zero-shot |
| `config_5x5_200m_8k_peaks` | 8817 | Griglia più grande, double peak |
| `config_6x6_200m_10k_flat` | 10390 | Griglia massima, zero-shot |
| `config_6x6_200m_10k_peaks` | 11093 | Griglia massima, double peak |

---

## Output Attesi

### Grafici (generati automaticamente)
1. **Loss con media mobile** — uno per modello, con linea verticale al cambio config durante il training (solo modelli addestrati)
2. **Travel Time con media mobile** — come sopra
3. **Phase Percentage** — un grafico per modello, media su tutte le configurazioni di test
4. **Barre Travel Time & Throughput per config di test** — un grafico per ogni config di test, confronta tutti i modelli (escluse le ablation) affiancati; nel grafico Throughput c'è una linea orizzontale con il numero totale di veicoli spawnati da quella configurazione (vedi `descrizione_configurazioni.md`)

### Tabelle
5. **Tabella Travel Time** — righe = modelli, colonne = config di test; formato CSV + LaTeX
6. **Tabella Throughput** — come sopra, il valore è la percentuale di veicoli completati rispetto al totale spawnato

> [!NOTE]
> I grafici a barre (punto 4) includono solo i modelli principali (Pro, Paper, FixedTime, MaxPressure). Le ablation hanno grafici separati (Loss + TT) ma NON compaiono nel grafico a barre finale per non sovraffollarlo. Le ablation compaiono invece nelle tabelle.

---

## Struttura di File e Cartelle Proposta

```
CodiceTesi/
├── scripts/
│   ├── generate_synthetic_data.py   # (già pronto)
│   ├── train.py                     # (da modificare per supportare ablation flags)
│   ├── test.py                      # (da modificare per output CSV strutturati)
│   ├── plot_results.py              # [NEW] genera tutti i grafici da CSV aggregati
│   ├── run_experiment.py            # [NEW] orchestratore completo (train → test → plot)
│   └── ...
├── results/
│   ├── metastgat_pro/
│   │   ├── training_log.csv
│   │   └── test_<config_name>.csv
│   ├── metastgat_paper/
│   ├── ablation_environment/
│   ├── ablation_temporal/
│   ├── ablation_rl_core/
│   ├── ablation_replay_stability/
│   ├── fixedtime/
│   └── maxpressure/
│   └── plots/                       # [NEW] grafici finali
│       ├── loss_<model>.png
│       ├── travel_time_<model>.png
│       ├── phase_pct_<model>.png
│       ├── bars_<config>.png
│       ├── table_travel_time.csv
│       ├── table_travel_time.tex
│       ├── table_throughput.csv
│       └── table_throughput.tex
```

---

## Modifiche al Codice Esistente

### `scripts/train.py` — Aggiunta flag ablation
Aggiungere parametri `argparse` per attivare/disattivare ogni blocco funzionale:
- `--reward-mode [custom|paper]` → formula reward: `custom` (weighted pressure, attuale) vs `paper` (-P_i)
- `--no-vision-cutoff` → rimuove il cutoff sul campo visivo (VISION_CUTOFF_M, ~144m) (usa visibilità globale come nel paper)
- `--no-wait-vec` → stato dim=20 senza `wait_vec` (rimuove la terza componente)
- `--no-action-mask` → rimuove la maschera azioni invalide (consente stessa fase ≥3 volte)
- `--no-bptt` → training single-step senza BPTT e senza burn-in ($L=1$)
- `--no-double-dqn` → DQN standard (target = $r + \gamma \max Q$)
- `--no-per` → buffer FIFO flat da 10k transizioni, campionamento uniforme, no IS weights
- `--no-huber` → MSE loss invece di Huber Loss
- `--no-soft-update` → hard update periodico del target network
- `--no-grad-clip` → disabilita gradient clipping
- `--no-warmup` → rimuove i 10 episodi di warm-up iniziali
- `--no-cyclic-exploration` → rimuove l'episodio random periodico ogni 10 ep
- `--no-tanh-meta` → rimuove Tanh finale dai meta-learner SMK/TMK
- `--ablation [full|paper|environment|temporal|rl_core|replay_stability]` → **preset** che attiva automaticamente la combinazione corretta di flag per ogni variante

**Mappa preset `--ablation`:**

| Preset | Flag attivati automaticamente |
|--------|-------------------------------|
| `full` | (nessuno — modello avanzato completo) |
| `paper` | tutti i `--no-*` + `--reward-mode paper` + `--no-vision-cutoff` |
| `environment` | `--reward-mode paper` + `--no-vision-cutoff` + `--no-wait-vec` + `--no-action-mask` |
| `temporal` | `--no-bptt` |
| `rl_core` | `--no-double-dqn` |
| `replay_stability` | `--no-per` + `--no-huber` + `--no-soft-update` + `--no-grad-clip` + `--no-warmup` + `--no-cyclic-exploration` + `--no-tanh-meta` |

### `scripts/test.py` — Output CSV strutturato
- Salvare per ogni episode di test: `episode, travel_time, throughput, phase_percentages[]`
- Salvare in `results/<model>/test_<config_name>.csv`

### `scripts/run_experiment.py` — [NUOVO] Orchestratore
```
python scripts/run_experiment.py \
  --train-configs configs/config_4x4_200m_2k_flat.json configs/config_4x4_200m_2k_peak.json \
  --test-configs configs/config_4x4_200m_6k_flat.json ... \
  --episodes-per-config 5 \
  --models all   # oppure: metastgat_pro fixedtime maxpressure
```
- Lancia `train.py` per ogni modello addestrabile
- Lancia `test.py` per ogni modello × ogni config di test
- Chiama `plot_results.py` alla fine

### `scripts/plot_results.py` — [NUOVO] Generazione grafici
- Legge tutti i CSV da `results/`
- Genera i 4 tipi di grafici descritti sopra
- Genera le 2 tabelle in CSV e `.tex`

---

## Piano di Implementazione (Fasi)

- [ ] **Fase 1** – Modificare `train.py` per supportare i flag di ablation e salvare `training_log.csv` strutturato
- [ ] **Fase 2** – Modificare `test.py` per salvare CSV strutturati per config
- [ ] **Fase 3** – Scrivere `run_experiment.py` (orchestratore)
- [ ] **Fase 4** – Scrivere `plot_results.py` (generazione grafici e tabelle)
- [ ] **Fase 5** – Test end-to-end con `--episodes-per-config 5` su Docker
- [ ] **Fase 6** – Lancio produzione con `--episodes-per-config 50`

---

## Piano di Verifica

### Automatico
```bash
# Quick smoke test (5 ep per config, tutti i modelli)
python scripts/run_experiment.py --episodes-per-config 5 --models all

# Verifica che le cartelle siano create correttamente
ls results/
# Verifica che i CSV esistano
ls results/metastgat_pro/
```

### Manuale
- Controllare che i grafici abbiano la linea verticale nella posizione giusta
- Verificare che la linea orizzontale nel grafico throughput corrisponda ai valori in `descrizione_configurazioni.md`
- Controllare che le tabelle `.tex` si compilino senza errori in LaTeX
