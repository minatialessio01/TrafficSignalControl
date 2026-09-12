# MetaSTGAT — Replica e Ablation Study

Replica e studio di ablazione del modello **MetaSTGAT** (Wang et al., 2022):
> *"Meta-learning based spatial-temporal graph attention network for traffic signal control"*
> Knowledge-Based Systems 250 (2022) 109166

Progetto di tesi: implementazione da zero dell'architettura, ambiente di simulazione multi-intersezione su CityFlow, generazione parametrica dei dataset, sistema di ablation study a preset, e un esperimento di generalizzazione mirato (traffico multi-variante + arteria asimmetrica) per mettere alla prova sia il modello avanzato sia le baseline classiche.

---

## Indice della documentazione

Questo repository usa un file Markdown per argomento, pensato per essere letto (anche da un assistente IA) come contesto strutturato per la stesura della tesi:

| Documento | Cosa descrive |
|---|---|
| [descrizione_src.md](descrizione_src.md) | Codice sorgente (`src/`): ambiente MDP, architettura MetaSTGAT/STGAT e sotto-moduli, agenti (DQN/MaxPressure/FixedTime), replay buffer, logger |
| [descrizione_environment.md](descrizione_environment.md) | L'ambiente di simulazione in dettaglio: CityFlow, gestione verde/giallo/rosso, formulazione MDP, ablation lato ambiente, replay/visualizzazione |
| [descrizione_metriche.md](descrizione_metriche.md) | Ogni metrica di valutazione prodotta dal codice: travel time (3 varianti), throughput, percentuali di fase, coda del travel time, equità direzionale N/S vs W/E |
| [descrizione_scripts.md](descrizione_scripts.md) | Script eseguibili (`scripts/`): training, test, orchestrazione della pipeline, generazione dati, grafici |
| [descrizione_configurazioni.md](descrizione_configurazioni.md) | Dataset e config CityFlow: roadnet, flussi di traffico, nomenclatura, densità calibrate, arteria e varianti multi-seed per la generalizzazione |
| [descrizione_modelli.md](descrizione_modelli.md) | **Il catalogo dei modelli**: quali sono stati testati e in cosa differiscono, lungo i 3 assi paper/alpha/ablation — punto di partenza, fonde `metastgat_diff_analysis.md` e `proposte_ablation.md` |
| [metastgat_diff_analysis.md](metastgat_diff_analysis.md) | Confronto sistematico codice vs paper originale, differenza per differenza, con estratti di codice (dettaglio dietro `descrizione_modelli.md` §1) |
| [proposte_ablation.md](proposte_ablation.md) | Le 4 proposte alternative di raggruppamento per l'ablation study, con vantaggi/svantaggi di ciascuna (dettaglio dietro `descrizione_modelli.md` §3, solo la Proposta 1 è implementata) |
| [piano_tesi.md](piano_tesi.md) | Piano di stesura della tesi vera e propria: struttura dei capitoli, stato di ciascuna sezione (pronta/da completare), notazione matematica unificata, bibliografia nota, convenzioni LaTeX |
| [descrizione_gcn_sonar.md](descrizione_gcn_sonar.md) | Seconda parte del progetto: MetaSTGNN (GAT→GCN) e MetaSTSONAR (GAT→SONAR) — design, decisioni D1-D6, verifica pre-training, come riprodurre |

`implementation_plan.md` e `task.md` (spec/pianificazione iniziale della tesi) sono stati rimossi il 12/9/2026: riferivano script (`run_experiment.py`) e config (`config_4x4_200m_2k_flat`, ecc.) mai più esistiti nell'albero attuale, completamente superati dalla pipeline reale (`main.py`/`train.py`/`test.py`) e dalle config effettivamente in uso — nessuna informazione persa, recuperabili dalla cronologia git se mai servisse. [istruzioni seconda parte.md](istruzioni%20seconda%20parte.md) resta invece il piano di riferimento per `descrizione_gcn_sonar.md` (implementato il 13/9/2026), non uno storico.

---

## Struttura del progetto

```
CodiceTesi/
├── README.md
├── docker/                    # Dockerfile + requirements (ambiente di esecuzione)
├── configs/                   # Config CityFlow (accoppiano roadnet+flow+parametri simulazione)
├── data/                      # Roadnet e flow generati parametricamente
├── src/
│   ├── environment/           # Wrapper CityFlow → MDP (cityflow_env.py)
│   ├── models/                # MetaSTGAT, STGAT e sotto-moduli (meta-GAT, meta-LSTM, ...)
│   ├── agents/                # DQN, MaxPressure, FixedTime, replay buffer
│   └── utils/                 # Logger di training, metriche episodiche
├── scripts/
│   ├── generate_synthetic_data.py   # Generazione roadnet+flow+config
│   ├── train.py                     # Training (preset ablation, multi-config, resume)
│   ├── test.py                      # Valutazione di un modello/baseline
│   ├── main.py                      # Punto di ingresso: train+test+plot per 1..N modelli
│   ├── plot_results.py              # Grafici e tabelle comparative
│   └── inspect_replay.py            # Replay MaxPressure/random senza training, per ispezione
└── results/                   # Checkpoint, log, metriche (creato automaticamente)
```

---

## Setup

CityFlow funziona nativamente solo su Linux. Su Windows la soluzione usata in questo progetto è **Docker** (via WSL2):

```bash
cd docker
docker build -t metastgat ..    # contesto = root del repo, non docker/
```

Esecuzione (monta il repo in `/workspace`, così le modifiche locali sono visibili subito senza ricostruire l'immagine):

```bash
docker run --rm -v <path-assoluto-repo>:/workspace -w /workspace metastgat \
    python3 -u scripts/main.py --model metastgat_pro --episodes-per-config 100
```

Su Windows con WSL2, da PowerShell: `wsl.exe docker run --rm -v /mnt/c/percorso/CodiceTesi:/workspace -w /workspace metastgat python3 -u <script> <args>`.

---

## Quick start

```bash
# 1. Genera i dati (roadnet + flow + config) — vedi descrizione_configurazioni.md per i parametri
python scripts/generate_synthetic_data.py --grid 4x4 --road-length 107 --road-length-label 100 \
    --duration 1800 --variance workday

# 2. Un solo modello, comodo per il debug (in-process, breakpoint funzionanti)
python scripts/main.py --model metastgat_pro --episodes-per-config 100

# 3. Tutta la pipeline: 8 modelli (MetaSTGAT pro/paper, 4 ablation, FixedTime, MaxPressure),
#    training + validazione + test di generalizzazione + grafici/tabelle finali
python scripts/main.py --models all
```

Per training/test più mirati (una config sola, preset di ablation specifico, resume da checkpoint) vedi gli esempi in [descrizione_scripts.md](descrizione_scripts.md).

---

## I modelli confrontati

| model_id | Modello | Ablation | Allenabile |
|---|---|---|---|
| `metastgat_pro` | MetaSTGAT | nessuna (completo) | sì |
| `metastgat_paper` | MetaSTGAT | replica fedele del paper | sì |
| `ablation_environment` | MetaSTGAT | senza reward custom/visibilità/wait/mask | sì |
| `ablation_temporal` | MetaSTGAT | senza BPTT+burn-in | sì |
| `ablation_rl_core` | MetaSTGAT | senza Double DQN | sì |
| `ablation_replay_stability` | MetaSTGAT | senza PER/Huber/grad-clip/warmup | sì |
| `fixedtime` | Fasi cicliche a tempo fisso | — | no (solo baseline) |
| `maxpressure` | MaxPressure (Varaiya 2013) | — | no (solo baseline) |

Dettaglio di ogni meccanismo di ablation (M1-M10) e della loro mappatura sui preset in [metastgat_diff_analysis.md](metastgat_diff_analysis.md) e [proposte_ablation.md](proposte_ablation.md).

---

## Checkpoint e risultati

Ogni modello allenato scrive in `results/<model_id>/`:
- `checkpoint_epXXXX.pt` — checkpoint periodico (ogni 10 episodi), `best_model.pt` — miglior travel time, `final_model.pth` — ultimo episodio, `selected_model.pth` — vincitore della selezione automatica finale (usato di default da `test.py`).
- `training_log.csv` / `training_state.json` — metriche per episodio / stato corrente.
- `test_<config>.csv` / `test_summary_<config>.json` / `phase_pct_<config>.png` — per ogni config di test.

Un training interrotto (Ctrl+C, o `docker stop -s SIGINT` se in container) può essere ripreso esattamente da dove si trovava con `--resume <checkpoint>` — dettaglio completo in [descrizione_scripts.md](descrizione_scripts.md).

---

## Riferimento

```bibtex
@article{wang2022metastgat,
  title={Meta-learning based spatial-temporal graph attention network for traffic signal control},
  author={Wang, Min and Wu, Libing and Li, Man and Wu, Dan and Shi, Xiaochuan and Ma, Chao},
  journal={Knowledge-Based Systems},
  volume={250},
  pages={109166},
  year={2022},
  publisher={Elsevier}
}
```
