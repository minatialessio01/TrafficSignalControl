# Descrizione degli Script (`scripts/`)

Punti di ingresso eseguibili: generazione dati, training, valutazione, orchestrazione della pipeline completa, grafici. Per il significato dei singoli file di config/dati vedi [descrizione_configurazioni.md](descrizione_configurazioni.md); per la formulazione MDP vedi [descrizione_environment.md](descrizione_environment.md); per le metriche prodotte vedi [descrizione_metriche.md](descrizione_metriche.md).

```
scripts/
├── generate_synthetic_data.py   → genera roadnet + flow + config CityFlow
├── train.py                     → training di un modello (MetaSTGAT/STGAT/FixedTime)
├── test.py                      → valutazione di un modello già allenato (o baseline)
├── main.py                      → orchestra train.py + test.py + plot_results.py per 1..N modelli
├── plot_results.py              → grafici e tabelle comparative da results/
└── inspect_replay.py            → replay "usa e getta" con MaxPressure/random, per ispezionare una config prima di allenarci sopra
```

---

## `train.py` — training

Implementa l'Algorithm 1 dell'articolo (raccolta episodio → update del modello), con l'aggiunta di: preset di ablation, ciclo tra più varianti di training, warm-up, esplorazione ciclica, checkpoint/resume, interruzione pulita, selezione automatica finale.

### Preset di ablation (`--ablation`)

`apply_ablation_preset()` imposta un gruppo coerente di flag `--no-*` (vedi tabella sotto), sovrascrivibili singolarmente da riga di comando (i flag espliciti dell'utente hanno sempre precedenza sul preset). Corrispondono esattamente alla "Proposta 1" di [proposte_ablation.md](proposte_ablation.md) (raggruppamento per sottosistemi funzionali):

| Preset | reward_mode | Flag disattivati | Cosa isola |
|---|---|---|---|
| `pro` (default) | custom | nessuno | Modello avanzato completo — benchmark di riferimento |
| `paper` | paper | **tutti**: vision_cutoff, wait_vec, action_mask, bptt, double_dqn, per, huber, soft_update, grad_clip, warmup, cyclic_exploration, tanh_meta | Replica fedele della procedura originale Wang et al. 2022 |
| `environment` | paper | vision_cutoff, wait_vec, action_mask | Quanto incide l'ingegnerizzazione dell'MDP (reward, visibilità, wait_vec, anti-starvation) |
| `temporal` | custom | bptt | Quanto incide BPTT+burn-in (R2D2-style) sulla Meta-LSTM |
| `rl_core` | custom | double_dqn | Quanto incide Double DQN sulla sovrastima dei Q-value |
| `replay_stability` | custom | per, huber, soft_update, grad_clip, warmup, cyclic_exploration, tanh_meta | Quanto incidono PER/IS, ottimizzazione robusta, warm-up ed esplorazione ciclica |

Ogni flag atomico (`--reward-mode`, `--no-vision-cutoff`, `--no-wait-vec`, `--no-action-mask`, `--no-bptt`, `--no-double-dqn`, `--no-per`, `--no-huber`, `--no-soft-update`, `--no-grad-clip`, `--no-warmup`, `--no-cyclic-exploration`, `--no-tanh-meta`) resta comunque disponibile per combinazioni ad-hoc fuori dai 6 preset.

### Warm-up, training, esplorazione ciclica

1. **Warm-up** (`warmup_episodes=10`, disattivabile con `--no-warmup`): episodi con `epsilon=1.0` (azioni casuali), **non loggati**, il cui unico scopo è riempire il replay buffer prima che parta il vero training (gate `min_buffer_size=1000`). Con `--config` a più varianti, il warm-up cicla comunque tra le varianti (`_variant_idx(w_ep, n_variants)`), così il buffer iniziale non è tutto della stessa variante.
2. **Training vero**: per ogni episodio, `switch_variant(_variant_idx(episode, n_variants))` seleziona la config di questo episodio, poi si raccoglie l'episodio e si chiama `agent.update(n_updates=100)` — **100 gradient step per episodio, un numero fisso indipendente dal numero di step ambientali** (120 con episodi da 1800s): non c'è alcun vincolo che li leghi, è un iperparametro ereditato dal paper (rimasto 100 anche quando la durata degli episodi era diversa). L'epsilon decade una volta per episodio, verso `epsilon_end` raggiunto a `eps_fraction=0.6` degli episodi totali (`_eps_decay_for()`, ricalcolato in base a `--episodes` effettivo, non un tasso fisso).
3. **Esplorazione ciclica** (ogni 10 episodi, disattivabile con `--no-cyclic-exploration`): un episodio extra con `epsilon=1.0`, **non usato per l'update dei pesi** (si esegue subito dopo l'episodio normale di quello stesso indice, sulla stessa variante già attiva — non sceglie esplicitamente una config, eredita quella corrente). Con 3 varianti di training e `episode % 10 == 0`, su 100 episodi le occorrenze (10,20,...,90) cadono su tutte e 3 le varianti in modo perfettamente bilanciato (3 occorrenze a testa), per pura aritmetica (10 e 3 coprimi, 90/10 multiplo di 3) — non è una garanzia strutturale del codice, va riverificata se si cambiano questi numeri.

   **Ipotesi valutata e non applicata (11-12/9/2026)**: fermarla non appena `epsilon` raggiunge `epsilon_end`, sull'idea che un episodio interamente casuale a policy ormai quasi greedy inserisca nel buffer transizioni fuori distribuzione con `max_priority` (stessa priorità di una sequenza buona), che il PER ricampia subito prima di sapere se sono informative. Il segnale sulla loss (un picco ripetuto subito dopo ogni occorrenza tardiva) era reale, ma controllando per la variante di training attiva — che cambia comunque ad ogni episodio — l'effetto sul travel time dell'episodio successivo è risultato inconcludente (un caso su tre nettamente migliore, non peggiore). Non abbastanza per modificare un meccanismo esistente e deliberatamente documentato (M9 in `proposte_ablation.md`, pensato per esplorare "anche a training avanzato"): il comportamento resta quello originale. Da riconsiderare con un vero confronto A/B a valle di un training completo.

### Multi-variante di training (`--config` con più file)

`--config` accetta `nargs="+"`: se più di un file, si cicla una variante per episodio (`episodio % N`, sia in warm-up che nel training vero — vedi `_variant_idx()`). Pensato per varianti "sostanza-preservante" (stesso roadnet/densità, seed diverso — vedi [descrizione_configurazioni.md §4bis](descrizione_configurazioni.md)), non per un curriculum di config diverse per difficoltà. `switch_variant()` ricostruisce `CityFlowEnv`/`edge_index` solo quando la variante cambia realmente rispetto a quella già caricata (no-op altrimenti).

### Checkpoint, interruzione, resume

- **Periodico**: ogni `checkpoint_interval=10` episodi (`logger.should_save_checkpoint()`), file `checkpoint_epXXXX.pt`.
- **Best model**: `best_model.pt`, aggiornato ogni volta che il travel time dell'episodio migliora il minimo storico (`agent.save_best()`).
- **Ctrl+C**: `TrainingLogger` intercetta SIGINT, ma non interrompe a metà episodio — imposta un flag, l'episodio in corso finisce, poi si salva un checkpoint d'emergenza e si stampa il comando `--resume` esatto. **Se il processo gira in un container Docker headless**, Ctrl+C da tastiera non arriva: va inviato un SIGINT esplicito al processo (es. `docker stop -s SIGINT -t <timeout>`, con timeout abbastanza lungo da coprire la fine dell'episodio in corso).
- **`--resume <checkpoint>`**: ripristina pesi, optimizer, `episode`, `epsilon`, `best_travel_time` e **il replay buffer** (bug corretto l'11/9/2026 — senza, il buffer ripartiva vuoto). Il ciclo tra varianti di training riprende automaticamente allineato, perché `_variant_idx()` dipende solo dal numero assoluto di episodio, non da uno stato salvato a parte.
- **`--stop-at N`**: ferma il training a un episodio specifico (utile per test/debug), salva un checkpoint.
- Alla fine di un training **non interrotto**, si genera `training_curves.png` (TT/throughput/loss vs episodio) e si esegue la **selezione finale** (sotto).

### Selezione finale del modello (`run_final_selection`)

A fine training confronta `final_model.pth` (ultimo episodio) e `best_model.pt` (miglior travel time mai visto) su `--select-best-config` (una config di validazione a seed fisso, default `config_4x4_100m_6k_flat.json`), con `--select-best-episodes=1` episodio ciascuno (a epsilon=0 e simulazione deterministica, ripetere non aggiunge informazione). Il vincitore è copiato in `selected_model.pth`, che `test.py` cerca con priorità massima. Disattivabile con `--no-select-best`.

### Modelli addestrabili

`--model {MetaSTGAT, STGAT, FixedTime, CoLight}`. `FixedTime` salta tutta l'infrastruttura DQN (`run_fixedtime()`: nessun training, solo valutazione su pochi episodi). `CoLight` è referenziato nel parser ma il modulo `src/agents/colight_agent.py` **non esiste più** nell'albero corrente — scegliere `--model CoLight` fallisce all'import.

### Uso

```bash
# Training singola config
python scripts/train.py --config configs/config_4x4_100m_train1.json

# 3 varianti sostanza-preservante in ciclo (un episodio ciascuna)
python scripts/train.py --config configs/config_4x4_100m_train1.json \
    configs/config_4x4_100m_train2.json configs/config_4x4_100m_train3.json

# Con selezione automatica finale su una config di validazione
python scripts/train.py --config configs/config_4x4_100m_train1.json --episodes 100 \
    --select-best-config configs/config_4x4_100m_6k_flat.json --output-dir results/metastgat_pro

# Preset di ablation
python scripts/train.py --config configs/config_4x4_100m_train1.json --ablation environment

# Riprendere un training interrotto
python scripts/train.py --config configs/config_4x4_100m_train1.json \
    --resume results/metastgat_pro/checkpoint_ep0030.pt --episodes 100
```

---

## `test.py` — valutazione

Carica un checkpoint (priorità di ricerca in `--output-dir`: `selected_model.pth` → `final_model.pth` → `best_model.pt` → `checkpoint_ep*.pt` più recente) e lo valuta per `--n-eval` episodi su una config, con `epsilon=0` (nessuna esplorazione). Default `--n-eval=1`: con flow/seed CityFlow fissi e `laneChange=False` la simulazione è deterministica bit-per-bit, ripeterla non riduce alcuna varianza reale.

Per ogni episodio salva/stampa (dettaglio formule in [descrizione_metriche.md](descrizione_metriche.md)):
- `travel_time` (metrica di ranking, include i veicoli non ancora arrivati) e `travel_time_completed_only` (solo arrivati, informativo);
- `throughput`, `total_vehicles`;
- percentuali di scelta di ciascuna delle 8 fasi (`phase_pct_0..7`);
- coda del travel time dei soli arrivati: `tt_max`, `tt_std`, `tt_p50/p90/p95/p99`;
- equità direzionale: `wait_avg_ns/max_ns`, `wait_avg_ew/max_ew`, `wait_worst`.

Output: `test_<config>.csv` (una riga per episodio), `test_summary_<config>.json` (medie), `phase_pct_<config>.png` (barplot fasi). `--model {MetaSTGAT, STGAT, FixedTime, MaxPressure}` — `FixedTime`/`MaxPressure` non richiedono checkpoint. `--ablation` seleziona il preset per configurare `CityFlowEnv` **esattamente come in training** (deve corrispondere, altrimenti dimensioni di stato/meta-feature non combaciano col checkpoint).

```bash
python scripts/test.py --config configs/config_4x4_100m_6k_peak.json \
    --model-id metastgat_pro --output-dir results/metastgat_pro
```

---

## `main.py` — punto di ingresso unico della pipeline sperimentale

Esegue **in-process** (nessun subprocess — `import train as train_mod` / `import test as test_mod`, quindi debuggabile con breakpoint) l'intera pipeline train→validazione→test→plot per uno o più modelli.

**`MODEL_REGISTRY`** — gli 8 modelli confrontabili:

| model_id | `--model` | `--ablation` | Allenabile |
|---|---|---|---|
| `metastgat_pro` | MetaSTGAT | pro | sì |
| `metastgat_paper` | MetaSTGAT | paper | sì |
| `ablation_environment` | MetaSTGAT | environment | sì |
| `ablation_temporal` | MetaSTGAT | temporal | sì |
| `ablation_rl_core` | MetaSTGAT | rl_core | sì |
| `ablation_replay_stability` | MetaSTGAT | replay_stability | sì |
| `fixedtime` | FixedTime | — | no (solo test) |
| `maxpressure` | MaxPressure | — | no (solo test) |

**Suddivisione dati** (default, sovrascrivibile da CLI — vedi [descrizione_configurazioni.md](descrizione_configurazioni.md)):
- `DEFAULT_TRAIN_CONFIGS` — le 3 varianti seed di `config_4x4_100m_train*.json`, cicliche.
- `DEFAULT_VALIDATION_CONFIG` — `config_4x4_100m_6k_flat.json`, usata da `train.py --select-best-config`.
- `DEFAULT_TEST_CONFIGS` — le altre 7 config (stress/generalizzazione topologica), mai viste in training/selezione.

Per ogni `model_id` in `--models` (o `--model` per uno solo): se allenabile, `train_one_model()` chiama `train_mod.run_training()` con `--config` = tutte le train-configs, poi `test_one_model()` valuta su **sia** le `--train-configs` **che** le `--test-configs` (prassi introdotta il 13/9/2026: prima si testava solo sulla generalizzazione, ma `compare_models.py` confronta i modelli anche sulle config di training — vedi sotto — quindi ogni modello va sempre testato su tutte e 10). Libera la memoria CUDA tra un modello e l'altro (`gc.collect()`+`torch.cuda.empty_cache()`, essendo tutto in-process). Salva `experiment_summary.json` e, a meno di `--no-plot`, chiama `plot_results.py` in-process a fine pipeline.

```bash
# Un solo modello (debug)
python scripts/main.py --model metastgat_pro --episodes-per-config 100

# Tutti gli 8 modelli
python scripts/main.py --models all
```

---

## `plot_results.py` — grafici e tabelle comparative

Legge `results/<model_id>/` per tutti i modelli presenti e genera in `results/plots/`:
- `loss_<model>.png` / `travel_time_training_<model>.png` — curve di training con media mobile (da `training_log.csv`).
- `phase_pct_<model>.png` — distribuzione media delle fasi in test (media su tutte le config di test).
- `bars_tt_<config>.png` / `bars_tp_<config>.png` — barplot TT/throughput per config di test, **solo sui 4 modelli principali** (`MAIN_MODELS = [metastgat_pro, metastgat_paper, fixedtime, maxpressure]`, per non affollare il grafico — le ablation restano nelle tabelle).
- `table_travel_time.csv/.tex`, `table_throughput.csv/.tex` — tabelle complete (tutti i modelli × tutte le config), con il valore migliore di ogni colonna in grassetto nel `.tex`. Il throughput in tabella è **percentuale sui veicoli spawnati** (`avg_throughput/total_vehicles*100`), diversamente dal valore assoluto stampato da `test.py`.

Palette colori fissa per modello (`MODEL_COLORS`), CVD-safe, coerente in ogni grafico. Eseguibile standalone (`python scripts/plot_results.py --results-dir results`) o richiamato in-process da `main.py`.

---

## `compare_models.py` — confronto mirato TT/equità, un grafico per config

Complementare a `plot_results.py`: dove quello aggrega tutte le config in un unico barplot per metrica, questo genera **un grafico per configurazione** (le 3 di training + le 7 di test, tutte e 10 di default), pensato per un confronto diretto tra un numero ridotto di modelli (2-3 tipicamente) su travel time medio/massimo ed equità direzionale N/S-W/E fianco a fianco. Nato per confrontare varianti dello stesso modello con un solo iperparametro diverso (es. `metastgat_pro_0.5` vs `metastgat_pro_0.2`, lo sweep sul peso `alpha` del reward), ma funziona con qualunque lista di `model_id` già testati con `test.py`/`main.py`.

Usa sempre le metriche "caso peggiore incluso" (`tt_max`, `wait_max_ns`/`wait_max_ew`) come valori ufficiali — mai le varianti `_resolved`/solo-stalli-conclusi, tenute solo come colonna informativa nella tabella stampata a schermo — scelta deliberata: un veicolo mai arrivato o uno stallo mai risolto è il caso peggiore reale, escluderlo premierebbe un modello che si comporta peggio (vedi [descrizione_metriche.md](descrizione_metriche.md) §1 e §5).

Nei grafici (non nella tabella stampata a schermo, che segue sempre l'ordine di `--model-ids`), le barre sono ordinate dal peggiore al migliore per TT medio **su quella specifica config** (13/9/2026, richiesto per leggere i grafici più rapidamente) — l'ordine può quindi cambiare da un grafico all'altro se il ranking dei modelli non è lo stesso su tutte le config.

```bash
python scripts/compare_models.py --model-ids metastgat_pro_0.5 metastgat_pro_0.2 maxpressure
```

Richiede che `results/<model_id>/test_summary_<config>.json` esista già per ogni combinazione modello/config — da qui la prassi (vedi sopra) di testare ogni modello anche sulle config di training, non solo su quelle di generalizzazione: senza quel passaggio le colonne di training del confronto resterebbero vuote.

---

## `generate_synthetic_data.py` — generazione dati

Vedi [descrizione_configurazioni.md §5](descrizione_configurazioni.md) per la guida completa (parametri, esempi, semantica di `--variance`/`--artery-row`/`--seed`). In breve: genera una griglia rettangolare M×N (`make_simple_roadnet`, geometria Bezier per le svolte) e un flow di traffico (`make_flow`, DFS sui percorsi possibili + allocazione proporzionale dei veicoli per rotta) con 4 modalità di distribuzione temporale (`flat`/`peak`/`peaks`/`workday`), supporto a un'arteria più trafficata (`--artery-row`/`--artery-boost`, ridistribuzione non additiva del budget veicoli) e a varianti "sostanza-preservante" (`--seed`, rumore su pesi delle rotte/intervalli di spawn a parità di profilo aggregato).

---

## `inspect_replay.py` — ispezione di una config senza training

Genera un replay (`replay.txt`+`roadnet_log.json`, visualizzabile nel frontend CityFlow) usando **MaxPressure** (default, un controllore reale seppur semplice — il suo throughput è un segnale onesto di quanto la rete regga il traffico) o una politica **random** (stress-test estremo, non un baseline onesto) — senza alcun training o checkpoint. Usato sistematicamente durante la calibrazione delle densità di traffico (vedi le note di revisione in `descrizione_configurazioni.md`): se anche MaxPressure non riesce a smaltire il traffico assegnato, la config è probabilmente sovraccarica indipendentemente da quale modello RL la userà.

```bash
python scripts/inspect_replay.py --config configs/config_4x4_100m_train1.json \
    --output-dir analisi_configurazioni2/config_4x4_100m_train1 --policy maxpressure
```

---

## Cosa NON c'è più

- `download_real_data.py`, le cartelle `data/hangzhou`/`data/jinan`: il progetto ha abbandonato i dataset reali di CoLight in favore di reti sintetiche interamente parametriche (vedi `descrizione_configurazioni.md`).
- `scripts/legacy/` (vecchi script di ablation "a componenti" e generalizzazione zero-shot manuale): rimossa, sostituita dal sistema di preset `--ablation` e dalla generalizzazione integrata in `main.py`/`test.py`.
- `run_experiment.py`: la sua logica di orchestrazione multi-modello è confluita in `main.py`, che fa lo stesso lavoro in-process (più comodo da debuggare) più la modalità a singolo modello.
- `compare_distributions.py` (violin plot per-veicolo): rimosso il 13/9/2026, scelta deliberata di semplificare il confronto a un solo tipo di grafico (`compare_models.py`). `test.py` continua a scrivere `raw_distributions_<config>.json` e `CityFlowEnv.get_raw_vehicle_waits()`/`get_raw_travel_times()` restano nel codice ma non hanno più alcun consumatore: nessuno script li legge più.
