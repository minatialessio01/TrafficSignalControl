# Task Tracker – Pipeline Sperimentale Tesi

> **Come usarlo**: Spuntare ogni voce con `[x]` appena completata. Se il cambio LLM avviene a metà, il nuovo modello deve leggere questo file + `implementation_plan.md` per riprendere dal primo `[ ]` non completato.
>
> **Contesto rapido**: Stiamo costruendo una pipeline che addestra 6 modelli (Pro, Paper, 4 ablation) su 2 config, li testa su 8 config, e genera grafici+tabelle automaticamente.
> Ablation (Proposta 1): `ablation_environment` (spegne M1-M4), `ablation_temporal` (M5), `ablation_rl_core` (M6), `ablation_replay_stability` (M7-M10).

---

## FASE 1 — Modifiche a `train.py` (flag ablation)

> File: `scripts/train.py`

- [x] **1.1** Aggiungere `--reward-mode [custom|paper]`
- [x] **1.2** Aggiungere `--no-vision-cutoff`
- [x] **1.3** Aggiungere `--no-wait-vec`
- [x] **1.4** Aggiungere `--no-action-mask`
- [x] **1.5** Aggiungere `--no-bptt`
- [x] **1.6** Aggiungere `--no-double-dqn`
- [x] **1.7** Aggiungere `--no-per`
- [x] **1.8** Aggiungere `--no-huber`
- [x] **1.9** Aggiungere `--no-soft-update`
- [x] **1.10** Aggiungere `--no-grad-clip`
- [x] **1.11** Aggiungere `--no-warmup`
- [x] **1.12** Aggiungere `--no-cyclic-exploration`
- [x] **1.13** Aggiungere `--no-tanh-meta`
- [x] **1.14** Aggiungere `--ablation [full|paper|environment|temporal|rl_core|replay_stability]` con `apply_ablation_preset(args)`
- [x] **1.15** Aggiungere `--output-dir`
- [x] **1.16** Verifica `training_log.csv` colonne (esistente, confermato)

---

## FASE 2 — Modifiche a `cityflow_env.py` (supporto flag ablation)

> File: `src/environment/cityflow_env.py`

- [x] **2.1** Aggiungere parametro `reward_mode: str = "custom"` al costruttore
- [x] **2.2** Aggiungere parametro `use_vision_cutoff: bool = True` al costruttore
- [x] **2.3** Aggiungere parametro `use_wait_vec: bool = True` al costruttore
- [x] **2.4** Aggiungere parametro `use_action_mask: bool = True` al costruttore
- [x] **2.5** Nella funzione reward: `reward_mode == "paper"` → formula `-P_i` originale
- [x] **2.6** Nella funzione stato: `not use_vision_cutoff` → rimuovere il filtro sul campo visivo (VISION_CUTOFF_M, ~144m)
- [x] **2.7** Nella funzione stato: `not use_wait_vec` → concatenare solo `[n_vec, p_vec]` (dim=20)
- [x] **2.8** In `get_invalid_actions`: `not use_action_mask` → return `{}`
- [x] **2.9** `observation_dim` aggiornato per restituire 20 o 32 in base a `use_wait_vec`

---

## FASE 3 — Modifiche a `dqn_agent.py` (supporto flag ablation)

> File: `src/agents/dqn_agent.py`

- [x] **3.1** Aggiungere `use_double_dqn: bool = True` al costruttore
- [x] **3.2** Aggiungere `use_per: bool = True` al costruttore
- [x] **3.3** Aggiungere `use_huber: bool = True` al costruttore
- [x] **3.4** Aggiungere `use_soft_update: bool = True` al costruttore
- [x] **3.5** Aggiungere `use_grad_clip: bool = True` al costruttore
- [x] **3.6** Aggiungere `use_bptt: bool = True` → forza `seq_len=1, burn_in=0` se False
- [x] **3.7** Nel `update`: `not use_double_dqn` → `q_next.max(dim=-1).values`
- [x] **3.8** Nel `update`: `not use_bptt` → già gestito tramite seq_len=1 nel costruttore
- [x] **3.9** Nel `update`: `not use_per` → IS weights = tensor di 1.0 costante
- [x] **3.10** Nel `update`: `not use_huber` → `mse_loss`
- [x] **3.11** Nel `update`: `not use_grad_clip` → salta `clip_grad_norm_`
- [x] **3.12** In `_update_target`: `not use_soft_update` → hard copy

---

## FASE 4 — Modifiche a `meta_knowledge_learner.py` e `metastgat.py`

> File: `src/models/meta_knowledge_learner.py`, `src/models/metastgat.py`

- [x] **4.1** Aggiungere `use_tanh_output: bool = True` al costruttore di `MetaKnowledgeLearner`
- [x] **4.2** Nel `forward`: `use_tanh_output` → Tanh; else → `nn.Identity()`
- [x] **4.3** Flag propagato da `metastgat.py` → `MetaKnowledgeLearner` (via `use_tanh_meta`)
- [x] **4.4** Flag propagato da `train.py` → `build_model` → `MetaSTGAT`

---

## FASE 5 — Modifiche a `test.py` (output CSV strutturato)

> File: `scripts/test.py`

- [x] **5.1** Aggiungere `--output-dir` (default: `results/<model_id>/`)
- [x] **5.2** Aggiungere `--model-id` (default: `metastgat_pro`)
- [x] **5.3** Salvare per episodio di test: `episode, travel_time, throughput, phase_pct_0..7` → `results/<model_id>/test_<config_name>.csv`
- [x] **5.4** Usare `final_model.pth` come checkpoint primario; il training deve salvarlo alla fine
- [x] **5.5** Stampare summary al termine: TT medio, throughput medio, config name

---

## FASE 6 — Nuovo `run_experiment.py` (orchestratore)

> File: `scripts/run_experiment.py` [NUOVO]

- [x] **6.1** Accettare `--train-configs`, `--test-configs`, `--episodes-per-config`, `--models [all|...]`
- [x] **6.2** Definire `MODEL_REGISTRY` dict con 8 modelli e rispettivi `--ablation` preset
- [x] **6.3** Per ogni modello addestrabile: lanciare `train.py` in subprocess
- [x] **6.4** Per ogni modello × config di test: lanciare `test.py` in subprocess
- [x] **6.5** Chiamare `plot_results.py` al termine
- [x] **6.6** Salvare `results/experiment_summary.json` (timestamp, modelli, config, path checkpoint)
- [x] **6.7** Gestire errori subprocess: loggare + continuare con il prossimo modello

---

## FASE 7 — Nuovo `plot_results.py` (grafici e tabelle)

> File: `scripts/plot_results.py` [NUOVO]

- [x] **7.1** Funzione `load_training_logs(results_dir)` → dict `{model_id: DataFrame}`
- [x] **7.2** Funzione `load_test_results(results_dir)` → dict `{model_id: {config: DataFrame}}`
- [x] **7.3** Grafico **Loss con media mobile** (window=5): per ogni modello, linea verticale al cambio config → `plots/loss_<model_id>.png`
- [x] **7.4** Grafico **Travel Time training con media mobile**: come 7.3 → `plots/travel_time_training_<model_id>.png`
- [x] **7.5** Grafico **Phase Percentage** (barplot media su config test) → `plots/phase_pct_<model_id>.png`
- [x] **7.6** Grafico **Barre TT per config test**: 4 modelli principali affiancati → `plots/bars_tt_<config>.png`
- [x] **7.7** Grafico **Barre Throughput per config test**: 4 modelli + linea orizzontale veicoli totali → `plots/bars_tp_<config>.png`
- [x] **7.8** Tabella **Travel Time** (righe=modelli, colonne=config test) → `plots/table_travel_time.csv` + `.tex`
- [x] **7.9** Tabella **Throughput** (percentuale completati) → `plots/table_throughput.csv` + `.tex`
- [x] **7.10** Palette colori coerente; stile `seaborn-v0_8-darkgrid`

---

## FASE 8 — Test End-to-End

- [ ] **8.1** Smoke test 3 modelli: `python scripts/run_experiment.py --episodes-per-config 3 --models metastgat_pro fixedtime maxpressure`
- [ ] **8.2** Verificare `results/metastgat_pro/training_log.csv` (colonne corrette)
- [ ] **8.3** Verificare `results/metastgat_pro/test_config_4x4_200m_6k_flat.csv`
- [ ] **8.4** Verificare generazione grafici in `results/plots/`
- [ ] **8.5** Smoke test completo: `--models all --episodes-per-config 3`
- [ ] **8.6** Verificare linea verticale al cambio config nei grafici di training
- [ ] **8.7** Verificare tabelle `.tex` compilabili senza errori
- [ ] **8.8** Git commit + push `"Fase 8 OK: pipeline sperimentale funzionante"`

---

## FASE 9 — Lancio Produzione

- [ ] **9.1** Lanciare `run_experiment.py --episodes-per-config 50 --models all` su Docker/WSL2
- [ ] **9.2** Monitorare convergenza (TT in calo entro i primi 20 ep)
- [ ] **9.3** Generare grafici finali
- [ ] **9.4** Push finale risultati su GitHub
