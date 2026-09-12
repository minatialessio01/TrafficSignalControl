# Descrizione del Codice Sorgente (`src/`)

Questo documento descrive in dettaglio ogni modulo di `src/`: cosa fa, quali equazioni del paper MetaSTGAT implementa, quali flag di ablation lo attraversano, e come si collega agli altri moduli. Per la formulazione MDP completa (stato/azione/reward/metriche) vedi [descrizione_environment.md](descrizione_environment.md) — qui ci si concentra sull'organizzazione del codice, non la si ripete. Per le differenze puntuali rispetto al paper vedi [metastgat_diff_analysis.md](metastgat_diff_analysis.md).

```
src/
├── environment/   → wrapper CityFlow → MDP (ambiente RL)
├── models/        → architetture di rete: MetaSTGAT, STGAT, sotto-moduli
├── agents/        → controllori: DQN (MetaSTGAT/STGAT), MaxPressure, FixedTime, replay buffer
└── utils/         → logger di training, metriche episodiche
```

---

## `src/environment/`

### `cityflow_env.py` — il wrapper MDP

Trasforma il motore CityFlow (simulatore microscopico in C++, binding Python) in un ambiente multi-agente: un agente per intersezione semaforizzata, azione = indice di fase, stato/reward calcolati per intersezione. Dettaglio completo di formule e metriche in [descrizione_environment.md](descrizione_environment.md); qui solo l'inventario dei metodi.

**Setup (una tantum in `__init__`)**:
- `_get_signalized_intersections()` — filtra le intersezioni `virtual` (bordi griglia, senza semaforo).
- `_build_adjacency()` — grafo dei vicini: due intersezioni sono adiacenti se collegate da una strada diretta; troncato ai `num_neighbors` (default 4) più vicini per distanza euclidea se ce ne sono di più.
- `_get_lanes_per_intersection()` — ordina le 12 corsie in ingresso di ogni intersezione in un ordine canonico fisso N,S,W,E × [sinistra, dritto, destra], deducendo la direzione dal confronto tra le coordinate riga/colonna di inizio/fine strada codificate nell'id (`road_<r>_<c>_to_<r'>_<c'>`). Rami mancanti (intersezioni di bordo) → corsie fittizie `missing_*`, sempre zero. Questo ordinamento è la base di **tutte** le metriche per corsia (stato, reward, fairness N/S vs W/E).
- `_get_road_lengths()` — lunghezza *reale percorribile* di ogni corsia, non la distanza punto-a-punto del roadnet: CityFlow tronca `get_vehicle_distance()` alla `width` dell'intersezione a **entrambe** le estremità (un veicolo lascia la corsia già `width` metri prima del punto centrale dichiarato). Bug corretto l'11/9/2026, verificato empiricamente (corsia dichiarata 167m, distanza massima reale osservata ≈127m = 167−20−20). Prima della correzione `VISION_CUTOFF_M` (basato su questa lunghezza) risultava più restrittivo del dovuto.
- `_build_phase_lanelinks()` — per ogni intersezione e ogni fase, le coppie (corsia-in, corsia-out) abilitate, lette direttamente da `roadLinks[*].laneLinks[*]` del roadnet. Usato **solo** da `MaxPressureAgent` per la pressione classica (formula di Varaiya, corsia intera, nessun cutoff di visibilità) — è deliberatamente un canale diverso da `n_vec`/`_get_observations()`, pensati per lo stato RL.

**Ciclo MDP**:
- `reset()` → stato iniziale; azzera contatori episodio (`arrived_tt`, `vehicle_wait_times`, `max_wait_ns`/`max_wait_ew`, ecc.).
- `step(actions)` → esegue `GREEN_TIME+YELLOW_TIME` secondi sulla fase scelta, poi `RED_TIME` secondi di tutto-rosso reale (fase dedicata `ALL_RED_PHASE`, mai nello spazio azioni), poi ripristina la fase verde a livello di bookkeeping. Aggiorna in un colpo solo: tempi di attesa (`vehicle_wait_times`, campionati una volta a fine step, non per sotto-step), le due mappe di massima attesa N/S ed E/W (`max_wait_ns`/`max_wait_ew`, per `get_direction_fairness_stats()`), gli arrivi/spawn per il travel time (`arrived_tt`, `spawn_times`). Ritorna `(observations, rewards, done, info)`.
- `_get_observations()` → costruisce `n_vec`/`wait_vec`/`p_vec` per ogni intersezione (dettaglio dimensioni/normalizzazione in descrizione_environment.md §6).
- `_compute_rewards()` → le due formule `reward_mode` (`custom`/`paper`, dettaglio in descrizione_environment.md §6).
- `get_invalid_actions()` → maschera anti-starvation (fase corrente invalida se scelta ≥2 volte consecutive), se `use_action_mask=True`. **Non** usata da `MaxPressureAgent` (è pura `argmax` senza vincoli, fedele a Varaiya 2013) — passata solo all'agente RL in `train.py`/`test.py`.

**Meta-feature per MetaSTGAT** (non usate dal reward, solo dal modello):
- `get_spatial_meta_features(iid)` → pressione/veicoli per corsia + distanza dai vicini, dim `N_LANES*2 + num_neighbors = 28`.
- `get_temporal_meta_features(iid, history, history_len)` → proxy di coda + storico ultimi 5 step di `n_vec`, dim `N_LANES + N_LANES*5 = 72`.

Nessuno dei due canali porta coordinate assolute o l'identità dell'intersezione — rilevante per la discussione sulla generalizzazione posizionale dell'arteria in [descrizione_configurazioni.md §4bis](descrizione_configurazioni.md).

**Metriche** (dettaglio formule/interpretazione in [descrizione_metriche.md](descrizione_metriche.md)):
- `get_average_travel_time(include_unfinished=True)` — metrica di ranking di default.
- `get_completed_only_travel_time()` / `get_original_average_travel_time()` — varianti "solo arrivati", solo informative.
- `get_throughput()` — veicoli completati.
- `get_travel_time_stats()` — max/std/p50/p90/p95/p99 del travel time dei soli arrivati.
- `get_direction_fairness_stats()` — massima attesa osservata per intersezione, separata N/S vs W/E, mediata/max sulle intersezioni.
- `get_lane_vehicle_count()` — conteggio nativo per corsia (nessun cutoff), usato da MaxPressure.

**Costruzione grafo per PyG**: `get_adjacency_matrix()`, `get_edge_index()` — quest'ultimo è quello effettivamente usato da `train.py`/`test.py` per costruire il tensore `edge_index` passato ai modelli. Vedi nota sotto su `graph_builder.py`.

### `graph_builder.py` — **non usato dalla pipeline attiva**

Helper per costruire oggetti `torch_geometric.data.Data`/`Batch` a partire da adiacenza + feature dei nodi. Esportato da `src/environment/__init__.py` ma **nessuno script lo importa**: `train.py`/`test.py` costruiscono l'`edge_index` direttamente con `CityFlowEnv.get_edge_index()`, che è più semplice per il caso d'uso corrente (stessa topologia per tutti i campioni di un batch, replicata block-diagonal in `DQNAgent.update()`). Codice morto ma innocuo, tenuto per un eventuale uso futuro con feature dei nodi costruite fuori da `CityFlowEnv` o con topologie diverse per campione.

---

## `src/models/` — architettura MetaSTGAT (Section 4 del paper)

Pipeline forward (una chiamata per timestep, per tutte le N intersezioni in parallelo — parameter sharing, non un modello per intersezione):

```
stato (N, 32) ─┬─→ DualStateEncoder ──→ e_i (temporale) ──→ MetaLSTM ──→ x_i ─┐
               └─→                  └─→ e_j (spaziale)  ────────────────────┼─→ MetaGAT (CST: Q=e_j,K=V=x_i, meta=TMK) ─┐
feature spaziali ──→ SMK-Learner ──→ SMK(i) ───────────────────────────────→ MetaGAT (CS: Q=K=V=e_j, meta=SMK)  ─┼─→ concat → Linear → Q-values (N, 8)
feature temporali ─→ TMK-Learner ──→ TMK(i) ─(usato anche da MetaLSTM)──────────────────────────────────────────┘
```

- **`state_encoder.py`** (Eq. 3-4) — `DualStateEncoder`: due MLP a 2 strati (ReLU) con pesi indipendenti, stessa architettura, inizializzazioni diverse. Producono `e_i` (branch temporale → LSTM) ed `e_j` (branch spaziale → query GAT) dallo stesso stato in input. Usato identicamente da `MetaSTGAT` e da `STGAT`.

- **`meta_knowledge_learner.py`** (Section 4.3.1) — `MetaKnowledgeLearner`: MLP a 2 strati (ReLU, poi `Tanh` opzionale — `use_tanh_output`, disattivabile con `--no-tanh-meta`) che mappa feature grezze in un embedding a 64 dim. Due sottoclassi identiche nell'architettura ma con input diversi: `SpatialMetaKnowledgeLearner` (28 dim, → SMK) e `TemporalMetaKnowledgeLearner` (72 dim, → TMK). Questi embedding sono le "meta-conoscenze" che generano dinamicamente i pesi di Meta-GAT e Meta-LSTM — non partecipano direttamente al calcolo dei Q-value.

- **`meta_lstm.py`** (Section 4.3.3, Eq. 20-21) — `MetaDense3` genera, da `TMK(i)`, i pesi `W_Φ` e i bias `b_Φ` per **tutti e 4 i gate** di una LSTM (forget/input/output/cell): `W: (N, 4, 2·D_h, D_h)`, `b: (N, 4·D_h)`. `MetaLSTMCell` è un'implementazione manuale della cella LSTM (non `nn.LSTMCell`, che non supporta pesi diversi per ogni elemento del batch) che accetta questi pesi come argomenti anziché come parametri fissi — il pattern standard per le hypernetwork. Ogni intersezione ha quindi, di fatto, una propria LSTM con pesi generati al volo dalle sue feature temporali locali.

- **`meta_gat.py`** (Section 4.3.2, Eq. 14-17) — `MetaDense` genera `W`/`b` per testa da un meta-embedding (SMK o TMK); `MetaGATLayer` implementa multi-head attention su grafo dove il dot-product `Q·K` è prima trasformato da una matrice dinamica per-nodo-per-testa (`W_h @ Q_h`) anziché scalata da uno scalare fisso — interpretazione scelta per l'equazione ambigua del paper (`W·(Q·K)` vs `(W·Q)·K`), discussa in `metastgat_diff_analysis.md`. Softmax per-nodo (`_edge_softmax`) via `scatter_reduce_`/`scatter_add_`, niente dipendenza da PyTorch Geometric per l'attention in sé. Usato due volte per ogni forward pass di `MetaSTGAT`: modulo **CST** (Q=e_j, K=V=x_i, meta=TMK) e modulo **CS** (Q=K=V=e_j, meta=SMK).

- **`stgat.py`** — baseline **senza** meta-learning: stessa architettura ad alto livello (encoder duale → LSTM → 2×GAT → head) ma con `nn.LSTMCell` standard e `StandardGATLayer` (pesi fissi, niente hypernetwork). Serve da termine di paragone per isolare il contributo del meta-learning rispetto a un modello ST-GAT "vanilla" con la stessa capacità nominale.

- **`metastgat.py`** — assembla tutti i sotto-moduli sopra in `MetaSTGAT.forward()`: Dual Encoder → SMK/TMK learners → Meta-LSTM (con TMK) → Meta-GAT CST (con TMK) + Meta-GAT CS (con SMK) → concat → `Linear(hidden_dim*2, n_actions)`. Espone `init_hidden()` per azzerare `(h,c)` a inizio episodio. `use_tanh_meta` è l'unico flag di ablation dell'architettura vera e propria (passato ai due meta-learner).

---

## `src/agents/` — i controllori

- **`dqn_agent.py`** — `DQNAgent` avvolge sia `MetaSTGAT` che `STGAT` (rilevato con `isinstance(model, MetaSTGAT)`) in un agente DQN multi-intersezione con **parameter sharing** (un solo modello per tutte le intersezioni, batch sulla dimensione N). Punti chiave:
  - **Selezione azione** (`select_actions`): epsilon-greedy con corto-circuito a `epsilon>=1.0` (warm-up/esplorazione ciclica) — salta il forward pass della rete perché ogni intersezione sceglierebbe comunque a caso, risparmiando il costo dominante per step su CPU. Applica `invalid_actions` (dall'action mask dell'ambiente) mettendo `-inf` sui Q-value delle fasi vietate prima dell'argmax.
  - **Stato ricorrente**: `h_state`/`c_state` persistono tra gli step di un episodio (`reset_hidden()` li azzera a inizio episodio); durante il training, l'assenza del vero stato iniziale nelle sequenze campionate dal buffer è compensata dal **burn-in**.
  - **`update()`** (Algorithm 1, line 9-13) — esegue `n_updates` (default 100) mini-batch SGD **per episodio**, non uno per step ambientale: numero fisso, indipendente dal numero di step dell'episodio (120 con episodi da 1800s). Ogni update: campiona `batch_size` sequenze di lunghezza `seq_len` dal `ReplayBuffer` (PER o uniforme secondo `use_per`), fa `burn_in` step senza gradiente per allineare `(h,c)`, poi BPTT sui restanti step con **Double DQN** (se `use_double_dqn`) o DQN standard, **Huber loss** (se `use_huber`, altrimenti MSE) pesata dagli IS weights del PER, **gradient clipping** (`max_norm=1.0`, se `use_grad_clip`), e aggiornamento del target network **soft** (Polyak, τ=0.01, se `use_soft_update`) o hard-copy, ogni `target_update_freq=100` step totali. L'errore TD medio per sequenza aggiorna le priorità nel buffer. Epsilon decade (`epsilon *= epsilon_decay`) una volta a fine `update()`, indipendentemente da quale variante di training (vedi `train.py`) sia stata usata per l'episodio appena raccolto.
  - **Checkpoint** (`save_checkpoint`/`load_checkpoint`): salva **tutto** lo stato necessario per un resume esatto — pesi del modello online e target, stato dell'optimizer, `episode`, `total_steps`, `epsilon`, `best_travel_time`, e (dall'11/9/2026) lo **stato del replay buffer** (`replay_buffer.state_dict()`), la cui assenza rendeva un `--resume` funzionalmente un training da zero con pesi pre-allenati (buffer vuoto → gate `min_buffer_size` e diversità del campionamento PER entrambi inefficaci nei primi update dopo il resume). `torch.load(..., weights_only=False)` esplicito, necessario da PyTorch 2.6+ per deserializzare l'oggetto custom `Transition` dentro lo stato del buffer (sicuro: si caricano solo checkpoint auto-prodotti, mai file di terzi).
  - Tutti i flag `use_*` del costruttore sono i sette meccanismi di ablation lato agente (Double DQN, PER, Huber, soft update, grad clip, BPTT, Tanh meta) — vedi tabella in `metastgat_diff_analysis.md`/`proposte_ablation.md` per la mappa completa M1-M10.

- **`replay_buffer.py`** — `ReplayBuffer`: **Prioritized Experience Replay sequenziale ed episodico**, non un buffer flat di transizioni singole. Le transizioni si accumulano in `current_episode` durante l'episodio; `end_episode(seq_len)` le chiude in una entry (`ep_dict`) e registra come chiavi campionabili tutte le finestre `(ep_id, start_idx)` di lunghezza `seq_len` con priorità iniziale = `max_priority`. `sample_sequences()` campiona proporzionalmente a `priority^alpha`, con IS weights normalizzati by `beta` (annealing lineare verso 1.0). `update_priorities()` aggiorna la priorità di ogni sequenza con `|TD_error|+ε`. Eviction FIFO per episodio intero quando si supera `capacity` (mai l'ultimo episodio rimasto, per evitare un buffer vuoto). `state_dict()`/`load_state_dict()` serializzano l'intero stato (inclusi gli oggetti `Transition`, con `__slots__` per compattezza) per il checkpoint/resume di `dqn_agent.py`.

- **`maxpressure_agent.py`** — `MaxPressureAgent`: baseline classico (Varaiya, 2013), **stateless**, nessun parametro da allenare. Fedele all'implementazione di riferimento LibSignal: per ogni fase, pressione = Σ su tutti i movimenti (lane-link) abilitati da quella fase di `(veicoli sulla corsia di provenienza − veicoli sulla corsia di destinazione)`, usando `CityFlowEnv.phase_lanelinks` e `get_lane_vehicle_count()` (corsia intera, **nessun** cutoff di visibilità — deliberatamente diverso da `n_vec`, che è una scelta di design specifica dello stato RL). Sceglie `argmax` sulla pressione, **senza alcuna maschera anti-starvation**: a differenza dell'agente RL, non riceve mai `get_invalid_actions()`. Il vincolo di durata minima di fase del paper LibSignal (`t_min`) è già implicitamente soddisfatto perché ogni decisione dura comunque `STEP_TIME=15s` per costruzione dell'ambiente.

- **`fixedtime_agent.py`** — `FixedTimeAgent`: baseline non intelligente, cicla le 8 fasi in ordine fisso (0,1,2,...,7,0,...) cambiando fase **ogni singolo step di decisione** (`FIXED_GREEN_TIME=1`, cioè ogni 15s — stessa cadenza di tutti gli altri modelli/controllori, nessun fattore di scala arbitrario tra i cicli). Non osserva l'ambiente: la fase dipende solo dal contatore di step passato a `select_actions(current_step=...)`.

---

## `src/utils/`

- **`logger.py`** — `TrainingLogger`: scrive `training_log.csv` (una riga per episodio: travel time, throughput, reward, loss, epsilon, buffer size, **`wait_max_ns`/`wait_max_ew`** — equità direzionale, aggiunta l'11/9/2026, già calcolata gratuitamente da `CityFlowEnv` ad ogni step ma prima non letta) e `training_state.json` (stato corrente leggibile senza Python). Gestisce **SIGINT** (Ctrl+C): non interrompe a metà episodio, imposta un flag che `train.py` controlla a fine episodio corrente, salva un checkpoint d'emergenza e stampa il comando `--resume` esatto. Su resume (`resume=True`), se l'header del CSV su disco è più vecchio (mancano colonne aggiunte dopo l'inizio di quel run) lo **migra automaticamente** riscrivendolo con l'header corrente (colonne mancanti = vuote sulle righe vecchie) prima di continuare in append — evita un CSV con colonne extra disallineate rispetto all'header.
- **`metrics.py`** — `EpisodeMetrics` (accumula travel time/throughput/reward/loss passo-passo, espone medie e valori finali dell'episodio) e `RunningMetrics` (media mobile sugli ultimi 10 episodi, "the average value of the last ten tests" del paper — usata per il best-model tracking e il riepilogo finale di `train.py`).

---

## Note per chi scrive la tesi da questo codice

- L'architettura è interamente **parameter-sharing**: un solo `MetaSTGAT`/`STGAT` per tutte le intersezioni della rete, non un modello per incrocio — è ciò che rende il modello indipendente dalla topologia e quindi (in teoria) generalizzabile a griglie diverse (vedi i test 5×5/6×6 in `descrizione_configurazioni.md`).
- I flag di ablation non sono sparsi arbitrariamente: sono organizzati in **preset** (`--ablation pro/paper/environment/temporal/rl_core/replay_stability`, definiti in `scripts/train.py`) che corrispondono esattamente alla Proposta 1 di `proposte_ablation.md` — vedi quel documento per la razionale scientifica di ogni raggruppamento.
- `graph_builder.py` è l'unico modulo di `src/` non attraversato dalla pipeline corrente — vale la pena saperlo prima di descriverlo come parte attiva dell'architettura in tesi.
