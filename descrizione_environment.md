# Descrizione dell'Ambiente di Simulazione (CityFlow + wrapper MDP)

Questo documento raccoglie tutto ciò che serve per descrivere in tesi l'ambiente di simulazione: cos'è CityFlow, come il wrapper `src/environment/cityflow_env.py` lo trasforma in un MDP multi-agente, come sono gestite le fasi semaforiche (verde/giallo/rosso), come sono calcolate le metriche, e come funziona la visualizzazione via replay. Per la nomenclatura di config/dati vedi [descrizione_configurazioni.md](descrizione_configurazioni.md) (non duplicata qui).

---

## 1. Cos'è CityFlow

[CityFlow](https://github.com/cityflow-project/CityFlow) è un simulatore di traffico microscopico open source (motore in C++, binding Python), pensato per il controllo semaforico multi-intersezione via RL su reti stradali di grandi dimensioni (a differenza di SUMO, privilegia velocità di simulazione rispetto al dettaglio fisico).

Concetti chiave del motore:

- **Simulazione a passi discreti**: `engine.next_step()` avanza la simulazione di **1 secondo simulato** alla volta. Non esiste un modo per avanzare di più secondi in una sola chiamata: per simulare N secondi si chiama `next_step()` N volte.
- **Config JSON**: un file di configurazione (es. [configs/config_4x4_200m_2k_flat.json](configs/config_4x4_200m_2k_flat.json)) accoppia un `roadnetFile` (topologia) a un `flowFile` (immissione veicoli) e fissa parametri come `interval` (intervallo di spawn, 1.0s), `seed`, `maxStep` (durata dell'episodio in secondi simulati — **1800s** in tutte le nostre config), `saveReplay`/`replayLogFile`/`roadnetLogFile` (per la visualizzazione).
- **Roadnet**: definisce la topologia — intersezioni, strade, corsie, e per ogni intersezione semaforizzata l'elenco delle **fasi** (`lightphases`, vedi §2).
- **Flow**: definisce quando e dove i veicoli vengono immessi (route, orario, volume). Generato parametricamente da [scripts/generate_synthetic_data.py](scripts/generate_synthetic_data.py) — dettagli in `descrizione_configurazioni.md`.
- **API usate dal wrapper**: `set_tl_phase(intersection_id, phase_index)` per comandare un semaforo, `get_lane_vehicles()`, `get_vehicle_distance()`, `get_vehicle_speed()`, `get_vehicles()`, `get_vehicle_count()`, `get_average_travel_time()` (nativa, solo veicoli arrivati).

**Limite nativo importante**: CityFlow **non ha un concetto di "giallo" o di fisica del semaforo ambra**. Una fase (`lightphase`) è solo un insieme binario di movimenti abilitati/disabilitati (`availableRoadLinks`); non esistono stati intermedi nativi. Qualunque comportamento di tipo giallo/tutto-rosso va costruito a mano definendo fasi aggiuntive nel roadnet e pilotandole esplicitamente da codice — è esattamente quello che facciamo (§4).

---

## 2. Struttura del roadnet

Il roadnet ([data/roadnet_4x4_200m.json](data/roadnet_4x4_200m.json) e varianti) è un JSON con due liste principali:

- **`intersections`**: ogni intersezione ha un `id`, una posizione, un flag `virtual`. Le intersezioni `virtual: true` sono i bordi della griglia (dove le strade "escono" dalla mappa, senza semaforo); quelle `virtual: false` sono le intersezioni reali e semaforizzate. Nel nostro dataset: 16 (griglia 4x4), 25 (5x5), 36 (6x6) intersezioni reali per roadnet.
- **`roads`**: segmenti stradali diretti tra due intersezioni, con `nLane` corsie ciascuno.
- **`trafficLight.lightphases`**: per ogni intersezione reale, la lista delle fasi disponibili. Ogni fase è `{"time": <durata_suggerita>, "availableRoadLinks": [indici]}` — gli indici si riferiscono ai `roadLinks` dell'intersezione (coppie corsia-in → corsia-out abilitate in quella fase). `set_tl_phase(id, k)` attiva la k-esima fase di questa lista.

Nel nostro roadnet ogni intersezione reale ha **8 fasi verdi** (indici 0-7), mappate 1:1 sulle 8 azioni dell'agente RL, più — dopo la modifica descritta al §4 — una **9ª fase di tutto-rosso** (indice 8, `availableRoadLinks: []`), mai selezionabile dall'agente, usata internamente dal wrapper.

---

## 3. Le 8 fasi semaforiche e la mappatura verde

Definite in [cityflow_env.py:64-73](src/environment/cityflow_env.py#L64) (`GREEN_LANES_PER_PHASE`), nomenclatura N/S/E/W (Nord/Sud/Est/Ovest) + T/L (Through=dritto, Left=sinistra):

| Fase | Nome | Movimenti abilitati |
|---|---|---|
| 0 | NTST | Nord dritto+destra, Sud dritto+destra |
| 1 | NLSL | Nord sinistra, Sud sinistra |
| 2 | NTNL | Nord sinistra+dritto+destra (solo Nord) |
| 3 | STSL | Sud sinistra+dritto+destra (solo Sud) |
| 4 | WTET | Ovest dritto+destra, Est dritto+destra |
| 5 | WLEL | Ovest sinistra, Est sinistra |
| 6 | ETEL | Est sinistra+dritto+destra (solo Est) |
| 7 | WTWL | Ovest sinistra+dritto+destra (solo Ovest) |

**Nota sulla svolta a destra**: non è mai "sempre verde" (free-right, come spesso semplificato in letteratura) — ogni svolta, inclusa la destra, deve aspettare la fase che la autorizza esplicitamente, esattamente come dritto e sinistra. Scelta dichiarata esplicitamente nel codice (docstring di `cityflow_env.py`), utile da menzionare in tesi come deviazione (più realistica) rispetto a semplificazioni comuni in altri lavori.

Ogni intersezione ha 12 corsie in ingresso modellate (`N_LANES=12`: 4 direzioni × 3 manovre), ordinate canonicamente N,S,W,E × [sinistra, dritto, destra]. Se un ramo manca fisicamente (intersezione di bordo con meno di 4 strade), viene riempito con corsie fittizie `missing_*` sempre a zero veicoli.

---

## 4. Gestione di verde, giallo e rosso (con la modifica introdotta)

Ogni azione dell'agente dura `STEP_TIME = 15s` simulati, suddivisi in tre costanti ([cityflow_env.py:54-63](src/environment/cityflow_env.py#L54)):

```
GREEN_TIME  = 10s
YELLOW_TIME = 3s
RED_TIME    = 2s
STEP_TIME   = GREEN_TIME + YELLOW_TIME + RED_TIME = 15s
```

Comportamento effettivo dentro `step()` ([cityflow_env.py:340-368](src/environment/cityflow_env.py#L340)):

1. **`GREEN_TIME + YELLOW_TIME` (13s)**: la fase scelta dall'agente resta attiva; i movimenti abilitati da quella fase restano verdi per tutti e 13 i secondi. **Il giallo non è un comportamento distinto**: per i 3 secondi "gialli" i veicoli continuano a defluire esattamente come in verde pieno. Questa è una scelta esplicita e dichiarata (non un bug): equivale a considerare il giallo come tempo di verde effettivo, una semplificazione comune quando il simulatore non ha fisica nativa del giallo.
2. **`RED_TIME` (2s)**: l'ambiente forza esplicitamente la **fase di tutto-rosso** (indice `ALL_RED_PHASE = 8`, aggiunta al roadnet) su ogni intersezione — nessun movimento abilitato per nessuno. Questo garantisce che l'incrocio si svuoti prima che il prossimo step liberi un movimento potenzialmente conflittuale con quello appena terminato.
3. Al termine dei 2s di rosso, la fase verde viene ripristinata a livello di motore (bookkeeping) prima di restituire il controllo — il prossimo `step()` imposterà comunque una fase fresca in base alla nuova azione.

**Perché è stato aggiunto**: prima di questa modifica, i 5 secondi "giallo+rosso" venivano eseguiti sotto la **stessa identica fase verde** delle prime 10 (nessuna seconda chiamata a `set_tl_phase`) — funzionalmente equivalente a 15s di verde continuo, senza alcun istante di tutto-rosso. Questo lasciava una finestra reale (seppur stretta: un'intersezione larga 20m si attraversa in ~2-4s, quindi il rischio riguarda solo i veicoli entrati negli ultimi 1-3s di verde) in cui un veicolo ancora dentro l'incrocio al cambio fase poteva sovrapporsi a un veicolo di un movimento conflittuale appena liberato dalla fase successiva.

**Come è stato implementato**: aggiunta una 9ª fase `{"time": 2, "availableRoadLinks": []}` a tutte le 93 intersezioni semaforizzate nei 4 roadnet (`roadnet_4x4_200m`, `roadnet_4x4_300m`, `roadnet_5x5_200m`, `roadnet_6x6_200m`), tutte con esattamente 8 fasi verdi preesistenti — modifica meccanica e uniforme, non manuale. `ALL_RED_PHASE = N_PHASES = 8` non è mai nello spazio delle azioni dell'agente (che resta 0-7): è puramente un dettaglio interno di `step()`.

**Verifica effettuata sul replay** (vedi anche §7): ispezionando `replay.txt` riga per riga per una singola intersezione, gli stati di 3 corsie campione risultano:

```
step  0-12  (13s, verde+giallo):  ['r','g','g']   ← fase invariata
step 13-14  ( 2s, tutto-rosso):   ['r','r','r']   ← TUTTE rosse contemporaneamente
step 15+    (nuova azione):       ['g','r','r']   ← fase successiva, verde su corsia diversa
```

Conferma diretta, riga per riga del replay reale, che il tutto-rosso è genuinamente simulato (non solo "contabilizzato") e che le fasi verdi successive sono effettivamente diverse (non lo stesso schema ripetuto).

**Impatto sulle metriche**: essendo applicato in modo identico a *tutti* i modelli (Pro, Paper, tutte le ablation, FixedTime, MaxPressure passano tutti dallo stesso `step()`), il confronto relativo fra modelli resta valido. Cambiano leggermente (in modo uniforme) i valori assoluti: 2 dei 15 secondi per ciclo (~13%) diventano tempo morto reale invece che verde esteso, quindi ci si aspetta un travel time medio leggermente più alto e un throughput leggermente più basso per tutti, rispetto ai numeri raccolti prima della modifica.

**Effetto a cascata sul campo visivo**: il cutoff di visibilità (`VISION_CUTOFF_M`, §6/§8) era definito come "distanza percorribile da un veicolo nella finestra temporale di un'azione", calcolata come `STEP_TIME * velocità_urbana_di_riferimento` = 15s × 11.1 m/s ≈ 167m. Da quando i 2s di `RED_TIME` sono tutto-rosso reale (nessun veicolo avanza in quella finestra), la finestra "utile" per coprire distanza è tornata a `GREEN_TIME + YELLOW_TIME` = 13s, non più 15s. Il cutoff è stato quindi ridefinito come costante derivata — `VISION_CUTOFF_M = (GREEN_TIME + YELLOW_TIME) * VEHICLE_SPEED_MS = 13 * 11.1 ≈ 144.3m` ([cityflow_env.py:70-79](src/environment/cityflow_env.py#L70)) — invece di restare un numero fisso (167) scollegato dalla nuova dinamica. Se `GREEN_TIME`/`YELLOW_TIME` cambiano ancora in futuro, il cutoff si aggiorna automaticamente.

---

## 5. Anti-starvation (non è "giallo", ma è un altro vincolo sulle fasi)

`use_action_mask=True` (default, disattivato nei preset `paper`/`environment`) maschera la fase corrente come azione non valida se è già stata scelta **2 volte consecutive** ([cityflow_env.py:769-788](src/environment/cityflow_env.py#L769)) — impedisce che una direzione tenga il verde all'infinito mentre le altre sono ferme. È un vincolo indipendente dal meccanismo verde/giallo/rosso: agisce sulla *scelta* dell'azione (a monte), non sulla *fisica* del cambio fase (a valle, dentro lo `step()`).

---

## 6. Formulazione MDP

- **Stato** `s_i^t`, per intersezione, dim 32 (avanzato) o 20 (paper): `[n_vec (12), wait_vec (12 se avanzato), p_vec (8)]`.
  - `n_vec`: numero di veicoli per corsia, normalizzato (`/30`, clip [0,1]). In modalità avanzata (`use_vision_cutoff=True`) conta solo veicoli entro **`VISION_CUTOFF_M` (~144m, derivazione in §4 "Effetto a cascata sul campo visivo")** dal semaforo; in modalità paper conta tutta la corsia (visibilità globale).
  - `wait_vec`: tempo di attesa massimo sulla corsia, normalizzato (`/100`, clip [0,1]); presente solo se `use_wait_vec=True`.
  - `p_vec`: fase corrente, one-hot a 8 bit.
- **Azione** `a_i^t`: indice di fase 0-7 (mai la fase di tutto-rosso, interna).
- **Reward** `r_i^t`, due modalità (`reward_mode`):
  - **`custom`** (avanzata): `reward = clip((Passed - Incoming - alpha*max_red_wait_time - wasted_green_penalty) / 100, [-20, 5])` — multi-obiettivo (throughput, coda sulle corsie rosse, penalità se un verde non fa passare nessuno).
  - **`paper`** (originale Wang et al. 2022): `reward = -P_i / 100`, dove `P_i` = pressione (veicoli in ingresso − veicoli in uscita). La divisione per 100 è una riscalatura numerica aggiunta per stabilità del training (vedi commento in [cityflow_env.py:504-520](src/environment/cityflow_env.py#L504)) — non altera la definizione di pressione né la policy ottima indotta.
- **Grafo**: le intersezioni sono nodi di un grafo, con archi verso i **vicini fisici** (intersezioni collegate da una strada diretta, troncato a `num_neighbors=4` più vicini per distanza euclidea). `get_edge_index()` produce il formato PyTorch Geometric usato dai modelli spaziali (GAT/GCN/ecc.).
- **Meta-feature** (usate dal meta-learner del modello, non dal reward): `get_spatial_meta_features` (pressione/veicoli per corsia + distanza dai vicini) e `get_temporal_meta_features` (proxy di coda + storico ultimi 5 step) — dettagli architetturali nel codice modello, non ambiente in senso stretto.

---

## 7. Metriche: le tre varianti di travel time (e perché esistono)

- **`get_average_travel_time(include_unfinished=True)`** — **metrica di default, usata per training/ranking**. Include anche i veicoli ancora in rete al momento della misura, contati con il tempo già trascorso (lower bound). Introdotta per evitare survivorship bias: senza questo, un modello che ingolfa la rete e fa passare solo pochi veicoli "fortunati" risulterebbe premiato con un travel time basso calcolato su un campione piccolo e non rappresentativo.
- **`get_completed_only_travel_time()`** — solo veicoli arrivati (equivalente a `include_unfinished=False`). Utile come dato informativo aggiuntivo, mai per scegliere tra modelli.
- **`get_original_average_travel_time()`** — chiama direttamente `engine.get_average_travel_time()` nativo di CityFlow (stessa convenzione "solo arrivati"). Serve solo per confronto diretto con un numero riportato in letteratura, non per confrontare i nostri modelli tra loro.
- **`get_throughput()`** — veicoli che hanno completato il viaggio: `len(tutti gli spawnati) - veicoli ancora in rete`.

---

## 8. Flag di ablation che modificano l'ambiente

| Flag | Default | Effetto |
|---|---|---|
| `reward_mode` | `custom` | `paper` → reward = pressione grezza (§6) invece del reward multi-obiettivo |
| `use_vision_cutoff` | `True` | `False` → visibilità globale sulla corsia invece del cutoff a `VISION_CUTOFF_M` (~144m) |
| `use_wait_vec` | `True` | `False` → stato a 20 dim (senza `wait_vec`), come nel paper originale |
| `use_action_mask` | `True` | `False` → nessun vincolo anti-starvation (§5) |

Combinazioni attivate dai preset `--ablation` (full/paper/environment/temporal/rl_core/replay_stability) — vedi `scripts/train.py` per la mappa esatta.

---

## 9. Replay e visualizzazione

- **Generazione**: quando `saveReplay=True` nella config CityFlow, il motore scrive due file per episodio: `replayLogFile` (`replay.txt`, posizioni veicoli + stato semafori) e `roadnetLogFile` (`roadnet.log`, rinominato `roadnet_log.json` a fine run — geometria statica per il renderer).
- **Formato di `replay.txt`**: una riga per secondo simulato, `carLogs;tlLogs`. `carLogs` è una lista di veicoli `x y angolo id laneChange lunghezza larghezza`. `tlLogs` è una lista di blocchi `roadId stato1 stato2 ...` (uno stato per corsia di quella strada), dove ogni stato è `'g'` (verde), `'r'` (rosso) o assente/`'i'` (non applicabile). **Questo è il meccanismo con cui abbiamo verificato empiricamente il tutto-rosso** (§4): durante `RED_TIME` tutte le corsie di ogni intersezione mostrano `'r'` simultaneamente.
- **Visualizzatore**: il frontend web ufficiale di CityFlow è vendorizzato in `CityFlow/frontend/` (`index.html` + `script.js`, libreria PixiJS). Legge `roadnet.json` (geometria) e `replay.txt` (stato per step) dalla stessa cartella; la funzione `drawStep()` in `script.js` fa il parsing riga per riga e colora le corsie in base allo stato `'g'`/`'r'` letto dal replay — quindi il replay **mostra esattamente e onestamente** ciò che la simulazione fa, non introduce interpretazioni proprie.
- **Come aprirlo**: copiare `replay.txt` e `roadnet_log.json` (rinominato `roadnet.json`) in `CityFlow/frontend/`, poi aprire `CityFlow/frontend/index.html` in un browser (serve un server statico locale per il fetch dei file, es. `python -m http.server` dalla cartella `CityFlow/frontend/`). `train.py` fa già questa copia automaticamente a fine training.

---

## 10. Semplificazioni note da dichiarare onestamente in tesi

- Il giallo (`YELLOW_TIME=3s`) è trattato come verde esteso, non come stato fisico distinto — scelta dichiarata, non equivoca dopo la modifica del §4 (il tutto-rosso invece è reale).
- CityFlow non simula la fisica di collisione punto-per-punto tra veicoli di movimenti diversi all'interno del poligono di intersezione; la sicurezza della transizione di fase è garantita **a livello di simulazione dei movimenti abilitati/disabilitati** (nessun veicolo nuovo entra su un movimento rosso), non da un motore fisico di collisione.
- Il campo visivo (`use_vision_cutoff`, `VISION_CUTOFF_M`) e il proxy di coda basato sul conteggio veicoli (non sulla velocità, non disponibile per corsia in CityFlow) sono approssimazioni dichiarate nel codice stesso.
- `maxStep=1800` (30 minuti simulati) per episodio, `STEP_TIME=15s` → **120 azioni per episodio** per ogni intersezione.
