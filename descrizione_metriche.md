# Descrizione delle Metriche di Valutazione

Riferimento unico per ogni numero che il codice produce in fase di test/training: cosa misura esattamente, come si calcola, e come va interpretato. Le metriche sono implementate in [`src/environment/cityflow_env.py`](src/environment/cityflow_env.py) (calcolo) e stampate/salvate da [`scripts/test.py`](scripts/test.py) e [`scripts/train.py`](scripts/train.py) (reporting). Per la formulazione MDP (stato/azione/reward) vedi [descrizione_environment.md](descrizione_environment.md) — qui solo le metriche di **valutazione**, non il segnale di reward usato per allenare.

---

## 1. Travel Time — tre varianti, un solo motivo per cui esistono

Tutte e tre misurano lo stesso concetto (tempo dallo spawn all'arrivo di un veicolo) ma su popolazioni diverse di veicoli:

| Metodo | Popolazione inclusa | Uso |
|---|---|---|
| `get_average_travel_time(include_unfinished=True)` (default) | Veicoli arrivati **+** veicoli ancora in rete (contati col tempo già trascorso, un lower bound) | **Metrica di ranking** — usata per il best-model tracking in training e per confrontare i modelli in test |
| `get_completed_only_travel_time()` | Solo veicoli arrivati | Informativa, mai per scegliere tra modelli |
| `get_original_average_travel_time()` | Solo veicoli arrivati (nativo del motore CityFlow, `engine.get_average_travel_time()`) | Solo per confronto diretto con un numero riportato in letteratura (il paper riporta ~430-470s su una rete sintetica 4×4) |

**Perché la prima è quella di default**: senza contare i veicoli non arrivati, un modello che ingolfa la rete e lascia passare solo pochi veicoli "fortunati" su corsie libere otterrebbe un travel time basso calcolato su un campione piccolo e non rappresentativo — nel caso estremo di zero arrivi, la metrica "solo arrivati" restituirebbe 0.0, il punteggio migliore possibile per il peggior comportamento immaginabile. Le altre due restano utili come dato di contesto, mai come criterio di scelta.

In `test.py`, la colonna `travel_time` del CSV/summary è sempre la prima (ranking); `travel_time_completed_only` è riportata a fianco solo come riferimento.

## 2. Throughput

`get_throughput()` = `len(tutti i veicoli spawnati) - veicoli ancora in rete` = veicoli che hanno **completato** il viaggio entro la fine dell'episodio. In `plot_results.py`/`generate_tables()` viene convertito in percentuale (`throughput / total_vehicles_spawned * 100`) per confrontare config con volumi di traffico diversi sulla stessa scala.

## 3. Percentuali di fase (`phase_pct_0..7`)

Frazione di step di decisione in cui ciascuna delle 8 fasi è stata scelta, su tutte le intersezioni e tutto l'episodio (`phase_counts[p] / totale_step_fase * 100`). Non è una metrica di prestazione in sé, ma un diagnostico: rivela se un modello ha collassato su poche fasi (es. MaxPressure tende a preferire fortemente le fasi "dritto", coerente con la sua logica greedy sulla pressione) o distribuisce le scelte in modo più uniforme.

## 4. Coda del Travel Time (`get_travel_time_stats()`)

La media da sola può nascondere casi estremi: un modello con media bassa ma coda lunga (pochi veicoli bloccati molto a lungo) è qualitativamente diverso da uno con distribuzione uniforme, anche a parità di media.

**Include sempre i veicoli non ancora arrivati** (`include_unfinished=True`, default — corretto il 13/9/2026 per essere coerente con `get_average_travel_time()`, che lo fa già): un veicolo mai arrivato entro fine episodio è contato col tempo già trascorso (lower bound del suo vero travel time). Scelta deliberata, non un dettaglio tecnico: un veicolo mai arrivato è quasi sempre il candidato più ovvio per il caso peggiore in assoluto, e un `tt_max` che lo escludesse premierebbe (bias di sopravvivenza) un modello che ingolfa la rete e non fa mai arrivare i suoi veicoli più problematici — esattamente il problema che `get_average_travel_time` evita già di default. Prima della correzione, `tt_max`/`std`/percentili erano calcolati solo su `self.arrived_tt` (soli arrivati), inconsistente con la media.

| Campo | Significato |
|---|---|
| `max` | Travel time del veicolo nel caso peggiore (arrivato più lentamente, o mai arrivato — tempo trascorso finora) |
| `std` | Deviazione standard |
| `p50` | Mediana |
| `p90` / `p95` / `p99` | Percentili — quanto impiega il 10%/5%/1% dei veicoli più lenti |

Se non ci sono ancora veicoli spawnati/arrivati, tutti i campi sono 0.0. Stampata da `test.py` per episodio e in media nel summary finale; **non** loggata da `train.py` (solo l'equità direzionale lo è, vedi §5 — la coda del TT non è stata ritenuta necessaria per il monitoraggio episodio-per-episodio del training, ma è disponibile a costo zero se servisse aggiungerla).

## 5. Equità direzionale (`get_direction_fairness_stats()`)

Nata per rispondere a una domanda specifica: **su una rete con un'arteria più trafficata, alcuni veicoli aspettano sistematicamente più di altri a seconda della direzione da cui arrivano?** MaxPressure garantisce l'ottimalità del throughput di rete ma nessuna garanzia di equità (sceglie sempre la fase con pressione istantanea più alta); il reward "custom" ha invece un termine esplicito anti-starvation. Questa metrica è il modo per verificare empiricamente se quel termine si traduce in un vantaggio reale.

**Cosa conta esattamente** — è una classificazione **per corsia fisica di approccio all'incrocio in questo istante**, non un'origine/destinazione del viaggio:

- Ad ogni intersezione, le 12 corsie in ingresso sono raggruppate per orientamento usando l'ordine canonico di `_get_lanes_per_intersection()` (N,S,W,E × 3 corsie ciascuna): indici 0-5 = **N/S** (chi arriva da Nord o da Sud, quindi attraversa l'incrocio in direzione verticale), indici 6-11 = **W/E** (chi arriva da Ovest o da Est, direzione orizzontale — quella dell'arteria).
- Un veicolo fermo sulla corsia sinistra della strada che arriva da Ovest conta come "W/E", indipendentemente da dove è partito il suo viaggio o dove è diretto — non è un'origine/destinazione, è "su quale corsia sta aspettando adesso".
- Per ogni intersezione, ad ogni `step()` si aggiorna il **massimo storico** (mai un valore istantaneo o una media) del tempo di attesa **consecutivo** (si azzera appena il veicolo si muove, non è un'attesa cumulativa sull'intero viaggio) osservato tra tutti i veicoli presenti sulle corsie di quel gruppo, usando `get_lane_vehicles()` (corsia intera, nessun cutoff di visibilità — a differenza di `wait_vec` dello stato RL) incrociato con `vehicle_wait_times`.
- A fine episodio, `avg_wait_ns`/`avg_wait_ew` sono la media di questi massimi-per-intersezione su tutte le intersezioni; `max_wait_ns`/`max_wait_ew` sono il massimo assoluto; `worst_wait` è il peggiore tra i due gruppi.

**Esempio concreto**: con un'arteria Est-Ovest sulla riga 1 (vedi [descrizione_configurazioni.md §4bis](descrizione_configurazioni.md)), un'intersezione su quella riga vede molto più traffico W/E che N/S. Se un modello favorisce sistematicamente la direzione con più pressione istantanea (il comportamento di MaxPressure), `max_wait_ns` su quelle intersezioni può superare `max_wait_ew` **nonostante** l'arteria sia la direzione W/E — è esattamente la corsia minoritaria (N/S) a pagare il prezzo dello squilibrio, e questa metrica lo rende visibile.

**`max_wait_ns`/`max_wait_ew` vs `max_wait_ns_resolved`/`max_wait_ew_resolved`** (scoperta e aggiunta il 13/9/2026): il record più alto può provenire da un veicolo **ancora fermo nell'istante esatto in cui l'episodio finisce** — l'attesa "vera" potrebbe essere anche più lunga di quella riportata, semplicemente non lo sappiamo perché la simulazione si è fermata, non perché il veicolo sia stato servito. Verificato con un caso reale: `config_4x4_100m_6k_peak`/MaxPressure riportava `max_wait_ns=1125s`; ricostruendo la traiettoria dal replay grezzo (posizione per veicolo, secondo per secondo — indipendente da qualunque bookkeeping interno), il veicolo responsabile (`flow_907_1`) risultava fermo nello stesso identico punto da t=679s fino a **t=1799s, l'ultimissimo secondo dell'episodio** — uno stallo *censurato* (right-censored), non concluso.

Non è un bug: è l'esatto analogo, per l'attesa, di "veicoli non arrivati" per il travel time (§1) — `max_wait_ns`/`max_wait_ew` sono la metrica **primaria** proprio perché nascondere questi casi premierebbe (bias di sopravvivenza) chi lascia un veicolo bloccato per sempre: uno stallo che non finisce mai non fa mai scattare un *nuovo* record dopo l'ultimo istante osservato, quindi sparirebbe da una metrica che considerasse solo gli stalli conclusi.

`max_wait_ns_resolved`/`max_wait_ew_resolved` sono invece il massimo **solo** tra gli stalli conclusi entro la fine dell'episodio (il veicolo responsabile del record si è poi mosso di nuovo — il timestamp del record è precedente all'ultimo istante possibile). Valore uguale o più piccolo di quello censurato, verificabile end-to-end in un replay come un singolo episodio di stallo con inizio e fine osservabili in un'unica intersezione.

**Scelta finale (13/9/2026)**: `compare_models.py`, `test.py` e ogni altro confronto tra modelli usano sempre `max_wait_ns`/`max_wait_ew` (censura inclusa) come metrica ufficiale, mai `_resolved` — per lo stesso principio del §1: un veicolo/stallo che non si risolve mai è il caso peggiore in assoluto, e va mostrato come tale, non escluso perché "non ancora concluso". `_resolved` resta calcolata e salvata in `test_summary_*.json`/CSV (colonna secondaria) solo per chi vuole isolare un esempio concreto e ispezionarlo in un replay, come fatto per la scoperta sopra — non è mai il numero che compare nei grafici o nelle tabelle di confronto principali.

**Risultato empirico osservato** (test MaxPressure vs FixedTime sulle config con arteria, prima dell'introduzione di questa metrica nel training log): l'attesa massima N/S di MaxPressure agli incroci sull'arteria (~529s media/660s max) risultava **peggiore** di quella di FixedTime (~308s media/555s max), nonostante MaxPressure avesse un travel time medio nettamente migliore — la sua reattività greedy sulla pressione istantanea causa uno starvation della direzione minoritaria più marcato di un round-robin ingenuo. È il tipo di risultato che questa metrica è pensata per far emergere.

Stampata da `test.py` per episodio e in summary; loggata da `train.py` (`wait_max_ns`/`wait_max_ew` in `training_log.csv`, aggiunta l'11/9/2026) episodio per episodio — il calcolo avviene comunque ad ogni `step()` indipendentemente dal fatto che venga letto, quindi non ha costo aggiuntivo.

**Distribuzione per-veicolo (`get_raw_vehicle_waits()`, aggiunta il 13/9/2026)**: `max_wait_ns`/`max_wait_ew` hanno un solo punto per intersezione (16 nei nostri roadnet) — troppo pochi per un grafico di distribuzione onesto (es. un violin plot). `get_raw_vehicle_waits()` traccia invece, per **ogni veicolo**, la sua massima attesa consecutiva mai sperimentata su una corsia N/S o W/E durante l'intero viaggio (0.0 se non ci è mai passato) — migliaia di punti invece di 16, con la stessa convenzione include_unfinished delle altre metriche. Non sostituisce `max_wait_ns`/`max_wait_ew` come metrica di ranking (quella resta per-intersezione, pensata apposta per isolare *dove* nella rete si concentra lo squilibrio) — è un dato complementare, salvato da `test.py` in `raw_distributions_<config>.json`. Nota (13/9/2026): lo script che lo consumava (`compare_distributions.py`, violin plot) è stato rimosso per semplificare il confronto a un solo tipo di grafico (`compare_models.py`); il dato resta calcolato e salvato, senza consumatore al momento. Controllo incrociato di correttezza (ancora valido): il massimo della distribuzione per-veicolo coincide esattamente con `max_wait_ns`/`max_wait_ew` (lo stesso veicolo che ha causato il record di rete compare anche qui col suo valore più alto).

## 6. Maschera anti-starvation vs equità — due meccanismi indipendenti

Da non confondere:
- **`get_invalid_actions()`** (§5 di `descrizione_environment.md`) agisce **a monte**, sulla scelta dell'azione: vieta di ripetere la stessa fase più di 2 volte consecutive. Si applica **solo** all'agente RL (mai a MaxPressure, che è pura `argmax` senza vincoli — fedele all'implementazione di riferimento LibSignal).
- **L'equità direzionale** (§5 sopra) è **a valle**, una misura osservazionale di quanto a lungo aspetta la corsia sfavorita — non impone alcun vincolo, si limita a registrare cosa succede. Un modello può rispettare rigorosamente l'anti-starvation (mai la stessa fase 3 volte di fila) e avere comunque una forte disparità N/S vs W/E, se torna sistematicamente più spesso sulla fase che serve l'arteria non appena il vincolo dei 2 consecutivi glielo permette di nuovo.

---

## Dove guardare nel codice

| Metrica | Metodo | File |
|---|---|---|
| Travel time (3 varianti) | `get_average_travel_time`, `get_completed_only_travel_time`, `get_original_average_travel_time` | `cityflow_env.py` |
| Throughput | `get_throughput` | `cityflow_env.py` |
| Coda travel time | `get_travel_time_stats` | `cityflow_env.py` |
| Equità direzionale | `get_direction_fairness_stats` (aggiorna `max_wait_ns`/`max_wait_ew` in `step()`) | `cityflow_env.py` |
| Percentuali fase | `phase_counts` (accumulo locale) | `test.py` |
| Reporting per episodio/summary | `evaluate()` | `test.py` |
| Reporting training log | `run_training()` → `logger.log_episode`/`print_episode` | `train.py`, `src/utils/logger.py` |
