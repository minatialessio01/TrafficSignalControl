# Descrizione di stato, meta-learner e reward

Cosa riceve in input la Q-network (stato + meta-feature spaziali/temporali) e
cosa la allena (reward) nella configurazione **Pro** (default, `--ablation
pro`, quella usata dai modelli `_official`) — per ciascun elemento: come si
calcola, cosa rappresenta, perché è stato scelto. Le differenze rispetto al
paper originale (Wang et al., 2022) e agli altri preset di ablation sono in
`descrizione_modelli.md` §1/§3; qui il focus è sul "perché così" più che sul
confronto riga per riga col paper. Fonte primaria del codice:
`src/environment/cityflow_env.py` (`_get_observations`,
`get_spatial_meta_features`, `get_temporal_meta_features`,
`_compute_rewards`).

---

## 1. Stato (input diretto alla Q-network — 32 dim)

| Componente | Dim | Formula | Perché |
|---|---:|---|---|
| `n_vec` | 12 | Veicoli per corsia in ingresso, entro `VISION_CUTOFF_M` (~144m) dal semaforo se `use_vision_cutoff=True`, normalizzato `/30` | Il dato base: quanti veicoli attendono su ciascuna delle 12 corsie in ingresso. Mai le corsie in uscita — quelle restano nel meta-learner (§3), non nello stato diretto. |
| `wait_vec` | 12 | Tempo di attesa del veicolo più fermo per corsia (entro lo stesso cutoff), saturato a 100s e normalizzato in [0,1] | Componente di equità/fairness nello stato: la pressione dice "quanto è congestionato", `wait_vec` dice "da quanto tempo qualcuno soffre" — un'informazione che MaxPressure non ha (nessun termine anti-starvation nel suo argmax puro). |
| `p_vec` | 8 | One-hot della fase corrente | Necessario perché la rete sappia da quale configurazione di movimenti parte prima di scegliere la prossima fase. |

`VISION_CUTOFF_M = (GREEN_TIME + YELLOW_TIME) · VEHICLE_SPEED_MS ≈ 144m`: la
distanza percorribile da un veicolo nella finestra temporale in cui la fase
corrente può ancora farlo muovere prima del prossimo cambio — non un numero
arbitrario, deriva dalla dinamica del semaforo stesso (dettaglio completo in
`descrizione_environment.md` §4).

**Pressione per fase nello stato — sperimentata, non adottata di default.**
`use_phase_pressure_state` (flag `--phase-pressure-state`, off di default)
aggiunge in coda un vettore di 8 valori, la stessa quantità che MaxPressure
usa per il suo argmax (§4 sotto), calcolata rispettando `VISION_CUTOFF_M`
(stato 32→40 dim). Un solo tentativo di training
(`metastgat_pro_0.2_meta_v2_state_pressure`) è stato interrotto a metà
episodio per una revisione di formula concorrente (§7) — nessun dato
utilizzabile, resta un'opzione disponibile ma non validata.

---

## 2. Perché un meta-learner (e non solo più input diretto)

Il modello genera i pesi di GAT (spaziale) e LSTM (temporale) da due
sotto-reti dedicate — **SMK** (Spatial Meta-Knowledge) e **TMK** (Temporal
Meta-Knowledge) — invece di usare pesi fissi condivisi tra tutti i nodi e
istanti, come fanno quasi tutti i metodi concorrenti (vedi
`descrizione_letteratura_correlata.md` §6: nessuno dei 5 SOTA letti usa un
meta-hypernetwork). Due motivi per mantenerla comunque:

1. **Il paper originale la giustifica esplicitamente** (§4.3): *"When
   calculating the attention score, [...] the nodes in the default graph are
   fixed, [...] the weights of the two nodes are fixed. However, [...] one
   attention mechanism cannot be applied to two connected intersections at
   all times."* — un GAT/LSTM a pesi fissi applica la stessa regola di
   aggregazione a ogni nodo e istante, mentre il traffico cambia sia nel
   tempo sia da un incrocio all'altro.
2. **Motivazione di ricerca**: sperimentare un meccanismo di meta-learning
   non ancora provato in questo dominio, indipendentemente dall'esistenza di
   alternative più semplici.

**Nota onesta per il capitolo limiti**: l'ablation study del paper originale
(RQ2) non separa mai "più informazione" da "informazione usata per generare
pesi invece che come input diretto" — il loro baseline STGAT non riceve mai
le feature SMK/TMK in nessuna forma, nemmeno concatenate. Il progetto ha già
in `src/models/stgat.py` l'esperimento che risolverebbe la domanda (STGAT con
le stesse feature di MetaSTGAT, concatenate invece che meta-generate), mai
eseguito — vedi `descrizione_letteratura_correlata.md` §6 per il dettaglio.

**Regola di ammissione per ogni feature meta**: è utile solo se **varia** —
nel tempo per lo stesso nodo (→ TMK) o tra nodi anche se costante nel tempo
per un nodo specifico, es. un angolo ha sempre 2 vicini (→ resta SMK utile).
L'unico caso inutile è un valore uguale per tutti i nodi e tutti gli istanti:
il meta-learner genererebbe sempre lo stesso peso, equivalente a non avere
meta-learning per quella feature. Verifica concreta di questa regola:
`distanze dai vicini`, usata dal paper originale, viola la regola su questo
progetto — su griglie a lunghezza di strada uniforme, la distanza normalizzata
tra vicini reali è quasi sempre 1.0, una costante travestita da feature.
Sostituita da `real_degree` (§3).

---

## 3. SMK — meta-learner spaziale (26 dim)

| # | Feature | Dim | Come si calcola | Perché |
|--:|---|---:|---|---|
| 1 | `lane_pressure` | 12 | Per ciascuna delle 12 corsie in ingresso: `(veicoli_in_corsia − media_uscite_raggiungibili) / 30`. Le uscite raggiungibili sono solo quelle che quella specifica corsia può raggiungere secondo il roadnet (`lane_reachable_outgoing`, da `roadLinks[i].laneLinks[j]`), non una media su tutte le uscite dell'incrocio — una corsia di sola-destra e una dritta non portano alle stesse corsie di uscita. | Dato grezzo su cui si basano le feature 3-5: la "forma" della congestione attorno al nodo, corsia per corsia, per pesare l'attenzione verso ciascun vicino. |
| 2 | `real_degree` | 1 | `len(vicini_reali) / num_neighbors` — 0.5 angolo, 0.75 bordo, 1.0 interno | Condiziona *come* aggregare l'attenzione: un nodo d'angolo deve pesare più forte i pochi vicini che ha, altrimenti l'attenzione media diluisce il segnale sui posti di padding. Sostituisce `distances` (degenere, §2). |
| 3 | `pressure_diff_neighbors` | 4 | Per ciascun vicino reale (0 per i padding): `pressione_media_propria − pressione_media_vicino`, dove "pressione media" è la media delle 12 `lane_pressure` del nodo. Il grado di padding dipende solo dai vicini reali del nodo CORRENTE, non da quelli del vicino (una proprietà già incorporata nella sua `lane_pressure`). | Confronto relativo diretto ("sono più o meno congestionato di lui"), pensato per guidare verso quale vicino spostare peso nell'attenzione. |
| 4 | `traffic_asymmetry_ns_ew` | 1 | `media(lane_pressure[N/S]) − media(lane_pressure[E/O])` | Descrive la forma della congestione (quale asse è più carico) osservata in questo istante, non una posizione nota a priori — evita di overfittare alla riga dell'arteria vista in training. |
| 5 | `phase_pressure` | 8 | Per ciascuna delle 8 fasi: stessa formula per movimento usata da `MaxPressureAgent` (§4), rispettando `VISION_CUTOFF_M` sia in ingresso sia in uscita (entrambi misurati come "vicino a questa intersezione", sui due lati opposti del movimento). | Il vero valore che MaxPressure usa per decidere, dato al meta-learner invece che allo stato diretto — verifica se questo aiuta il modello ad avvicinarsi al comportamento di MaxPressure. Unica feature SMK che rinuncia in parte all'argomento di frugalità (§4-A: piena fedeltà a MaxPressure entro comunque lo stesso raggio di visibilità del resto del sistema). |

Scartate: identità/ruolo fisso del nodo (romperebbe la generalizzazione a
topologie mai viste in training); `n_vehicles` grezzo (ripeteva `n_vec` dello
stato).

---

## 4. TMK — meta-learner temporale (25 dim)

| # | Feature | Dim | Come si calcola | Perché |
|--:|---|---:|---|---|
| 1 | `queue_trend` | 12 | `n_vec(adesso) − n_vec(3 step fa)`, per corsia (finestra di `history` + stato corrente) | Derivata discreta — la coda cresce o cala — non un livello assoluto (già dato da `n_vec`). Segnale che dovrebbe modulare i gate forget/input della LSTM. |
| 2 | `phase_dwell_time` | 1 | `min(consecutive_phases, CAP) / CAP`, `CAP=2` con maschera anti-starvation attiva (il vero massimo raggiungibile: la maschera forza un cambio al 3° step consecutivo) | **Non** uno smaltimento di coda (il range reale è troppo corto per rappresentarlo) — indica quanto è vincolata la PROSSIMA decisione: a 1 la scelta è libera su tutte le 8 fasi, a 2 la maschera eliminerà la fase corrente dalle opzioni valide al passo dopo. La LSTM può anticipare questo vincolo strutturale imminente. |
| 3 | `queue_volatility` | 12 | Deviazione standard di `n_vec` per corsia sulla finestra disponibile (fino a 5 punti) | Quanto fidarsi dell'osservazione corrente (alta volatilità → possibile rumore) — indipendente dal trend: una coda può essere stabile-alta, stabile-bassa o instabile. |

Scartate: storico grezzo di `n_vec` su 60 dim (ridondante con la memoria
`(h,c)` che la LSTM mantiene già da sola — si ridava in pasto al meta-learner
la storia che serve a decidere come processare quella stessa storia);
**posizione nel ciclo di traffico giornaliero** — proposta e ritirata: il
flusso di training ha 5 fasce di intensità mappate sull'episodio, ma le
config di test usano flussi `flat`/`peak`, non lo stesso schema. Un
meta-learner che impara "quando l'orologio segna X, reagisci così" applica un
pregiudizio temporale sbagliato a ogni step di decisione in generalizzazione
— non un fatto neutro ignorato se non serve, ma un segnale dinamico attivo e
sistematicamente scorretto fuori distribuzione. `queue_trend`/
`queue_volatility` ottengono la stessa consapevolezza di "regime" in modo
reattivo (calcolato da ciò che succede davvero), quindi generalizzano a
qualunque forma di flusso.

`phase_pressure` (SMK, sopra) **non** è ripetuta qui: è un dato spaziale per
fase in un dato istante, non una sua dinamica nel tempo — ripeterla identica
non aggiungeva informazione che il TMK potesse elaborare diversamente dal SMK.

---

## 5. Reward

| Reward | Formula | Quando si usa |
|---|---|---|
| **Custom** (default, modello Pro) | `(Passed − Incoming − α·MaxRedWait − Penalty) / 100`, clip `[-20, 5]` — `Passed − Incoming` è un termine throughput, `α·MaxRedWait` un termine di equità pesato dallo sweep su α (0.5/0.2/0.1/0.0), `Penalty=50` se un verde non fa passare nessuno pur avendo veicoli in coda | Preset `pro`/`temporal`/`rl_core`/`replay_stability` |
| **Paper** (Wang et al., 2022) | `-P_i / 100`, `P_i` = veicoli in ingresso − veicoli in uscita dell'intero incrocio | Preset `paper`/`environment` (fedeltà all'originale per l'ablation study) |

**Perché non pressione pura per il modello Pro**: la reward custom è già,
nello spirito, "pressione + attesa" — incentiva il deflusso reale
(throughput) e penalizza sia lo stallo di corsie secondarie sia il verde
concesso a vuoto, un'informazione che la pura pressione del paper non cattura
(nessun termine di equità).

**Sperimentato e scartato**: sostituire il termine throughput con la
pressione stile paper (`use_pressure_reward_term`, primo termine =
`Outgoing − Incoming_ora` invece di `Passed − Incoming`, resto della formula
invariato). Testato su 100 episodi (`metastgat_pro_0.0_meta_v3_pressure_reward`):
TT di selezione 189.7s, indistinguibile dalla reward originale (187.2s) —
allenare direttamente sulla pressione invece che sul throughput non ha
chiuso il gap con MaxPressure. Il gap sembra strutturale (vedi §6), non un
problema di segnale di reward.

---

## 6. Perché non copiare MaxPressure fino in fondo

MaxPressure (`src/agents/maxpressure_agent.py`) sceglie, per ogni incrocio, la
fase con `argmax` della pressione per movimento
(`Σ_{(in,out)} count(in) − count(out)`, sui lane-link esatti di quella fase),
a **visibilità completa** sull'intera corsia e senza alcun termine di equità.
Resta il miglior modello sul travel time medio, sia in training che in
generalizzazione. Ogni scelta di questo progetto che se ne allontana si
appoggia a una di due argomentazioni:

**(A) Costo/fattibilità della raccolta dati nel mondo reale.** Il modello
limita la visibilità a `VISION_CUTOFF_M` (~144m, sensori a corto raggio vicino
al semaforo, non l'intera strada) e, nel meta-learner, usa solo dati statici
(geometria) o già raccolti per un altro scopo (tempo in fase) — mai nuovi
sensori dedicati (eccetto `phase_pressure`, §3, che resta comunque entro lo
stesso raggio). Argomento: *il sistema assume un'infrastruttura di sensori
realistica e minima, non la visibilità totale e istantanea che un algoritmo
euristico può permettersi in simulazione.*

**(B) Generalizzabilità del modello.** La pressione per fase richiede di
conoscere in anticipo `phase_lanelinks` — specifico dello schema semaforico e
della topologia. Le componenti che restano vicine al conteggio per-corsia
grezzo (`n_vec`, `wait_vec`) si adattano a qualunque schema senza modifiche
strutturali; introdurre `phase_pressure` è un compromesso deliberato — chiude
parte del gap ma lega il modello, come MaxPressure stesso, a una
rappresentazione fissa dello schema. Per questo il progetto non si spinge
oltre (niente `outgoing_vec` per-corsia esplicito nello stato): ogni ulteriore
avvicinamento a MaxPressure eroderebbe la generalità già dimostrata su tre
topologie (4x4/5x5/6x6, 100m/200m).

---

## 7. Correzioni di formula (13/9/2026)

Tre bug/imprecisioni corretti nella stessa revisione, che invalidano
numericamente ogni risultato ottenuto prima di questa data con le feature
meta sopra (i modelli `_official`, avviati dopo, le incorporano già):

- **`lane_pressure`** sottraeva una media *globale* (tutte le corsie in
  uscita dell'incrocio) da ogni corsia in ingresso, indipendentemente da
  quali uscite fossero davvero raggiungibili da quella corsia — corretto a
  media *locale* (§3.1).
- **`phase_pressure`** ignorava sempre `use_vision_cutoff` (piena visibilità
  anche con cutoff attivo) — corretto per rispettarlo come il resto del file
  (§3.5), e spostata dal TMK al solo SMK.
- **`phase_dwell_time`**: cap di normalizzazione ricalibrato da 10
  (arbitrario) a 2 (il vero massimo con maschera anti-starvation attiva) —
  §4.2.

Formule attuali già incorporate sopra; questa sezione resta solo come nota
storica per interpretare eventuali risultati più vecchi trovati in
`results/`.
