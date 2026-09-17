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
`_compute_rewards`); per la loss che usa la reward per allenare la Q-network
(§8), `src/agents/dqn_agent.py` (`update`).

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
| **Custom** (default, modello Pro) | `(Passed − Incoming − α·MaxRedWait − Penalty) / 100`, clip `[-10, 0]` — `Passed − Incoming` è un termine throughput, `α·MaxRedWait` un termine di equità pesato dallo sweep su α (0.5/0.2/0.1/0.0), `Penalty=50` se un verde non fa passare nessuno pur avendo veicoli in coda | Preset `pro`/`temporal`/`rl_core`/`replay_stability` |
| **Paper** (Wang et al., 2022) | `-P_i / 100`, `P_i` = veicoli in ingresso − veicoli in uscita dell'intero incrocio | Preset `paper`/`environment` (fedeltà all'originale per l'ablation study) |

**Il clip `[-10, 0]` (corretto il 14/9/2026, era `[-20, 5]`)**: `Passed` è un
sottoinsieme di `Incoming` per costruzione (`Passed = |in_t0 \ in_t1|`), quindi
`Passed ≤ Incoming` sempre e il numeratore è sempre `≤ 0` — il lato `+5` del
vecchio clip non era mai raggiungibile, non per rarità ma per costruzione
matematica. Sul lato negativo: con strade da 100m, `VISION_CUTOFF_M=144.3m`
supera la lunghezza della strada, quindi il cutoff è 0 e si contano tutti i
veicoli della corsia (~13/corsia con veicolo 5m + minGap 2.5m, ~156 su
`N_LANES=12` nel gridlock totale su tutte le corsie); `MaxRedWait` è limitato
dalla durata dell'episodio (120 step da 15s = 1800s max, non un secondo di
più). Worst case fisico con α=0.5 (il più alto dello sweep):
`-156 - 0.5·1800 - 50 = -1106`, cioè **-11.06** dopo `/100` — il vecchio `-20`
non era mai raggiunto nemmeno nel caso limite assoluto. Il nuovo clip
`[-10, 0]` riflette il range realmente utilizzabile con queste config; non
cambia alcun risultato già ottenuto, dato che il vecchio clip non tagliava mai
nulla nella pratica (i modelli allenati prima di questa data restano validi).

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

**Sperimentato e scartato (15/9/2026, rimosso dal codice lo stesso giorno)**:
due varianti aggiuntive esplorate nella stessa giornata, entrambe eliminate
su decisione esplicita — la reward ufficiale resta `Passed − Incoming` per
il modello Pro e `-P_i` con `incoming` a t+1 per `paper` (nessuna delle due
tocca il codice attuale, che non ha più i flag corrispondenti):

- `use_pressure_incoming_t0` — usava `incoming` a t0 invece che a t+1 in
  `P_i`, per coerenza causale con l'azione appena scelta. Scartato: si
  preferisce mantenere t+1 su entrambi i lati di `P_i` per `paper`.
- `use_phase_pressure_reward_term` — sostituiva il primo termine con
  `-phase_pressure(fase scelta)` (la stessa quantità di `MaxPressureAgent`,
  ma solo sulla fase effettivamente scelta, non sull'argmax di tutte le 8).
  Corretto un bug di scala il giorno stesso (mancava un `*30` per essere
  sulla stessa scala di `Passed - Incoming`), testato su
  `metastgat_pro_0.0_phasepressure_seed7_official` (alpha=0.0, 100 episodi):
  risultati giudicati insoddisfacenti, addestramento interrotto e modello
  eliminato.

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

---

## 8. Come la reward allena la rete: Huber loss vs MSE

La reward di §5 diventa un segnale di apprendimento tramite il target TD e la
loss che confronta quel target con la previsione della rete
(`DQNAgent.update`, [dqn_agent.py:440-475](src/agents/dqn_agent.py#L440-L475)):

```
q_target = r + gamma * Q(s', a'*; target_network)     # a'* = argmax se Double DQN
errore   = q_pred - q_target
```

`use_huber=True` (default, disattivato solo dai preset `paper`/`replay_stability`)
sceglie **Huber loss** (in PyTorch, `smooth_l1_loss`, `δ=1.0` di default — mai
cambiato in questo codice) al posto della **MSE** che userebbe altrimenti:

$$
L_\delta(\text{errore}) =
\begin{cases}
\tfrac{1}{2}\,\text{errore}^2 & \text{se } |\text{errore}| \le \delta \\[4pt]
\delta\left(|\text{errore}| - \tfrac{1}{2}\delta\right) & \text{se } |\text{errore}| > \delta
\end{cases}
$$

**Come funziona, in pratica**: è una funzione ibrida, quadratica vicino a zero
e lineare oltre la soglia `δ`. Per un errore piccolo (`|errore| ≤ 1`) coincide
quasi con la MSE (stesso comportamento, stesso gradiente che cresce
proporzionalmente all'errore) — utile perché vicino al target la quadratica dà
un gradiente che si annulla dolcemente man mano che l'errore si riduce,
favorendo una convergenza fine. Per un errore grande (`|errore| > 1`), invece
di continuare a crescere col quadrato (MSE: un errore di 100 pesa 10000 volte
un errore di 1), cresce **linearmente** (un errore di 100 pesa "solo" 100
volte un errore di 1): un singolo TD-error anomalo (es. un target instabile
subito dopo un update del target network, o un episodio con un evento raro
tipo il bug dei veicoli "teletrasportati" in gridlock, vedi
`_compute_rewards`) non genera un gradiente enorme che stravolge i pesi in un
solo step — l'aggiornamento resta proporzionato invece che dominato
dall'outlier. Questo è il motivo per cui si dice che la Huber loss è "più
robusta agli outlier della MSE": non li ignora, li pesa meno che
quadraticamente.

**Nella pipeline di questo codice**, la loss è calcolata elemento per
elemento (`reduction='none'`) su un tensore `(batch, step_BPTT, intersezioni)`,
mediata prima sulla sequenza/intersezioni di ciascun campione
(`loss_per_seq`), poi pesata dagli **IS weights** del PER (se `use_per=True`,
altrimenti pesi costanti 1.0 — campionamento uniforme) e infine mediata sul
batch. Il gradiente risultante passa poi da `use_grad_clip`
(`clip_grad_norm_=1.0`, se attivo) prima di `optimizer.step()` — un secondo
livello di protezione dagli stessi outlier, indipendente dalla scelta
Huber/MSE ma con lo stesso obiettivo (vedi M8 in `descrizione_modelli.md` §1).
L'errore TD (`|q_pred - q_target|`, non la loss) è anche riusato, separatamente,
per aggiornare le priorità del PER — la stessa quantità che alimenta la loss
alimenta anche quali sequenze verranno ricampionate più spesso.

---

## 9. Ottimizzatore: RMSprop, non Adam

`DQNAgent` usa **RMSprop** (`torch.optim.RMSprop`, lr=1e-3,
[dqn_agent.py:168-169](src/agents/dqn_agent.py#L168-L169)) per **tutti** i
modelli di questo progetto, non solo per il preset `paper` — a differenza
degli altri dettagli di training (Huber, PER, Double DQN, soft update, grad
clip), l'ottimizzatore **non è un flag di ablation**: non esiste un modo per
scegliere Adam da riga di comando.

**Perché RMSprop**: è la scelta dell'articolo originale (Wang et al. 2022,
Section 5.1) — vedi `metastgat_diff_analysis.md`. È stata ereditata
inalterata quando il progetto ha introdotto tutte le altre modifiche (reward,
stato, self-loop, meta-learner, Double DQN, PER, ...): a differenza di quelle,
non è mai stata messa in discussione né testata in alternativa. Non è quindi
una scelta deliberata per il modello Pro, è un default rimasto tale per
mancanza di un esperimento dedicato, non per un argomento tecnico specifico a
favore di RMSprop.

**Adam non è "necessariamente migliore"**: sono ottimizzatori adattivi
imparentati — Adam è essenzialmente RMSprop più un termine di momento (media
mobile del gradiente stesso, non solo del suo quadrato) più una correzione di
bias nelle prime iterazioni. In letteratura RL il quadro è misto: il DQN
originale (Mnih et al., 2015) usa RMSprop, mentre molte implementazioni più
recenti (es. gli algoritmi di Stable-Baselines3) di default usano Adam. Il
momento di Adam può aiutare la convergenza su gradienti lisci, ma in un
contesto con target non stazionari (il target network cambia nel tempo,
seppur lentamente con soft update) può anche accumulare una direzione
"vecchia" che non riflette più il target corrente — non un argomento
decisivo in nessuna delle due direzioni, solo una differenza di
comportamento. Nessun esperimento in questo progetto ha confrontato i due:
la domanda resta aperta, e sarebbe un candidato naturale per un nuovo asse di
ablation (`--optimizer {rmsprop,adam}`) se si volesse chiuderla con un dato
invece che con un'aspettativa.

---

## 10. Correzione del tempo di attesa (14/9/2026) — invalida i risultati precedenti

**Il bug**: `vehicle_wait_times[veh]` si azzerava ogni volta che la velocità
del veicolo era `≥ 0.1 m/s` nell'istante campionato (una volta a step, dopo
i 2s di tutto-rosso) — non distingueva "il veicolo ha attraversato
l'incrocio" da "il veicolo si è solo spostato in avanti nella coda senza
attraversare". Un veicolo bloccato per molte fasi, che ogni tanto avanza di
qualche metro quando i veicoli davanti a lui defluiscono, vedeva il proprio
contatore azzerarsi ripetutamente pur non essendo mai passato.

**Verifica empirica** (fixed-time su `config_4x4_200m_6k_peak`, un episodio
da 120 step): su tutti i reset di `vehicle_wait_times`, quelli con il
veicolo ancora presente tra gli "in ingresso" della stessa intersezione
(quindi non transitato — reset spuri) erano **10.743**, contro **8.018**
reset legittimi (veicolo uscito dal set, presumibilmente attraversato). Il
tempo di attesa perso nei reset spuri: mediana **45s**, massimo osservato
**330s** in un singolo evento.

**Il fix** ([cityflow_env.py, `step()`](src/environment/cityflow_env.py)):
`vehicle_wait_times` ora cresce finché il veicolo resta presente nel set
"in ingresso" di una qualunque intersezione (stessa definizione già usata
per `incoming_t0`/`incoming_t1`), indipendentemente dalla sua velocità
istantanea, e si azzera solo quando esce da quel set (attraversato, o mai
stato vicino a un incrocio). Riverifica sullo stesso scenario dopo il fix:
**zero** reset spuri. Nessun nuovo flag: era un bug di definizione, non un
asse di ablation legittimo, quindi corretto direttamente per tutti i modelli
(preset `pro`/`temporal`/`rl_core`/`replay_stability`), a partire dal
prossimo modello allenato.

**Perché non la semplificazione "solo il primo veicolo in coda"** (proposta
come possibile alleggerimento): anche isolando il veicolo in testa a ogni
corsia, servirebbe comunque un contatore persistente per sapere da quanto
tempo è lì — non elimina il bisogno di uno stato per-veicolo, elimina solo
la copertura per gli altri veicoli in coda, che servono comunque per
`vehicle_max_wait_ns/ew` (metriche di equità per-veicolo, non solo il
peggiore per corsia). Il ciclo per-veicolo già esisteva identico nel codice
precedente (`get_vehicle_speed()` itera su tutti i veicoli della
simulazione): la correzione non aggiunge costo computazionale rispetto a
prima.

**Cosa invalida e cosa no** — la reward custom ha DUE termini, solo uno
tocca `vehicle_wait_times`:

| Quantità | Dipende dal bug? | Nota |
|---|---|---|
| `Passed - Incoming` (throughput, 1° termine) | No | Calcolato da `incoming_t0`/`incoming_t1`, mai da `vehicle_wait_times` |
| `alpha * MaxRedWait` (2° termine, anti-starvation) | **Sì** | Sottostimato per ogni α > 0: la vera intensità della penalità anti-starvation applicata durante il training era più debole di quanto il valore nominale di α suggerisse |
| `avg_travel_time` / TT medio (tutte le tabelle di questa sessione) | No | Calcolato da `spawn_times`/`arrived_tt`, indipendente da `vehicle_wait_times` |
| `wait_max_ns`/`wait_max_ew`, equità direzionale (tutte le tabelle di questa sessione) | **Sì** | Sottostima sistematica, verificata sopra |
| `wait_vec` nello stato (dim 12, se `use_wait_vec=True`) | Sì (indirettamente) | Satura più spesso a 1.0 dopo il fix (i veicoli davvero fermi a lungo ora si vedono) — non un problema, il cap a 100s era già pensato per questo |

Conseguenza pratica: **i confronti di TT medio fatti in questa sessione
restano validi** (α=0.2/0.1/0.0, `_official`, ablation study — tutti
indipendenti da questo bug). **Non restano validi** i confronti di equità
(`wait_max_ns/ew`) e qualunque conclusione sulla forza reale del termine
anti-starvation per α > 0 — inclusi `metastgat_pro_0.2_official` e
`metastgat_pro_0.1_official` (allenati col codice pre-fix). Eccezione:
l'attuale `metastgat_pro_0.00_official` (rinominato il 15/9/2026 da
`metastgat_pro_0.0_official` per coerenza con lo sweep {0.08, 0.04, 0.00};
promosso il 14/9/2026 al posto del vecchio, era
`metastgat_pro_0.0_seed7_official`, TT medio aggregato migliore sulle 10
config — 229.4s contro 233.2s) ha la sua **suite di test** girata
DOPO il fix (i singoli `test.py` per config sono processi nuovi, importano
il codice corrente), quindi le sue metriche di equità sono già corrette —
mentre `pro_0.2`/`pro_0.1` no: non sono confrontabili 1:1 sull'equità finché
non si ripete anche il loro test. Dato che α=0.0 azzera comunque il secondo
termine della reward, il training di `pro_0.0` non era comunque diluito dal
bug in nessuna delle due versioni.

---

## 11. Ricalibrazione di alpha e del clip (15/9/2026)

Conseguenza diretta della correzione in §10: con `MaxRedWait` tipicamente
più grande, lo stesso α di prima sposterebbe la reward più verso l'equità di
quanto il modello Pro sia mai stato pensato per fare (il throughput deve
restare l'obiettivo primario, l'equità un correttivo, non il contrario).

**Misura** (non su fixed-time come in §10, ma guidando i checkpoint
`_official` veri — `pro_0.2`/`pro_0.1`/`pro_0.0` — attraverso l'ambiente, per
restare ancorati al comportamento dei modelli che contano):

| Quantità | mediana | Note |
|---|---:|---|
| `Passed - Incoming` (throughput) | −13 / −14 | Mai positivo (dimostrazione in `cityflow_env.py`, nota sul clip) |
| `MaxRedWait` PRIMA del fix | 60-120 | Dipende dal modello (meno anti-starvation → codone più ferme) |
| `MaxRedWait` DOPO il fix | 180-210 | Rapporto ADESSO/PRIMA: 2.3-3.3x su pro_0.2/pro_0.1, 1.5-1.75x su pro_0.0 |
| `-phase_pressure` (grezza, flag da allora rimosso — vedi §5) | ≈ −15 | Dopo il fix di scala *30 — prima del fix era ≈ −0.5 |

**Calcolo di partenza**: α ∈ {0.07, 0.035, 0.0}, ciascun valore riscalato sul
rapporto proprio di quel modello (non un fattore unico per tutti), verificato
contro il punto di parità throughput/equità (α tale che `α·MaxRedWait` =
`|Passed-Incoming|` in scala tipica, risultato ≈0.065-0.077 in modo stabile
su mediana/p90/p99): 0.07 coincide quasi esattamente con quel punto.

**Sweep ufficiale fissato (Alessio, 15/9/2026)**: α ∈ {0.08, 0.04, 0.00} —
valori tondi al posto degli equivalenti calcolati sopra, stessa logica (0.08
resta il tetto vicino alla parità, non un valore da superare). **Sostituisce
{0.2, 0.1, 0.0}. Da qui in avanti: se viene chiesto un nuovo modello Pro
senza specificare quale di questi tre alpha usare, va chiesto prima di
allenare, non assunto un default.**

**Clip**: worst case fisico con α=0.08 (il più alto del nuovo sweep):
`-156 - 0.08×1800 - 50 = -350` → **-3.50** dopo `/100`.

> **Aggiornamento 16/9/2026 (Alessio)**: clip abbassato da `[-3, 0]` a
> `[-2, 0]`. A differenza del bound precedente (-3, scelto apposta per
> restare appena sotto il worst case teorico di -3.50 e intervenire quindi
> solo nei casi davvero estremi), **-2 è nettamente più stretto del worst
> case fisico** (-3.50): il clip ora interviene su una fascia molto più
> ampia di step "cattivi ma non estremi", non solo sulla coda peggiore in
> assoluto. Con α=0.08, basta ad esempio `MaxRedWait ≈ 700s` (con
> `Incoming-Passed≈100` e `wasted_green_penalty=50`) per toccare già il
> nuovo floor, contro i ~1800s (quasi l'intero episodio) necessari prima.
> Conseguenza pratica: la ricompensa perde granularità proprio nella fascia
> di attese "gravi ma non catastrofiche" (700-1800s) — una compressione
> concettualmente analoga a quella già discussa per il clip di `wait_vec`
> nello stato (§1, `use_soft_wait_scale`), ma qui sul segnale di training,
> non sulla sola osservazione. Va ricalcolato (o rivalutato) se in futuro
> risultasse che il training soffre di questa perdita di granularità sulle
> code peggiori.

Nessun modello è stato ancora ri-allenato con questi nuovi valori di α — la
tabella e le conclusioni di `descrizione_modelli.md` §2 restano quelle dello
sweep vecchio, da ripetere quando si rifà la coda `_official`.
