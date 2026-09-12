# Piano di Stesura della Tesi

> **Scopo di questo file**: non è la tesi, è la mappa che useremo per scriverla.
> Quando mi chiederai di redigere in LaTeX una parte già pronta, io (o chi
> riprenderà questa sessione) devo partire da qui: struttura concordata, quali
> sezioni hanno già tutto il materiale necessario, quali no, quali formule usare
> e come, quali fonti citare, e le regole di stile da rispettare capitolo per
> capitolo. Aggiornare questo file ogni volta che cambia lo scope (es. quando
> la parte GCN/SONAR di `istruzioni seconda parte.md` sarà pronta).

---

## 1. Vincoli editoriali (validi per ogni capitolo)

- **Lunghezza minima**: 60 pagine complessive. Stima per capitolo in §3.
- **Registro**: tesi magistrale in intelligenza artificiale — formale, terminologia
  tecnica corretta (RL, GNN, teoria del traffico), niente semplificazioni da
  articolo divulgativo. Il lettore ideale è un relatore/correlatore che conosce
  RL e GNN ma non necessariamente CityFlow o i dettagli di questo progetto.
- **Anti-slop obbligatorio**: prima di considerare chiusa una sezione, applicare
  la skill `stop-slop` (e se serve una riscrittura più aggressiva, `no-ai-slop`).
  Non è un passaggio opzionale di rifinitura: va fatto su ogni sezione appena
  scritta, non solo a tesi finita. Checklist minima da questa skill, da tenere
  a mente *mentre si scrive*, non solo in revisione:
  - Niente incipit standard da tesi ("Il presente lavoro si propone di...",
    "Nel presente capitolo verrà illustrato..."): affermare subito il contenuto.
  - Niente schema meccanico "Innanzitutto / Inoltre / Infine" o "Tuttavia /
    Pertanto / Dunque" ripetuto paragrafo per paragrafo.
  - Niente catene di passivo ("viene generato... viene poi suddiviso... viene
    infine utilizzato..."): nominare il soggetto quando aiuta la chiarezza.
  - Niente affermazioni vaghe ("i risultati sono significativamente
    migliori"): sempre il numero, il confronto, l'unità di misura.
  - Niente avverbi di riempimento (fondamentalmente, sostanzialmente,
    chiaramente, notevolmente) e niente "è importante sottolineare che...".
  - Niente tre aggettivi in fila, niente finale riassuntivo generico
    ("In conclusione, si può affermare che..."): chiudere su un dato concreto.
  - Niente trattini lunghi (—): virgole, due punti o parentesi.
  - Varietà di ritmo tra frasi: non tre periodi di fila della stessa lunghezza.
- **Onestà sperimentale**: questo progetto ha già una cultura precisa di
  "riportare il caso peggiore, non nascondere le code" (vedi
  `descrizione_metriche.md` — veicoli non arrivati inclusi nel travel time,
  attese censurate incluse nell'equità direzionale). La tesi deve mantenere
  lo stesso standard: mai abbellire un risultato, dichiarare esplicitamente le
  semplificazioni (già elencate in `descrizione_environment.md` §10) invece di
  ometterle.
- **Formule matematiche**: ogni simbolo va definito alla prima occorrenza:
  notazione unificata in tutta la tesi (tabella in §4). Nessuna formula va
  scritta a memoria per "sembrare completa": se il dettaglio esatto non è
  verificato in questo file o nei documenti di riferimento, il capitolo va
  marcato "da verificare nel codice" invece di inventare una plausibile
  approssimazione (vale soprattutto per MetaGAT/MetaLSTM, vedi §4.4).

---

## 2. Struttura macro concordata

1. **Introduzione**
2. **Background** (RL, GNN, Traffic Signal Control come campo)
3. **TSC e Modelli** (formalizzazione del problema, ambiente, baseline, MetaSTGAT e varianti)
4. **Confronto** (metodologia sperimentale, risultati, discussione)
5. **Conclusioni**

Le sezioni sotto sono la proposta di dettaglio capitolo per capitolo, con stato
attuale. `[PRONTO]` = materiale/dati già disponibili nel repository, si può
scrivere ora. `[PARZIALE]` = si può scrivere una parte, il resto dipende da
lavoro non ancora fatto. `[BLOCCATO]` = dipende da codice/esperimenti non
ancora esistenti.

---

## 3. Dettaglio capitoli, stato, fonti, stima pagine

### Capitolo 1 — Introduzione `[SCRITTO]` (bozza in `tesi/capitoli/cap1_introduzione.tex`, 13/9/2026)

- Motivazione: congestione urbana, limiti del controllo semaforico a tempi
  fissi e a logica reattiva locale (MaxPressure), promessa del RL multi-agente
  con condivisione spaziale dell'informazione (GNN).
- Contributo di questo lavoro: replica da zero di MetaSTGAT (Wang et al.,
  2022) su un ambiente di simulazione multi-intersezione (CityFlow), sistema
  di ablation a preset per isolare il contributo di ogni scelta implementativa
  rispetto al paper originale, esperimento di generalizzazione dedicato
  (traffico multi-seed + arteria asimmetrica), e — parte non ancora scritta,
  vedi Cap. 3-4 — confronto tra meccanismo spaziale GAT, GCN e SONAR a parità
  di resto dell'architettura.
- Struttura della tesi (un paragrafo per capitolo).
- Fonte: `README.md` (già ha executive summary del progetto), tutto
  `descrizione_*.md` per i dettagli da citare senza ripeterli qui.

### Capitolo 2 — Background `[SCRITTO]` (bozza in `tesi/capitoli/cap2_background.tex`, 13/9/2026)

**2.1 Reinforcement Learning**
- MDP: tupla (S, A, P, R, γ), politica π, funzione valore, funzione
  Q, equazione di Bellman, Bellman ottimale.
- Q-learning tabellare → DQN (Mnih et al., 2015): rete neurale come
  approssimatore di Q, experience replay, target network.
- Estensioni usate nel progetto, da introdurre qui in astratto (il dettaglio
  implementativo va nel Cap. 3): Double DQN (van Hasselt et al., 2016),
  Prioritized Experience Replay (Schaul et al., 2016), Huber loss, soft
  update (Polyak averaging), BPTT + burn-in per reti ricorrenti in un
  contesto RL (R2D2, Kapturowski et al., 2019).
- Fonte per le formule corrette: §4.2 di questo file.

**2.2 Graph Neural Networks**
- Motivazione: perché rappresentare una rete stradale come grafo (nodi =
  intersezioni, archi = strade dirette tra intersezioni vicine).
- GCN (Kipf & Welling, 2017) come meccanismo di base: aggregazione media
  pesata dai vicini.
- GAT (Veličković et al., 2018): attenzione tra nodi vicini, formula completa
  in §4.3 — è il meccanismo usato dal modello Pro di questo progetto.
- Accenno a SONAR (paper NeurIPS 2025 allegato al repository,
  `NeurIPS-2025-sonar-...pdf`) come meccanismo alternativo a "propagazione
  ondulatoria" per informazione a lungo raggio — qui solo l'idea generale;
  il design applicato al nostro problema è ancora da fare (vedi Cap. 3,
  `istruzioni seconda parte.md` §4) e va scritto quando quel lavoro sarà
  concluso, **non prima**, per non promettere in Background un contributo che
  poi cambia in corso d'opera.
- Meta-learning applicato ai pesi di una GNN (hypernetwork: una rete genera i
  pesi di un'altra rete, condizionandoli su feature locali) — concetto
  generale qui, dettaglio implementativo (MetaDense, MetaGATLayer) nel Cap. 3.

**2.3 Traffic Signal Control come problema**
- Tassonomia: controllo a tempi fissi, controllo reattivo (MaxPressure,
  Varaiya 2013), controllo appreso (RL a singolo agente, RL multi-agente con
  comunicazione/coordinamento tramite GNN).
- Perché il TSC multi-intersezione è naturalmente un problema multi-agente
  cooperativo con osservabilità parziale locale.
- Simulatori usati in letteratura (SUMO, CityFlow) e perché questo progetto
  usa CityFlow (Zhang et al., 2019) — velocità di simulazione, formato dati.
- Fonte: `descrizione_environment.md` §1.

### Capitolo 3 — TSC e Modelli `[PARZIALE]` (3.1-3.5 scritte, 3.6 bloccata)

Bozza in `tesi/capitoli/cap3_tsc_e_modelli.tex` (13/9/2026). Le formule di
Meta-GAT/Meta-LSTM che §3.4 segnalava come "da verificare riga per riga" sono
state effettivamente verificate leggendo `meta_gat.py`, `meta_lstm.py`,
`meta_knowledge_learner.py`, `metastgat.py` e `state_encoder.py` prima di
scriverle: il flag di cautela di questa sezione è quindi risolto per la
parte architetturale (resta valido solo per il lavoro GCN/SONAR non ancora
iniziato, §3.6).

**3.1 Formalizzazione del problema** `[SCRITTO]`
- MDP applicato: stato per intersezione (dim 32 avanzato / 20 paper), azione
  (indice di fase 0-7), reward (due modalità, formule in §4.5), struttura a
  grafo delle intersezioni.
- Fonte: `descrizione_environment.md` §6, `metastgat_diff_analysis.md` §1.

**3.2 Ambiente di simulazione** `[SCRITTO]`
- CityFlow: motore, formato roadnet/flow, fasi semaforiche (8 fasi + gestione
  giallo/tutto-rosso), vincolo anti-starvation sull'azione.
- Dataset: reti calibrate (100m/200m), le tre densità (train/flat/peak), le 3
  varianti seed per il training, l'arteria Est-Ovest per l'esperimento di
  generalizzazione.
- Fonte: `descrizione_environment.md` (tutto), `descrizione_configurazioni.md`
  (tutto), `descrizione_src.md` per `cityflow_env.py`.

**3.3 Baseline classiche** `[SCRITTO]`
- Fixed-Time: rotazione a tempo fisso, nessun input dallo stato.
- MaxPressure (Varaiya, 2013): formula di pressione classica, `argmax` sulla
  fase a pressione massima, nessun vincolo di equità (rilevante per il Cap. 4).
- Fonte: `descrizione_src.md`, `descrizione_environment.md` nota su
  `_build_phase_lanelinks()`.

**3.4 Architettura MetaSTGAT** `[SCRITTO, verificato riga per riga il 13/9/2026]`
- State encoder duale, meta-knowledge learner (spaziale e temporale),
  Meta-GAT layer (pesi generati da hypernetwork condizionato su
  meta-feature), Meta-LSTM, testa Q-value.
- **Attenzione**: le formule esatte di MetaGAT/MetaLSTM (dimensioni dei
  tensori di pesi generati, come il meta-feature entra nell'hypernetwork)
  vanno riverificate direttamente in `src/models/metastgat.py` e
  `src/models/meta_knowledge_learner.py` al momento della stesura — non
  scriverle a memoria da questo file, che riporta solo la struttura, non
  ogni indice. Scheletro di riferimento in §4.4.
- Fonte: `descrizione_src.md` §`src/models/`, `metastgat_diff_analysis.md` §2.

**3.5 Procedura di training e differenze dal paper originale** `[SCRITTO]`
- Sequenza PER episodica, BPTT con burn-in (`seq_len=4, burn_in=2`), Double
  DQN, Huber loss + IS weights, soft update, gradient clipping.
- Ogni differenza dal paper è motivata (non solo elencata): vedi
  `metastgat_diff_analysis.md` §1-3 per il confronto puntuale codice/paper,
  già scritto in quella forma, da trasporre in prosa di tesi.
- Sistema di ablation a preset (`pro`/`paper`/`environment`/`temporal`/
  `rl_core`/`replay_stability`) e razionale scientifica del raggruppamento:
  fonte `proposte_ablation.md` (già argomentata, sezione "Raccomandazione per
  la tesi" alla fine).

**3.4bis — Inquadramento narrativo di 3.4-3.5: dal paper al modello Pro** `[SCRITTO, integrato come §3.4.6 "Dal paper al modello Pro" nel capitolo]`

Il capitolo deve raccontare il modello Pro come un **punto di arrivo**: si parte
dall'architettura e dalla procedura di training del paper originale (già
descritte in 3.4-3.5), poi si presentano le differenze del codice come una
sequenza di scelte motivate, non come un elenco neutro. Fonte completa di ogni
differenza: `metastgat_diff_analysis.md` (già organizzato in Sommario
Esecutivo + dettaglio puntuale + Cap. 4 "Riepilogo"). Da lì, tre categorie
narrative distinte — **non trattarle tutte come "miglioramenti"**, sarebbe
disonesto rispetto a quanto già scritto in quel documento stesso:

**(A) Miglioramenti con motivazione tecnica solida** (già argomentati, si
possono scrivere in tesi come miglioramenti veri e propri):
| Modifica | Motivazione | Fonte |
|---|---|---|
| `wait_vec` nello stato (dim 32 vs 20) | Segnale più informativo del solo conteggio veicoli: un'attesa di 90s è più urgente di una di 5s a parità di coda | §1.1 |
| Maschera azioni invalide (max 2 fasi consecutive) | Previene il collasso su una singola fase (verde fisso), comportamento degenerativo osservabile nelle prime fasi di training senza vincolo | §1.2 |
| Double DQN | Riduce il bias di sovrastima del valore Q rispetto al DQN standard (van Hasselt et al., 2016) | §3.3 |
| BPTT + burn-in (R2D2-style) | Corregge l'"hidden state staleness": senza burn-in lo stato LSTM usato in training (inizializzato a zero) non corrisponde a quello prodotto durante la raccolta dati | §3.2 |
| Huber loss + IS weights da PER | Più robusta agli outlier della MSE; il campionamento prioritario concentra il training sulle transizioni a errore TD più alto | §3.1, §3.4 |
| Soft update (Polyak, τ=0.01) | Target network più stabile di un hard-copy periodico | §3.5 |
| Gradient clipping (`max_norm=1.0`) | Stabilità numerica necessaria con un componente ricorrente (LSTM) nel grafo di computazione | §3.7 |
| Tanh finale nei meta-learner | Normalizza l'embedding generato in un range limitato [-1,1], invece di lasciarlo libero | §2.2 |
| Cutoff di visibilità (`VISION_CUTOFF_M ≈ 144m`) | Realismo fisico: rappresenta la distanza percorribile da un veicolo nella finestra temporale di un'azione (`(GREEN_TIME+YELLOW_TIME) · v_urbana`), non un numero arbitrario — derivazione completa in `descrizione_environment.md` §4 | `descrizione_environment.md` §4/§6 |
| Buffer PER dimensionato a 4.800 sequenze (vs 10.000 transizioni piatte del paper) | **Confermato dall'utente**: non è un ridimensionamento subìto, è calibrato deliberatamente sulla durata dell'episodio (~40 episodi di storia con episodi da 1800s) — la metrica giusta di confronto con il paper non è il numero grezzo di transizioni ma la copertura in episodi, e su quella base il dimensionamento è coerente, non ridotto | §3.1 |

**(B) Scelte interpretative per risolvere ambiguità del paper** (da presentare
come disambiguazione necessaria, non come deviazione arbitraria — il paper
stesso è sotto-specificato in questi punti):
| Punto ambiguo | Scelta fatta | Perché è ragionevole |
|---|---|---|
| Meta-GAT: `W·(Q·K)` (paper) vs `(W·Q)·K` (codice) | `(W·Q)·K`, con `W` matrice `(hs,hs)` per nodo/head | Più vicina alla pratica standard delle hypernetwork; il paper non specifica la dimensione di `W`, lasciando aperta l'interpretazione |
| Meta-LSTM: pesi per 1 gate (notazione paper) vs 4 gate (codice) | Pesi generati per tutti e 4 i gate (`forget, input, output, cell`) | Una LSTM funzionante richiede tutti e 4 i gate: la notazione del paper è una semplificazione, non un'architettura alternativa praticabile |
| Fattore di scala `/√(head_size)` vs `/√(D_h)` | `/√(head_size)` | Corretto per multi-head attention quando `D_h` è già diviso tra le head, coerente con Vaswani et al. (2017) |

**(C) Scelte da presentare con cautela, non come "miglioramenti" dimostrati**
— qui la tesi deve essere onesta sul fatto che manca ancora un confronto
sperimentale diretto (nessun preset `paper`/`environment`/`rl_core`/
`replay_stability` è stato allenato al momento di questo piano). **Decisioni
di framing confermate dall'utente il 13/9/2026**:

- **Reward riscritta** (`custom` multi-obiettivo vs `-P_i` puro del paper):
  **scelta progettuale, giudizio rimandato al Cap. 4**. Nel Cap. 3 va
  presentata con la sua motivazione teorica (il reward a pura pressione può
  essere instabile; la formula riscritta promuove throughput esplicito e
  penalizza attese prolungate, §1.3), ma **senza dichiararla già superiore**
  al paper: il verdetto empirico (travel time/throughput a parità di resto
  dell'architettura, `reward_mode=custom` vs `paper`) resta esplicitamente
  aperto e va riportato nel Cap. 4 quando l'ablation `paper` sarà disponibile.
  Frase-tipo da evitare nel Cap. 3: "il reward riscritto migliora il
  training". Frase-tipo corretta: "il reward riscritto introduce due termini
  assenti nel paper originale [...]; l'effetto netto sulle metriche
  oggettive è discusso nel Cap. 4 alla luce dei risultati dell'ablation
  `reward_mode`".
- **Episodi (50-100 vs 200×3 processi paralleli del paper) ed epsilon decay
  aggressivo (~20 episodi)**: **non solo un limite subìto**. Va dichiarato
  onestamente il vincolo di risorse (un solo processo, mai 3 paralleli come
  nel paper — questo resta un limite da menzionare anche nelle limitazioni
  del Cap. 5), ma il numero di episodi usato è risultato **sufficiente per
  la convergenza** nei nostri esperimenti: il meccanismo di selezione del
  miglior modello (`run_final_selection`, best travel time su una config di
  validazione) mostra che il training raggiunge il suo miglior risultato
  ben prima di esaurire il budget di episodi, non che si interrompe
  prematuramente. Nel Cap. 3/Cap. 4, quando si scrive questa parte, riportare
  il numero di episodio in cui `best_travel_time` è stato raggiunto (da
  `training_log.csv`/`training_state.json` del modello specifico) come
  evidenza a supporto, invece di limitarsi ad affermarlo.
- **Feature "location" assente nel TMK**: **omissione dichiarata come
  limitazione**, senza cercare una motivazione a posteriori. Va nel Cap. 3
  (differenze dal paper) e ripresa nelle limitazioni del Cap. 5, in linea con
  lo stesso standard di onestà già applicato altrove nel progetto (vedi §1
  di questo file).

**3.6 Estensioni spaziali: MetaSTGNN e MetaSTSONAR** `[BLOCCATO]`
- GAT → GCN (MetaSTGNN, 1 e 2 layer) e GAT → SONAR (MetaSTSONAR, profondità
  parametrizzata da `L` ricorrenze) a parità del resto dell'architettura.
- Dipende dal lavoro descritto in `istruzioni seconda parte.md` (design,
  decisioni ancora da confermare in quel documento §5, implementazione,
  training). Non scrivere questa sezione finché quel lavoro non è concluso:
  il rischio è dover riscrivere formule e figure due volte.

### Capitolo 4 — Confronto `[PARZIALE]` (4.1-4.5 scritte, 4.6 bloccata)

Bozza in `tesi/capitoli/cap4_confronto.tex` (13/9/2026), 4.1-4.4 con dati
reali (estratti direttamente dai `test_summary_*.json`, non dai grafici),
4.5 con un solo confronto completo (`temporal` vs `pro`) più preciso di come
inizialmente pianificato qui: due tabelle separate (train/test), non un'unica
media a 10 config, per non nascondere il divario in-distribuzione vs
generalizzazione. Le due figure incluse (bar chart) sono le versioni a 4
modelli generate prima dello sweep completo su alpha, con didascalia che lo
dichiara esplicitamente: da rigenerare a valle di questo capitolo con
`compare_models.py --model-ids metastgat_pro_0.5 metastgat_pro_0.2
metastgat_pro_0.1 metastgat_pro_0.0 maxpressure ablation_temporal` (più gli
altri preset di ablation quando pronti).

**4.1 Metriche di valutazione** `[SCRITTO]`
- Fonte: `descrizione_metriche.md`.

**4.2 Protocollo sperimentale** `[SCRITTO]`
- Include una tabella dei 6 preset di ablation (riusata anche in §4.5) e un
  listato del comando di training per riproducibilità, e un paragrafo
  esplicito sul determinismo bit-per-bit di CityFlow e sulle sue conseguenze
  per l'interpretazione statistica dei risultati (nessun error bar possibile
  senza generare varianti a seed multiplo aggiuntive).
- Fonte: `descrizione_scripts.md` (`train.py`, `test.py`, `main.py`).

**4.3 Risultati: baseline classiche e sweep su alpha** `[SCRITTO]`
- Tabelle train/test per MaxPressure + Pro ($\alpha\in\{0.5,0.2,0.0\}$):
  $\alpha=0.2$ risulta il miglior compromesso TT medio/massimo tra le
  varianti Pro (persino $tt_{\max}$ più basso di MaxPressure), $\alpha=0.0$
  il peggiore su ogni colonna. `ablation_temporal` discusso a parte in §4.5
  (confronto a parità di $\alpha=0.5$, non in questa tabella).
- **Ancora da aggiungere quando pronti**: FixedTime, `metastgat_pro_0.1`
  (training in corso), i 4 preset di ablation restanti.

**4.4 Equità direzionale: il ruolo di alpha** `[SCRITTO]`
- Sostituisce l'idea originaria di una sezione "esperimento di
  generalizzazione" separata: dato che l'arteria Est-Ovest è ormai presente
  in **tutte** le 10 configurazioni (non uno scenario a parte, vedi
  `descrizione_configurazioni.md` §4bis), l'analisi direzionale sulle
  10 config già testate *è* l'esperimento di generalizzazione. Risultato
  chiave: $\alpha=0.0$ collassa l'equità quasi al livello di MaxPressure,
  confermando il ruolo causale del termine anti-starvation; siccome
  $\alpha=0.0$ è anche il peggiore su travel time (non solo su equità),
  i dati escludono un semplice compromesso equità-contro-throughput.
  Limite dichiarato esplicitamente nel testo: nessuna stima di significatività
  statistica classica, solo consistenza di direzione su più configurazioni
  (simulazione deterministica, vedi §4.2).

**4.5 Studio di ablation (risultati preliminari)** `[PARZIALE]`
- Un solo confronto completo per ora: `ablation_temporal` vs Pro
  ($\alpha=0.5$ fisso). BPTT+burn-in disattivato peggiora TT medio e massimo
  in entrambi i regimi, con un peggioramento relativo maggiore in
  generalizzazione che in training (+18.6\% vs +7.1\% su TT medio) — indizio,
  non prova, di un contributo di BPTT anche alla generalizzazione, non solo
  al fit in-distribuzione. Sezione esplicitamente segnalata come da
  completare quando `paper`/`environment`/`rl_core`/`replay_stability`
  saranno pronti (in training durante la notte tra il 13 e il 14/9/2026).

**4.6 Confronto GAT vs GCN vs SONAR** `[BLOCCATO]`
- Dipende interamente dal Cap. 3.6. Non scrivere prima.

### Capitolo 5 — Conclusioni `[BLOCCATO]`
- Per definizione va scritto per ultimo: riprende i risultati del Cap. 4
  (compresa la parte GCN/SONAR) e non può essere anticipato senza rischiare
  di dover riscrivere le conclusioni quando arrivano risultati nuovi.
- Struttura suggerita quando sarà il momento: risposta alla domanda di
  ricerca, limiti dichiarati (stessi di `descrizione_environment.md` §10, da
  richiamare non da nascondere), lavoro futuro.

**Stima totale**: 5-7 + 15-18 + 18-22 + 15-20 + 4-6 ≈ **57-73 pagine**, con
margine sopra le 60 richieste anche escludendo la parte GCN/SONAR (che la
allungherebbe ulteriormente in Cap. 3.6/4.6/5).

---

## 4. Notazione matematica unificata

Da usare identica in tutta la tesi. Ogni formula qui sotto è già verificata
contro il codice in sessioni precedenti (fonte indicata); per le uniche due
non ancora verificate riga per riga (§4.4) è segnalato esplicitamente.

### 4.1 MDP generico

- Tupla $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)$.
- Politica $\pi(a \mid s)$, ritorno $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$.
- Funzione valore-stato $V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s]$.
- Funzione valore-azione $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s, a_t=a]$.
- Equazione di Bellman ottimale:
  $$Q^*(s,a) = \mathbb{E}_{s'}\left[r + \gamma \max_{a'} Q^*(s', a') \,\middle|\, s,a\right]$$

### 4.2 DQN e le sue estensioni (fonte: `metastgat_diff_analysis.md` §3)

- **Target DQN standard**: $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$
- **Target Double DQN** (van Hasselt et al., 2016), usato nel codice:
  $$y = r + \gamma\, Q\!\left(s', \operatorname*{argmax}_{a'} Q(s', a'; \theta);\ \theta^-\right)$$
- **Huber loss** (Smooth L1, $\delta=1$), usata al posto della MSE del paper:
  $$
  L_\delta(y, \hat y) =
  \begin{cases}
  \tfrac{1}{2}(y-\hat y)^2 & \text{se } |y-\hat y| \le \delta \\
  \delta\left(|y-\hat y| - \tfrac{1}{2}\delta\right) & \text{altrimenti}
  \end{cases}
  $$
- **Loss pesata da PER** (con importance-sampling weights $w_i$):
  $$L(\theta) = \frac{1}{B}\sum_{i=1}^{B} w_i \cdot L_\delta\big(y_i,\, Q(s_i,a_i;\theta)\big)$$
- **Prioritized Experience Replay** (Schaul et al., 2016): priorità
  $p_i = |\delta_i| + \varepsilon$ (con $\delta_i$ errore TD), probabilità di
  campionamento $P(i) = p_i^\alpha / \sum_k p_k^\alpha$, peso IS
  $w_i = \left(\frac{1}{N \cdot P(i)}\right)^{\!\beta} \big/ \max_j w_j$
  ($\beta$ annealed verso 1 durante il training — verificare il valore
  esatto di annealing nel codice, `replay_buffer.py`, prima di riportarlo).
- **Soft update / Polyak averaging** ($\tau = 0.01$ nel codice):
  $$\theta^- \leftarrow \tau\,\theta + (1-\tau)\,\theta^-$$
- **BPTT con burn-in** (R2D2-style, Kapturowski et al., 2019): su una
  sequenza campionata di lunghezza $L=4$, i primi $burn\_in=2$ step
  aggiornano lo stato ricorrente $(h,c)$ senza calcolare gradiente, i
  successivi $L - burn\_in$ step partecipano al calcolo della loss.

### 4.3 GAT (Veličković et al., 2018)

Per un nodo $i$ con vicinato $\mathcal{N}(i)$, embedding di input $h_i$,
matrice di proiezione $W$ e vettore di attenzione $\vec a$:

$$e_{ij} = \text{LeakyReLU}\!\left(\vec a^\top [W h_i \,\|\, W h_j]\right), \qquad
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

$$h_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}\, W h_j\right)$$

Con $K$ teste di attenzione, concatenazione (layer intermedi) o media (layer
finale) degli output delle singole teste.

### 4.4 Meta-GAT e Meta-LSTM `[DA VERIFICARE riga per riga nel codice]`

Lo scheletro concettuale (da `metastgat_diff_analysis.md` §2, commenti nel
codice) è: i pesi $W$ e i bias $b$ di ogni layer non sono parametri globali
condivisi, ma **generati da un hypernetwork** (`MetaDense`/`MetaDense3`)
condizionato su un vettore di meta-feature $m_i$ specifico del nodo/intersezione:

$$W_i, b_i = \text{MetaDense}(m_i)$$

con $W_i$ di forma $(N, H, d_h, d_h)$ per il Meta-GAT ($H$ = numero di teste)
e, per il Meta-LSTM, pesi per tutti e 4 i gate contemporaneamente, forma
$(N, 4, 2d_h, d_h)$ e bias $(N, 4d_h)$. **Non trascrivere le equazioni
complete del forward (attenzione GAT con pesi per-nodo, le 4 equazioni di
gate LSTM standard con $W_i$ al posto di un $W$ condiviso) finché non sono
state rilette direttamente in `src/models/metastgat.py` e
`src/models/meta_knowledge_learner.py`**: questo file ne riporta la struttura
dimensionale, non la formula riga per riga, e un errore di indice qui
diventerebbe un errore di tesi.

### 4.5 Reward dell'ambiente (fonte: `metastgat_diff_analysis.md` §1.3,
`descrizione_environment.md` §6 — già verificate contro il codice)

- **Modalità `custom`** (default, usata dal modello Pro):
  $$r_i = \operatorname{clip}\!\left(\frac{\text{Passed}_i - \text{Incoming}_i - \alpha \cdot \text{MaxRedWait}_i - \text{Penalty}_i}{100},\ -20,\ 5\right)$$
  dove $\text{Penalty}_i = 50$ se $\text{Passed}_i = 0 \land \text{Incoming}_i > 0$
  (verde sprecato), altrimenti $0$; $\alpha$ è l'iperparametro anti-starvation
  (sweep condotto su $\alpha \in \{0.2, 0.5\}$ in questo progetto).
- **Modalità `paper`** (Wang et al., 2022, riproduzione fedele):
  $$r_i = -\frac{P_i}{100}, \qquad P_i = \text{Incoming}_i - \text{Outgoing}_i$$

### 4.6 Metriche di valutazione (fonte: `descrizione_metriche.md`, già
verificate contro il codice)

- **Travel time** (convenzione ufficiale, include veicoli non arrivati):
  $$
  tt_v = \begin{cases}
  t_{\text{arrivo}}(v) - t_{\text{spawn}}(v) & \text{se } v \text{ è arrivato} \\
  t_{\text{now}} - t_{\text{spawn}}(v) & \text{altrimenti (lower bound)}
  \end{cases}
  \qquad
  \overline{tt} = \frac{1}{|V|}\sum_{v \in V} tt_v
  $$
- **Equità direzionale**: per ogni intersezione $i$ e gruppo direzionale
  $g \in \{\text{N/S}, \text{W/E}\}$, massimo storico dell'attesa
  *consecutiva* (si azzera quando il veicolo si muove) su tutte le corsie di
  quel gruppo:
  $$\text{max\_wait}_g = \max_{i \in I} \left(\max_{v \in \text{lane}_g(i),\, t} w_{v,t}\right)$$
  calcolato **includendo** gli stalli ancora in corso al termine
  dell'episodio (censura a destra) come valore ufficiale — mai la variante
  `_resolved` (solo stalli conclusi) nei grafici/tabelle principali.

### 4.7 MaxPressure (Varaiya, 2013)

Per un'intersezione $i$ e una fase $\phi$, pressione:
$$P(i, \phi) = \sum_{\ell \,\in\, \text{in}(\phi)} n_\ell - \sum_{\ell \,\in\, \text{out}(\phi)} n_\ell$$
con $n_\ell$ = numero di veicoli sulla corsia $\ell$ (corsia intera, nessun
cutoff di visibilità, a differenza dello stato RL — vedi
`descrizione_environment.md`, nota su `_build_phase_lanelinks()`). Azione:
$\phi^*(i) = \operatorname*{argmax}_\phi P(i,\phi)$, nessun vincolo di equità
o anti-starvation.

---

## 5. Bibliografia nota (da formalizzare in BibTeX al momento della stesura)

| Riferimento | Uso nella tesi |
|---|---|
| Wang et al., *"Meta-learning based spatial-temporal graph attention network for traffic signal control"*, Knowledge-Based Systems 250 (2022) 109166 | Paper di riferimento, Cap. 3 (architettura e procedura di training originali) |
| Mnih et al., *"Human-level control through deep reinforcement learning"*, Nature 518 (2015) | DQN, Cap. 2.1 |
| van Hasselt, Guez, Silver, *"Deep Reinforcement Learning with Double Q-learning"*, AAAI 2016 | Double DQN, Cap. 2.1 / 3.5 |
| Schaul et al., *"Prioritized Experience Replay"*, ICLR 2016 | PER, Cap. 2.1 / 3.5 |
| Kapturowski et al., *"Recurrent Experience Replay in Distributed Reinforcement Learning"* (R2D2), ICLR 2019 | BPTT + burn-in, Cap. 3.5 |
| Veličković et al., *"Graph Attention Networks"*, ICLR 2018 | GAT, Cap. 2.2 / 3.4 |
| Kipf & Welling, *"Semi-Supervised Classification with Graph Convolutional Networks"*, ICLR 2017 | GCN, Cap. 2.2 (e Cap. 3.6 quando pronto) |
| Paper SONAR (NeurIPS 2025, PDF già nel repository) | SONAR, Cap. 2.2 (accenno) e Cap. 3.6 (quando pronto) — **estrarre titolo/autori esatti dal PDF al momento della citazione**, non tenuto a memoria qui |
| Varaiya, *"Max pressure control of a network of signalized intersections"*, Transportation Research Part C 36 (2013) | MaxPressure, Cap. 2.3 / 3.3 |
| Zhang et al., *"CityFlow: A Multi-Agent Reinforcement Learning Environment for Large Scale City Traffic Scenarios"*, WWW 2019 | Simulatore, Cap. 2.3 / 3.2 |

---

## 6. Convenzioni pratiche LaTeX

- Un file `.tex` per capitolo (`cap1_introduzione.tex`, ...), `main.tex` con
  `\input{}` in sequenza — non un unico file monolitico, per poter lavorare
  capitolo per capitolo come da questo piano.
- Pacchetti minimi: `amsmath`, `amssymb` (formule), `graphicx` (figure già
  generate in `results/plots/`), `booktabs` (tabelle, coerente con
  `table_travel_time.tex`/`table_throughput.tex` già esportate dal codice),
  `hyperref` + `cleveref` (riferimenti incrociati a equazioni/figure/tabelle),
  `algorithm`/`algorithmic` o `algorithm2e` se si include lo pseudocodice del
  training loop, `biblatex` con backend `biber` (o `natbib` se il template
  della sede richiede uno stile diverso — verificare requisiti formali
  dell'ateneo prima di iniziare `main.tex`, non assunti qui).
- Figure già pronte da includere (non rigenerare): bar chart per
  configurazione in `results/plots/compare_per_config/`, grafici di
  training/loss in `results/<model_id>/`. Verificare che siano aggiornate ai
  risultati finali (non a run intermedi) prima di includerle in figura
  definitiva. (I violin plot per-veicolo sono stati rimossi il 13/9/2026.)
- Tabelle: `table_travel_time.tex`/`table_throughput.tex` sono già in formato
  `booktabs`-compatibile, generate da `scripts/plot_results.py` — da
  aggiornare a fine sperimentazione, non da ricreare a mano.
- **Front matter completato il 13/9/2026** (`tesi/main.tex` + due nuovi file
  `tesi/capitoli/sommario.tex` e `tesi/capitoli/notazione.tex`): frontespizio
  con campi segnaposto (`\nomeUniversita`, `\nomeDipartimento`, `\nomeCorso`,
  `\nomeRelatore`, `\nomeCandidato`, `\annoAccademico` — definiti come
  `\newcommand` in testa a `main.tex`, da sostituire prima della consegna),
  pagina bianca subito dopo, numerazione romana per Sommario/Indice/Elenco
  figure/Elenco tabelle/Notazione, numerazione araba da Cap. 1. Aggiunto
  anche `\hypersetup{colorlinks=true, linkcolor=..., ...}` con un unico blu
  scuro sobrio: di default hyperref disegna una cornice colorata rettangolare
  attorno a ogni link interno (voci di indice, citazioni, riferimenti a
  equazioni/figure/tabelle), poco elegante in un documento lungo — va
  mantenuto per ogni capitolo futuro, non solo per quelli già scritti.
- Il Sommario scritto il 13/9/2026 copre solo Cap. 1-4: **va riscritto (non
  solo esteso) quando Cap. 3.6/4.6/5 saranno pronti**, non è un testo
  definitivo.
- **Passata di espansione/precisione del 13/9/2026** (Cap. 1-4, su richiesta
  esplicita "non dare nulla per scontato"): aggiunti ε-greedy in Cap. 2
  (mancava del tutto), tabella delle 8 fasi semaforiche, narrazione completa
  del tutto-rosso (perché è stato aggiunto, verifica su replay grezzo),
  sezione "Semplificazioni dichiarate", intera generazione parametrica del
  dataset (calibrazione onda verde, 5 fasce del flusso "giornata
  lavorativa", storia della ricalibrazione densità 5x5/6x6, arteria+3 seed)
  in Cap. 3; warm-up/n\_updates=100/esplorazione ciclica (incluso l'A/B test
  inconcludente su quando disattivarla) in §3.5; esempio concreto
  `flow_907_1` (censura dell'equità direzionale) e percentuali di fase in
  Cap. 4. **Bibliografia verificata sulle fonti primarie**, non più
  segnaposto: autori MetaSTGAT e SONAR letti direttamente dalla prima
  pagina dei PDF allegati al repository (`MetaSTGAT.pdf`,
  `NeurIPS-2025-sonar-...pdf`); aggiunta `suttonbarto2018` (citazione
  standard per MDP/Bellman, prima assente); risultato preciso di
  Max-Pressure (Teorema 2 di Varaiya 2013: stabilizzante ogni volta che
  esiste una politica stabilizzante) e cifra di velocità di CityFlow
  (~20-25× SUMO) verificati via ricerca web, non parafrasati a memoria.
  Corretto anche un errore di riferimento trovato durante la rilettura
  (Cap. 4 citava un'"Equazione 4.6 del Capitolo 3" per il travel time che in
  realtà non esisteva in Cap. 3: la formula ora è definita direttamente in
  Cap. 4 dove serve).

---

## 7. Cosa manca prima di poter chiudere alcune sezioni

- **Cap. 3.6 / 4.6 / Cap. 5 (parziale)**: sostituzione GAT → GCN → SONAR,
  design ancora da confermare (`istruzioni seconda parte.md` §5), codice non
  scritto, nessun training effettuato.
- **Cap. 4.3/4.5**: risultati completi di tutti i preset di ablation (solo
  `pro` e `temporal` avviati al momento della stesura di questo piano;
  mancano `environment`, `rl_core`, `replay_stability`).
- **Cap. 4.4**: verificare se l'esperimento di generalizzazione (arteria +
  multi-seed) è già stato eseguito per MetaSTGAT Pro con lo stesso dettaglio
  con cui è stato fatto per MaxPressure/FixedTime, o va completato.

Tutto il resto (Cap. 1, Cap. 2, Cap. 3.1-3.5, Cap. 4.1-4.2) ha già il
materiale necessario nei documenti `descrizione_*.md`,
`metastgat_diff_analysis.md`, `proposte_ablation.md` e nei risultati in
`results/`: può essere scritto in LaTeX ora, capitolo per capitolo, seguendo
questo piano.
