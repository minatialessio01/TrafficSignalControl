# Descrizione dei Modelli — il "catalogo" del progetto

> Fonde e sostituisce, come riferimento principale, [`proposte_ablation.md`](proposte_ablation.md)
> e [`metastgat_diff_analysis.md`](metastgat_diff_analysis.md) (12/9/2026): i due documenti
> restano nel repository con tutto il loro dettaglio (rispettivamente: le 4 proposte di
> raggruppamento alternative per l'ablation study, e l'analisi riga-per-riga codice-vs-paper con
> gli estratti di codice), ma per capire **quali modelli sono stati effettivamente testati e in
> cosa differiscono l'uno dall'altro** questo è il documento da leggere per primo.
>
> L'idea è quella di un catalogo/concessionaria: ogni modello è un'"auto in esposizione" con la
> propria scheda tecnica, e la domanda a cui questo file risponde per ciascuno è sempre la stessa
> — *rispetto a cosa cambia, e di quanto?* I modelli di questo progetto non variano lungo un solo
> asse, ma lungo **tre assi indipendenti** (più un paio di esperimenti fuori asse, in fondo):
>
> 1. **Rispetto al paper** (Wang et al., 2022) — quante e quali delle modifiche proprie di questo
>    progetto sono attive: il modello "Pro" le ha tutte, il modello "Paper" nessuna, i 4 preset di
>    ablation ne disattivano un sottoinsieme mirato.
> 2. **Rispetto ad alpha** — a parità di tutto il resto (preset `pro`), quanto pesa la penalità
>    anti-starvation nella reward: 4 valori testati, 0.5/0.2/0.1/0.0.
> 3. **Rispetto all'ablation** — quale sottosistema funzionale del modello Pro viene disattivato
>    per isolarne il contributo (Proposta 1 di `proposte_ablation.md`, l'unica implementata).
>
> Ogni modello ha una cartella `results/<model_id>/` con le proprie metriche (vedi
> `results/REORGANIZATION.md` per la struttura interna) e compare in almeno uno dei grafici
> generati da `scripts/compare_models.py` (`results/plots/compare_*/`).

---

## 0. Il piazzale: tutti i modelli testati

> ⚠️ **Self-loop nel Meta-GAT, decisione del 12/9/2026 (dettaglio in §4)**:
> `src/models/metastgat.py` aggiunge ora sempre self-loop a `edge_index`, per
> ogni preset — non più un esperimento a parte ma l'architettura standard.
> Le prime 9 righe di questa tabella (tutto tranne le baseline classiche e i
> tre `*_self_loop`) sono state allenate **prima** di questa modifica e non
> sono più ad armi pari con un modello allenato dopo: vanno ri-allenate.

| `model_id` | Asse | Cosa lo distingue in una frase |
|---|---|---|
| `metastgat_pro_0.5` | baseline Pro / alpha | Modello avanzato completo (tutte le M1-M10 attive), $\alpha=0.5$ — il "modello di serie" da cui partono tutti i confronti. **Da ri-allenare** (self-loop) |
| `metastgat_pro_0.2` | alpha | Come sopra, $\alpha=0.2$ — miglior compromesso TT/equità tra le varianti alpha (vedi §2). **Da ri-allenare** |
| `metastgat_pro_0.1` | alpha | Come sopra, $\alpha=0.1$. **Da ri-allenare** |
| `metastgat_pro_0.0` | alpha | Come sopra, $\alpha=0.0$ — nessuna penalità anti-starvation. **Da ri-allenare** |
| `metastgat_paper` | paper | Tutte le M1-M10 disattivate: riproduzione fedele di Wang et al. 2022 con lo stesso codice del modello Pro. **Da ri-allenare** |
| `ablation_environment` | ablation | Disattiva M1-M4 (reward custom, cutoff visivo, `wait_vec`, maschera anti-starvation): torna all'MDP del paper, RL core avanzato. **Da ri-allenare** |
| `ablation_temporal` | ablation | Disattiva M5 (BPTT $L=4$ + burn-in): Meta-LSTM allenata single-step. **Da ri-allenare** |
| `ablation_rl_core` | ablation | Disattiva M6 (Double DQN): torna a DQN standard. **Da ri-allenare** |
| `ablation_replay_stability` | ablation | Disattiva M7-M10 (PER/IS, Huber, soft-update, grad clip, warm-up, esplorazione ciclica, Tanh meta). **Da ri-allenare** |
| `fixedtime` | baseline classica | Nessun apprendimento: fasi a durata fissa cicliche, non richiede checkpoint — non toccato dalla modifica |
| `maxpressure` | baseline classica | Euristico max-pressure (Varaiya, 2013), non richiede checkpoint — non toccato dalla modifica |
| `metastgat_pro_0.2_self_loop` | esperimento fondativo | Come `metastgat_pro_0.2` con self-loop — l'esperimento che ha stabilito il nuovo standard (vedi §4) |
| `metastgat_pro_0.0_self_loop` | esperimento fondativo | Come sopra su `metastgat_pro_0.0` — stesso esito, miglioramento ancora più marcato |
| `ablation_environment_self_loop` | esperimento fondativo | Come `ablation_environment`, con self-loop — verifica se il miglioramento regge anche quando l'MDP torna a quello del paper (in addestramento) |

Tutti i modelli allenabili condividono la stessa procedura (`scripts/train.py`, Algorithm 1 del
paper con le aggiunte descritte in `descrizione_scripts.md`), le stesse 3 config di training
cicliche (`config_4x4_100m_train{1,2,3}.json`) e la stessa selezione finale automatica
(`final_model.pth` vs `best_model.pt` su una config di validazione) — quello che cambia da un
`model_id` all'altro è **sempre e solo** uno dei flag descritti nelle sezioni seguenti, mai la
pipeline attorno.

---

## 1. Rispetto al paper: le 10 modifiche (M1-M10)

Il modello Pro (`--ablation pro`, il default) e il modello Paper (`--ablation paper`) usano
**esattamente lo stesso codice**: la differenza è interamente nei flag `--no-*` che il preset
`paper` attiva. Questo è deliberato — non è una riscrittura separata "modalità paper", è lo stesso
modello con 10 interruttori spenti, quindi il confronto isola davvero l'effetto delle modifiche e
non anche eventuali differenze di implementazione accidentali.

| ID | Modifica | Paper originale (Wang et al., 2022) | Codice di questo progetto | Perché |
|:--:|:---------|:-------------------------------------|:---------------------------|:-------|
| **M1** | Reward multi-obiettivo pesata | Pura pressione: $r_i=-P_i$ | `Passed − Incoming − α·MaxRedWait − Penalty`, normalizzata /100, clip $[-20,5]$ | Incentiva il deflusso reale, penalizza sia lo stallo di corsie secondarie sia il verde concesso a vuoto |
| **M2** | Campo visivo limitato | Visibilità teorica sull'intera strada | Solo veicoli entro `VISION_CUTOFF_M` (~144m) dal semaforo | Simula la portata reale di sensori/telecamere |
| **M3** | Stato a 3 componenti | Dim 20: `[n_vec(12) \|\| p_vec(8)]` | Dim 32: `[n_vec(12) \|\| wait_vec(12) \|\| p_vec(8)]` | Aggiunge l'urgenza temporale (attesa massima per corsia), non solo la presenza |
| **M4** | Maschera anti-starvation | Nessun vincolo sulle azioni consecutive | Una fase è invalida se scelta ≥2 volte di fila | Impedisce il collasso su "verde fisso permanente" |
| **M5** | BPTT + burn-in (R2D2-style) | DQN single-step, stato LSTM isolato tra transizioni | Sequenze $L=4$, burn-in=2 senza gradiente + 2 step di BPTT | Corregge l'*hidden state staleness* della Meta-LSTM |
| **M6** | Double DQN | $y = r+\gamma\max_{a'}Q(s',a';\theta^-)$ | $y = r+\gamma Q(s',\arg\max_{a'}Q(s',a';\theta);\theta^-)$ | Elimina la sovrastima sistematica dei Q-value |
| **M7** | PER con IS weights | Buffer FIFO piatto, 10.000 transizioni, campionamento uniforme | PER sequenziale (priorità TD), pesi Importance Sampling con $\beta$-annealing | Campiona più spesso le transizioni informative, corregge il bias con IS |
| **M8** | Huber loss + grad clip + soft update | MSE, nessun clipping, update periodico hard | Smooth L1 pesata IS, `clip_grad_norm_=1.0`, Polyak $\tau=0.01$ | Stabilità numerica in presenza di picchi di traffico |
| **M9** | Warm-up + esplorazione ciclica | Epsilon decay passivo, nessun warm-up | 10 episodi random iniziali (buffer non loggato) + 1 episodio random ogni 10 | Evita update su buffer vuoto, esplora stati rari anche a training avanzato |
| **M10** | Tanh sui meta-learner | MLP lineare, nessuna attivazione finale | `nn.Tanh()` finale su SMK/TMK | Confina gli embedding generati in $[-1,1]$, previene la divergenza delle hypernetwork |

**Modello Pro = tutte e 10 attive. Modello Paper = tutte e 10 disattivate.** I 4 preset di
ablation (§3) ne disattivano solo un sottoinsieme mirato, per isolare l'effetto di una singola
area invece che dell'intero pacchetto.

Oltre alle 10 modifiche comportamentali, esistono differenze **interpretative** dove la notazione
del paper è ambigua e il codice ha dovuto scegliere (nessuna delle due letture è "sbagliata",
sono scelte implementative legittime): la formula del Meta-GAT è implementata come $(W{\cdot}Q){\cdot}K$
anziché $W{\cdot}(Q{\cdot}K)$; la Meta-LSTM genera pesi per tutti e 4 i gate (il paper scrive la
dimensione di un solo gate); la feature "location" del TMK del paper non è implementata (nessun
canale porta coordinate assolute, si veda `descrizione_configurazioni.md` §4bis sul perché questo
è stato deliberatamente messo alla prova con l'esperimento delle 3 arterie su righe diverse).
Dettaglio riga per riga con estratti di codice: [`metastgat_diff_analysis.md`](metastgat_diff_analysis.md).

**Numeri**: su `config_4x4_100m_train1` (in-distribuzione) il modello Paper ha TT medio 334.9s
contro 215.2s del Pro ($\alpha=0.5$); sulla generalizzazione più dura (6x6 peak) 538.2s contro
326.8s — un divario che si allarga, non si restringe, fuori distribuzione. Tabella completa e
per-config: `results/plots/table_travel_time.csv` e `results/plots/compare_ablation_study/`.

---

## 2. Rispetto ad alpha: il peso della penalità anti-starvation

$\alpha$ è l'unico iperparametro che cambia tra `metastgat_pro_0.5`, `_0.2`, `_0.1`, `_0.0`: tutto
il resto (preset `pro`, quindi tutte le M1-M10 attive) resta identico. Compare nella reward
custom (M1):

$$r_i = \operatorname{clip}\!\left(\frac{\text{Passed}_i - \text{Incoming}_i - \alpha \cdot \text{MaxRedWait}_i - \text{Penalty}_i}{100},\ -20,\ 5\right)$$

$\alpha$ pesa quanto la reward penalizza un'attesa massima prolungata su una corsia col rosso:
$\alpha=0$ rimuove del tutto questo termine (la reward diventa "solo throughput", con lo stesso
rischio di starvation di corsie secondarie che l'M1 originale voleva risolvere); $\alpha$ alto
penalizza più severamente le attese lunghe, a costo di un possibile compromesso sul throughput
puro.

| `model_id` | $\alpha$ | TT medio train1 (s) | TT medio 6x6 peak (s) |
|---|---|---:|---:|
| `metastgat_pro_0.5` | 0.5 | 215.2 | 326.8 |
| `metastgat_pro_0.2` | 0.2 | 199.6 | 320.7 |
| `metastgat_pro_0.1` | 0.1 | 194.7 | 338.6 |
| `metastgat_pro_0.0` | 0.0 | 213.6 | 348.9 |

Nessuna delle 4 varianti domina su ogni config (vedi `results/plots/compare_per_config/` per il
quadro completo sulle 10 config): $\alpha=0.1$ ha il TT medio più basso in training ma non
generalizza altrettanto bene; **$\alpha=0.2$ risulta il miglior compromesso complessivo** tra TT
medio e massimo attraverso l'insieme delle config (piano_tesi.md §4.3), mentre **$\alpha=0.0$ è
sistematicamente il peggiore**, non solo sul travel time ma anche sull'equità direzionale: la sua
attesa massima N/S e W/E collassa quasi al livello di MaxPressure, confermando che il termine
anti-starvation non è ridondante rispetto al solo throughput (piano_tesi.md §4.4). Nessuna delle
due metriche isolate basta da sola a scegliere un $\alpha$: da qui la scelta di riportare sempre
TT e equità direzionale fianco a fianco nei grafici di `compare_models.py`, mai uno senza l'altro.

---

## 3. Rispetto all'ablation: i 4 sottosistemi funzionali

Dei 10 interruttori M1-M10, la Proposta 1 di `proposte_ablation.md` (l'unica implementata) li
raggruppa in 4 aree funzionali coerenti, ciascuna disattivata da un preset `--ablation` dedicato.
A differenza del sweep su alpha (un solo numero), qui ogni preset spegne un **gruppo** di flag
insieme, perché sono componenti che cooperano strettamente e non avrebbe senso isolarle una a una
in questo raggruppamento (per l'isolamento fattore-per-fattore puro, vedi la Proposta 2
alternativa, non implementata, in `proposte_ablation.md` §3).

| Preset | `reward_mode` | Flag `--no-*` attivati (disattiva quella modifica) | Cosa isola | TT medio train1 / 6x6 peak (s) |
|---|---|---|---|---:|
| `pro` (= `metastgat_pro_0.5`) | custom | nessuno | Benchmark di riferimento | 215.2 / 326.8 |
| `environment` | paper | `vision-cutoff`, `wait-vec`, `action-mask` | Quanto incide l'ingegnerizzazione dell'MDP (M1-M4, tranne M1 che qui resta su `paper`\*) | 197.0 / 311.3 |
| `temporal` | custom | `bptt` | Quanto incide BPTT+burn-in sulla Meta-LSTM (M5) | 229.8 / 390.6 |
| `rl_core` | custom | `double-dqn` | Quanto incide Double DQN sulla sovrastima dei Q-value (M6) | 214.7 / 335.7 |
| `replay_stability` | custom | `per`, `huber`, `soft-update`, `grad-clip`, `warmup`, `cyclic-exploration`, `tanh-meta` | Quanto incidono PER/IS, ottimizzazione robusta, warm-up ed esplorazione ciclica (M7-M10) | 295.3 / 432.6 |
| `paper` (= `metastgat_paper`) | paper | tutti | Riproduzione fedele Wang et al. 2022 (M1-M10 tutte disattivate) | 334.9 / 538.2 |

\* Nota su `environment`: usa `reward_mode=paper` (quindi disattiva anche M1, la reward pesata,
non solo M2-M4) — l'unico preset in cui `reward_mode` non è indipendente dagli altri flag del
gruppo, perché la reward custom dipende da `wait_vec`/dal cutoff visivo per essere ben definita
nello stesso modo. Vedi `descrizione_scripts.md` per la mappa preset→flag completa e il dettaglio
di ogni flag atomico.

**Lettura dei risultati** (unico confronto completo scritto in tesi finora, piano_tesi.md §4.5):
`ablation_temporal` vs Pro peggiora sia in training (+7.1% TT medio) sia, in misura maggiore, in
generalizzazione (+18.6%) — indizio che BPTT+burn-in non è solo un dettaglio di ottimizzazione ma
contribuisce alla capacità di generalizzare, non solo a convergere più in fretta sulle config viste
in training. `replay_stability` è il preset col degrado più marcato in assoluto (295.3s contro i
215.2s del Pro, +37%): il gruppo che disattiva è il più numeroso (7 flag), quindi l'effetto
osservato è cumulativo, non attribuibile a una singola causa — coerente con la scelta dichiarata
di questo raggruppamento (vedi "Svantaggi" della Proposta 1 in `proposte_ablation.md`).

---

## 4. Fuori asse: esperimenti non ancora (o mai) integrati

Due famiglie di modelli non rientrano nei tre assi sopra, perché nascono da domande diverse:

**Self-loop nel Meta-GAT — ora parte permanente dell'architettura** (esperimenti:
`metastgat_pro_0.2_self_loop`, `metastgat_pro_0.0_self_loop`, `ablation_environment_self_loop`):
il GAT di questo progetto (`MetaGATLayer`) non aggiungeva mai self-loop all'`edge_index`, una
deviazione dalla formulazione GAT standard individuata durante l'audit di codice del 12/9/2026
(vedi `descrizione_gcn_sonar.md` §0) — troppo invasiva da correggere silenziosamente sul modello
Pro già validato (richiede ri-allenare tutto), quindi prima trasformata in un esperimento
controllato: stesso `metastgat_pro_0.2`/`_0.0`, con self-loop aggiunti solo per la durata di
quell'addestramento specifico (patch temporanea, tramite `add_self_loops()` in
`src/models/meta_gcn.py`, la stessa già riusata da `MetaSTGNN`).

> **Correzione (12/9/2026)**: la prima valutazione di `metastgat_pro_0.2_self_loop` aveva
> riportato una "regressione catastrofica in generalizzazione" (attesa quasi al tetto dei 1800s
> su 6+ config di test). Era un artefatto di misura, non un risultato reale: lo script di test
> lancia un processo Python **separato per ognuna delle 10 config**, e il codice con i self-loop
> era stato ripristinato alla versione pulita subito dopo la conferma sulla **prima** config
> invece che dopo la fine di tutte e 10 — le config successive (tutte le 7 di generalizzazione)
> hanno quindi valutato pesi allenati **con** self-loop usando un'architettura **senza**, un
> disallineamento silenzioso (le shape dei parametri sono identiche, self-loop non aggiunge pesi,
> quindi non c'è nessun errore di caricamento che lo segnali). Rifatta la valutazione aspettando
> la fine dell'intera suite prima di ripristinare il codice, il quadro si ribalta: **miglioramento
> consistente su tutte e 10 le config, non solo su quelle di training**.

| $\alpha$ | TT medio, media 10 config (base → self-loop) | TT medio, media sulle 7 di generalizzazione |
|---|---|---|
| 0.2 | 254.7s → 230.5s (**-9.5%**) | 280.8s → 254.0s (**-9.5%**) |
| 0.0 | 281.6s → 229.8s (**-18.4%**) | 311.8s → 254.9s (**-18.2%**) |

Il miglioramento è più marcato proprio dove il modello Pro è più debole ($\alpha=0.0$, la
variante peggiore del sweep in §2) e si mantiene identico in proporzione tra config di training e
di generalizzazione — il segno di un effetto reale del meccanismo di attenzione (i self-loop
permettono a un nodo di pesare anche il proprio stato invece di dipendere solo dai vicini), non di
un adattamento alla topologia 4×4 vista in training.

**Decisione (12/9/2026): applicata in via permanente.** `src/models/metastgat.py` aggiunge ora
sempre i self-loop, per ogni preset di ablation, senza flag per disattivarli — non è più un
esperimento a parte ma il comportamento standard di `MetaSTGAT`. Conseguenza diretta: **ogni
modello della tabella in §0 allenato prima di questa modifica non è più ad armi pari** con uno
allenato dopo (stessa natura del bug di misura appena descritto, ma permanente invece che
transitoria se non si interviene) — `metastgat_pro_0.5/0.2/0.1/0.0`, `metastgat_paper` e i 4
preset di ablation vanno ri-allenati per restare comparabili. Non ancora fatto al momento della
stesura di questa nota. La stessa correzione è stata estesa per coerenza a **ogni** layer GAT del
progetto, non solo a quello di MetaSTGAT: anche `StandardGATLayer` (`src/models/stgat.py`, usato
dal baseline `STGAT`) aveva la stessa lacuna strutturale — corretta, ma senza alcun costo di
ri-addestramento perché nessun modello STGAT è mai stato allenato in questo progetto. SONAR non
riceve la stessa modifica per un motivo strutturale, non di scelta: nella sua azione Laplaciana un
self-loop contribuirebbe sempre con un termine nullo (differenza di un nodo con se stesso), quindi
sarebbe un'aggiunta inerte — vedi `descrizione_gcn_sonar.md` §0.

**GCN e SONAR al posto del GAT** (`metastgnn_1l`/`metastgnn_2l`, `metastsonar_l2`/`_l4`, non
ancora in `results/`): sostituiscono il meccanismo spaziale (Meta-GAT) con una GCN o con SONAR
(propagazione ondulatoria, Trenta et al., NeurIPS 2025) a parità del resto dell'architettura,
per confrontare tre meccanismi di aggregazione spaziale invece di uno. Implementati e verificati
(shape/gradienti/non-regressione), ma non ancora sottoposti a un training completo al momento
della stesura di questo file — dettaglio in `descrizione_gcn_sonar.md`.

---

## 5. Per approfondire

- Dettaglio riga-per-riga codice-vs-paper (con estratti di codice): [`metastgat_diff_analysis.md`](metastgat_diff_analysis.md)
- Le 4 proposte alternative di raggruppamento per l'ablation study (solo la Proposta 1 è implementata): [`proposte_ablation.md`](proposte_ablation.md)
- Flag `--ablation`/`--no-*` e come li applica `train.py`: [`descrizione_scripts.md`](descrizione_scripts.md)
- Definizione precisa di ogni metrica citata qui (TT medio, TT massimo, equità direzionale): [`descrizione_metriche.md`](descrizione_metriche.md)
- Configurazioni di training/test/validazione (arterie, seed, densità): [`descrizione_configurazioni.md`](descrizione_configurazioni.md)
- Struttura di `results/<model_id>/` e convenzione dei grafici di confronto: [`results/REORGANIZATION.md`](results/REORGANIZATION.md)
