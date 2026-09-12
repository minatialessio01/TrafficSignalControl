# Descrizione: MetaSTGNN e MetaSTSONAR (Seconda Parte)

Documento di riferimento per la seconda parte del progetto: sostituzione del
meccanismo spaziale di MetaSTGAT (GAT) con una GCN (`MetaSTGNN`) e con SONAR
(`MetaSTSONAR`), come pianificato in [istruzioni seconda parte.md](istruzioni%20seconda%20parte.md).
Quel file resta il riferimento per il *perche'* di ogni scelta; qui si
documenta *cosa e' stato effettivamente implementato*, le decisioni prese
sui punti aperti (D1-D6), e cosa e' stato verificato prima di lasciarlo a
disposizione per il training vero.

> **Stato (13/9/2026)**: codice scritto, verificato con test di forma/
> parametri/gradienti e uno smoke test end-to-end (3 episodi, tutte e 5 le
> varianti). **Nessun training completo e' stato ancora lanciato** su questi
> modelli: la sezione "Prossimi passi" in fondo elenca cosa manca prima di
> avere risultati comparabili in tesi.

---

## 0. Prima di tutto: un bug trovato nel codice esistente

Prima di scrivere codice nuovo, come richiesto, e' stata fatta un'analisi
del codice esistente per cercare bug. Ne sono emersi diversi (elenco completo
nella cronologia della sessione); i piu' rilevanti, e gli unici corretti
prima di procedere perche' toccano codice che la seconda parte riusa
direttamente, sono:

- **`--no-per` non rendeva uniforme il campionamento del replay buffer**
  (`src/agents/replay_buffer.py`): `sample_sequences()` campionava sempre
  per priorita' TD indipendentemente dal flag; `--no-per` si limitava a
  spegnere la correzione IS lato loss, non il campionamento a monte. I
  preset `paper` e `replay_stability` non riproducevano quindi davvero un
  buffer piatto/uniforme come documentato. Corretto: il buffer ora accetta
  `use_per` e campiona uniformemente quando disattivato.
- **`use_vision_cutoff=False` non raggiungeva `_get_outgoing_vehicles_count()`
  ne' `get_spatial_meta_features()`** (`src/environment/cityflow_env.py`):
  entrambi applicavano sempre il cutoff di visibilita', anche nei preset
  `paper`/`environment` che lo disattivano esplicitamente per il resto
  dell'ambiente. Corretto in entrambi i punti.
- **`DQNAgent.is_meta` riconosceva solo `MetaSTGAT` per tipo**
  (`src/agents/dqn_agent.py`, `isinstance(model, MetaSTGAT)`): un problema
  diretto per questa seconda parte, dato che `MetaSTGNN`/`MetaSTSONAR`
  richiedono `spatial_meta`/`temporal_meta` esattamente come `MetaSTGAT` ma
  non ereditano da quella classe. Senza la correzione, i due modelli nuovi
  non avrebbero mai ricevuto le loro feature meta. Corretto ampliando il
  controllo a una tupla `_META_MODEL_CLASSES = (MetaSTGAT, MetaSTGNN, MetaSTSONAR)`.
- Altri quattro bug minori (persistenza di `beta`/`total_episodes` nei
  checkpoint, un edge case di eviction nel replay buffer, un edge case di
  slicing con `seq_len` disallineato dopo un `--resume` tra preset diversi,
  la media di percentili invece del percentile sul pool in `test.py` con
  `--n-eval > 1`) sono stati corretti per correttezza generale ma non
  toccano risultati gia' generati (il primo perche' nessun modello
  `paper`/`replay_stability` era ancora stato allenato fino in fondo al
  momento della correzione, gli altri perche' non attivati dall'uso reale
  fatto finora della pipeline).

**Una scoperta inizialmente non corretta, poi risolta in via definitiva
(12/9/2026)**: `MetaGATLayer` (`src/models/meta_gat.py`) non aggiungeva mai
self-loop a `edge_index` — un nodo non riceveva mai il proprio stesso valore
come input della propria uscita di attenzione, a differenza della
formulazione originale di Velickovic et al. (2018), che include esplicitamente
il nodo stesso nel proprio vicinato. Correggerlo cambia il comportamento di
**ogni** modello Pro/ablation gia' allenato in questo progetto: per questo
non e' stato corretto silenziosamente qui, ma trasformato in un esperimento
controllato (`metastgat_pro_0.2_self_loop`, `metastgat_pro_0.0_self_loop`,
`ablation_environment_self_loop`) prima di decidere. Risultato — dopo aver
scoperto e corretto un bug nella prima misura (vedi `descrizione_modelli.md`
§4 per il dettaglio): miglioramento del travel time medio del 9.5%
(alpha=0.2) e 18.4% (alpha=0.0), identico in proporzione tra config di
training e di generalizzazione. **Deciso di applicarla in `src/models/metastgat.py`
in via permanente**: ora fa parte dell'architettura base, non e' un flag di
ablation, si applica a ogni preset senza eccezioni. Conseguenza diretta:
ogni modello MetaSTGAT allenato prima di questa modifica (tutti i model_id
sotto `results/` tranne i tre `*_self_loop`) va ri-allenato per restare
comparabile ad armi pari con uno allenato dopo — la stessa mancata
architettura in training/inferenza e' esattamente il bug appena scoperto e
corretto nella misura, e ripresentarsi in modo permanente sarebbe peggio,
non meglio, di lasciarla com'era.

`MetaGCNLayer` continua a ricevere self-loop separatamente (vedi §2): la GCN,
a differenza della GAT, si rompe concettualmente senza self-loop (la
normalizzazione di grado esclude altrimenti ogni nodo dal proprio stesso
output), quindi li ha sempre aggiunti a prescindere da questa decisione.
**SONAR non ha ricevuto la stessa modifica**: nella sua azione Laplaciana
$(L^aX)_v=\sum_u a_{uv}(X_v-X_u)$ un self-loop contribuirebbe con il termine
$a_{vv}(X_v-X_v)=0$, sempre nullo per costruzione (la differenza di un nodo
con se stesso e' zero) — aggiungerlo sarebbe un'operazione inerte, non
un'estensione "per coerenza" del beneficio osservato su GAT. L'asimmetria
GAT/GCN vs SONAR nel vedere il proprio stato resta quindi reale, ma per due
motivi diversi: GAT ora lo vede (self-loop aggiunto, beneficio misurato),
SONAR non lo vedra' mai tramite questo meccanismo (nullo per la sua stessa
formula) — un'eventuale via per dargli comunque accesso al proprio stato
passerebbe da un cambio della formula stessa, non da un self-loop, e resta
fuori dallo scope di questa decisione. Aggiornamento (12/9/2026): la stessa
correzione e' stata poi estesa per coerenza a `StandardGATLayer`
(`src/models/stgat.py`), che aveva la stessa lacuna strutturale -- nessun
modello STGAT era mai stato allenato in questo progetto, quindi li' la
modifica non ha alcun costo di ri-addestramento.

---

## 1. Mappa dei file

| File | Ruolo |
|---|---|
| `src/models/meta_gcn.py` | **Nuovo.** `MetaGCNLayer` (Meta-GCN) + `add_self_loops()` |
| `src/models/meta_sonar.py` | **Nuovo.** `MetaSONARLayer` (Meta-SONAR) |
| `src/models/metastgnn.py` | **Nuovo.** `MetaSTGNN` (MetaSTGAT con Meta-GAT → Meta-GCN) |
| `src/models/metastsonar.py` | **Nuovo.** `MetaSTSONAR` (MetaSTGAT con Meta-GAT → Meta-SONAR) |
| `src/models/metastgat.py` | **Modificato.** Aggiunto `num_layers` (1 o 2) allo stack Meta-GAT CST/CS |
| `src/agents/dqn_agent.py` | **Modificato.** `is_meta` riconosce anche i due modelli nuovi (bug, §0) |
| `scripts/train.py` | **Modificato.** `--model MetaSTGNN\|MetaSTSONAR`, `--num-layers`, `--sonar-recurrences`, `--sonar-step-size`, `--no-sonar-dissipation`, `--no-sonar-forcing` |
| `scripts/test.py` | **Modificato.** Stessi flag di `train.py`, devono combaciare col checkpoint valutato |
| `scripts/compare_spatial_mechanisms.py` | **Nuovo.** Registro separato (D5) per allenare/testare i 5-6 modelli del confronto GAT/GCN/SONAR |

Tutto il resto dell'architettura (`DualStateEncoder`, `SpatialMetaKnowledgeLearner`,
`TemporalMetaKnowledgeLearner`, `MetaLSTM`) e' condiviso e invariato nelle tre
varianti: solo il meccanismo che produce `z_CST`/`z_CS` dal grafo cambia.

---

## 2. MetaSTGNN (GCN)

`MetaGCNLayer` (`meta_gcn.py`) sostituisce l'attenzione di Meta-GAT con la
regola di aggregazione classica della GCN (Kipf & Welling, 2017):
normalizzazione simmetrica del grado, fissa e dipendente solo dalla
topologia, **senza** alcun meccanismo di compatibilita' appreso per coppia
di nodi (niente softmax su uno score Q·K). Il principio "meta" e' preservato
identico a Meta-GAT: il peso che trasforma il valore del vicino non e' un
parametro fisso condiviso, ma generato da SMK(i)/TMK(i) via `MetaDense`
(riusata as-is da `meta_gat.py`):

$$
h_v' = \mathrm{ReLU}\!\left(\sum_{u \,\in\, \mathcal{N}(v)\cup\{v\}} \frac{1}{\sqrt{\deg(u)\deg(v)}} \cdot W^{(v)} h_u + b^{(v)}\right)
$$

con $W^{(v)}, b^{(v)}$ generati da `MetaDense(meta\_embedding_v)`, non
condivisi tra i nodi. Self-loop aggiunti esplicitamente (§0) da
`add_self_loops()`, chiamata una sola volta per `edge_index` distinto (cache
per `id()` del tensore in `MetaSTGNN`, non ricalcolata a ogni step — vedi
`istruzioni seconda parte.md` §3.3).

`num_layers` (1 default, o 2) impila layer **indipendenti** (pesi propri,
`nn.ModuleList`), applicati sia al modulo CST sia al CS (decisione D6, vedi
§4). Con `num_layers=2`, la ReLU interna di ogni `MetaGCNLayer` funge gia' da
non-linearita' tra un layer e il successivo (nessuna attivazione esterna
aggiuntiva nello stack, a differenza di Meta-GAT — vedi §3).

---

## 3. MetaSTGAT: `num_layers` (GAT-1L / GAT-2L)

Stessa idea di stacking applicata a `MetaGATLayer`, che pero' non ha
un'attivazione interna: l'ELU va quindi applicata esplicitamente **tra** un
layer e il successivo, mai dopo l'ultimo (altrimenti `num_layers=1`
smetterebbe di essere identico al comportamento precedente questa modifica —
verificato esplicitamente, vedi §6).

Per il modulo CST (asimmetrico: query=$e_j$, key/value=$x_i$ dalla
Meta-LSTM), la query resta **fissa** a $e_j$ su entrambi i layer; key/value si
raffinano da un layer al successivo (l'output del primo layer diventa
key/value del secondo). Per il modulo CS (simmetrico), lo stack e' quello
standard: $Q=K=V=$ output del layer precedente. Scelta dichiarata, non
l'unica possibile (l'alternativa "aggiorna anche la query" e' egualmente
legittima) — annotata qui invece che lasciata implicita nel codice.

Con `num_layers=1`, entrambi i moduli eseguono esattamente un `MetaGATLayer`
senza alcuna ELU aggiuntiva: comportamento bit-per-bit identico alla
versione di `metastgat.py` precedente questa modifica (verificato, §6).

---

## 4. MetaSTSONAR (SONAR)

`MetaSONARLayer` (`meta_sonar.py`) modella la propagazione come un'onda
smorzata (Trenta, Gravina, Bacciu, NeurIPS 2025, Eq. 5):

$$\ddot X(t) = -L^a X(t) W - D(X(t)) \odot \dot X(t) + F(X(t))$$

discretizzata su $L$ passi (`n_recurrences`) con passo $h$ (`step_size`),
**a pesi condivisi tra le ricorrenze** (a differenza dello stack GAT/GCN
sopra, dove i due layer hanno pesi indipendenti — e' il motivo per cui SONAR
si confronta "a parita' di $L$", non "a parita' di numero di layer",
§0/tabella iniziale di `istruzioni seconda parte.md`):

$$a_{uv} = \mathrm{ReLU}\big((x_u - x_v) \cdot W_{\mathrm{res}}^{(v)} + b_{\mathrm{res}}^{(v)}\big), \qquad
  (L^a X)_v = \sum_{u} a_{uv}(X_v - X_u)$$

$$V \leftarrow V - h\,(L^aX + D(X)\odot V - F(X)), \qquad X \leftarrow X + h\,\tanh(V)$$

Solo $W_{\mathrm{res}}^{(v)}, b_{\mathrm{res}}^{(v)}$ (la resistenza
adattiva per arco) sono generati da `MetaDense(meta\_embedding)`: e' l'analogo
diretto del peso dinamico di Meta-GAT/Meta-GCN (Eq. 14-17 del paper
MetaSTGAT). Dissipazione $D(\cdot)$ e forzante $F(\cdot)$ restano MLP a pesi
**fissi** (decisione D3, §5): sono filtri locali del nodo, non meccanismi di
comunicazione tra nodi, quindi il collegamento concettuale con SMK/TMK
("il vicinato e' simile o diverso da me") e' piu' debole che per la
resistenza. $V^0 = \text{query} \cdot W_V$ (non $X^0 \cdot W_V$ come nel
paper): scelta di design per dare al modulo CST un'analogia residua con
"Q e K asimmetrici", assente nel paper originale (che non ha il concetto di
query/key). Nessuna persistenza di $V$ tra uno step ambientale e il
successivo (decisione D2, §5): ogni chiamata riparte da $V^0$ fresco.

**Validato contro il codice ufficiale**
(<https://github.com/gravins/SONAR>, `graph_transfer_task/models/sonar.py`)
prima di fissare l'implementazione, per due dettagli che il paper lascia
impliciti e che sono stati adottati qui:
- la resistenza $a_{uv}$ e' **ricalcolata a ogni iterazione** (non fissata
  all'inizio del blocco), dato che dipende dallo stato corrente dei nodi;
- l'aggiornamento di $X$ usa $\tanh(V)$, non $V$ grezzo: un accorgimento di
  stabilita' non esplicitato nel paper (nel repo ufficiale l'attivazione e'
  configurabile, default `tanh`).

`n_recurrences` di default 2 (raccomandazione D1, §5); `scripts/
compare_spatial_mechanisms.py` include anche `metastsonar_l4` (L=4) come
variante opzionale.

---

## 5. Decisioni D1-D6 (da `istruzioni seconda parte.md` §5)

| # | Decisione | Scelta presa |
|---|---|---|
| D1 | Profondita' spaziale di SONAR | **Opzione (a)**: $L$ passi *dentro* ogni chiamata (un blocco per step ambientale), non forzato a $L=1$. $L=2$ di default, $L=4$ disponibile come seconda variante opzionale. Bias sperimentale dichiarato: SONAR ottiene un vantaggio strutturale di poter guardare piu' lontano nello spazio a ogni singolo step, che e' esattamente il suo punto di forza dichiarato nel paper. |
| D2 | Persistenza di $V$ tra gli step | **Nessuna persistenza**: $V^0$ ricalcolato da zero a ogni chiamata. Piu' semplice, non invasivo sulla firma di `forward()`/`DQNAgent`. Possibile lavoro futuro. |
| D3 | Profondita' del "meta" in SONAR | **Solo la resistenza** e' meta-condizionata; dissipazione e forcing restano MLP fisse. Se in fase di training SONAR non impara nulla di interessante, la prima cosa da provare e' meta-condizionare anche questi due termini (stesso pattern `MetaDense`), prima di concludere che il meccanismo non funziona per il traffico. |
| D4 | Budget di parametri | **Non ancora deciso**: i modelli sono stati implementati a `hidden_dim` uguale (64), non a conteggio di parametri pareggiato. Conteggio effettivo misurato (griglia 4x4, $N=16$): vedi tabella sotto. **Da decidere prima di generare le tabelle comparative definitive** (vedi "Prossimi passi"). |
| D5 | Quali modelli confrontare | **Registro separato** (`scripts/compare_spatial_mechanisms.py`), tutti allenati con preset `pro`, nessuna combinazione con gli altri preset di ablation. 5 modelli di default (MetaSTGAT-2L, MetaSTGNN-1L/2L, MetaSTSONAR-L2), MetaSTGAT-1L e' il `metastgat_pro_0.5` gia' allenato nella prima parte (non riallenato), MetaSTSONAR-L4 incluso solo con `--include-sonar-l4`. |
| D6 | Stacking a 2 layer: CST, CS o entrambi | **Entrambi** (per GAT e per GCN, in modo coerente tra le due famiglie): isola la domanda "la profondita' del meccanismo spaziale nel suo complesso aiuta?" a livello di confronto principale. "Solo CST"/"solo CS" restano una possibile ablation di follow-up, non implementata qui. |

**Conteggio parametri misurato** (griglia 4x4, $N=16$, $d_1=64$, verificato
con backward pass su tensori fittizi, §6):

| Modello | Parametri |
|---|---|
| MetaSTGAT (1L, = Pro gia' allenato) | 2.321.104 |
| MetaSTGAT (2L) | 2.463.064 |
| MetaSTGNN (1L) | 2.720.074 |
| MetaSTGNN (2L) | 3.261.004 |
| MetaSTSONAR (L=2 o L=4, identico) | 2.229.066 |

Coerente con l'atteso (§6 di `istruzioni seconda parte.md`): GAT-2L/GCN-2L
hanno visibilmente piu' parametri delle rispettive varianti 1L (layer
indipendenti); SONAR-L2 e SONAR-L4 hanno lo **stesso** conteggio (pesi
condivisi tra ricorrenze). MetaSTGNN ha piu' parametri di MetaSTGAT a parita'
di `hidden_dim`: la `MetaDense` di Meta-GCN genera un'unica matrice $D_h
\times D_h$ per nodo (non spezzata per testa come in Meta-GAT), un dettaglio
che chi decide D4 deve tenere presente.

---

## 6. Verifica prima del training (istruzioni seconda parte.md §6)

Eseguita il 13/9/2026, prima di lanciare qualunque training lungo:

1. **Test di forma**: tutti e 6 le combinazioni (GAT 1L/2L, GCN 1L/2L, SONAR
   L2/L4) producono `q_values:(16,8)`, `h:(16,64)`, `c:(16,64)` con tensori
   fittizi su una griglia $4\times4$ ($N=16$), nessun NaN/Inf in uscita.
2. **Conteggio parametri**: misurato per tutte le combinazioni, tabella §5.
   GAT-2L>GAT-1L, GCN-2L>GCN-1L, SONAR-L2==SONAR-L4: tutti confermati.
3. **Controllo gradienti**: un singolo `.backward()` su una loss fittizia
   (somma dei Q-value) per ciascuna combinazione — nessun parametro con
   gradiente `None` o non finito, inclusi tutti i componenti di
   `MetaDense`/resistenza/dissipazione/forzante e di entrambi i layer nelle
   varianti a 2L. SONAR con `h=0.1` (default) non mostra segni di
   esplosione del gradiente ne' a $L=2$ ne' a $L=4$.
4. **Non-regressione**: l'output di `MetaSTGAT(num_layers=1)` e' stato
   confrontato, a pesi identici, con una chiamata diretta del singolo
   `MetaGATLayer` sottostante (bypassando il ciclo di stacking) —
   `torch.allclose` conferma output identico: nessuna ELU spuria introdotta
   dalla modifica per `num_layers`, il modello Pro gia' validato non e'
   stato alterato.
5. **Smoke test end-to-end** (Docker, 3 episodi, `--no-warmup
   --no-select-best`, `config_4x4_100m_train1.json`): completato senza errori
   per tutte e 5 le combinazioni (MetaSTGAT-2L, MetaSTGNN-1L, MetaSTGNN-2L,
   MetaSTSONAR-L2, MetaSTSONAR-L4) — training, checkpoint, replay buffer e
   integrazione con `CityFlowEnv`/`DQNAgent` funzionano end-to-end, non solo
   il forward/backward isolato del punto 1-3.

**Non ancora fatto** (esplicitamente fuori scope di questa implementazione,
vedi "Prossimi passi"): punto 5 di `istruzioni seconda parte.md` §6
(confronto del degrado relativo 4x4→5x5→6x6 tra profondita' diverse) richiede
risultati di training completo, non disponibili al momento della stesura di
questo documento.

---

## 7. Come allenare i modelli di questo confronto

```bash
# Un solo modello (debug)
python scripts/compare_spatial_mechanisms.py --model metastgnn_1l

# I 5 modelli principali (MetaSTGAT-1L/Pro e' gia' allenato, non ripetuto)
python scripts/compare_spatial_mechanisms.py --models all

# Includendo anche il secondo valore di L per SONAR (6 modelli)
python scripts/compare_spatial_mechanisms.py --models all --include-sonar-l4
```

Ogni modello e' allenato sulle stesse 3 varianti di training e con lo stesso
preset `pro` usato per `metastgat_pro_0.5`, poi testato sulle stesse 10
configurazioni (3 training + 7 generalizzazione) usate per ogni altro modello
di questo progetto. I grafici comparativi si generano con lo script gia'
esistente:

```bash
python scripts/compare_models.py --model-ids metastgat_pro_0.5 metastgnn_1l metastgat_2l metastgnn_2l metastsonar_l2
```

(`metastgat_2l`, non `metastgat_pro_0.5`, e' l'output di
`compare_spatial_mechanisms.py` per la variante a 2 layer — nomi scelti per
restare corti nelle legende dei grafici, vedi `MODEL_COLORS` in
`compare_models.py` se si vuole assegnare loro un colore fisso invece del
fallback automatico.)

---

## 8. Prossimi passi (non ancora fatti)

- **Decidere D4** (budget di parametri: uguale `hidden_dim` o conteggio
  pareggiato) prima di lanciare il training definitivo — la tabella di §5
  mostra che le 4 famiglie non sono a parita' di parametri con le impostazioni
  di default.
- **Lanciare il training completo** dei 5 (o 6) modelli via
  `scripts/compare_spatial_mechanisms.py` — non incluso in questa
  implementazione, solo smoke-testato.
- **Verificare la generalizzazione 5x5/6x6** (punto 5 di `istruzioni seconda
  parte.md` §6): confrontare il degrado *relativo* (non solo assoluto) tra
  varianti a profondita' diversa, dato il rischio di disallineamento
  treno/test descritto in D1.
- ~~Decidere se correggere il bug dei self-loop mancanti in Meta-GAT~~ **Deciso
  (12/9/2026)**: applicata in via permanente in `src/models/metastgat.py`
  dopo l'esperimento controllato (§0) — miglioramento consistente del 9.5%/
  18.4% di TT medio (alpha=0.2/0.0), identico in proporzione su training e
  generalizzazione. **Conseguenza da gestire**: ogni modello MetaSTGAT nella
  campagna sperimentale attuale (`metastgat_pro_0.5/0.2/0.1/0.0`,
  `metastgat_paper`, i 4 preset di ablation) e' stato allenato PRIMA di
  questa modifica e va ri-allenato per restare comparabile — non ancora
  fatto al momento della stesura di questa nota.
