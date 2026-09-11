# Istruzioni Seconda Parte — MetaSTGNN e MetaSTSONAR

> **Per il te del futuro**: questo file è il piano per la seconda parte della tesi.
> Leggilo tutto prima di scrivere codice — in particolare la sezione "Decisioni da
> confermare prima di iniziare", perché la parte SONAR ha scelte progettuali non
> ovvie che il paper originale non deve risolvere (lui non ha il concetto di
> "meta-knowledge"), ma noi sì.

**Aggiornamento scope (aggiunto dopo la stesura iniziale)**: il confronto non è
più a 3 modelli ma a **5, eventualmente 6**:

| Modello | Meccanismo spaziale | Profondità | Stato |
|---|---|---|---|
| MetaSTGAT-1L | GAT | 1 layer | **Già esiste** — è l'attuale `MetaSTGAT` (Pro), nessun lavoro nuovo |
| MetaSTGAT-2L | GAT | 2 layer | Nuovo — vedi §2.1 |
| MetaSTGNN-1L | GCN | 1 layer | Nuovo — §3 (design originale, invariato) |
| MetaSTGNN-2L | GCN | 2 layer | Nuovo — §2.1 applicata a `MetaGCNLayer` |
| MetaSTSONAR-L\* | SONAR | `L` ricorrenze (§4) | Nuovo — §4, valore di `L` da decidere, raccomandazione in D1 |
| MetaSTSONAR-L\*\* *(opzionale)* | SONAR | secondo valore di `L` | Solo se si decide di testare due profondità SONAR (→ 6 modelli totali) |

GAT e GCN si confrontano "a parità di hop" (1 e 2), SONAR si confronta a parità
di *L* interno perché la sua profondità non si ottiene impilando layer a pesi
distinti (costo di parametri) ma ricorrenze a pesi condivisi (§0) — motivo per
cui il suo asse di confronto è `L`, non "numero di layer" in senso stretto. Vedi
D1 per la raccomandazione sul valore di `L`.

## 0. Cosa dice il paper SONAR (riferimento)

Il file allegato (`NeurIPS-2025-sonar-...pdf`) descrive **SONAR** (Structured
Oscillatory Graph Neural Network with Adaptive Resistance), un DE-GNN che modella
la propagazione dell'informazione su grafo come un'onda:

- Stato del nodo `X(t)`, equazione d'onda del second'ordine:
  `Ẍ(t) = -L^a X(t) W - D(X(t)) ⊙ Ẋ(t) + F(X(t))` (Eq. 5)
  dove `L^a` è il Laplaciano pesato (resistenza adattiva `a_uv` per arco, sempre
  positiva via ReLU), `D(X)` è un termine dissipativo (≥0, via ReLU) e `F(X)` una
  forza esterna (senza vincoli di segno).
- Riscritta come sistema del prim'ordine con la velocità ausiliaria `V(t)=Ẋ(t)`,
  discretizzata (Eq. 7) con passo `h`:
  ```
  X^{l+1} = X^l + h·V^{l+1}
  V^{l+1} = V^l - h·(L^a X^l W + D(X^l)⊙V^l - F(X^l))
  ```
  con `V^0 = X^0 · W_V` (proiezione lineare appresa dell'input).
- Un "blocco" SONAR esegue `L` passi di discretizzazione (iperparametro
  `n_recurrences`), poi applica una MLP non lineare all'uscita (Eq. 8); più blocchi
  si impilano passando `X^{(i+1),0} = MLP(X^{(i),L})`.
- `a_uv`, `D(·)`, `F(·)` sono implementati nel paper come MLP **fisse** (pesi
  standard, non generati da un meta-learner) che prendono in input gli stati dei
  nodi correnti.

**Non esiste nel paper alcun concetto di "meta-knowledge"**: quella è la parte che
dobbiamo aggiungere noi, mantenendo lo stesso principio già usato da Meta-GAT nel
codice — vedi §2.

**Codice pubblico ufficiale**: <https://github.com/gravins/SONAR>. Prima di
implementare da zero `laplacian_action`/il loop di discretizzazione/le MLP di
resistenza-dissipazione-forcing (§4.1), confronta la tua implementazione con
quella del repo per verificare segni, convenzioni src/dst e dettagli numerici
(es. come inizializzano `V^0`, come gestiscono l'update di `a_uv` per batch di
grafi diversi, eventuali trick di stabilità su `h` non menzionati esplicitamente
nel paper). Non è detto che l'interfaccia del repo (probabilmente pensata per
grafi statici stile LRGB/Peptides, non per una sequenza di grafi ripetuti step
per step come nel nostro caso) sia riusabile as-is, ma è la fonte di verità più
affidabile per i dettagli implementativi che il paper lascia ambigui — usala
per validare la propria versione, non necessariamente per importarla di peso.

## 1. Terminologia: GNN vs GCN (chiarimento richiesto)

**Non sono la stessa cosa.** GNN (Graph Neural Network) è la categoria generale di
reti neurali che operano su grafi — GCN, GAT, GraphSAGE, GIN, e anche SONAR sono
*tutte* GNN. GCN (Graph Convolutional Network, Kipf & Welling 2017) è **una
specifica famiglia** di GNN, con una regola di aggregazione precisa:

```
h_v' = σ( Σ_{u ∈ N(v)∪{v}} (1/√(deg(u)·deg(v))) · W · h_u )
```

cioè aggregazione dei vicini pesata dalla normalizzazione simmetrica del grado
(fissa, dipende solo dalla topologia), **senza** un meccanismo di attenzione
(niente softmax su uno score di compatibilità Q·K come in GAT). Quando dici
"sostituendo le GAT con le GCN" il significato è chiaro (la classe specifica GCN),
quindi va bene così — l'equivalenza "GCN = GNN" no.

## 2. Dove vive la GAT nel codice attuale (mappa esatta)

`MetaSTGAT` (`src/models/metastgat.py`) usa la GAT **solo** in due punti, entrambi
istanze di `MetaGATLayer` (`src/models/meta_gat.py`):

| Attributo | Ruolo | Query | Key | Value | Meta-embedding |
|---|---|---|---|---|---|
| `self.meta_gat_cst` | CST module (Eq. 16-17) | `e_j` | `x_i` (output Meta-LSTM) | `x_i` | `tmk` |
| `self.meta_gat_cs`  | CS module (Eq. 14-15)  | `e_j` | `e_j` | `e_j` | `smk` |

Il resto dell'architettura (`DualStateEncoder`, `SpatialMetaKnowledgeLearner`,
`TemporalMetaKnowledgeLearner`, `MetaLSTM`) **non tocca la GAT e non va toccato**:
il Meta-LSTM è già "meta" a modo suo (pesi generati da `MetaDense3(TMK)`,
`src/models/meta_lstm.py`) e resta identico in entrambe le nuove varianti.

Il meccanismo "meta" dentro `MetaGATLayer.forward()` (righe 148-153) è:
```python
m = relu(meta_fc1(meta_embedding))          # (N, D_h)
W = meta_fc_W(m).view(N, H, hs, hs)          # pesi PER NODO, generati da SMK/TMK
b = meta_fc_b(m)                              # bias per nodo/testa
# W trasforma la query prima del dot-product con la key (Eq. 15/17)
```
**Questo è il principio da preservare identico** in `MetaGCNLayer` e
`MetaSONARLayer`: un embedding SMK(i)/TMK(i) genera via MetaDense i pesi che
governano come l'informazione si propaga per il nodo i — non pesi fissi
condivisi da tutti i nodi.

### 2.1 Da 1 a 2 layer (GAT e GCN, stessa ricetta per entrambe)

`MetaSTGAT` (Pro) usa oggi **1 layer** per `meta_gat_cst` e 1 per `meta_gat_cs`
(un solo hop di propagazione tra vicini per ciascun modulo, per chiamata). La
richiesta di confrontare anche una versione a 2 layer vale sia per GAT sia per
GCN: implementala **una volta sola**, come parametro generico, non come 4 file
separati.

**Design consigliato**: aggiungi `num_layers: int = 1` al costruttore di
`MetaSTGAT` (e allo stesso modo di `MetaSTGNN`, §3). Quando `num_layers == 2`,
sia `meta_gat_cst` sia `meta_gat_cs` (o i loro equivalenti GCN) diventano una
lista di 2 layer indipendenti (`nn.ModuleList`), ciascuno con il proprio
`MetaDense`/pesi propri (non condivisi tra i due — è deliberatamente diverso da
SONAR, dove invece i pesi *sono* condivisi tra le `L` ricorrenze, vedi §0):

```python
self.meta_gat_cs = nn.ModuleList([
    MetaGATLayer(hidden_dim, num_heads, meta_dim) for _ in range(num_layers)
])
...
# forward:
h = e_j
for layer in self.meta_gat_cs:
    h = layer(query=h, key=h, value=h, edge_index=edge_index, meta_embedding=smk)
    h = F.elu(h)   # non-linearita' tra un layer e l'altro — senza, 2 layer lineari
                   # collassano nell'espressivita' di 1 solo layer
```

Per `meta_gat_cst` (dove query=`e_j` esterno e key/value=`x_i` dalla Meta-LSTM,
non simmetrico come CS) il secondo layer prende come key/value l'output del
primo, mantenendo `e_j` originale come query per entrambi i layer (query fissa,
key/value che si raffinano) — oppure aggiorna anche la query con l'output del
layer precedente, se preferisci un vero stack "tutto si raffina insieme". Non è
ovvio quale sia meglio: **annotalo come scelta e sii coerente tra CST e CS**.

**Decisione aperta (nuova, aggiungila a D1-D5 come D6)**: se `num_layers=2`
vada applicato a **entrambi** i moduli (CST e CS) o a **uno solo**. Applicarlo a
entrambi è la lettura più naturale di "il modello a 2 layer" ma raddoppia i
parametri di entrambi i moduli contemporaneamente, rendendo più difficile
isolare se il guadagno (se c'è) viene dalla profondità spaziale del CST o del
CS. Raccomandazione: applicalo a entrambi per il confronto principale
GAT-1L/2L vs GCN-1L/2L vs SONAR (tesi centrale: "la profondità del meccanismo
spaziale nel suo complesso aiuta?"), e tieni "solo CST" o "solo CS" come
eventuale ablation di follow-up se il tempo lo permette — non bloccante.

Con `num_layers=1` il comportamento è identico a oggi (nessuna regressione sul
modello Pro già validato) — verificalo con un test di forma prima/dopo.

## 3. MetaSTGNN (GAT → GCN)

### 3.1 Design

Nuovo file `src/models/meta_gcn.py`, classe `MetaGCNLayer`, **stessa interfaccia**
di `MetaGATLayer` per essere un drop-in replacement:
```python
def forward(self, query, key, value, edge_index, meta_embedding) -> torch.Tensor:
    # query non serve alla GCN (nessun meccanismo di compatibilità Q·K) —
    # tienilo in firma solo per compatibilità di interfaccia, ignoralo nel corpo.
```

Passi:
1. **Aggiungi self-loop** a `edge_index` (pratica standard GCN: `Â = A + I`) — nel
   nostro caso `edge_index` viene da `graph_builder.py`/`CityFlowEnv.get_edge_index()`;
   verifica se include già self-loop (probabile di no, essendo pensato per GAT dove
   l'auto-contributo non serve perché non c'è normalizzazione di grado). Se manca,
   aggiungilo con `torch.cat([edge_index, self_loops], dim=1)` dentro il layer o
   una volta sola fuori (meglio: fuori, per non rifarlo ogni forward — vedi §3.3).
2. **Genera il peso dinamico per nodo** (riusa la classe `MetaDense` già presente
   in `meta_gat.py`, che produce esattamente `W: (N, D_h, D_h), b: (N,1)` da un
   embedding meta — è già pronta, non serve riscriverla):
   ```python
   self.meta_dense = MetaDense(meta_dim, hidden_dim, hidden_dim)  # out_w_dim = D_h
   W, b = self.meta_dense(meta_embedding)   # W: (N, D_h, D_h)
   ```
3. **Normalizzazione simmetrica del grado** (fissa, dipende solo dalla topologia,
   quindi calcolabile una volta e non ad ogni forward — vedi §3.3):
   ```python
   deg = torch.zeros(N, device=...)
   deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
   norm = (deg[src] * deg[dst]).clamp(min=1).rsqrt()   # (E,) = 1/sqrt(deg_u*deg_v)
   ```
4. **Trasforma e aggrega** (per ogni arco src→dst, trasforma il valore del nodo
   sorgente con il peso dinamico del nodo **destinazione** — coerente con la
   convenzione già usata da `MetaGATLayer`, dove `W[dst]` trasforma la query del
   nodo destinazione):
   ```python
   V_src = value[src]                                    # (E, D_h)
   V_transformed = torch.bmm(V_src.unsqueeze(1), W[dst]).squeeze(1)  # (E, D_h)
   weighted = norm.unsqueeze(-1) * V_transformed + b[dst]  # (E, D_h)
   out = torch.zeros(N, D_h, device=...)
   out.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)
   return F.relu(out)   # GCN classica ha una non-linearità in uscita
   ```

Nota: niente `num_heads`/softmax/attention score — la GCN non ha un meccanismo di
compatibilità, la "importanza" di un vicino è fissata dalla topologia (norma di
grado), non appresa dinamicamente per coppia di nodi. Questa è la differenza
concettuale reale rispetto a GAT/SONAR, non un dettaglio da nascondere.

**`num_layers` (1 o 2)**: applica a `MetaSTGNN` la stessa ricetta di stacking
descritta in §2.1 per `MetaSTGAT` — stesso parametro costruttore, stessa lista
`nn.ModuleList` di `MetaGCNLayer` indipendenti, stessa non-linearità (`ReLU`,
già presente in uscita da ogni `MetaGCNLayer`) tra un layer e il successivo, e
stessa decisione D6 su CST/CS da applicare in modo coerente con la versione GAT
(se decidi "entrambi i moduli" per GAT, fai lo stesso per GCN — il confronto
tra i due deve isolare solo GAT-vs-GCN, non anche una scelta di stacking
diversa tra i due).

### 3.2 File da creare/modificare

- **Nuovo** `src/models/meta_gcn.py` — `MetaGCNLayer` come sopra.
- **Nuovo** `src/models/metastgnn.py` — copia quasi identica di `metastgat.py`,
  con `self.meta_gat_cst`/`self.meta_gat_cs` sostituiti da `MetaGCNLayer`, classe
  rinominata `MetaSTGNN`. Tienila come file separato (non un flag dentro
  `MetaSTGAT`) — sono due architetture distinte da confrontare, mescolarle in un
  solo file con `if self.spatial_type == "gcn"` renderebbe il codice illeggibile.
- **`scripts/train.py`**: in `build_model()`, aggiungi il branch
  `elif args.model == "MetaSTGNN": model = MetaSTGNN(...)` (stessi argomenti di
  `MetaSTGAT`, `num_heads` va ignorato/non passato dato che non esistono teste).
  Aggiungi `"MetaSTGNN"` a `choices=[...]` di `--model`. Aggiungi anche un nuovo
  flag `--num-layers` (default 1, scelte 1/2) passato sia a `MetaSTGAT` sia a
  `MetaSTGNN` (§2.1) — così i 4 modelli GAT/GCN × 1L/2L si ottengono con
  `--model MetaSTGAT|MetaSTGNN --num-layers 1|2`, senza bisogno di 4 file/classi
  separate.
- **`scripts/test.py`**: stessa aggiunta nel branch di costruzione modello, nei
  `choices` di `--model`, e lo stesso `--num-layers` (deve combaciare con come è
  stato allenato il checkpoint che si sta valutando).
- **`src/agents/dqn_agent.py`**: verifica che `DQNAgent` non assuma da nessuna
  parte che il modello sia `MetaSTGAT`/`STGAT` per nome/tipo (dovrebbe già essere
  agnostico, dato che li tratta tutti come `nn.Module` con lo stesso
  `forward(states, edge_index, spatial_meta, temporal_meta, h_prev, c_prev)` — ma
  controllalo, non darlo per scontato).

### 3.3 Ottimizzazione non-obbligatoria ma consigliata

`edge_index` e quindi `norm`/self-loop non cambiano tra uno step e l'altro
all'interno dello stesso episodio (la topologia della griglia è fissa) — calcolarli
una volta in `CityFlowEnv.get_edge_index()` o cache-arli nel layer alla prima
`forward()` evita di rifare `scatter_add_` su ogni singolo step di ogni episodio.
Non bloccante per un primo prototipo funzionante, ma se noti training lento è il
primo posto da guardare.

## 4. MetaSTSONAR (GAT → SONAR)

Questa è la parte con vere decisioni di design da prendere — leggi prima §5.

### 4.1 Design proposto (default consigliato)

Nuovo file `src/models/meta_sonar.py`, classe `MetaSONARLayer`, stessa interfaccia:
```python
def forward(self, query, key, value, edge_index, meta_embedding) -> torch.Tensor:
```

Mappatura sui simboli del paper (Eq. 4-8): `value` è la condizione iniziale
`X^0` del blocco SONAR (il tensore che vogliamo propagare/aggiornare — stesso
ruolo di "value" in `MetaGATLayer`); `query`/`key` **non hanno un ruolo diretto
nell'equazione d'onda** (il paper propaga uno stato unico, non ha Q/K/V separati)
— per restare nell'interfaccia comune, `query` può essere usata come input allo
stato iniziale della velocità (`V^0 = query · W_V`, invece di `X^0 · W_V` come nel
paper — questo permette a `meta_gat_cst` "equivalente" di avere comunque due
sorgenti di informazione distinte, `e_j` per la velocità e `x_i` per la posizione,
mantenendo un minimo di analogia con "Q e K sono asimmetrici" del modulo CST
originale). Se questo ti sembra artificioso in fase di implementazione, l'alternativa
più fedele al paper è ignorare `query` e usare `V^0 = value · W_V` per entrambi i
moduli CS e CST — è una scelta legittima, annotala nella tesi.

Passi (per ogni chiamata, cioè per ogni environment-step — un "blocco" SONAR):

1. **Resistenza adattiva meta-condizionata** (l'equivalente SONAR del peso
   dinamico di Meta-GAT). Per ogni arco (u,v), il paper calcola
   `a_uv = ReLU(MLP(x_u, x_v))` con MLP a pesi fissi. Noi la rendiamo "meta"
   generando i pesi di quella MLP dal meta-embedding del nodo destinazione,
   riusando ancora `MetaDense`:
   ```python
   self.meta_dense_resistance = MetaDense(meta_dim, hidden_dim, 1)  # out_w_dim=1
   # produce W_res: (N, D_h, 1), b_res: (N, 1) per nodo destinazione
   ```
   poi, per ogni arco: `a_uv = relu( (x_u - x_v) · W_res[dst] + b_res[dst] )`
   — nota il segno `(x_u - x_v)`, coerente con la definizione di gradiente sul
   grafo (Eq. 2 del paper).
2. **Dissipazione e forza esterna**: per un primo prototipo, **tienile fisse**
   (MLP standard applicate allo stato corrente, non meta-condizionate) — sono
   meccanismi di filtraggio locale del nodo, non di comunicazione tra nodi, quindi
   il collegamento concettuale con SMK/TMK ("il vicinato è simile o diverso da
   me") è più debole. Meta-condizionarle è un'estensione naturale per dopo (vedi
   §5, decisione D3), non un requisito per dire "il meta-knowledge è implementato
   correttamente" — quello lo garantisce già la resistenza adattiva del punto 1,
   che è l'analogo diretto di Eq. 14-17.
3. **Loop di discretizzazione** (L passi, iperparametro `n_recurrences`, nuovo):
   ```python
   V = query @ self.W_V           # (N, D_h) — velocità iniziale
   X = value                       # (N, D_h) — posizione iniziale
   for _ in range(self.n_recurrences):
       LaX = laplacian_action(X, edge_index, a_uv_per_edge)  # scatter_add, vedi sotto
       D = F.relu(self.dissipation_mlp(X))          # (N, D_h), >=0
       Fext = self.forcing_mlp(X)                    # (N, D_h), libero
       V = V - self.h * (LaX @ self.W + D * V - Fext)
       X = X + self.h * V
   return X
   ```
   dove `laplacian_action(X, edge_index, a_uv)` calcola `(L^a X)_v = Σ_u a_uv(X_v - X_u)`
   via scatter, esattamente come la Eq. 2 del paper — implementabile con
   `scatter_add_` su `a_uv.unsqueeze(-1) * (X[dst] - X[src])` accumulato su `dst`
   (occhio al segno e a chi è sorgente/destinazione, ricontrolla contro Eq. 2 prima
   di fidarti).
4. `self.h` (step size) e `self.n_recurrences` (L) sono nuovi iperparametri del
   layer — esponili nel costruttore di `MetaSONARLayer` e propagali su fino a
   `MetaSTSONAR.__init__` e poi a `build_model()`/CLI (nuovi flag
   `--sonar-step-size` default 0.1, `--sonar-recurrences` default 5-10 — parti
   dai valori di Tabella 6 del paper per il caso "Hetero", che è il setting più
   vicino al nostro come scala di grafo).

### 4.2 File da creare/modificare

- **Nuovo** `src/models/meta_sonar.py` — `MetaSONARLayer` come sopra.
- **Nuovo** `src/models/metastsonar.py` — copia di `metastgat.py`, classe
  `MetaSTSONAR`, con `meta_gat_cst`/`meta_gat_cs` sostituiti da `MetaSONARLayer`.
- **`scripts/train.py`** e **`scripts/test.py`**: come per MetaSTGNN (§3.2), più i
  due nuovi flag CLI per `h` e `n_recurrences` (con default ragionevoli, non
  obbligare l'utente a specificarli sempre).
- Verifica **conteggio parametri**: SONAR ha più componenti (resistenza,
  dissipazione, forcing, `W_V`, `W`) di Meta-GAT — se vuoi un confronto onesto a
  "budget di parametri comparabile" tra i 3 modelli (pratica standard in
  letteratura, vedi il paper SONAR stesso che fissa un budget di 500k parametri
  per LRGB), calcola i parametri di tutti e 3 con `sum(p.numel() for p in
  model.parameters())` (già stampato da `build_model()`, riga
  `print(f"[Model] {args.model} | Parametri: {n_params:,}")`) e aggiusta
  `hidden_dim` per pareggiarli **prima** di lanciare gli esperimenti definitivi.

## 5. Decisioni da confermare prima di iniziare

Queste non hanno una risposta "ovviamente corretta" — deciderle ora evita di
riscrivere l'architettura a metà tesi:

- **D1 — Profondità della propagazione spaziale**: `MetaGATLayer` fa **un solo
  hop** per chiamata (la profondità nel tempo la dà la ricorrenza LSTM/Meta-LSTM
  sull'episodio, non un GNN profondo). SONAR invece è pensato per essere eseguito
  con **L passi interni per chiamata** (un "blocco"). Il design in §4.1 fa L passi
  *dentro* ogni singola chiamata (cioè dentro un singolo step ambientale), il che
  rompe la simmetria "1 hop spaziale per step" che hanno oggi MetaSTGAT e la GCN.
  Alternative: (a) tenerlo così (L passi per step, come proposto) — SONAR ottiene
  comunque un vantaggio strutturale di poter guardare più lontano nello spazio a
  ogni singolo step, che è esattamente il suo punto di forza dichiarato nel paper,
  quindi è difendibile in tesi come "abbiamo sfruttato la caratteristica di SONAR
  di propagare a lungo raggio in un solo blocco"; (b) forzare L=1 per un confronto
  "a parità di hop spaziali per step" più simile a un'ablation pulita, sacrificando
  parte del punto di forza di SONAR. **Scegli e scrivilo esplicitamente nella
  tesi**, qualunque sia la scelta — è un bias sperimentale che va dichiarato.

  **Raccomandazione concreta sul valore di `L`** (chiesta esplicitamente, tenendo
  conto che si allena solo su griglia 4x4 ma si testa anche su 5x5 e 6x6): il
  vincolo che conta non è "quanto è grande L in assoluto" ma **quanto L è grande
  rispetto al diametro del grafo** (distanza massima in hop tra due intersezioni,
  con `num_neighbors=4` cioè solo vicini di griglia diretti):

  | Griglia | Nodi | Diametro (hop, Manhattan) |
  |---|---|---|
  | 4x4 (train) | 16 | 3+3 = **6** |
  | 5x5 (test) | 25 | 4+4 = **8** |
  | 6x6 (test) | 36 | 5+5 = **10** |

  Se `L` si avvicina o supera il diametro della griglia di training (6), il
  modello impara a sfruttare informazione **globale** (ogni nodo "vede" quasi
  tutta la 4x4 in un blocco) — ma in test su 6x6 lo stesso `L` copre solo una
  frazione molto più piccola del grafo (`L`/10 invece di `L`/6): è un
  disallineamento treno/test del campo recettivo, un rischio concreto di
  generalizzazione peggiore su griglie più grandi, oltre al classico
  over-smoothing noto in letteratura GNN oltre 3-4 hop di propagazione.

  **Valore consigliato: `L=2`** — resta ben sotto il diametro anche della
  griglia più piccola (6), è direttamente confrontabile "a parità di hop" con la
  variante a 2 layer di GAT/GCN (§2.1), e minimizza il rischio di
  disallineamento treno/test. Se si vuole mostrare il vero punto di forza di
  SONAR (propagare più lontano di un GNN impilato, a costo di parametri
  costante grazie ai pesi condivisi tra ricorrenze, §0) e si opta per **due
  valori di L → 6 modelli totali**, il secondo valore consigliato è **`L=4`**:
  il doppio della profondità massima testata per GAT/GCN, ma ancora
  abbondantemente sotto il diametro di tutte e tre le griglie (4/6, 4/8, 4/10),
  quindi senza saturare la 4x4 in training. Evita `L>=6`: a quel punto il
  modello vedrebbe l'intera griglia di training ad ogni blocco, il caso peggiore
  per il disallineamento appena descritto.
- **D2 — Persistenza della velocità V tra gli step**: nel design proposto, `V`
  viene ricalcolata da zero (`V^0 = query · W_V`) a ogni chiamata/step ambientale,
  mai portata avanti da uno step al successivo (coerente col fatto che ogni
  "blocco" SONAR nel paper riparte da `X^{(i),0}` fresco). Un'alternativa è far
  persistere `V` come stato ricorrente aggiuntivo (accanto a `h_prev`/`c_prev`
  della LSTM), il che renderebbe SONAR più simile a un vero stato "di lungo
  periodo" ma richiede cambiare la firma di `forward()` di `MetaSTSONAR` (un
  input/output in più) e la gestione in `dqn_agent.py`/`train.py` dove oggi si
  passano solo `h_prev`/`c_prev`. **Consigliato**: partire senza persistenza (più
  semplice, meno invasivo), notarlo come possibile lavoro futuro.
- **D3 — Quanto in profondità va il "meta"**: nel design proposto solo la
  resistenza adattiva è meta-condizionata (dissipazione e forcing restano MLP
  fisse). Se durante l'implementazione risulta che il modello non impara nulla di
  interessante, la prima cosa da provare è meta-condizionare *anche* dissipazione
  e/o forcing allo stesso modo (stesso pattern `MetaDense`), prima di concludere
  che "SONAR non funziona per il traffico".
- **D4 — Budget di parametri**: confrontare i 5-6 modelli a `hidden_dim` uguale
  (semplice, ma GAT-2L/GCN-2L avranno più parametri di GAT-1L/GCN-1L per via dei
  layer indipendenti, §2.1, e SONAR ne avrà ancora di più per le MLP aggiuntive
  — mentre SONAR-L2 e SONAR-L4, se testati entrambi, avranno lo stesso
  conteggio a parità di `hidden_dim`, §0) oppure a conteggio di parametri
  pareggiato tra le famiglie (più onesto, richiede tuning di `hidden_dim` per
  GCN/SONAR, e persino tra 1L e 2L della stessa famiglia se si vuole isolare
  l'effetto della profondità a parità di capacità). Deciderlo prima di generare
  tabelle comparative.
- **D5 — Quali modelli confrontare esattamente**: la richiesta attuale è 5
  modelli (MetaSTGAT-1L, MetaSTGAT-2L, MetaSTGNN-1L, MetaSTGNN-2L, MetaSTSONAR
  con un valore di `L`), eventualmente 6 se si testano due valori di `L` per
  SONAR (vedi tabella a inizio file e D1). Verifica se vuoi anche le ablation
  (environment/temporal/rl_core) applicate a **ciascuna** delle varianti
  spaziali (esplosione combinatoria: 5-6 varianti spaziali × 4 ablation = 20-24
  modelli) o se l'ablation study resta solo sulla variante GAT-1L (Pro, come
  oggi) e questo secondo confronto è indipendente e più ristretto (solo i 5-6
  modelli, stesso training completo "pro" per tutti, cioè tutti allenati con lo
  stesso preset — nessuna ablation attiva). **Raccomandazione**: la seconda
  opzione (confronto ristretto, separato dallo studio di ablation) — 20+ modelli
  renderebbero sia il tempo di calcolo sia le tabelle comparative difficili da
  presentare in modo leggibile in tesi, e le due domande di ricerca sono
  concettualmente separate ("quale meccanismo spaziale è migliore" vs "quali
  componenti del framework Pro contano"). Questo determina se estendere
  `MODEL_REGISTRY` in `scripts/main.py` o tenere questo esperimento separato con
  un proprio piccolo script/registro — con la raccomandazione sopra, un registro
  separato (5-6 voci, tutte preset "pro") è più pulito.
- **D6 — Stacking a 2 layer su CST, CS, o entrambi**: vedi §2.1. Raccomandato:
  entrambi, per isolare in modo netto la domanda "profondità del meccanismo
  spaziale nel suo complesso", tenendo "solo CST"/"solo CS" come possibile
  ablation di follow-up.

## 6. Piano di verifica prima del training vero

Nell'ordine, prima di lanciare qualunque training lungo:

1. **Test di forma**: istanzia `MetaSTGAT`/`MetaSTGNN` con `num_layers=1` **e**
   `num_layers=2`, e `MetaSTSONAR` con il/i valore/i di `L` scelti, con tensori
   fittizi (`torch.randn`) delle dimensioni giuste (`N=16` nodi come la griglia
   4x4) e verifica che `forward()` non esploda e restituisca `(N, n_actions)`,
   `(N, D_h)`, `(N, D_h)` come oggi. In particolare verifica che
   `MetaSTGAT(num_layers=1)` dia risultati identici alla versione attuale
   (nessuna regressione sul modello Pro già validato).
2. **Conteggio parametri**: stampalo per tutti i 5-6 modelli (via `build_model()`,
   già presente) e decidi D4. Aspettati che GAT-2L/GCN-2L abbiano visibilmente
   più parametri di GAT-1L/GCN-1L (layer indipendenti, §2.1), mentre SONAR-L2 vs
   SONAR-L4 abbiano lo **stesso** conteggio (pesi condivisi tra ricorrenze, §0) —
   se non è così, c'è un bug nello stacking.
3. **Controllo gradienti**: un singolo `.backward()` su una loss fittizia, verifica
   che tutti i parametri (inclusi quelli di `MetaDense`/resistenza/dissipazione/
   forcing, e di entrambi i layer nelle varianti 2L) abbiano un gradiente
   non-None e non-NaN — SONAR in particolare, per `h` troppo grande, può
   esplodere (vedi Teorema 3.4 e §D "Limitations" nel paper: il bound di
   sensitività cresce con `h`), quindi controlla anche che la loss non diverga
   già nei primi update con gli `h`/`L` di default (incluso `L=4` se lo testi).
4. **Smoke test reale** (docker, come già fatto per il resto della pipeline):
   `python scripts/train.py --model MetaSTGAT --num-layers 2 --config configs/config_4x4_200m_2k_flat.json --episodes 3 --no-warmup --no-select-best`
   e lo stesso per `MetaSTGNN` (1L e 2L) e `MetaSTSONAR` (per ciascun `L`
   scelto), prima di qualunque run lungo.
5. **Generalizzazione 5x5/6x6** (specifico per la profondità spaziale, non solo
   per SONAR): dato il rischio di disallineamento treno/test descritto in D1,
   quando arrivano i risultati sulle config 5x5/8k e 6x6/10k confronta il
   *degrado relativo* (non solo il valore assoluto) di travel time/throughput
   tra le varianti a profondità diversa (1L vs 2L, L2 vs L4) — se una variante
   più profonda generalizza peggio (degrado relativo maggiore passando da 4x4 a
   6x6) è un risultato di per sé interessante da riportare, non un fallimento
   dell'esperimento.

## 7. Riepilogo checklist file

- [ ] `src/models/meta_gcn.py` (nuovo)
- [ ] `src/models/meta_sonar.py` (nuovo)
- [ ] `src/models/metastgnn.py` (nuovo)
- [ ] `src/models/metastsonar.py` (nuovo)
- [ ] `src/models/metastgat.py`: aggiungi `num_layers` (§2.1) — deve restare identico a oggi con `num_layers=1`
- [ ] `scripts/train.py`: `--model` choices + `build_model()` + `--num-layers` (1/2, per MetaSTGAT e MetaSTGNN) + flag CLI SONAR (`--sonar-step-size`, `--sonar-recurrences`/`L`)
- [ ] `scripts/test.py`: stessa cosa (incluso `--num-layers`, deve combaciare col checkpoint)
- [ ] Registro dei 5-6 modelli da confrontare (vedi tabella a inizio file e D5) — separato da `MODEL_REGISTRY` delle ablation, salvo diversa decisione
- [ ] Decisioni D1-D6 prese e scritte da qualche parte (anche solo aggiornando questo file con le risposte), incluso il valore di `L` scelto per SONAR (raccomandazione: L=2, eventuale secondo valore L=4 — vedi D1)
- [ ] Verifica §6 completata prima di qualunque training lungo, incluso il controllo di non-regressione su `MetaSTGAT(num_layers=1)`
- [ ] Confronto con il codice pubblico <https://github.com/gravins/SONAR> per validare segni/convenzioni prima del training vero (vedi §0)
