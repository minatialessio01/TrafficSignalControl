# Analisi delle Differenze: Codice vs Paper MetaSTGAT

> 📖 **Per capire quali modelli sono stati testati e in cosa differiscono, parti da
> [`descrizione_modelli.md`](descrizione_modelli.md)** (12/9/2026): quel documento fonde questo
> file con `proposte_ablation.md` in un unico catalogo. Questo file resta il riferimento di
> dettaglio riga-per-riga con gli estratti di codice, che `descrizione_modelli.md` §1 riassume
> in una singola tabella (M1-M10).
>
> **Scopo**: Identificare sistematicamente tutte le differenze tra l'implementazione
> presente nel repository e l'architettura/procedura descritta nell'articolo originale
> *"MetaSTGAT: Meta-learning Spatial-Temporal Graph Attention Network for Traffic Signal Control"*
> (Wang et al., Knowledge-Based Systems, 2022).
>
> **Nota (11/9/2026)**: ognuna delle differenze qui elencate è stata da allora formalizzata
> in un flag di ablation individualmente disattivabile (`--no-vision-cutoff`, `--no-bptt`,
> `--no-per`, ecc. in `scripts/train.py`), organizzati in 6 preset (`pro`/`paper`/`environment`/
> `temporal`/`rl_core`/`replay_stability`). Il preset `paper` disattiva **tutte** le differenze
> elencate qui, riproducendo la procedura originale con lo stesso codice usato per il modello
> avanzato — vedi [descrizione_scripts.md](descrizione_scripts.md) per la mappa preset→flag e
> [proposte_ablation.md](proposte_ablation.md) per la razionale scientifica del raggruppamento
> (che è, letteralmente, i 10 "M1..M10" di questo documento riorganizzati in sottosistemi).
> Alcuni valori numerici citati sotto (buffer size, episodi di default) sono stati rivisti dopo
> la stesura originale di questo documento — vedi le note puntuali dove rilevante.

---

## Sommario Esecutivo

| Area | Differenza | Impatto atteso |
|------|-----------|----------------|
| MDP – Stato | Aggiunto `wait_vec` (12 dim), stato dim=32 vs 20 | Positivo – info aggiuntive |
| MDP – Reward | Formula completamente riscritta (Throughput-based) vs `-P_i` | Significativo |
| MDP – Azione | Maschera invalid actions (max 2 consecutive) | Positivo |
| Training – Buffer | PER sequenziale (2.400) vs buffer piatto (10.000) | Significativo |
| Training – BPTT + Burn-in | Sequenze L=4, burn-in=2 vs single-step | Significativo |
| Training – Double DQN | Double DQN vs DQN standard | Moderato |
| Training – Loss | Huber loss + IS weights vs MSE | Moderato |
| Training – Epsilon decay | 0.9 to 0.01 in 20 ep vs non specificato | Moderato |
| Training – Episodi | 50 ep default (variabile per run) vs 200 ep nel paper | Minore |
| Architettura GAT | (W·Q)·K vs W·(Q·K) — interpretazione ambigua | Da verificare |
| Meta-LSTM | W generato per tutti e 4 i gate vs notazione paper ambigua | Interpretazione |
| Meta-knowledge features | wait_vec nel TMK; assenza "location" | Misto |
| Output MetaSTGAT | Restituisce x_i (hidden LSTM) — corretto ma notazione diversa | Comportamentale |

---

## 1. MDP – State, Action, Reward

### 1.1 Stato (State)

**Paper (Section 3.2):**
> "State of intersection is composed of the number of vehicles in each lane and the current
> signal phase."
>
> Stato = `[n_vec (12), p_vec (8)]` --- **dimensione 20**

**Codice** ([cityflow_env.py](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/environment/cityflow_env.py)):
```python
# STATE_DIM = N_LANES + N_LANES + N_PHASES = 12 + 12 + 8 = 32
observations[iid] = np.concatenate([n_vec, wait_vec, p_vec])  # dim=32
```

**Differenza**: E' stata aggiunta una terza componente `wait_vec` (12 dim) che rappresenta il
**tempo massimo di attesa normalizzato per corsia** (0-1, saturazione a 100s). Questo porta
la dimensione dello stato da 20 a **32**.

> **Motivazione**: Il wait time e' un segnale piu' informativo del semplice conteggio veicoli.
> Un veicolo fermo da 90 secondi e' piu' urgente di uno fermo da 5 secondi. Il paper usa il
> wait time solo nei meta-learner, non nello stato principale.

---

### 1.2 Azione (Action)

**Paper (Section 3.2):**
> "At each timestamp the agent takes one phase as its action [...] the duration of the green light."
>
> Nessuna menzione di vincoli sulle azioni consecutive.

**Codice** ([cityflow_env.py L674-685](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/environment/cityflow_env.py#L674-L685)):
```python
def get_invalid_actions(self):
    # Maschera la fase corrente se scelta >= 2 volte consecutive
    if self.consecutive_phases.get(iid, 0) >= 2:
        invalid_actions[iid] = [phase]
```

**Differenza**: Aggiunto un vincolo che **invalida una fase se gia' selezionata 2 volte di fila**,
forzando la rotazione. Il paper non menziona questo meccanismo.

> **Motivazione**: Evita che l'agente si blocchi su una singola fase (verde fisso),
> comportamento degenerativo tipico nelle prime fasi del training.

---

### 1.3 Reward

**Paper (Section 3.2, Eq. 1):**
```
r_i = -P_i
```
dove `P_i` = pressione = (veicoli in ingresso) - (veicoli in uscita).

**Codice** ([cityflow_env.py L464-519](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/environment/cityflow_env.py#L464-L519)):
```python
passed   = len(in_t0 - in_t1)      # veicoli usciti dalla finestra visiva
incoming = len(in_t0)               # veicoli in ingresso al tempo t
wasted_green_penalty = 50.0 if (passed == 0 and incoming > 0) else 0.0

raw_reward = passed - incoming - (alpha * max_red_wait_time) - wasted_green_penalty
normalized_reward = raw_reward / 100.0
rewards[iid] = max(min(normalized_reward, 5.0), -20.0)
```

**Differenze**:

| Aspetto | Paper | Codice |
|---------|-------|--------|
| Formula base | `-P_i` (pressione) | `Passed - Incoming - alpha·MaxWait - Penalty` |
| Penalita' attesa corsie rosse | Non presente | `-alpha·max_red_wait_time` (alpha=0.5) |
| Penalita' verde sprecato | Non presente | `-50.0` se nessun veicolo passa |
| Normalizzazione | Non presente | Divisione /100, clip [-20, +5] |
| Finestra veicoli | Intera intersezione | Solo entro VISION_CUTOFF_M dal semaforo (~144m) |

> **Motivazione**: Il reward originale (pura pressione) puo' essere instabile. La formula
> riscritta promuove throughput esplicito e penalizza attese prolungate.

---

## 2. Architettura del Modello

### 2.1 State Encoder (DualStateEncoder)

**Paper (Section 4.2.1, Eq. 3-4):**
> Two-layer MLP con stesse dimensioni ma inizializzazioni diverse.
> Output: `e_i` (branch temporale -> LSTM) e `e_j` (branch spaziale -> GAT).
> Attivazione: ReLU.

**Codice** ([state_encoder.py](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/models/state_encoder.py)):
```python
class StateEncoder(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
```

**Differenza**: Architettura conforme al paper. Dimensione input 32 vs 20 per l'aggiunta di `wait_vec`.

---

### 2.2 Meta-Knowledge Learners (SMK/TMK)

**Paper (Section 4.3.1):**
> "Two-layer fully connected network with different initializations."
>
> **SMK features**: distance, lane pressure, number of vehicles
>
> **TMK features**: vehicle queue length, historical states, **location**

**Codice** ([meta_knowledge_learner.py](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/models/meta_knowledge_learner.py)):
```python
class MetaKnowledgeLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=64):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation     = nn.ReLU()
        self.out_activation = nn.Tanh()   # <- AGGIUNTO, non nel paper
```

**Differenze**:

| Aspetto | Paper | Codice |
|---------|-------|--------|
| Attivazione layer finale | Non specificata | `Tanh()` aggiunta |
| TMK features | queue + historical + **location** | queue + historical (**no location**) |
| Output activation | Non specificata | Tanh, range [-1, 1] |

> **Note**: "location" non e' stata implementata. Il `Tanh` finale e' una scelta
> implementativa per contenere l'embedding in un range limitato.

---

### 2.3 Meta-GAT (MetaGATLayer)

**Paper (Section 4.3.2, Eq. 14-17):**

CS Module (Eq. 14-15):
```
W_j, b_j = MetaDense(1)(SMK(i))
phi(Q_i, K_{i,u}) = [W_j · (Q_i · K_{i,u}) + b_j] / sqrt(D_h)
```

CST Module (Eq. 16-17):
```
W_k, b_k = MetaDense(2)(TMK(i))
phi(Q_i, K_{i,u}) = [W_k · (Q_i · K_{i,u}) + b_k] / sqrt(D_h)
```

**Codice** ([meta_gat.py](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/models/meta_gat.py)):
```python
# MetaDense genera W: (N, H, hs, hs) e b: (N, H)
W_h = W[dst, h, :, :]
Q_h_trans = (W_h @ Q_h.unsqueeze(-1)).squeeze(-1)  # W trasforma Q
dot = (Q_h_trans * K_h).sum(dim=-1) + b[dst, h]   # dot-product + bias
dot_scaled = dot / self.scale                       # / sqrt(head_size)
```

**Differenze**:

| Aspetto | Paper | Codice |
|---------|-------|--------|
| Formula | `W · (Q · K)` scalare | `(W·Q) · K` — W matrice che trasforma Q |
| Dimensione W | Ambigua (scalare?) | Matrice `(hs, hs)` per nodo per head |
| Bias b | Scalare `b in R` | Scalare per head `(N, H)` |
| Fattore di scala | `/sqrt(D_h)` | `/sqrt(head_size)` = `/sqrt(D_h/H)` |

> **Nota**: La formula del paper e' ambigua. L'implementazione usa la seconda interpretazione
> `(W·Q)·K`, piu' simile alle hypernetwork standard. La scala `/sqrt(head_size)` e'
> matematicamente corretta per multi-head attention.

---

### 2.4 Meta-LSTM (MetaDense3 + MetaLSTMCell)

**Paper (Section 4.3.3, Eq. 20-21):**
```
W_Phi, b_Phi = MetaDense(3)(TMK(i))
x_i^t = Meta-LSTM(e_i^t, h_{t-1} | W_Phi, b_Phi)
```
> "W_Phi in R^{2D_h x D_h}" (notazione per concatenazione [input, hidden])
>
> "b_Phi in R" (bias scalare)

**Codice** ([meta_lstm.py](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/models/meta_lstm.py)):
```python
# Genera pesi per TUTTI e 4 i gate LSTM:
self.fc_weight = nn.Linear(hidden_dim, 4 * 2 * hidden_dim * hidden_dim)
self.fc_bias   = nn.Linear(hidden_dim, 4 * hidden_dim)
# W: (N, 4_gates, 2*D_h, D_h),  b: (N, 4*D_h)
```

**Differenze**:

| Aspetto | Paper | Codice |
|---------|-------|--------|
| Dimensione W | `R^{2D_h x D_h}` (notaz. 1 gate) | `R^{4 x 2D_h x D_h}` (tutti 4 i gate) |
| Bias b | `b in R` (scalare) | `b in R^{4 x D_h}` (bias completo) |
| Gate coperti | Non esplicitato | Tutti e 4 (forget, input, output, cell) |

> **Interpretazione**: La notazione del paper e' semplificata. Un'LSTM funzionante richiede
> i pesi per tutti e 4 i gate. L'implementazione e' la scelta corretta per una hypernetwork LSTM.

---

### 2.5 Q-value Head

**Paper (Eq. 12):**
```
q~(o_i^t) = Dense(Concat(z_CST, z_CS))
```

**Codice** ([metastgat.py](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/models/metastgat.py)):
```python
self.q_head = nn.Linear(hidden_dim * 2, n_actions)
```

**Differenza**: Nessuna. Conforme al paper.

---

## 3. Procedura di Training

### 3.1 Replay Buffer

**Paper (Section 5.1):**
> "Sampling size: 1,000 and the experience replay buffer size: 10,000."
> Buffer standard (flat), campionamento uniforme, transizioni singole.

**Codice** ([replay_buffer.py](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/agents/replay_buffer.py)):

| Aspetto | Paper | Codice |
|---------|-------|--------|
| Tipo buffer | Standard FIFO flat | **PER sequenziale (episodico)** |
| Campionamento | Uniforme | **Proporzionale alle priorita' TD** |
| Dimensione | 10,000 transizioni | **4,800 transizioni** (~40 episodi di storia con episodi da 1800s; il costruttore di `DQNAgent` ha un default di 2,400, ma `scripts/train.py` lo sovrascrive esplicitamente — vedi `DEFAULTS["buffer_size"]`) |
| Unita' campionata | Singola transizione | **Sequenza di L=4 step** |
| IS weights | Non presenti | **Presenti (beta annealing)** |

Nota: con `--no-per` (preset `paper`/`replay_stability`) il buffer torna piatto e uniforme, con dimensione **10,000** — la stessa del paper — non 2,400/4,800 (quei valori sono specifici del PER sequenziale).

---

### 3.2 BPTT e Burn-in

**Paper (Algorithm 1):**
> Training DQN standard su singole transizioni. No BPTT, no burn-in.

**Codice** ([dqn_agent.py](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/agents/dqn_agent.py) — default costruttore, righe 76-77):
```python
seq_len = 4      # sequenze di training (rivisto da un originario 8 — vedi nota)
burn_in = 2      # step iniziali senza gradiente (rivisto da un originario 4)

for t in range(L):
    if t < self.burn_in:
        with torch.no_grad():  # solo aggiorna h/c
            ...
    else:
        ...  # BPTT attivo
```
Nota (13/9/2026): questo documento riportava ancora `L=8`/`burn_in=4`, i valori di una versione precedente del codice — il default attuale, verificato direttamente nel costruttore di `DQNAgent` e mai sovrascritto da `train.py`/`test.py`, è `seq_len=4`/`burn_in=2`. Corretto qui; vedi anche `proposte_ablation.md` (stessa correzione).

**Differenza**: Il codice implementa **R2D2-style training** (Kapturowski et al., ICLR 2019):
BPTT su sequenze di L=4 step + burn-in di 2 step. Non presente nel paper.

> **Motivazione**: Con il training DQN standard, lo stato LSTM usato nel training (zero)
> non corrisponde a quello prodotto durante la raccolta dati. BPTT + burn-in corregge
> parzialmente il problema "hidden state staleness".

---

### 3.3 DQN Standard vs Double DQN

**Paper (Algorithm 1, Eq. 13):**
```
target = r + gamma * max_a' Q(s', a'; Phi')   # DQN standard
```

**Codice** ([dqn_agent.py L351-357](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/agents/dqn_agent.py#L351-L357)):
```python
# Double DQN (van Hasselt et al., 2016):
best_next_actions = q_online_next.argmax(dim=-1, keepdim=True)
q_next_value      = q_next.gather(1, best_next_actions)
q_target          = rt + self.gamma * q_next_value
```

**Differenza**: Paper usa DQN standard; codice usa **Double DQN** per ridurre bias di sovrastima.

---

### 3.4 Loss Function

**Paper (Eq. 13):**
```
L(Phi) = sum_t sum_i [y_i^t - q~(o_i^t)]^2   # MSE
```

**Codice** ([dqn_agent.py L377-381](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/agents/dqn_agent.py#L377-L381)):
```python
loss = smooth_l1_loss(...)   # Huber loss invece di MSE
loss = torch.mean(loss_per_seq * is_weights_t)   # pesato IS weights
```

| Aspetto | Paper | Codice |
|---------|-------|--------|
| Loss type | MSE (L2) | **Huber Loss (Smooth L1)** |
| Pesatura campioni | Uniforme | **IS weights da PER** |

---

### 3.5 Target Network Update

**Paper**: Non specificato.

**Codice** ([dqn_agent.py L406-412](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/src/agents/dqn_agent.py#L406-L412)):
```python
# Soft update: tau=0.01, ogni 100 step
target_p.data.copy_(tau * online_p.data + (1 - tau) * target_p.data)
```

**Differenza**: **Soft update** (Polyak averaging, tau=0.01). Il paper non specifica.

---

### 3.6 Numero Episodi e Configurazione

**Paper**: 200 episodi, 3 processi paralleli, 1800s (synth) / 3600s (real).

**Codice**: `DEFAULTS["episodes"] = 50` in `scripts/train.py`, quasi sempre sovrascritto da riga di comando per run specifici (es. 70 o 100 episodi per `metastgat_pro` in sessioni di training successive a questo documento) — non esiste un "vero" numero fisso, è un parametro di ogni run. Singolo processo in ogni caso (mai 3 paralleli).

| Aspetto | Paper | Codice |
|---------|-------|--------|
| Episodi | 200 | 50 di default, tipico 50-100 a seconda del run (`--episodes`) |
| Processi paralleli | 3 | 1 |

---

### 3.7 Gradient Clipping

**Paper**: Non presente.

**Codice**: `clip_grad_norm_(max_norm=1.0)` — aggiunto per stabilita' LSTM.

---

## 4. Riepilogo

### 4.1 Modifiche che presumibilmente MIGLIORANO il paper originale

| # | Modifica | Razionale |
|---|----------|-----------|
| 1 | `wait_vec` nello stato | Segnale piu' ricco |
| 2 | Double DQN | Riduce bias sovrastima Q-value |
| 3 | BPTT + burn-in (R2D2) | Training piu' corretto per LSTM |
| 4 | PER | Campionamento piu' efficiente |
| 5 | Huber Loss | Piu' robusta agli outlier |
| 6 | Soft update target network | Aggiornamento piu' stabile |
| 7 | Gradient clipping | Stabilita' LSTM |
| 8 | Maschera azioni invalide | Previene comportamenti degenerativi |
| 9 | Tanh finale nei meta-learner | Embedding normalizzato |

### 4.2 Modifiche che DEVIANO o rendono difficile il confronto

| # | Modifica | Possibile impatto |
|---|----------|-------------------|
| 1 | Reward completamente riscritta | Confronto diretto col paper difficile |
| 2 | Buffer ridotto (2,400 vs 10,000) | Meno diversita' campioni |
| 3 | Epsilon decay aggressivo (20 ep) | Esplorazione insufficiente |
| 4 | 100 ep invece di 200 | Training incompleto |
| 5 | Assenza di "location" nel TMK | Feature mancante |
| 6 | Stato 32 vs 20 dim | Non confrontabile direttamente |

### 4.3 Modifiche INTERPRETATIVE (ambiguita' del paper)

| # | Differenza | Ambiguita' originale |
|---|------------|----------------------|
| 1 | W per tutti e 4 i gate LSTM | Paper scrive solo R^{2D_h x D_h} |
| 2 | (W·Q)·K invece di W·(Q·K) | Eq. 14-17 ambigue |
| 3 | Scala /sqrt(head_size) vs /sqrt(D_h) | Paper non distingue |
| 4 | Bias per head invece di bias scalare | Paper scrive b in R |

---

## 5. Implicazioni per il Confronto Sperimentale

### 5.1 Metriche oggettive (indipendenti dalla reward)

- **Average Travel Time (s)** -- metrica principale del paper (Tabella 3)
- **Throughput (veicoli completati)** -- metrica secondaria
- **Convergence speed** (episodi per raggiungere il best)

### 5.2 Strategia di confronto

1. Usare **solo metriche oggettive** (travel time, throughput) perche' il reward e' diverso.

2. **Ablation study** per isolare l'effetto di ogni modifica:
   - `reward_original` (-P_i) vs `reward_custom` (attuale)
   - DQN standard vs Double DQN
   - Replay uniforme vs PER
   - Single-step vs BPTT+burn-in

3. **Riferimento paper**: travel time ~430-470s sul synthetic 4x4 per MetaSTGAT.

> [!WARNING]
> La differenza nel **reward** e' quella che rende piu' difficile il confronto diretto.
> Verificare **sempre** con travel time e throughput, non con la reward di training.

> [!NOTE]
> Le differenze interpretative (Meta-GAT formula, Meta-LSTM dimensioni) non sono
> necessariamente errori: il paper usa spesso notazione semplificata. Le scelte
> implementative fatte sono giustificabili e in linea con la letteratura.

---

*Documento generato il 07/09/2026, aggiornato il 11/9/2026 (vedi nota introduttiva).*
