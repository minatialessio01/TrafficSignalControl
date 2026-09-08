# Proposte di Raggruppamento per lo Studio di Ablation

> **Contesto**: Questo documento formalizza le proposte per lo **studio di ablazione (Ablation Study)** della tesi, basandosi sull'analisi comparativa tra il modello **MetaSTGAT Avanzato** (implementazione corrente) e il modello **MetaSTGAT Originale** del paper (*Wang et al., 2022*), dettagliata in [`metastgat_diff_analysis.md`](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/metastgat_diff_analysis.md).
>
> **Vincoli rispettati**:
> - Ogni proposta include un **massimo di 6 modelli** (incluso il modello Full).
> - **Tutte le modifiche introdotte** rispetto al paper originale sono mappate e rappresentate in modo coerente e scientificamente solido.
> - Vengono fornite **4 proposte alternative**, ciascuna con una specifica razionale metodologica, vantaggi, svantaggi e impatto per la stesura della tesi.

---

## 1. Mappatura Completa delle Modifiche rispetto al Paper

Prima di strutturare i raggruppamenti, ricapitoliamo le **8 modifiche concrete** apportate all'architettura e alla pipeline:

| ID | Componente | Paper Originale (Wang et al., 2022) | Nostro Modello Avanzato (Codice Tesi) | Razionale / Ruolo |
|:--:|:-----------|:------------------------------------|:--------------------------------------|:-------------------|
| **M1** | **Funzione di Reward** | Pura pressione: $r_i = -P_i$ (veicoli in ingresso - in uscita) | Throughput-based multi-obiettivo: `passed - incoming - alpha*wait - wasted_penalty`, clip `[-20, 5]` | Incentiva l'effettivo deflusso e previene lo stallo di corsie secondarie. |
| **M2** | **Vettore di Stato (`wait_vec`)** | Dim = 20: `[n_vec (12), p_vec (8)]` (solo conteggi veicoli e fase attiva) | Dim = 32: `[n_vec (12), wait_vec (12), p_vec (8)]` (aggiunto tempo max di attesa per corsia) | Fornisce alla rete l'urgenza temporale dei veicoli in coda. |
| **M3** | **Maschera Azioni (Anti-Starvation)** | Nessun vincolo: una fase può essere scelta indefinitamente | `get_invalid_actions`: maschera una fase se selezionata $\ge 2$ volte consecutive | Impedisce il fenomeno del "verde fisso infinito" tipico nelle prime fasi di training. |
| **M4** | **Temporal Training (BPTT + Burn-in)** | Single-step transitions: training DQN convenzionale senza sequenza | Approccio R2D2: sequenze $L=8$ con **burn-in=4** (hidden state warm-up) e BPTT su 4 step | Risolve il problema dell'*hidden state staleness* nella componente ricorrente Meta-LSTM. |
| **M5** | **Algoritmo Q-Learning (Double DQN)** | DQN Standard: $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ | Double DQN: $y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$ | Mitiga la sistematica sovrastima dei Q-value tipica del DQN classico. |
| **M6** | **Experience Replay (PER + IS)** | Buffer FIFO piatto (10.000), campionamento uniforme | Prioritized Experience Replay (2.400) con pesi Importance Sampling (IS) e $\beta$-annealing | Campiona con frequenza maggiore le transizioni con alto TD-error. |
| **M7** | **Stabilizzatori di Training** | Loss MSE ($L2$), hard target update, no gradient clipping | Huber Loss (Smooth L1) pesata, Soft Polyak Update ($\tau=0.01$), Grad Clip (`max_norm=1.0`) | Previene esplosioni del gradiente e rende l'apprendimento monotonicamente più stabile. |
| **M8** | **Regolarizzazione Meta-Learner (Tanh)** | MLP lineare senza attivazione delimitata per gli embedding | `nn.Tanh()` finale su SMK e TMK (range `[-1, 1]`) | Mantiene limitati i pesi sintetizzati dalle hypernetwork Meta-GAT e Meta-LSTM. |

*(Nota: Le formalizzazioni matematiche necessarie per il funzionamento delle GNN/LSTM, come i 4 gate dell'LSTM e la matrice di proiezione $W \cdot Q$, rimangono attive in tutti i modelli in quanto implementazioni necessarie del paper).*

---

## 2. Proposta 1: Raggruppamento per "Sottosistemi Funzionali" (Macro-Aree Logiche)

### Concetto Guida
Si raggruppano le 8 modifiche in **4 macro-aree funzionali ad alta coesione**. Ciascun modello di ablazione disattiva un intero sottosistema per quantificare il valore aggiunto di quell'area logica rispetto al paper.

### Composizione del Gruppo (6 Modelli)

| # | Nome Modello | Modifiche Attive | Modifiche Rimosse (Disattivate) | Domanda Scientifica a cui Risponde |
|:--|:-------------|:-----------------|:--------------------------------|:-----------------------------------|
| **1** | **MetaSTGAT-Full** | Tutte (M1..M8) | Nessuna (Modello completo) | Benchmark di riferimento delle massime prestazioni raggiungibili. |
| **2** | **Abl-MDP** *(Environment)* | M4, M5, M6, M7, M8 | **M1, M2, M3** (Torna a reward $-P_i$, stato 20 dim senza `wait_vec`, no action mask) | *Quanto incide la nuova formulazione dell'ambiente/MDP (reward + stato arricchito + anti-starvation) sulle prestazioni?* |
| **3** | **Abl-Temporal** *(Recurrence)* | M1, M2, M3, M5, M6, M7, M8 | **M4** (Torna a training single-step senza BPTT né burn-in) | *L'addestramento R2D2 (BPTT sequenziale + burn-in) è davvero necessario per sfruttare la memoria LSTM, o basta il training standard?* |
| **4** | **Abl-RLCore** *(Value Estimation)* | M1, M2, M3, M4, M6, M7, M8 | **M5** (Torna a DQN standard invece di Double DQN) | *Quanto giova l'uso di Double DQN nell'eliminare la sovrastima dei Q-value in questo ambiente di traffico?* |
| **5** | **Abl-Optimization** *(Sampling & Stability)* | M1, M2, M3, M4, M5 | **M6, M7, M8** (Torna a Replay uniforme 10k, MSE loss, hard target update, no grad clip) | *L'insieme delle tecniche di campionamento prioritario (PER) e stabilizzazione numerica fa una reale differenza sui tempi di convergenza?* |
| **6** | **MetaSTGAT-Paper** | Nessuna | **Tutte (M1..M8)** | Replicazione fedele del modello originale di Wang et al. (2022). |

### Vantaggi e Svantaggi per la Tesi
- **Vantaggi**:
  - Copre esattamente il 100% delle modifiche senza lasciarne fuori nessuna.
  - Narrazione accademica molto lineare: capitoli dedicati a *"Modellazione MDP"*, *"Dinamica Temporale Recurrent"*, *"Algoritmo di Controllo"* e *"Stabilizzazione del Training"*.
  - Include direttamente il confronto con il modello del Paper all'interno dei 6 modelli.
- **Svantaggi**:
  - In `Abl-MDP` e `Abl-Optimization` sono aggregate più modifiche assieme: se una di queste componenti crea un effetto opposto a un'altra, l'effetto netto potrebbe nascondere dettagli secondari.

---

## 3. Proposta 2: Raggruppamento "Leave-One-Out ad Alto Impatto" (Isolamento Puntuale)

### Concetto Guida
Nello standard dei paper di Deep RL (es. ICLR / NeurIPS), la prassi metodologica più rigorosa è il **Leave-One-Out (LOO)**: si parte dal modello migliore (Full) e si disattiva **un singolo componente isolato alla volta**.
Per non superare i 6 modelli, le tecniche di regolarizzazione di base (M7: Huber/GradClip e M8: Tanh) e l'Action Masking (M3) vengono considerate standard infrastrutturali comuni e mantenute attive come "baseline robusta", mentre si testano i 4 veri "motori" algoritmici.

### Composizione del Gruppo (6 Modelli)

| # | Nome Modello | Modifica Disattivata | Descrizione Configurazione | Domanda Scientifica a cui Risponde |
|:--|:-------------|:---------------------|:---------------------------|:-----------------------------------|
| **1** | **MetaSTGAT-Full** | Nessuna | Modello Avanzato completo (M1..M8) | Benchmark di riferimento. |
| **2** | **Abl-NoCustomReward** | **M1** (Reward) | Usa la reward originale $-P_i$ (pressione), ma mantiene stato dim=32, Double DQN, PER, BPTT | *La nuova funzione di reward throughput-based supera davvero la reward di pressione originale a parità di rete e training?* |
| **3** | **Abl-NoWaitState** | **M2** (Stato) | Rimuove `wait_vec`: stato a dimensione 20 (solo veicoli e fase), tutto il resto invariato | *L'informazione del tempo di attesa nello stato apporta effettivo valore decisionale alla rete neurale?* |
| **4** | **Abl-NoDoubleDQN** | **M5** (DQN) | Usa Vanilla DQN standard (target max), tutto il resto invariato | *Qual è l'apporto isolato del meccanismo Double DQN sul controllo del traffico?* |
| **5** | **Abl-NoBPTT** | **M4** (BPTT) | Rimuove BPTT su sequenze e burn-in: training su singole transizioni ($L=1$) | *Quanto degrada la componente temporale Meta-LSTM se addestrata senza finestre sequenziali di BPTT?* |
| **6** | **Abl-NoPER** | **M6** (Replay) | Replay buffer uniforme standard (no TD-priority, no IS weights) | *Il campionamento prioritario (PER) accelera o migliora le performance rispetto al campionamento casuale uniforme?* |

### Vantaggi e Svantaggi per la Tesi
- **Vantaggi**:
  - **Massima purezza scientifica**: nessun problema di attribuzione ("attribution bias"). Qualsiasi variazione di travel time è attribuibile al 100% a quel singolo componente.
  - Tabelle e grafici a barre chiarissimi nella discussione dei risultati: un istogramma in cui ogni barra mostra la perdita di prestazione causata dalla rimozione di quello specifico componente.
- **Svantaggi**:
  - M7 (stabilizzatori numerici) e M3 (maschera) rimangono attivi come parte integrante dell'infrastruttura condivisa e non vengono spenti individualmente (anche se possono essere discussi come *default engineering choices*).

---

## 4. Proposta 3: Raggruppamento "Ingegneria del Dominio vs Algoritmi di Intelligenza Artificiale"

### Concetto Guida
Separa chiaramente le modifiche in base alla loro natura concettuale:
1. **Ingegneria del Dominio (Traffic & Transportation Engineering)**: cosa abbiamo detto alla rete sul problema del traffico (Reward, Stato con attese, Vincolo sulle fasi consecutive).
2. **Intelligenza Artificiale (Deep RL & Machine Learning)**: come la rete impara ed elabora le informazioni (Double DQN, Recurrent Temporal Training, Prioritized Experience Replay).

### Composizione del Gruppo (6 Modelli)

| # | Nome Modello | Settore Ablato | Dettaglio Modifiche | Valore per la Tesi |
|:--|:-------------|:---------------|:--------------------|:-------------------|
| **1** | **MetaSTGAT-Full** | Nessuno | Sistema completo al 100% | Configurazione di punta della tesi. |
| **2** | **Abl-TrafficEnv** | Dominio Traffico (M1, M2, M3) | Ambiente Paper puro (stato 20, reward $-P_i$, no mask) + Motore RL Avanzato | Risponde a: *"Se applichiamo il Deep RL più moderno ma manteniamo l'ambiente grezzo del paper, quanto otteniamo?"* |
| **3** | **Abl-RecurrentLearning** | Apprendimento Temporale (M4) | Rimuove BPTT sequenziale e burn-in (training single-step) | Risponde a: *"La gestione esplicita della memoria a lungo termine tramite BPTT è il fattore determinante per l'LSTM?"* |
| **4** | **Abl-ValueEstimator** | Teoria RL (M5) | Sostituisce Double DQN con Vanilla DQN | Risponde a: *"L'overestimation bias compromette la convergenza dei semafori?"* |
| **5** | **Abl-ExperienceMemory** | Memoria e Campionamento (M6, M7) | Sostituisce PER con buffer piatto uniforme e loss MSE | Risponde a: *"Il replay uniforme standard è sufficiente per compiti cooperativi multi-incrocio?"* |
| **6** | **MetaSTGAT-Paper** | Entrambi i settori (M1..M8) | Ripristina il modello originale del paper | Baseline di confronto originaria da superare. |

### Vantaggi e Svantaggi per la Tesi
- **Vantaggi**:
  - Molto apprezzata dai docenti e dalle commissioni di laurea perché dimostra che la tesi non è solo un esercizio di programmazione, ma affronta criticamente sia l'**ingegneria del problema reale (trasporti)** sia la **teoria del machine learning**.
  - Permette di trarre conclusioni chiare su quale delle due aree porti il maggiore beneficio.
- **Svantaggi**:
  - `Abl-ExperienceMemory` accoppia PER e stabilizzatori, pur essendo entrambi appartenenti all'area del campionamento e dell'ottimizzazione.

---

## 5. Proposta 4: Raggruppamento "Incrementale a Stadi" (Evoluzione Cumulativa dal Paper al Full)

### Concetto Guida
Invece di procedere per sottrazione (top-down), si dimostra l'evoluzione **bottom-up**: si parte dal Paper originale e si aggiunge un "blocco di innovazione" alla volta fino a completare il modello Full. Questo approccio è ideale per grafici temporali e curve a gradini ("progressione delle prestazioni").

### Composizione del Gruppo (5 Modelli)

```
[M0: Paper Originale]
       ↓ + (Reward throughput + wait_vec + action mask)
[M1: + Nuova Modellazione Traffico (MDP)]
       ↓ + (BPTT su sequenze L=8 + Burn-in=4)
[M2: + Addestramento Temporale Recurrent]
       ↓ + (Double DQN + Huber Loss + Polyak update)
[M3: + Stabilizzazione RL e Value Target]
       ↓ + (Prioritized Experience Replay - PER)
[M4: MetaSTGAT Avanzato Completo]
```

| # | Modello | Descrizione Tecnica Cumulativa | Delta introdotto rispetto al precedente |
|:--|:--------|:-------------------------------|:----------------------------------------|
| **1** | **Stage-0 (Paper Base)** | Il modello originale del paper (Stato 20, reward $-P_i$, no mask, DQN standard, buffer uniforme, single-step). | Baseline di partenza. |
| **2** | **Stage-1 (+ MDP Design)** | Stage-0 + **M1 (Reward), M2 (wait_vec), M3 (Action Mask)**. | Misura il guadagno della sola ridefinizione del problema di traffico. |
| **3** | **Stage-2 (+ Recurrence)** | Stage-1 + **M4 (BPTT sequenziale $L=8$ + Burn-in=4)**. | Aggiunge la corretta dinamica temporale per l'LSTM. |
| **4** | **Stage-3 (+ RL Engine)** | Stage-2 + **M5 (Double DQN) + M7 (Huber Loss, Soft Update)**. | Aggiunge l'algoritmo di stima del valore avanzato e la stabilità dei gradienti. |
| **5** | **Stage-4 (Full Model)** | Stage-3 + **M6 (Prioritized Experience Replay)** + **M8 (Tanh)**. | Modello finale avanzato completo. |

### Vantaggi e Svantaggi per la Tesi
- **Vantaggi**:
  - Conta solo **5 modelli** (risparmio di tempo di calcolo e GPU!).
  - Costruisce un "filo rosso" narrativo perfetto nella tesi: ogni capitolo dimostra un miglioramento incrementale del travel time ($Stage_0 \to Stage_1 \to Stage_2 \to Stage_3 \to Stage_4$).
  - Dimostra che ogni singola scelta progettuale è stata aggiunta per risolvere un limite tangibile della versione precedente.
- **Svantaggi**:
  - Non è un'ablazione "pura" di tipo Leave-One-Out (non isola l'effetto di un componente da solo sul modello full, ma ne misura l'impatto progressivo).

---

## 6. Tabella Comparativa delle 4 Proposte

| Criterio | Proposta 1 (Sottosistemi) | Proposta 2 (Leave-One-Out) | Proposta 3 (Dominio vs AI) | Proposta 4 (Incrementale a Stadi) |
|:---------|:-------------------------:|:--------------------------:|:--------------------------:|:---------------------------------:|
| **Numero di Modelli** | 6 modelli | 6 modelli | 6 modelli | **5 modelli** |
| **Copertura Modifiche** | 100% esaustiva | 100% (con base comune) | 100% esaustiva | 100% cumulativa |
| **Rigorosità Scientifica** | Alta | **Massima (Top-tier)** | Alta | Molto buona |
| **Facilità di Spiegazione in Tesi** | Molto alta | Alta | **Eccellente** | **Intuitiva al 100%** |
| **Isolamento dei Singoli Effetti** | Medio-Alto | **Puro (Singolo fattore)** | Medio-Alto | Incrementale |
| **Presenza Baseline Paper** | Sì (inclusa) | Modello Full al centro | Sì (inclusa) | Sì (punto di partenza) |

---

## 7. Raccomandazione Finale

Per massimizzare l'impatto accademico della tesi in base ai tuoi obiettivi:

1. **Se vuoi il massimo rigore scientifico e isolamento causale**: Scegli la **Proposta 2 (Leave-One-Out)**. Permette di affermare senza ombra di dubbio: *"La rimozione di X dal modello completo ha causato un degrado del Y% nel travel time"*.
2. **Se vuoi la massima chiarezza concettuale e coesione tematica**: Scegli la **Proposta 1 (Sottosistemi Funzionali)**. Suddivide perfettamente l'analisi tra Ambiente, Ricorrenza, Algoritmo RL e Ottimizzazione.
3. **Se vuoi ottimizzare i tempi di calcolo con una narrazione fluida**: Scegli la **Proposta 4 (Incrementale)**. Con soli 5 modelli mostri l'evoluzione passo-passo che giustifica l'intero lavoro di tesi.
