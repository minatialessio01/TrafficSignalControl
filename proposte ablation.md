# Proposte di Raggruppamento per lo Studio di Ablation

> **Contesto**: Questo documento formalizza le proposte per lo **studio di ablazione (Ablation Study)** della tesi, basandosi sull'analisi comparativa tra il modello **MetaSTGAT Avanzato** (implementazione corrente) e il modello **MetaSTGAT Originale** del paper (*Wang et al., 2022*), dettagliata in [`metastgat_diff_analysis.md`](file:///c:/Users/user/Documents/Antigravity/CodiceTesi/metastgat_diff_analysis.md).
>
> **Vincoli rispettati**:
> - Ogni proposta include un **massimo di 6 modelli** (incluso il modello Full).
> - **Tutte le modifiche introdotte** rispetto al paper originale (incluse le 9 note di implementazione specifiche) sono mappate e rappresentate in modo coerente e scientificamente solido.
> - Vengono fornite **4 proposte alternative**, ciascuna con una specifica razionale metodologica, vantaggi, svantaggi e impatto per la stesura della tesi.

---

## 1. Mappatura Completa di Tutte le Modifiche rispetto al Paper

Ecco il censimento sistematico di tutte le modifiche introdotte nel progetto rispetto al modello base descritto in letteratura (*Wang et al., 2022*):

| ID | Modifica / Meccanismo | Paper Originale (Wang et al., 2022) | Nostro Modello Avanzato (Codice Tesi) | Razionale / Ruolo Metodologico |
|:--:|:----------------------|:------------------------------------|:--------------------------------------|:-------------------------------|
| **M1** | **Reward Multi-Obiettivo Pesata** | Pura pressione: $r_i = -P_i$ (veicoli in ingresso - in uscita) | Weighted Pressure: throughput passato - veicoli entranti - $\alpha \cdot \text{max\_red\_wait}$ - penalità verde a vuoto (-50), normalizzata /100 e clippata in `[-20, 5]` | Incentiva l'effettivo deflusso, previene lo stallo di corsie secondarie e punisce fasi verdi concesse a corsie vuote. |
| **M2** | **Campo Visivo Limitato (Cutoff 167m)** | Visibilità teorica infinita (tutta la lunghezza della strada) | Solo veicoli entro 167m dal semaforo (`cutoff = max(0, road_len - 167m)`) sia per conteggi che per code | Simula la portata reale dei sensori/telecamere fisiche all'incrocio ed evita di penalizzare veicoli lontani centinaia di metri. |
| **M3** | **Stato Concatenato a 3 Componenti** | Dim = 20: `[n_vec (12) \|\| p_vec (8)]` (solo conteggi veicoli e fase attiva) | Dim = 32: `[n_vec (12) \|\| wait_vec (12) \|\| p_vec (8)]` con tempi massimi di attesa normalizzati (0-1) | Fornisce alla rete l'urgenza temporale dei veicoli fermi in coda oltre alla loro semplice presenza. |
| **M4** | **Maschera Azioni Anti-Starvation** | Nessun vincolo: una fase può essere mantenuta all'infinito | `get_invalid_actions`: maschera una fase se selezionata $\ge 2$ volte consecutive (evita la stessa fase per 3 o più volte di fila) | Impedisce il collasso della policy nel "verde fisso permanente", forzando una rotazione minima delle fasi. |
| **M5** | **Aggiornamento Pesi LSTM & BPTT con Burn-in** | Single-step DQN standard (no BPTT, transizioni isolate, stato LSTM azzerato o fisso) | Training R2D2 su sequenze $L=8$: **Burn-in di 4 step** (allineamento di $h, c$ senza gradiente) + BPTT sui 4 step successivi; pesi generati per tutti e 4 i gate ($f, i, o, c$) | Risolve l'*hidden state staleness*, garantendo che la memoria temporale della LSTM sia fisicamente consistente durante l'aggiornamento. |
| **M6** | **Double DQN (Stima del Valore)** | DQN Standard: $y = r + \gamma \max_{a'} Q(s', a'; \theta^-)$ | Double DQN: $y = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)$ | Elimina la sistematica sovrastima dei Q-value tipica dell'operatore max nel Q-learning classico. |
| **M7** | **Experience Replay (PER con IS) & Buffer Size** | Buffer uniforme FIFO piatto da 10.000 transizioni singole | Prioritized Experience Replay (2.400 sequenze) con campionamento pesato su TD-error e correzione Importance Sampling (IS weights con $\beta$-annealing) | Campiona con frequenza maggiore le transizioni più informative/critiche correggendo il bias con i pesi IS. |
| **M8** | **Huber Loss con Gradient Clipping & Soft Update** | MSE Loss ($L2$), nessun clipping, hard update periodico | Smooth L1 (Huber Loss) pesata da IS, Gradient Clipping (`max_norm=1.0`), Soft Polyak Update ($\tau=0.01$) | Stabilizza numericamente la convergenza, proteggendo la rete da gradienti esplosivi in presenza di picchi di traffico. |
| **M9** | **Esplorazione Ciclica e Warmup** | Epsilon decay passivo indefinito senza warmup | **Warm-up**: 10 episodi iniziali random ($\epsilon=1.0$) per pre-popolare il buffer; **Ciclica**: 1 episodio random forzato ogni 10 episodi per rompere minimi locali | Evita aggiornamenti su buffer vuoto/povero e garantisce l'esplorazione di stati rari anche a training avanzato. |
| **M10**| **Regolarizzazione Meta-Learner (Tanh)** | MLP lineare senza attivazione delimitata | `nn.Tanh()` finale per confinare gli embedding dei pesi sintetizzati in `[-1, 1]` | Previene la divergenza dei parametri generati dalle hypernetwork Meta-GAT e Meta-LSTM. |

---

## 2. Proposta 1: Raggruppamento per "Sottosistemi Funzionali" (Macro-Aree Logiche)

### Concetto Guida
Le 10 modifiche vengono aggregate in **4 macro-aree funzionali omogenee**. Ciascun modello di ablazione disattiva un intero sottosistema per quantificare il valore aggiunto di quell'area logica rispetto al paper.

### Composizione del Gruppo (6 Modelli)

| # | Nome Modello | Modifiche Attive | Modifiche Disattivate (Ablate) | Domanda Scientifica a cui Risponde |
|:--|:-------------|:-----------------|:-------------------------------|:-----------------------------------|
| **1** | **MetaSTGAT-Full** | Tutte (M1..M10) | Nessuna (Modello avanzato completo) | Benchmark di riferimento delle massime prestazioni raggiungibili. |
| **2** | **Abl-Environment** *(MDP & Sensori)* | M5..M10 | **M1, M2, M3, M4** (Torna a reward pura pressione $-P_i$, visibilità infinita, stato a 20 dim senza `wait_vec`, no action mask) | *Quanto incide l'ingegnerizzazione dell'ambiente (reward pesata, cutoff sensori 167m, wait times e anti-starvation) sul controllo del traffico?* |
| **3** | **Abl-TemporalLSTM** *(Recurrence & BPTT)* | M1..M4, M6..M10 | **M5** (Torna a training single-step senza BPTT né burn-in; LSTM senza warm-up dinamico) | *L'addestramento R2D2 (BPTT sequenziale + burn-in per allineare l'hidden state) è indispensabile per valorizzare la memoria LSTM?* |
| **4** | **Abl-RLCore** *(Value Estimation)* | M1..M5, M7..M10 | **M6** (Torna a DQN standard invece di Double DQN) | *Quanto incide la sovrastima dei Q-value sulle decisioni semaforiche agli incroci?* |
| **5** | **Abl-ReplayStability** *(Sampling & Ottimizzazione)* | M1..M6 | **M7, M8, M9, M10** (Buffer uniforme 10k senza PER/IS, MSE loss senza grad clip, no warmup/esplorazione ciclica) | *L'infrastruttura di memoria (PER + IS), l'ottimizzazione robusta (Huber + clip) e il regime di esplorazione garantiscono una convergenza superiore?* |
| **6** | **MetaSTGAT-Paper** | Nessuna | **Tutte (M1..M10)** | Replicazione fedele dell'architettura e della procedura del paper originale. |

### Vantaggi e Svantaggi
- **Vantaggi**: Copertura al 100% di tutte le modifiche; narrazione per la tesi impeccabile (capitoli: Ambiente/MDP, Ricorrenza Temporale, Algoritmo RL, Stabilità & Memoria).
- **Svantaggi**: I blocchi Ambiente e ReplayStability aggregano più componenti strettamente cooperanti.

---

## 3. Proposta 2: Raggruppamento "Leave-One-Out ad Alto Impatto" (Isolamento Puntuale)

### Concetto Guida
Lo standard scientifico dei paper di riferimento (NeurIPS, ICLR, AAAI) prevede di testare il **Leave-One-Out (LOO)** sui singoli fattori chiave. Gli stabilizzatori di base (M8: Huber/clip, M9: warmup, M10: Tanh, M4: maschera) vengono mantenuti come standard condiviso di robustezza, mentre si isolano i 4 grandi driver metodologici.

### Composizione del Gruppo (6 Modelli)

| # | Nome Modello | Modifica Disattivata | Descrizione Configurazione | Domanda Scientifica a cui Risponde |
|:--|:-------------|:---------------------|:---------------------------|:-----------------------------------|
| **1** | **MetaSTGAT-Full** | Nessuna | Modello Avanzato completo (M1..M10) | Benchmark di riferimento. |
| **2** | **Abl-NoCustomReward** | **M1, M2** (Reward & Visibilità) | Torna alla pressione pura $-P_i$ e campo visivo globale, mantenendo stato a 32 dim, Double DQN, PER, BPTT | *La formulazione della reward multi-obiettivo con visibilità limitata a 167m supera la pressione teorica pura a parità di rete?* |
| **3** | **Abl-NoWaitState** | **M3** (Stato) | Rimuove `wait_vec`: stato a 20 dim (solo veicoli e fase), tutto il resto invariato | *L'informazione sul tempo massimo di attesa delle corsie apporta un reale vantaggio decisionale?* |
| **4** | **Abl-NoBPTT-LSTM** | **M5** (BPTT & Burn-in) | Training standard single-step ($L=1$ senza burn-in per la LSTM) | *Qual è l'impatto dell'hidden state staleness sulla capacità di previsione temporale della rete?* |
| **5** | **Abl-NoDoubleDQN** | **M6** (DQN) | Q-learning standard con target max, tutto il resto invariato | *Qual è l'apporto netto del Double DQN nel prevenire la divergenza delle stime di valore?* |
| **6** | **Abl-NoPER** | **M7** (PER & IS) | Buffer con campionamento uniforme (senza priorità TD e senza pesi IS) | *Il campionamento prioritario (PER) con pesi IS produce traiettorie di apprendimento migliori rispetto al replay casuale?* |

### Vantaggi e Svantaggi
- **Vantaggi**: **Massima purezza e rigore accademico**. Non esiste ambiguità di attribuzione: il calo di prestazioni è dovuto esclusivamente al singolo fattore disattivato.
- **Svantaggi**: Lascia invariati gli aspetti di ottimizzazione numerica (Huber, grad clip, warmup), considerati "buone pratiche ingegneristiche" di default.

---

## 4. Proposta 3: Raggruppamento "Ingegneria del Dominio vs Algoritmi di Intelligenza Artificiale"

### Concetto Guida
Separa chiaramente i contributi tra:
1. **Ingegneria del Problema di Traffico (Domain Engineering)**: Reward pesata, visibilità limitata dei sensori, attese nello stato, vincolo di rotazione fasi.
2. **Ingegneria del Machine Learning (Deep RL & Architecture)**: BPTT + Burn-in, Double DQN, PER con pesi IS, stabilizzatori di training.

### Composizione del Gruppo (6 Modelli)

| # | Nome Modello | Settore Disattivato | Dettaglio Tecnico | Valore Aggiunto per la Tesi |
|:--|:-------------|:--------------------|:------------------|:----------------------------|
| **1** | **MetaSTGAT-Full** | Nessuno | Sistema completo al 100% | Configurazione di punta della tesi. |
| **2** | **Abl-TrafficDomain** | Dominio Traffico (M1, M2, M3, M4) | Ambiente Paper originale (stato 20, reward $-P_i$, visibilità globale, no mask) + Motore RL Avanzato completo | *Se applichiamo il Deep RL più evoluto ma manteniamo la modellazione del traffico grezza del paper, quanto si perde?* |
| **3** | **Abl-RecurrentDynamics** | Dinamica LSTM (M5) | Rimuove BPTT su sequenze e burn-in (training single-step per LSTM) | *La gestione esplicita della continuità temporale tramite BPTT è il fattore abilitante per la Meta-LSTM?* |
| **4** | **Abl-ValueEstimator** | Teoria RL (M6) | Sostituisce Double DQN con Vanilla DQN | *Quanto pesa il bias di sovrastima nelle decisioni coordinate della rete?* |
| **5** | **Abl-MemoryAndExploration**| Memoria & Esplorazione (M7, M8, M9) | Rimuove PER/IS (buffer uniforme piatto), MSE loss, no warmup/esplorazione ciclica | *Quanto contano le strategie avanzate di replay e di diversificazione dell'esplorazione per evitare minimi locali?* |
| **6** | **MetaSTGAT-Paper** | Entrambi i settori (M1..M10) | Ripristina il modello originale del paper | Benchmark originario di partenza da superare. |

### Vantaggi e Svantaggi
- **Vantaggi**: Divide in modo eccellente il lavoro della tesi tra "comprensione dei trasporti urbani" e "innovazione metodologica di intelligenza artificiale".
- **Svantaggi**: Richiede di spiegare bene la distinzione concettuale tra le due sfere.

---

## 5. Proposta 4: Raggruppamento "Incrementale a Stadi" (Evoluzione Cumulativa dal Paper al Full)

### Concetto Guida
Approccio *bottom-up*: si parte dal modello originale del Paper e si aggiunge un livello di innovazione alla volta. Questo approccio è ideale per mostrare una curva a gradini delle prestazioni dove ogni aggiunta dimostra una riduzione sistematica del Travel Time.

### Composizione del Gruppo (5 Modelli)

```
[Stage 0: MetaSTGAT Paper Originale]
       ↓ + (Reward weighted pressure + Cutoff 167m + wait_vec + Anti-starvation mask)
[Stage 1: + Modellazione del Problema di Traffico (MDP)]
       ↓ + (BPTT su sequenze L=8 + Burn-in=4 + Warmup pesi LSTM)
[Stage 2: + Addestramento Temporale Recurrent]
       ↓ + (Double DQN + Huber Loss + Gradient Clipping + Soft Target Update)
[Stage 3: + Algoritmo RL Robusto & Stima del Valore]
       ↓ + (Prioritized Experience Replay con IS + Esplorazione Ciclica)
[Stage 4: MetaSTGAT Avanzato Completo]
```

| # | Modello | Descrizione Tecnica Cumulativa | Innovazione rispetto allo stadio precedente |
|:--|:--------|:-------------------------------|:--------------------------------------------|
| **1** | **Stage-0 (Paper Base)** | Il modello originale del paper (Stato 20, reward $-P_i$, visibilità infinita, no mask, DQN standard, buffer uniforme, single-step). | Baseline di partenza di Wang et al. |
| **2** | **Stage-1 (+ Traffic Design)** | Stage-0 + **M1 (Weighted Reward), M2 (Cutoff 167m), M3 (wait_vec), M4 (Anti-starvation)**. | Isola il guadagno derivante dalla sola corretta formulazione del traffico urbano. |
| **3** | **Stage-2 (+ Recurrent Dynamics)** | Stage-1 + **M5 (BPTT $L=8$ + Burn-in=4 + Allineamento hidden state LSTM)**. | Integra la reale comprensione delle serie storiche temporali. |
| **4** | **Stage-3 (+ RL Engine & Stability)** | Stage-2 + **M6 (Double DQN) + M8 (Huber Loss, Grad Clip, Soft Update)**. | Elimina la sovrastima e stabilizza i gradienti. |
| **5** | **Stage-4 (Full Model)** | Stage-3 + **M7 (PER con pesi IS) + M9 (Warmup & Esplorazione Ciclica) + M10 (Tanh)**. | Modello finale avanzato completo. |

### Vantaggi e Svantaggi
- **Vantaggi**:
  - Conta solo **5 modelli** (massima efficienza nei tempi di calcolo!).
  - Costruisce una progressione narrativa naturale: ogni stadio risolve un collo di bottiglia specifico del precedente.
- **Svantaggi**:
  - L'effetto di ogni blocco è misurato in sequenza cumulativa, non isolato in senso "Leave-One-Out".

---

## 6. Tabella di Raffronto delle 4 Proposte

| Criterio | Proposta 1 (Sottosistemi) | Proposta 2 (Leave-One-Out) | Proposta 3 (Dominio vs AI) | Proposta 4 (Incrementale a Stadi) |
|:---------|:-------------------------:|:--------------------------:|:--------------------------:|:---------------------------------:|
| **Numero Totale Modelli** | 6 modelli | 6 modelli | 6 modelli | **5 modelli** |
| **Copertura Modifiche** | 100% esaustiva | 100% (con base comune) | 100% esaustiva | 100% cumulativa |
| **Rigorosità Scientifica** | Molto alta | **Massima (Top-tier)** | Molto alta | Ottima |
| **Narrativa della Tesi** | Lineare per macro-aree | Analitica fattore per fattore | **Ideale per commissione mista** | **Evoluzione passo-passo** |
| **Isolamento Effetti** | Per sottosistema | **Singolo fattore puro** | Per macro-settore | Incrementale cumulativo |
| **Costo Computazionale** | Standard (6 run) | Standard (6 run) | Standard (6 run) | **Minimo (5 run)** |

---

## 7. Raccomandazione per la Tesi

- Se intendi dimostrare che il lavoro ha seguito il metodo scientifico più rigoroso per isolare le singole componenti: **Scegli la Proposta 2 (Leave-One-Out)**.
- Se vuoi strutturare i capitoli della tesi in modo pulito e bilanciato tra trasporti e machine learning: **Scegli la Proposta 1 o la Proposta 3**.
- Se vuoi ridurre i tempi di addestramento mantenendo una narrazione impeccabile dell'evoluzione del progetto: **Scegli la Proposta 4**.
