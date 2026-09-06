# Indice della Tesi: Traffic Signal Control con Deep Reinforcement Learning

Di seguito una proposta per strutturare la tua tesi di laurea. Questa struttura segue un approccio accademico classico (Introduzione $\rightarrow$ Stato dell'Arte $\rightarrow$ Metodologia $\rightarrow$ Esperimenti $\rightarrow$ Conclusioni), ottimizzato per esaltare il lavoro svolto su **MetaSTGAT** e il confronto rigoroso con **CoLight**.

---

## 1. Introduzione
*Obiettivo: Introdurre il lettore al problema, motivare la ricerca e dichiarare i contributi della tesi.*
- **1.1 Contesto e Motivazioni:** L'importanza dell'ottimizzazione del traffico urbano (inquinamento, tempi di percorrenza, smart cities).
- **1.2 Definizione del Problema:** I limiti dei sistemi semaforici tradizionali (es. a tempo fisso) e le sfide del controllo adattivo su larga scala.
- **1.3 Contributi della Tesi:** Presentazione sintetica di *MetaSTGAT*, dell'integrazione di un simulatore realistico (CityFlow) e dell'analisi comparativa con modelli State-of-the-Art (SOTA) come *CoLight*.
- **1.4 Struttura della Tesi:** Breve riassunto dei capitoli successivi.

## 2. Background e Stato dell'Arte
*Obiettivo: Fornire le basi teoriche necessarie a comprendere il progetto e analizzare le soluzioni già esistenti.*
- **2.1 Reinforcement Learning (RL):** Concetti base (MDP, Agent, Environment, State, Action, Reward, Q-Learning).
- **2.2 Graph Neural Networks (GNN):** Introduzione ai grafi, Graph Convolutional Networks (GCN) e Graph Attention Networks (GAT).
- **2.3 Traffic Signal Control (TSC):**
  - Approcci tradizionali (Fixed-time, MaxPressure).
  - Approcci basati su RL Multi-Agente (MARL).
- **2.4 Analisi di CoLight:** Spiegazione dell'architettura di *CoLight* (Index-free neighborhood cooperation, Attention mechanism).

## 3. Metodologia: Formulazione del Problema
*Obiettivo: Descrivere come il problema del traffico reale è stato tradotto in un problema matematico risolvibile da un algoritmo.*
- **3.1 Il Simulatore (CityFlow):** Perché è stato scelto, come modella le reti stradali e la dinamica dei veicoli.
- **3.2 Formulazione dell'MDP (Markov Decision Process):**
  - **Spazio degli Stati (State):** Descrizione vettoriale a 32 dimensioni (Code, Tempi di attesa massimi, Fasi correnti).
  - **Spazio delle Azioni (Action):** Le 8 fasi semaforiche possibili e i vincoli (es. rotazione obbligatoria per evitare starvation).
  - **Funzione di Ricompensa (Reward):** Analisi profonda della nostra reward ibrida (Throughput - Max Wait Time - Penalità verde sprecato). Analisi delle differenze rispetto alle reward ingenue (es. solo code).

## 4. Architettura del Modello: MetaSTGAT
*Obiettivo: Cuore tecnico della tesi. Descrivere nel dettaglio il modello da te proposto.*
- **4.1 Panoramica dell'Architettura:** Struttura generale del modello (Spatio-Temporal Graph Attention).
- **4.2 Estrazione delle Feature:**
  - *Spatial Meta-Features:* Come l'agente osserva i vicini.
  - *Temporal Meta-Features:* Come viene mantenuto lo storico degli stati.
- **4.3 Apprendimento e Ottimizzazione:** Modello di learning, iperparametri e logica di aggiornamento dei pesi.

## 5. Setup Sperimentale
*Obiettivo: Descrivere l'ambiente e le metriche utilizzate per validare il modello.*
- **5.1 Reti Stradali e Flussi di Traffico:** Descrizione delle topologie (es. Griglia sintetica 4x4) e delle distribuzioni di traffico testate (es. 200m).
- **5.2 Baselines (Modelli di Confronto):**
  - Presentazione del setup per CoLight (Integrazione della repo *LibSignal*, adattamento al nostro MDP).
  - Motivazioni per l'equità del test (stessa reward e stesso stato).
- **5.3 Metriche di Valutazione:** *Average Travel Time* e *Throughput*.
- **5.4 Dettagli di Training:** Parametri di addestramento (Epsilon decay, learning rate, numero di episodi).

## 6. Risultati e Discussione
*Obiettivo: Dimostrare la superiorità o i trade-off di MetaSTGAT basandosi sui dati raccolti.*
- **6.1 Analisi delle Performance Complessive:**
  - Confronto diretto MetaSTGAT vs CoLight.
  - Discussione dei grafici (es. Curve di apprendimento per Travel Time e Throughput con Medie Mobili).
- **6.2 Stabilità e Convergenza:** Analisi della deviazione standard, velocità di raggiungimento della convergenza.
- **6.3 Interpretazione dei Comportamenti (Opzionale):** Discussione su come il modello gestisce situazioni specifiche (es. picchi di traffico).

## 7. Conclusioni e Sviluppi Futuri
*Obiettivo: Tirare le somme e guardare al futuro.*
- **7.1 Riepilogo dei Risultati:** Sintesi dei traguardi raggiunti.
- **7.2 Limiti dello Studio:** Eventuali limitazioni (es. assenza di pedoni nel simulatore, assenza di mezzi pubblici).
- **7.3 Sviluppi Futuri (Future Work):** Ideazioni per migliorare MetaSTGAT o estenderlo a topologie asimmetriche (reti cittadine reali).

## Bibliografia (References)

---

### 💡 Consigli per scriverla in LaTeX:
1. **Struttura a file:** Non scrivere tutto nel `main.tex`. Crea una cartella `chapters/` e metti un file `.tex` per ogni capitolo (es. `chapters/01_introduction.tex`), includendoli con il comando `\input{chapters/01_introduction}`.
2. **Algoritmi:** Usa il pacchetto `algorithm2e` per formattare elegantemente in pseudocodice la tua funzione di Reward o il loop di RL.
3. **Equazioni:** Le funzioni matematiche (come la Reward complessa) scritte in ambienti `\begin{equation} ... \end{equation}` aggiungono moltissimo rigore accademico.
4. **Grafici:** Salva i nostri grafici di comparazione in `.pdf` o `.png` ad alta risoluzione (300+ dpi, cosa che ho già impostato in `compare.py`!) e includili con `\begin{figure}`. Assicurati che i label degli assi siano grandi abbastanza.
5. **Tabelle:** Per tabelle comparative complesse, puoi usare generatori online come *TablesGenerator* per farti restituire il codice LaTeX corretto.
