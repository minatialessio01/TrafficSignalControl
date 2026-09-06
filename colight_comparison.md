# Confronto: CoLight Paper originale vs Nostra Implementazione

Analizzando il paper *CoLight: Learning Network-level Cooperation for Traffic Signal Control* (Sezione 3) e il nostro codice (`src/environment/cityflow_env.py` e il wrapper che abbiamo creato), emergono diverse differenze chiave nel modo in cui l'agente "vede" il mondo e viene ricompensato.

Poiché abbiamo trapiantato la rete neurale pura di CoLight all'interno del **nostro** ambiente CityFlow (quello progettato per MetaSTGAT), il modello CoLight sta di fatto imparando regole molto diverse rispetto all'esperimento originale.

Ecco un riepilogo delle differenze sostanziali:

### 1. Spazio delle Osservazioni (Stato)
- **Paper originale**: L'osservazione $o_i^t$ per ogni agente è composta esclusivamente da due elementi:
  1. Fase semaforica corrente (vettore one-hot).
  2. Numero di veicoli presenti su ogni corsia in ingresso.
- **Nostra Implementazione**: Il nostro stato è un vettore a **32 dimensioni** molto più ricco:
  1. Fase semaforica corrente (8 bit one-hot).
  2. Numero di veicoli (`n_vec`) per le 12 corsie (filtrato entro i 167 metri dal semaforo e normalizzato diviso 30).
  3. **Tempo massimo di attesa** (`wait_vec`) per ognuna delle 12 corsie (calcolato sui veicoli quasi fermi e normalizzato diviso 100).
- **Impatto**: Il "nostro" CoLight è più avvantaggiato rispetto a quello del paper perché riceve in input l'informazione critica sui tempi di attesa dei veicoli, permettendogli di capire da quanto tempo le auto sono ferme, cosa che il CoLight originale non poteva fare.

### 2. Funzione di Reward (Ricompensa)
- **Paper originale**: Usa una metrica estremamente semplice basata sulla lunghezza delle code. 
  La reward è la somma negativa delle lunghezze delle code sulle corsie in ingresso: $r_i^t = -\sum_l u_{i,l}^t$.
- **Nostra Implementazione**: Utilizziamo una metrica composita e molto più avanzata, basata su un ibrido tra pressione e tempi di attesa:
  $r_i^t = (\text{Veicoli Usciti} - \text{Veicoli Entrati}) - (\alpha \times \text{Tempo Massimo di Attesa col Rosso}) - \text{Penalità Verde Sprecato}$
  Il tutto viene poi normalizzato diviso 100 e clippato in un range tra -20.0 e +5.0.
- **Impatto**: Il CoLight del paper cerca banalmente di minimizzare le auto in coda. Il nostro cerca attivamente di *massimizzare il throughput* (pressione), punendo contemporaneamente attese estreme e penalizzando le inefficienze (verde dato su corsie vuote).

### 3. Dinamica delle Azioni (Policy)
- **Paper originale**: Ad ogni step temporale $\Delta t$, l'agente sceglie liberamente una fase $p$ tra quelle disponibili.
- **Nostra Implementazione**: Limitiamo esplicitamente l'agente tramite una maschera di *Invalid Actions*. Se una fase semaforica è già stata mantenuta per 2 turni consecutivi, il nostro ambiente forza il modello a cambiarla.
- **Impatto**: La nostra versione evita la starvation perpetua (un verde infinito su un asse molto trafficato) imponendo una rotazione obbligatoria. Il paper lascia questa esplorazione esclusivamente alla rete neurale.

### 4. Ciclo e Transizione Semaforica
- **Paper originale**: Tra una fase verde e l'altra viene inserito un tempo di giallo di 3 secondi e un "all red" time di 2 secondi.
- **Nostra Implementazione**: Abbiamo lo stesso offset (3s giallo + 2s rosso per transizione), ma nel nostro environment "fingiamo" il giallo trattenendo la vecchia fase ma non facendo partire nuove auto. 

### Conclusione
Il CoLight che abbiamo addestrato usa l'**architettura neurale esatta** (Graph Attention Networks per la comunicazione con i vicini) decritta dagli autori in LibSignal, ma la applica a un **Problema di Ottimizzazione più difficile e ricco** (lo stesso MDP di MetaSTGAT). Questo garantisce che il paragone che abbiamo appena plottato tra MetaSTGAT e CoLight sia assolutamente *ad armi pari*: entrambi hanno accesso allo stesso stato (32-dim) e ricevono esattamente la stessa complessa Reward.
