# Descrizione delle Configurazioni e dei Dati (CityFlow)

Questo documento descrive lo scopo dei file presenti nelle cartelle `configs/` e `data/`, analizzandone la struttura attuale e proponendo una **nomenclatura standardizzata** per renderli più uniformi e facilmente comprensibili.

---

## 1. Cartella `data/`

La cartella `data/` contiene i file sorgente necessari al simulatore CityFlow per generare l'ambiente. Si dividono in due categorie principali: **Reti Stradali (Roadnets)** e **Flussi di Traffico (Flows)**.

### 1.1 Reti Stradali (`roadnet_*.json`)
Definiscono la topologia della rete, incluse le intersezioni, le corsie, i collegamenti (links) e la posizione dei semafori.
- `roadnet_3x3_100m.json`: Griglia 3x3 con strade lunghe 100 metri.
- `roadnet_4x4_200m.json`: Griglia 4x4 con strade lunghe 200 metri.
- `roadnet_5x5_200m.json`: Griglia 5x5 con strade lunghe 200 metri.

### 1.2 Flussi di Traffico (`flow_*.json`)
Definiscono come i veicoli vengono immessi nella rete (orari, percorsi, intervalli).
- `flow_1.3k_flat.json`: Flusso costante ("flat") con un arrival rate di 0.416 veicoli/s (tipicamente per griglie 4x4).
- `flow_1.9k_peak.json`: Flusso variabile ("peak") con picchi di traffico, arrival rate base 0.416.
- `flow_6.2k_flat.json`: Flusso costante che immette un volume elevato (es. ~6000 veicoli totali o parametri equivalenti), usato in configurazioni di stress.
- `flow_7.3k_peak.json`: Flusso con due picchi di traffico intensi (stress test).
- `flow_6.3k_flat.json`: Flusso costante specifico per la griglia 5x5 (per gestire il maggior numero di intersezioni periferiche).

---

## 2. Cartella `configs/`

La cartella `configs/` contiene i file di configurazione principali per CityFlow. Ogni file JSON accoppia una mappa (`roadnetFile`) a un flusso di traffico (`flowFile`) e definisce parametri come il seed, i file di log e la durata dell'episodio (`maxStep`).

- `config_4x4_200m_1.3k_flat.json`:
  - Mappa: `roadnet_4x4_200m.json`
  - Flusso: `flow_1.3k_flat.json`
- `config_4x4_200m_1.9k_peak.json`:
  - Mappa: `roadnet_4x4_200m.json`
  - Flusso: `flow_1.9k_peak.json` (presumibilmente, analogamente a config3)
- `config_4x4_200m_6.2k_flat.json`:
  - Mappa: `roadnet_4x4_200m.json`
  - Flusso: `flow_6.2k_flat.json`
- `config_4x4_200m_7.3k_peak.json`:
  - Mappa: `roadnet_4x4_200m.json`
  - Flusso: `flow_7.3k_peak.json` (presumibilmente)
- `config_5x5_200m_1.3k_flat.json`:
  - Simile al 4x4, ma su rete 5x5 (flusso normale).
- `config_5x5_200m_1.9k_peak.json`:
  - Rete 5x5 con flusso peak.
- `config_5x5_200m_6.3k_flat.json`:
  - Mappa: `roadnet_5x5_200m.json`
  - Flusso: `flow_6.3k_flat.json`

---

## 3. Proposta di Uniformazione della Nomenclatura

Attualmente, i file di flusso usano mischiati tassi di arrivo (es. `0.416`) e volumi o scenari (es. `6k`). I config usano "config3" o "config4" che sono poco descrittivi.

Ecco una proposta per una convenzione di nomi standardizzata e auto-esplicativa.

### 3.1 Nomi per i File Roadnet (Mappe)
La struttura attuale `roadnet_<grid>_<length>.json` è già ottima.
- ✅ `roadnet_4x4_200m.json`
- ✅ `roadnet_5x5_200m.json`

### 3.2 Nomi per i File Flow (Traffico)
Struttura proposta: `flow_<grid>_<k_veicoli>_<flat_or_peak/s>.json`
Questo chiarisce per quale mappa sono stati generati e che tipo di stress applicano.
*Esempi:*
- `flow_4x4_1.3k_flat.json` (sostituisce `flow_1.3k_flat.json`)
- `flow_4x4_1.9k_peak.json` (sostituisce `flow_1.9k_peak.json`)
- `flow_4x4_6.2k_flat.json` (sostituisce `flow_6.2k_flat.json`)
- `flow_4x4_7.3k_peaks.json` (sostituisce `flow_7.3k_peak.json`)
- `flow_5x5_6.3k_flat.json` (sostituisce `flow_6.3k_flat.json`)

### 3.3 Nomi per i Config CityFlow
Struttura proposta: `sim_<grid>_<length>_<volume_o_tasso>_<tipo_distribuzione>.json`
In questo modo dal nome del config si sa esattamente cosa si sta testando.
*Esempi:*
- `sim_4x4_200m_0.416_flat.json` (sostituisce `config_4x4_200m_1.3k_flat.json`)
- `sim_4x4_200m_0.416_peak.json` (sostituisce `config_4x4_200m_1.9k_peak.json`)
- `sim_4x4_200m_stress_flat.json` (sostituisce `config_4x4_200m_6.2k_flat.json`)
- `sim_4x4_200m_stress_doublepeak.json` (sostituisce `config_4x4_200m_7.3k_peak.json`)

### Passi successivi per l'implementazione
1. Rinominare i file nella cartella `data/`.
2. Modificare il parametro `"flowFile"` all'interno di ogni JSON di configurazione in `configs/` per puntare al nuovo nome.
3. Rinominare i file nella cartella `configs/`.
4. (Opzionale) Aggiornare eventuali riferimenti hardcoded all'interno degli script di training Python (`train.py`, ecc.) per farli puntare ai nuovi nomi.
