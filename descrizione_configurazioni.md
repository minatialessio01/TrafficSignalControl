# Descrizione delle Configurazioni e dei Dati (CityFlow)

Questo documento descrive lo scopo dei file presenti nelle cartelle `configs/` e `data/`, analizzandone la struttura attuale e la **nomenclatura standardizzata** applicata per renderli uniformi e facilmente comprensibili.

---

## 1. Cartella `data/`

La cartella `data/` contiene i file sorgente necessari al simulatore CityFlow per generare l'ambiente. Si dividono in due categorie principali: **Reti Stradali (Roadnets)** e **Flussi di Traffico (Flows)**.

### 1.1 Reti Stradali (`roadnet_*.json`)
Definiscono la topologia della rete, incluse le intersezioni, le corsie, i collegamenti (links) e la posizione dei semafori.
- `roadnet_4x4_200m.json`: Griglia 4x4 con strade lunghe 200 metri.
- `roadnet_4x4_300m.json`: Griglia 4x4 con strade lunghe 300 metri.
- `roadnet_5x5_200m.json`: Griglia 5x5 con strade lunghe 200 metri.
- `roadnet_6x6_200m.json`: Griglia 6x6 con strade lunghe 200 metri.

### 1.2 Flussi di Traffico (`flow_*.json`)
Definiscono come i veicoli vengono immessi nella rete (orari, percorsi, intervalli). Essendo vincolati alla topologia della mappa per cui sono stati generati, includono la dimensione della griglia nel nome.
Struttura: `flow_<grid>_<k_veicoli>_<flat_or_peak/s>.json`

- **4x4**:
  - `flow_4x4_2k_flat.json`: Flusso costante (~2000 veicoli)
  - `flow_4x4_2k_peak.json`: Flusso variabile a 3 fasi (~2000 veicoli)
  - `flow_4x4_6k_flat.json`: Flusso costante (~6000 veicoli)
  - `flow_4x4_6k_peak.json`: Flusso variabile a 3 fasi (~6000 veicoli)
  - `flow_4x4_6k_peaks.json`: Flusso variabile a 5 fasi (double peak) (~6000 veicoli)
- **5x5**:
  - `flow_5x5_8k_flat.json`: Flusso costante (~8000 veicoli)
  - `flow_5x5_8k_peaks.json`: Flusso variabile a 5 fasi (~8000 veicoli)
- **6x6**:
  - `flow_6x6_10k_flat.json`: Flusso costante (~10000 veicoli)
  - `flow_6x6_10k_peaks.json`: Flusso variabile a 5 fasi (~10000 veicoli)

---

## 2. Cartella `configs/`

La cartella `configs/` contiene i file di configurazione principali per CityFlow. Ogni file JSON accoppia in modo inequivocabile una mappa (`roadnetFile`) a un flusso di traffico (`flowFile`) e definisce parametri aggiuntivi.
Struttura: `config_<grid>_<length>_<k_veicoli>_<tipo_distribuzione>.json`

Di seguito l'elenco completo delle configurazioni attive con il conteggio esatto dei veicoli inseriti nei 1800 secondi di simulazione:

| File Configurazione | Veicoli Effettivi Spawnati | Note |
|----------------------------------------|---------------------------|------|
| `config_4x4_200m_2k_flat.json`         | 2083                      | Base |
| `config_4x4_200m_2k_peak.json`         | 2185                      | Base (3 fasi) |
| `config_4x4_200m_6k_flat.json`         | 6320                      | Stress |
| `config_4x4_200m_6k_peaks.json`        | 6522                      | Stress Double Peak (5 fasi) |
| `config_4x4_300m_6k_flat.json`         | 6320                      | Stress su strade lunghe |
| `config_4x4_300m_6k_peak.json`         | 6686                      | Stress su strade lunghe (3 fasi) |
| `config_5x5_200m_8k_flat.json`         | 8481                      | Rete estesa |
| `config_5x5_200m_8k_peaks.json`        | 8817                      | Rete estesa Double Peak (5 fasi) |
| `config_6x6_200m_10k_flat.json`        | 10390                     | Stress Rete Gigante |
| `config_6x6_200m_10k_peaks.json`       | 11093                     | Stress Rete Gigante Double Peak (5 fasi) |

---

## 3. Generazione Parametrica (`generate_synthetic_data.py`)

Tutti i file descritti in precedenza vengono generati in maniera completamente deterministica e parametrica dal nuovo script `scripts/generate_synthetic_data.py`.

### Utilizzo
```bash
python scripts/generate_synthetic_data.py --grid 4x4 --road-length 200 --vehicles 6k --variance peaks
```
- `--grid`: Dimensioni logiche (es. `4x4`, `3x5`).
- `--road-length`: Lunghezza strada (es. `200`, `300`).
- `--vehicles`: Quantità indicativa target di veicoli che lo script proverà a generare nei 1800s. Formati ammessi: esatti (`8000`) o k-shorthand (`8k`).
- `--variance`: Modalità di iniezione. 
  - `flat`: costante per tutti i 1800s.
  - `peak`: 3 fasi (Low, Peak, Low).
  - `peaks`: 5 fasi "Double Peak" (Low, Peak, Low, Peak, Low). Il file verrà nominato terminando per `_peaks.json`.

*(Lo script ottimizza le rotte usando DFS e distribuisce il traffico allocando la richiesta frazionalmente tra i percorsi trovati. I file flow e roadnet vengono scritti in `data/`, il master file in `configs/`.)*
