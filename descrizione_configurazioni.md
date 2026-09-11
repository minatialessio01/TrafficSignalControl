# Descrizione delle Configurazioni e dei Dati (CityFlow)

Questo documento descrive lo scopo dei file presenti nelle cartelle `configs/` e `data/`, analizzandone la struttura attuale e la **nomenclatura standardizzata** applicata per renderli uniformi e facilmente comprensibili.

> **Revisione del 2026-09-10**: l'intero set di dati precedente è stato cancellato e rigenerato da zero, per introdurre (a) reti calibrate per l'onda verde e (b) un unico flusso di training "giornata lavorativa" al posto del curriculum a due stage. Vedi §4 per i dettagli.

---

## 1. Cartella `data/`

La cartella `data/` contiene i file sorgente necessari al simulatore CityFlow per generare l'ambiente. Si dividono in due categorie principali: **Reti Stradali (Roadnets)** e **Flussi di Traffico (Flows)**.

### 1.1 Reti Stradali (`roadnet_*.json`)

Definiscono la topologia della rete, incluse le intersezioni, le corsie, i collegamenti (links) e la posizione dei semafori.

**Nota sulla nomenclatura "100m" vs "200m"**: l'etichetta nel nome file NON è il parametro fisico `--road-length` passato allo script, ma un nome "pulito" scelto per leggibilità. La distanza reale tra due incroci è sempre `--road-length + 60` (offset geometrico interno di `generate_synthetic_data.py`, verificato empiricamente). Corrispondenza usata in questo progetto:

| Etichetta file | `--road-length` fisico | Distanza reale tra incroci | Motivo |
|---|---|---|---|
| `100m` | 107 | **167m** | Calibrata per l'onda verde: un veicolo a velocità massima (11.11 m/s) che parte all'inizio di un verde percorre 11.11 × 15s (ciclo completo verde+giallo+rosso) ≈ 166.5m, arrivando al prossimo incrocio esattamente al tick di decisione successivo. Sfrutta il fatto che tutte le intersezioni cambiano fase in modo sincrono (non è possibile uno sfasamento tra incroci, quindi l'onda verde si ottiene solo calibrando la distanza). **Unica distanza usata per il training.** |
| `200m` | 200 | 260m | Spaziatura "storica" (non calibrata), mantenuta solo per il roadnet 4x4, solo come config di TEST zero-shot: stesso traffico (flat/peak 6k) di `100m`, ma senza il beneficio dell'onda verde — misura quanto un modello allenato sulla rete calibrata regge quando la spaziatura cambia. Non viene usata per il training. |

File presenti:
- `roadnet_4x4_100m.json`: griglia 4x4, calibrata onda verde (167m reali) — rete di **training**.
- `roadnet_4x4_200m.json`: griglia 4x4, spaziatura storica (260m reali) — solo test di generalizzazione alla spaziatura.
- `roadnet_5x5_100m.json`: griglia 5x5, calibrata onda verde (167m reali) — generalizzazione topologica.
- `roadnet_6x6_100m.json`: griglia 6x6, calibrata onda verde (167m reali) — generalizzazione topologica.

Nota: 5x5 e 6x6 esistono solo nella variante calibrata: servono a testare la generalizzazione a topologie più grandi, non l'effetto onda verde (già isolato sul 4x4 tramite 100m vs 200m).

### 1.2 Flussi di Traffico (`flow_*.json`)

Definiscono come i veicoli vengono immessi nella rete (orari, percorsi, intervalli). Un flow dipende solo dalla **topologia** della griglia (dimensione), non dalla lunghezza delle strade: lo stesso file può quindi essere abbinato a roadnet diversi con la stessa griglia (vedi `flow_4x4_train.json`, usato sia su 100m che su 200m).

- **`flow_4x4_train.json`** — sostituisce il vecchio curriculum a due config (flat poi peak). Flusso a 1800s che simula una giornata lavorativa completa in 5 fasce di intensità e durata diverse (mappatura 24h → 1800s, 75s/ora simulata). Vedi §4 per dettagli e razionale. **7119 veicoli effettivi.** Usato solo su `roadnet_4x4_100m.json` (il training avviene solo sulla rete calibrata).
- `flow_4x4_6k_flat.json`: flusso costante 1800s (**6320 veicoli**).
- `flow_4x4_6k_peak.json`: flusso a 3 fasi Low→Peak→Low, 1800s (**6686 veicoli**).
- `flow_5x5_9.4k_flat.json`: flusso costante 1800s (**9921 veicoli**).
- `flow_5x5_9.4k_peak.json`: flusso a 3 fasi, 1800s (**10213 veicoli**).
- `flow_6x6_11.5k_flat.json`: flusso costante 1800s (**11923 veicoli**).
- `flow_6x6_11.5k_peak.json`: flusso a 3 fasi, 1800s (**12452 veicoli**).

---

## 2. Cartella `configs/`

Ogni file JSON accoppia in modo inequivocabile un roadnet a un flow e definisce i parametri di simulazione CityFlow (durata, seed, replay). Struttura del nome: `config_<grid>_<road_length_label>m_<k_veicoli>_<tipo>.json` (per il flusso di training: `config_<grid>_<road_length_label>m_train.json`).

| File Configurazione | Veicoli Effettivi | Durata | Ruolo |
|---|---|---|---|
| `config_4x4_100m_train.json` | 7119 | 1800s | **Unica config di training** — rete calibrata onda verde |
| `config_4x4_100m_6k_flat.json` | 6320 | 1800s | Stress test, rete calibrata |
| `config_4x4_100m_6k_peak.json` | 6686 | 1800s | Stress test 3 fasi, rete calibrata |
| `config_4x4_200m_6k_flat.json` | 6320 | 1800s | Stress test, rete non calibrata |
| `config_4x4_200m_6k_peak.json` | 6686 | 1800s | Stress test 3 fasi, rete non calibrata |
| `config_5x5_100m_9.4k_flat.json` | 9921 | 1800s | Generalizzazione topologica (5x5) |
| `config_5x5_100m_9.4k_peak.json` | 10213 | 1800s | Generalizzazione topologica (5x5), 3 fasi |
| `config_6x6_100m_11.5k_flat.json` | 11923 | 1800s | Generalizzazione topologica (6x6) |
| `config_6x6_100m_11.5k_peak.json` | 12452 | 1800s | Generalizzazione topologica (6x6), 3 fasi |

> **Revisione del 2026-09-11**: i target veicoli di 5x5 e 6x6 sono stati **abbassati** rispetto alla versione precedente (10529/10849 → 9921/10213 per 5x5; 16459/17000 → 11923/12452 per 6x6). Il ragionamento originale ("in una griglia più grande la quota di bordo è minore, quindi serve più densità per uno stress comparabile") si è rivelato **sbagliato in pratica**: verificando con MaxPressure (`scripts/inspect_replay.py --policy maxpressure`), il 6x6 a 16459 veicoli riusciva a far arrivare solo il 63% dei veicoli target (throughput 78.8% di un già ridotto 80.2% di veicoli effettivamente spawnati) — congestione reale, non solo un controllore imperfetto. La stessa densità/incrocio del 4x4 (395 veicoli/incrocio su 1800s, quella di `config_4x4_100m_6k_flat`, validata: MaxPressure la gestisce quasi senza perdite) applicata a 5x5 la rende sana (throughput ~91-92%), ma il 6x6 ha richiesto un ulteriore taglio (a ~330 veicoli/incrocio) prima di arrivare a un throughput comparabile (~88-91%). Conclusione pratica: **le griglie più grandi non assorbono meglio il traffico a parità di densità per incrocio — ne assorbono peggio**, verosimilmente perché la congestione si accumula lungo percorsi che attraversano più incroci in sequenza. Numeri finali validati via MaxPressure:

| Config | Veicoli/incrocio (1800s) | Throughput MaxPressure (arrivati/target) |
|---|---|---|
| `config_4x4_100m_6k_flat` (riferimento) | 395 | ~94% |
| `config_5x5_100m_9.4k_flat` | 397 | ~92% |
| `config_6x6_100m_11.5k_flat` | 331 | ~90% |

---

## 3. Training: da curriculum a 2 stage a flusso unico

Il vecchio curriculum a due stage (`--config-b`/`--episodes-b` in `train.py`, Stage A su un flow flat poi Stage B su un flow peak) è stato **rimosso**: `train.py` allena ora solo su un'**unica** configurazione. `flow_4x4_train.json` contiene già la variabilità intra-episodio (basso→picco→alto→picco→basso) che il vecchio curriculum a due config approssimava in modo più grezzo switchando config a metà training. I titoli dei grafici (`training_curves.png`) mostrano di conseguenza solo il nome della config di training, senza più menzionare "Stage A"/"Stage B".

### Suddivisione train / validation / test

| Ruolo | Config | Uso |
|---|---|---|
| **Train** | `config_4x4_100m_train.json` | Unica config su cui si allena (`train.py --config`) |
| **Validation** | `config_4x4_100m_6k_flat.json` | Usata a fine training per scegliere tra `final_model.pth` e `best_model.pt` (`train.py --select-best-config`, `main.py --validation-config`) |
| **Test** | tutte le altre 7 config (`config_4x4_100m_6k_peak`, `config_4x4_200m_6k_flat/peak`, `config_5x5_100m_9.4k_flat/peak`, `config_6x6_100m_11.5k_flat/peak`) | Solo valutazione di generalizzazione (`main.py --test-configs`), mai viste in training o selezione |

---

## 4. Il flusso "giornata lavorativa" (`flow_4x4_train.json`)

> **Revisione del 2026-09-11 (durata)**: durata riportata da 3600s a 1800s e picco serale ammorbidito, dopo aver osservato che il gap di travel time rispetto a MaxPressure raddoppiava sulla config di training (non stazionaria) rispetto alla config di validazione (stazionaria).
>
> **Revisione del 2026-09-11 (densità)**: guardando il replay, anche MaxPressure non riusciva a smaltire il traffico durante la fascia "flat alto" — segno che i veicoli erano troppi in assoluto, non solo troppo concentrati nel picco. Rate ricalibrati usando come riferimento la densità sostenuta validata di `config_4x4_100m_6k_flat` (395 veicoli/incrocio su 1800s = 3.5 veicoli/secondo costanti, MaxPressure la gestisce quasi senza perdite): la fascia "alto" (quella più lunga, 525s) ora non supera mai questo ritmo, invece di superarlo del ~80%. Totale sceso da 7119 a 4621 veicoli. Vedi `confronto/` per il confronto diretto pro vs MaxPressure.

Mappa una giornata lavorativa di 24h su 1800 secondi di simulazione (75s = 1h simulata) in 5 fasce a durata e intensità diverse. Ogni intensità è un "rate" espresso in veicoli equivalenti per 1800s (stessa unità delle altre config, cioè un ritmo: rate/1800 = veicoli/secondo se sostenuto per l'intera fascia), convertito in veicoli assoluti in base alla durata reale della fascia. Le durate/orari sono hardcoded in `make_flow()` (frazioni di `duration`); i 4 rate sono parametrizzabili via `--rate-basso`/`--rate-peak-basso`/`--rate-alto`/`--rate-peak-alto`:

| # | Fascia | Orario virtuale | Intervallo simulato | Step di decisione (15s/step) | Rate (veic./1800s) | Veic./secondo | Veicoli in fascia (circa) |
|---|---|---|---|---|---|---|---|
| 1 | Flat basso (mattina presto) | 00–07 | 0s – 525s | step 0 – 35 | 2160 | 1.2 | ~630 |
| 2 | Peak basso (pendolari mattutini) | 07–09 | 525s – 675s | step 35 – 45 | 4500 | 2.5 | ~375 |
| 3 | Flat alto (ore diurne) | 09–16 | 675s – 1200s | step 45 – 80 | 6300 | **3.5** | ~1840 |
| 4 | Peak alto (fine giornata lavorativa) | 16–19 | 1200s – 1350s | step 80 – 90 | 7560 | 4.2 | ~630 |
| 5 | Flat basso (notte) | 19–24 | 1350s – 1800s | step 90 – 120 | 2160 | 1.2 | ~540 |

Il ritmo di "flat alto" (3.5 veic./s) è deliberatamente identico a quello di `config_4x4_100m_6k_flat` sostenuto per l'intero episodio — qui dura solo 525s (29% dell'episodio), quindi è per costruzione più leggero del riferimento validato, non solo uguale. Il picco serale ("peak alto", 4.2 veic./s) resta entro il range già validato da `config_4x4_100m_6k_peak` (che raggiunge 5.6 veic./s per 600s senza problemi).

Il "passo di decisione" è lo step dell'agente RL (`CityFlowEnv.step()`), che dura sempre `GREEN_TIME + YELLOW_TIME + RED_TIME` = 15s reali; l'episodio da 1800s corrisponde quindi a 120 step totali. I confini di fascia sono allineati esattamente a multipli di 15s, quindi ogni cambio di intensità coincide con l'inizio di un nuovo step di decisione.

Totale: **4621 veicoli** effettivi su 1800s (in precedenza 7119, e ancora prima 10806 su 3600s). La non-stazionarietà interna a un singolo episodio (leggero→medio→pesante→leggero) resta intenzionale: impedisce al modello di limitarsi a imparare una densità di traffico fissa.

---

## 5. Generazione Parametrica (`generate_synthetic_data.py`)

Tutti i file descritti in precedenza vengono generati in maniera completamente deterministica e parametrica da `scripts/generate_synthetic_data.py`.

### Utilizzo

```bash
# Rete calibrata onda verde (label "100m", fisico 107 -> 167m reali) + stress flat 6k
python scripts/generate_synthetic_data.py --grid 4x4 --road-length 107 --road-length-label 100 \
    --duration 1800 --variance flat --vehicles 6k

# Flusso "giornata lavorativa" (1800s, 5 fasce) sulla stessa rete calibrata
python scripts/generate_synthetic_data.py --grid 4x4 --road-length 107 --road-length-label 100 \
    --duration 1800 --variance workday
    # opzionale: --rate-basso --rate-peak-basso --rate-alto --rate-peak-alto (default 2160/4500/6300/7560)
```

- `--grid`: dimensioni logiche (es. `4x4`, `5x5`).
- `--road-length`: lunghezza stradale FISICA in metri (la distanza reale tra incroci è `road-length + 60`).
- `--road-length-label`: etichetta usata nei nomi file al posto di `--road-length` (default: uguale a `--road-length`). Usata per dare un nome "pulito" (es. `100m`) a un parametro fisico meno rotondo (es. `107`).
- `--duration`: durata simulazione in secondi (default 1800).
- `--vehicles`: quantità target di veicoli, esatta o in k-shorthand (es. `8000`, `8k`). Obbligatorio per `flat`/`peak`/`peaks`, ignorato per `workday`.
- `--variance`: modalità di iniezione.
  - `flat`: costante per tutta la durata.
  - `peak`: 3 fasi (Low 25% → Peak 50% → Low 25%), confini a 1/3 e 2/3 della durata.
  - `peaks`: 5 fasi "Double Peak" (Low-Peak-Low-Peak-Low, pesi 9/36.5/9/36.5/9%), confini a multipli di durata/5.
  - `workday`: 5 fasce a durata/intensità asimmetriche (vedi §4), mappate su `--duration` (default 1800s = giornata "compressa" in 30 min simulati); usa `--rate-basso`/`--rate-peak-basso`/`--rate-alto`/`--rate-peak-alto` invece di `--vehicles`.

*(Lo script ottimizza le rotte usando DFS e distribuisce il traffico allocando la richiesta frazionalmente tra i percorsi trovati. I file flow e roadnet vengono scritti in `data/`, il master file in `configs/`.)*

---

## 6. Cartella `analisi_configurazioni2/` (verifica, non permanente)

Cartella di controllo qualità: contiene, per ognuna delle 9 config sopra, una sotto-cartella con un replay generato da **MaxPressure** (`scripts/inspect_replay.py --policy maxpressure`, nessun training/checkpoint richiesto) più un riepilogo numerico (travel time, throughput). Usare MaxPressure invece di una politica casuale dà un segnale più onesto: è un controllore reale seppur semplice, quindi il suo throughput riflette quanto la rete regge il traffico assegnato, non solo quanto è inefficiente il controllore. Serve a ispezionare visivamente/numericamente se le quantità di veicoli scelte sono ragionevoli per ciascuna topologia — è stata proprio questa verifica (guardando i replay) a far scoprire il 2026-09-11 che le densità di 5x5/6x6 e della config di training erano eccessive (vedi le note di revisione sopra). Rigenerata dopo ogni cambio di densità delle config. Non fa parte della pipeline di training/test regolare (`scripts/inspect_replay.py` supporta anche `--policy random` per uno stress-test più estremo, se serve).
