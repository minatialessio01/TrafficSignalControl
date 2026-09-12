# Descrizione delle Configurazioni e dei Dati (CityFlow)

Questo documento descrive lo scopo dei file presenti nelle cartelle `configs/` e `data/`, analizzandone la struttura attuale e la **nomenclatura standardizzata** applicata per renderli uniformi e facilmente comprensibili.

> **Revisione del 2026-09-10**: l'intero set di dati precedente è stato cancellato e rigenerato da zero, per introdurre (a) reti calibrate per l'onda verde e (b) un unico flusso di training "giornata lavorativa" al posto del curriculum a due stage. Vedi §4 per i dettagli.
>
> **Revisione del 2026-09-12 (generalizzazione)**: due aggiunte per evitare che i modelli imparino ad ottimizzare una singola sequenza fissa di veicoli invece di una vera politica di controllo. Vedi §4bis:
> 1. **3 varianti seed della config di training** (`flow_4x4_train1.json`/`train2.json`/`train3.json`), cicilate un episodio alla volta da `train.py` — stessa densità/forma, percorsi e istanti di spawn diversi.
> 2. **Un'arteria Est-Ovest più trafficata** aggiunta a training, validazione e test — verificato che senza intervento il traffico è quasi uniforme (±10% tra le righe), quindi non emergeva da solo. Serve a testare se il termine anti-starvation del reward custom dà ai modelli RL un vantaggio che MaxPressure (nessuna garanzia di equità, solo di throughput) non ha. Riga fissa (1) per validazione/test; **variabile (1/2/0) tra le 3 varianti di training**, per non rischiare che il modello impari a riconoscere la posizione fisica dell'arteria invece di reagire alla pressione osservata.

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

Definiscono come i veicoli vengono immessi nella rete (orari, percorsi, intervalli). Un flow dipende solo dalla **topologia** della griglia (dimensione), non dalla lunghezza delle strade: lo stesso file può quindi essere abbinato a roadnet diversi con la stessa griglia (vedi `flow_4x4_train1.json`, usato sia su 100m che su 200m).

Tutti i flow elencati includono ora un'**arteria Est-Ovest boostata** (riga 1 per validazione/test; le 3 varianti di training hanno la riga rispettivamente 1/2/0 — vedi §4bis).

- **`flow_4x4_train1.json`** (seed 42), **`flow_4x4_train2.json`** (seed 43), **`flow_4x4_train3.json`** (seed 44) — 3 varianti "sostanza-preservante" dello stesso flusso "giornata lavorativa" (1800s, 5 fasce di intensità, mappatura 24h → 1800s). `train.py` cicla tra le 3 un episodio alla volta. Vedi §4/§4bis per dettagli e razionale. **4728 / 4723 / 4620 veicoli effettivi.** Usati solo su `roadnet_4x4_100m.json`.
- `flow_4x4_6k_flat.json`: flusso costante 1800s (**6361 veicoli**).
- `flow_4x4_6k_peak.json`: flusso a 3 fasi Low→Peak→Low, 1800s (**6585 veicoli**).
- `flow_5x5_9.4k_flat.json`: flusso costante 1800s (**9871 veicoli**).
- `flow_5x5_9.4k_peak.json`: flusso a 3 fasi, 1800s (**10107 veicoli**).
- `flow_6x6_11.5k_flat.json`: flusso costante 1800s (**11870 veicoli**).
- `flow_6x6_11.5k_peak.json`: flusso a 3 fasi, 1800s (**12403 veicoli**).

---

## 2. Cartella `configs/`

Ogni file JSON accoppia in modo inequivocabile un roadnet a un flow e definisce i parametri di simulazione CityFlow (durata, seed, replay). Struttura del nome: `config_<grid>_<road_length_label>m_<k_veicoli>_<tipo>.json` (per il flusso di training: `config_<grid>_<road_length_label>m_train<N>.json`, N=1/2/3).

| File Configurazione | Veicoli Effettivi | Durata | Ruolo |
|---|---|---|---|
| `config_4x4_100m_train1.json` (+ `train2`, `train3`) | 4728 / 4723 / 4620 | 1800s | **3 varianti di training** (cicliche, vedi §4bis) — rete calibrata onda verde, arteria riga 1/2/0 |
| `config_4x4_100m_6k_flat.json` | 6361 | 1800s | Validazione, rete calibrata, arteria riga 1 |
| `config_4x4_100m_6k_peak.json` | 6585 | 1800s | Stress test 3 fasi, rete calibrata, arteria riga 1 |
| `config_4x4_200m_6k_flat.json` | 6361 | 1800s | Stress test, rete non calibrata, arteria riga 1 |
| `config_4x4_200m_6k_peak.json` | 6585 | 1800s | Stress test 3 fasi, rete non calibrata, arteria riga 1 |
| `config_5x5_100m_9.4k_flat.json` | 9871 | 1800s | Generalizzazione topologica (5x5), arteria riga 1 |
| `config_5x5_100m_9.4k_peak.json` | 10107 | 1800s | Generalizzazione topologica (5x5), 3 fasi, arteria riga 1 |
| `config_6x6_100m_11.5k_flat.json` | 11870 | 1800s | Generalizzazione topologica (6x6), arteria riga 1 |
| `config_6x6_100m_11.5k_peak.json` | 12403 | 1800s | Generalizzazione topologica (6x6), 3 fasi, arteria riga 1 |

> **Revisione del 2026-09-11**: i target veicoli di 5x5 e 6x6 sono stati **abbassati** rispetto alla versione precedente (10529/10849 → 9921/10213 per 5x5; 16459/17000 → 11923/12452 per 6x6). Il ragionamento originale ("in una griglia più grande la quota di bordo è minore, quindi serve più densità per uno stress comparabile") si è rivelato **sbagliato in pratica**: verificando con MaxPressure (`scripts/inspect_replay.py --policy maxpressure`), il 6x6 a 16459 veicoli riusciva a far arrivare solo il 63% dei veicoli target (throughput 78.8% di un già ridotto 80.2% di veicoli effettivamente spawnati) — congestione reale, non solo un controllore imperfetto. La stessa densità/incrocio del 4x4 (395 veicoli/incrocio su 1800s, quella di `config_4x4_100m_6k_flat`, validata: MaxPressure la gestisce quasi senza perdite) applicata a 5x5 la rende sana (throughput ~91-92%), ma il 6x6 ha richiesto un ulteriore taglio (a ~330 veicoli/incrocio) prima di arrivare a un throughput comparabile (~88-91%). Conclusione pratica: **le griglie più grandi non assorbono meglio il traffico a parità di densità per incrocio — ne assorbono peggio**, verosimilmente perché la congestione si accumula lungo percorsi che attraversano più incroci in sequenza. Numeri finali validati via MaxPressure:

| Config | Veicoli/incrocio (1800s) | Throughput MaxPressure (arrivati/target) |
|---|---|---|
| `config_4x4_100m_6k_flat` (riferimento) | 395 | ~94% |
| `config_5x5_100m_9.4k_flat` | 397 | ~92% |
| `config_6x6_100m_11.5k_flat` | 331 | ~90% |

---

## 3. Training: da curriculum a 2 stage a flusso unico

Il vecchio curriculum a due stage (`--config-b`/`--episodes-b` in `train.py`, Stage A su un flow flat poi Stage B su un flow peak) è stato **rimosso**: `train.py` non alterna più due config *diverse per densità* a metà training. `flow_4x4_train1.json` contiene già la variabilità intra-episodio (basso→picco→alto→picco→basso) che il vecchio curriculum a due config approssimava in modo più grezzo. I titoli dei grafici (`training_curves.png`) mostrano di conseguenza solo il/i nome/i della config di training, senza più menzionare "Stage A"/"Stage B".

Dal 2026-09-12, `--config` accetta comunque più file (vedi §4bis) — ma a differenza del vecchio curriculum, le varianti sono **sostanza-preservante** (stesso profilo di densità, seed diverso) e si cicla tra loro un episodio alla volta per tutta la durata del training, non uno switch singolo a metà.

### Suddivisione train / validation / test

| Ruolo | Config | Uso |
|---|---|---|
| **Train** | `config_4x4_100m_train1.json` + `train2.json` + `train3.json` | 3 varianti su cui si allena, ciclate un episodio alla volta (`train.py --config`, vedi §4bis) |
| **Validation** | `config_4x4_100m_6k_flat.json` | Usata a fine training per scegliere tra `final_model.pth` e `best_model.pt` (`train.py --select-best-config`, `main.py --validation-config`) |
| **Test** | tutte le altre 7 config (`config_4x4_100m_6k_peak`, `config_4x4_200m_6k_flat/peak`, `config_5x5_100m_9.4k_flat/peak`, `config_6x6_100m_11.5k_flat/peak`) | Solo valutazione di generalizzazione (`main.py --test-configs`), mai viste in training o selezione |

---

## 4. Il flusso "giornata lavorativa" (`flow_4x4_train1.json`)

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

## 4bis. Generalizzazione: 3 varianti di training e arteria Est-Ovest

Due aggiunte del 2026-09-12, motivate da un rischio concreto di overfitting metodologico: CityFlow è deterministico dato lo stesso flow+seed, e `make_flow()` usava sempre `seed=42` — un modello vedeva quindi la sequenza *esatta* di veicoli (stessi istanti di spawn, stessi percorsi) a ogni episodio di training, potendo in teoria "imparare" quella sequenza specifica invece di una vera politica di controllo generalizzabile.

### 3 varianti seed (solo training)

`make_flow()` accetta ora `--seed` (CLI) e `--name-suffix` per generare più copie dello stesso identico profilo di densità/fasce, cambiando solo il rumore su pesi delle rotte (`rng.uniform(0.8,1.2)` per rotta) e sugli intervalli di spawn (`rng.gauss`). Il *profilo aggregato* (quanti veicoli, quando, con che forma) resta identico; cambiano i percorsi specifici enfatizzati e gli istanti esatti di spawn.

- `flow_4x4_train1.json` — seed 42 (4728 veicoli)
- `flow_4x4_train2.json` — seed 43 (4723 veicoli)
- `flow_4x4_train3.json` — seed 44 (4620 veicoli)

`train.py --config` accetta ora una lista di config (`nargs="+"`): se più di una, si cicla una variante per episodio (`episodio % N`, sia durante il warm-up che nel loop principale — vedi `_variant_idx()`/`switch_variant()`). **Validazione e test restano invece a seed singolo** (un solo rappresentante): il loro scopo è essere un metro di paragone stabile, la varianza deve venire dal lato training, non dal benchmark.

### Arteria Est-Ovest

Ipotesi: MaxPressure garantisce l'ottimalità del throughput di rete ma **nessuna garanzia di equità** — sceglie sempre la fase con pressione istantanea più alta, quindi una direzione strutturalmente più trafficata di altre potrebbe essere favorita ripetutamente, lasciando le corsie minori sistematicamente in attesa. Il reward "custom" ha invece un termine esplicito anti-starvation (`alpha * max_red_wait_time`) pensato esattamente per questo scenario. Su una rete uniforme (il caso di tutte le config precedenti) questo vantaggio potenziale del modello RL non ha mai l'occasione di manifestarsi.

**Controllo preliminare** (analisi statica di `flow_4x4_train1.json` prima di questa modifica): il carico per riga, sulle sole strade orizzontali interne, variava solo del ±10% (2044-2247 veicoli/riga) — un'asimmetria naturale trascurabile. Serviva costruire l'arteria deliberatamente.

**Implementazione**: `make_flow()` accetta `--artery-row R --artery-boost B`. Ogni rotta che attraversa un segmento orizzontale sulla riga `R` vede il proprio peso (prima della normalizzazione) moltiplicato per `B` — è una **ridistribuzione** dello stesso budget di veicoli, non un'aggiunta: le altre rotte ricevono proporzionalmente meno. Usato ovunque con `--artery-row 1 --artery-boost 3.0` (riga 1 = "la seconda strada" in ogni griglia, 0-indexed). Verificato: con boost 3.0 la riga 1 arriva a portare 2.3-2.8× il carico delle altre righe (era già stato provato empiricamente prima di applicarlo ovunque).

**Dove è stata applicata**: training (tutte e 3 le varianti seed), validazione (`config_4x4_100m_6k_flat`) e tutte e 7 le config di test — `--artery-boost 3.0` ovunque, ma con un solo seed ciascuna per validazione/test (non serve la stessa diversità di training per un benchmark fisso).

**Riga dell'arteria: fissa per validazione/test, variabile per training**. Le feature spaziali/temporali che alimentano SMK/TMK (`get_spatial_meta_features`/`get_temporal_meta_features`) sono costruite solo da conteggi di veicoli sulle proprie corsie e storico recente — nessun canale porta coordinate assolute o l'identità dell'incrocio, e il GAT condivide gli stessi pesi su tutti i nodi. In teoria una regola "reagisci a pressione alta e persistente" dovrebbe quindi generalizzare a un'arteria in una riga qualsiasi. Resta però un dubbio legittimo: il modo in cui l'attenzione del GAT *coordina* i vicini potrebbe specializzarsi sulla forma spaziale con cui la congestione si propaga se l'arteria fosse **sempre** nella stessa riga. Per non lasciare questa ipotesi non testata (costa pochissimo verificarla), le 3 varianti di training hanno l'arteria su righe diverse:
- `flow_4x4_train1.json` (seed 42) — arteria riga **1**
- `flow_4x4_train2.json` (seed 43) — arteria riga **2**
- `flow_4x4_train3.json` (seed 44) — arteria riga **0**

Validazione e le 7 config di test restano invece fisse su riga 1 (benchmark stabile).

**Verifica di sicurezza**: rigenerate le config, ho ricontrollato con MaxPressure (`inspect_replay.py`) che nessuna fosse andata in sovraccarico grave. Nessun collasso: il caso peggiore resta 6x6/peak (87.6% di throughput sui veicoli spawnati, contro il 90.9%/88.4% pre-arteria) — un aumento di difficoltà reale ma non catastrofico, coerente con l'obiettivo dell'esperimento.

---

## 5. Generazione Parametrica (`generate_synthetic_data.py`)

Tutti i file descritti in precedenza vengono generati in maniera completamente deterministica e parametrica da `scripts/generate_synthetic_data.py`.

### Utilizzo

```bash
# Rete calibrata onda verde (label "100m", fisico 107 -> 167m reali) + stress flat 6k
python scripts/generate_synthetic_data.py --grid 4x4 --road-length 107 --road-length-label 100 \
    --duration 1800 --variance flat --vehicles 6k

# Flusso "giornata lavorativa" (1800s, 5 fasce) sulla stessa rete calibrata --
# prima delle 3 varianti di training (--name-suffix numerico, mai vuoto: le
# 3 varianti si chiamano train1/train2/train3, non train/train_s2/train_s3)
python scripts/generate_synthetic_data.py --grid 4x4 --road-length 107 --road-length-label 100 \
    --duration 1800 --variance workday --name-suffix 1
    # opzionale: --rate-basso --rate-peak-basso --rate-alto --rate-peak-alto (default 2160/4500/6300/7560)

# Una seconda variante "sostanza-preservante" dello stesso flusso (seed diverso,
# nome diverso per non sovrascrivere la prima) + arteria Est-Ovest sulla riga 1
python scripts/generate_synthetic_data.py --grid 4x4 --road-length 107 --road-length-label 100 \
    --duration 1800 --variance workday --seed 43 --name-suffix 2 \
    --artery-row 1 --artery-boost 3.0
```

- `--grid`: dimensioni logiche (es. `4x4`, `5x5`).
- `--seed`: seed per il rumore su pesi delle rotte/intervalli di spawn (default 42). Variarlo a parità di tutto il resto produce una variante "sostanza-preservante" (stessa densità/forma, percorsi e istanti diversi) — vedi §4bis.
- `--artery-row`/`--artery-boost`: rende una riga orizzontale (0-based) un'arteria più trafficata, ridistribuendo il peso delle rotte che la attraversano (non aggiunge veicoli). Vedi §4bis.
- `--name-suffix`: suffisso per il nome di flow/config, per generare più varianti senza sovrascriverle.
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
