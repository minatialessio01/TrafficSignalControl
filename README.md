# MetaSTGAT — Replica da zero

Replica del modello **MetaSTGAT** (Wang et al., 2022):
> *"Meta-learning based spatial-temporal graph attention network for traffic signal control"*
> Knowledge-Based Systems 250 (2022) 109166

---

## Struttura del progetto

```
CodiceTesi/
├── README.md
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
├── configs/                  # Configurazioni CityFlow per ogni dataset
├── data/
│   ├── synthetic/            # Dati sintetici 4×4 (4 configurazioni)
│   ├── hangzhou/             # Dataset reale (16 intersezioni)
│   └── jinan/               # Dataset reale (12 intersezioni)
├── src/
│   ├── environment/          # Wrapper CityFlow + costruttore grafi
│   ├── models/               # Architettura MetaSTGAT
│   ├── agents/               # DQN agent, FixedTime baseline
│   └── utils/                # Metriche, logger, checkpoint manager
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── download_real_data.py
│   ├── train.py              # Training principale
│   ├── test.py               # Valutazione
│   └── ablation.py           # Ablation study
└── results/                  # Checkpoint, log, metriche (creato automaticamente)
```

---

## Installazione e Setup

CityFlow funziona nativamente su Linux. Poiché siamo su Windows, la soluzione raccomandata è **Docker**.

### Opzione A — Docker (Consigliata)

Assicurati di avere Docker installato e funzionante (es. Docker Desktop).

```bash
cd CodiceTesi/docker
docker build -t metastgat .
docker run -it --rm -v $(pwd)/..:/workspace metastgat bash
```

### Opzione C — Manuale (Linux/macOS)

```bash
# Crea ambiente virtuale
python3 -m venv venv
source venv/bin/activate

# Installa dipendenze
pip install -r docker/requirements.txt

# Installa CityFlow
pip install cityflow

# Installa PyG (PyTorch Geometric)
pip install torch_geometric
```

---

## Dataset

### Dati sintetici (generati localmente)

```bash
python scripts/generate_synthetic_data.py
```

Genera una griglia 4×4 con 4 configurazioni di traffico (combinazioni di arrival rate 0.388/0.416 e varianza flat/peak).

### Dataset reali (Hangzhou e Jinan)

```bash
python scripts/download_real_data.py
```

Scarica automaticamente i dataset dal repository CoLight (gli stessi usati nell'articolo originale).

---

## Training

```bash
# Training MetaSTGAT su dataset sintetico Config 1
python scripts/train.py \
    --config configs/synthetic_4x4_config1.json \
    --model MetaSTGAT \
    --episodes 200 \
    --output results/metastgat_config1

# Riprendere training da un checkpoint
python scripts/train.py \
    --config configs/synthetic_4x4_config1.json \
    --model MetaSTGAT \
    --episodes 200 \
    --resume results/metastgat_config1/checkpoint_ep050.pt \
    --output results/metastgat_config1

# Fermarsi a un episodio specifico (es. episodio 50)
python scripts/train.py \
    --config configs/synthetic_4x4_config1.json \
    --model MetaSTGAT \
    --episodes 200 \
    --stop-at 50 \
    --output results/metastgat_config1
```

> **Interruzione manuale**: premi `Ctrl+C` in qualsiasi momento. Il training salverà automaticamente un checkpoint con l'episodio corrente e potrà essere ripreso con `--resume`.

### Argomenti train.py

| Argomento | Default | Descrizione |
|---|---|---|
| `--config` | obbligatorio | File di configurazione CityFlow |
| `--model` | `MetaSTGAT` | `MetaSTGAT`, `STGAT`, `FixedTime` |
| `--episodes` | `200` | Numero totale di episodi |
| `--stop-at` | `None` | Ferma il training all'episodio N |
| `--resume` | `None` | Percorso checkpoint da cui riprendere |
| `--output` | `results/run` | Cartella di output |
| `--batch-size` | `20` | Batch size per l'aggiornamento DQN |
| `--lr` | `1e-3` | Learning rate RMSprop |
| `--gamma` | `0.85` | Discount factor RL |
| `--hidden-dim` | `64` | Dimensione hidden layer |
| `--num-heads` | `4` | Numero teste di attenzione |
| `--seed` | `42` | Random seed |

---

## Test e valutazione

```bash
python scripts/test.py \
    --config configs/synthetic_4x4_config1.json \
    --checkpoint results/metastgat_config1/best_model.pt \
    --model MetaSTGAT
```

Output: travel time medio (s) e throughput (veicoli) — stesse metriche dell'articolo.

---

## Ablation study

```bash
python scripts/ablation.py \
    --config configs/synthetic_4x4_config2.json \
    --episodes 100 \
    --output results/ablation
```

Allena in sequenza: `GAT-only`, `STGAT`, `MetaGAT`, `MetaLSTM`, `MetaSTGAT` e confronta i risultati.

---

## Checkpoint

I checkpoint vengono salvati in `results/<run_name>/`:
- `checkpoint_epXXX.pt` — checkpoint ogni 10 episodi
- `best_model.pt` — modello migliore per travel time
- `training_log.csv` — metriche per ogni episodio
- `training_state.json` — stato del training (episodio corrente, epsilon, ecc.)

Per vedere lo stato di un training interrotto:
```bash
cat results/metastgat_config1/training_state.json
```

---

## Riferimento

```bibtex
@article{wang2022metastgat,
  title={Meta-learning based spatial-temporal graph attention network for traffic signal control},
  author={Wang, Min and Wu, Libing and Li, Man and Wu, Dan and Shi, Xiaochuan and Ma, Chao},
  journal={Knowledge-Based Systems},
  volume={250},
  pages={109166},
  year={2022},
  publisher={Elsevier}
}
```
