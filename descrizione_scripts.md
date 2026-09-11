# Descrizione degli Script (`scripts/`)

Questa cartella contiene tutti gli script eseguibili (i "punti di ingresso") per lanciare esperimenti, addestrare i modelli, testarli, e generare o analizzare i dati per la tesi.

## Pipeline Sperimentale (Tesi)
- **`main.py`**: **Punto di ingresso unico** della pipeline sperimentale. Esegue in-process (nessun subprocess, quindi debuggabile con breakpoint normali) allenamento + test di UN solo modello (`--model`, comodo per il debug) oppure di PIÙ modelli in sequenza (`--models`, incluso `all`), e alla fine chiama `plot_results.py` per generare grafici e tabelle di confronto. Contiene il `MODEL_REGISTRY` con gli 8 modelli confrontati (Pro, Paper, 4 ablation, FixedTime, MaxPressure) e la suddivisione train/validation/test (`DEFAULT_TRAIN_CONFIGS`/`DEFAULT_VALIDATION_CONFIG`/`DEFAULT_TEST_CONFIGS`, vedi `descrizione_configurazioni.md`).
- **`train.py`**: Script di allenamento. Istanzia l'ambiente CityFlow, costruisce il modello (MetaSTGAT/STGAT) con i flag di ablation richiesti, e lo allena su un'**unica** configurazione (`--config`). A fine training confronta in automatico l'ultimo modello e il best model su una config di validazione (`--select-best-config`) e salva il vincitore come `selected_model.pth`.
- **`test.py`**: Carica un checkpoint (priorità: `selected_model.pth` > `final_model.pth` > `best_model.pt`) e lo valuta su una config, salvando CSV per episodio, JSON di riepilogo e plot delle percentuali di fase.
- **`plot_results.py`**: Legge tutti i risultati in `results/` e genera i grafici (curve di training, percentuali di fase, barre di confronto TT/Throughput con linea dei veicoli totali) e le tabelle (CSV + LaTeX) comparative finali.

## Generazione Dati
- **`generate_synthetic_data.py`**: Il generatore parametrico dei dataset sintetici. Permette di creare reti M x N e flussi flat/peak/peaks con volumi target esatti.

---

## 🧹 Note sulla Pulizia

- **`compare.py`**: rimosso in precedenza (percorsi fissi al vecchio CoLight).
- **`run_experiment.py`**: rimosso — la sua logica (orchestrazione multi-modello + train + test + plot) è stata assorbita da `main.py`, che fa lo stesso lavoro in-process (più comodo da debuggare) e aggiunge la modalità a singolo modello.
- **`legacy/`**: script superati ma tenuti come riferimento, non più parte della pipeline attiva:
  - `train_generalization.py` — vecchio esperimento "a mano" di generalizzazione zero-shot, con path a config ormai rinominate. Sostituito dalla valutazione di generalizzazione su config di test integrata in `main.py`/`test.py`.
  - `ablation.py` / `plot_ablation.py` — vecchio studio di ablation "a componenti" (GAT-only/STGAT/MetaGAT/MetaLSTM/MetaSTGAT). Sostituito dal sistema di flag (`--ablation environment/temporal/rl_core/replay_stability`) orchestrato da `main.py`/`plot_results.py`.
