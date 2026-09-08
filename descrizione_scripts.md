# Descrizione degli Script (`scripts/`)

Questa cartella contiene tutti gli script eseguibili (i "punti di ingresso") per lanciare esperimenti, addestrare i modelli, testarli, e generare o analizzare i dati per la tesi.

## Pipeline Principale
- **`train.py`**: È lo script fondamentale per l'addestramento. Instanzia l'ambiente CityFlow, carica il modello RL (es. MetaSTGAT, CoLight, DQN) e lo addestra su un file config specifico salvando checkpoint e log CSV.
- **`test.py`**: Carica un modello pre-addestrato (il miglior checkpoint salvato da `train.py`) e lo esegue sull'ambiente in fase di valutazione (senza esplorazione). Alla fine salva un `replay.txt` utile per guardare la simulazione graficamente.
- **`main.py`**: È un comodo "orchestratore". Se lanciato, esegue automaticamente prima il `train.py` per il numero di episodi indicato e, se va a buon fine, lancia subito dopo il `test.py` per valutare il modello appena salvato.

## Generalizzazione e Ablation
- **`train_generalization.py`**: Una variante speciale dell'addestramento ideata per testare la "Zero-shot Generalization". Addestra il modello su una griglia e lo testa su altre (modificando la config al volo) per dimostrare l'adattabilità della rete.
- **`ablation.py`**: Esegue lo studio di ablazione, addestrando e testando diverse "versioni depotenziate" del modello (es. togliendo un modulo) per dimostrare l'utilità di ogni suo singolo componente.
- **`plot_ablation.py`**: Legge i risultati prodotti da `ablation.py` e genera grafici (tramite Matplotlib/Seaborn) da inserire nella tesi.

## Generazione Dati
- **`generate_synthetic_data.py`**: Il generatore parametrico dei dataset sintetici che abbiamo appena perfezionato. Permette di creare reti M x N e flussi flat/peaks con volumi target esatti per creare nuovi stress test.

---

## 🧹 Note sulla Pulizia
- **`compare.py`**: Script obsoleto rimosso per mantenere la repository pulita (aveva percorsi fissi al vecchio CoLight).
