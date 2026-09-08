# Descrizione del Codice Sorgente (`src/`)

Questa cartella rappresenta il "cuore" del framework sviluppato per la tesi. Contiene tutta l'infrastruttura algoritmica, l'architettura delle reti neurali e le definizioni degli agenti.

## `src/environment/`
Interfaccia diretta con il motore di CityFlow.
- **`cityflow_env.py`**: Definisce l'ambiente aderendo agli standard OpenAI Gym/PettingZoo. Qui viene implementata la logica di interazione con l'ambiente: la funzione `step` per avanzare la simulazione, il calcolo della `reward` (basata sulle code) e l'osservazione dello stato.
- **`graph_builder.py`**: Crea il grafo stradale matematico per il passaggio di messaggi tramite GNN (PyTorch Geometric). Calcola quali incroci sono adiacenti.

## `src/agents/`
Contiene le diverse strategie e agenti di controllo dei semafori con cui MetaSTGAT verrà confrontato:
- **`dqn_agent.py`**: Implementazione base di una Deep Q-Network indipendente (senza collaborazione tra semafori).
- **`fixedtime_agent.py`**: Baseline non intelligente: cicla le fasi semaforiche a intervalli di tempo predefiniti fissi.
- **`maxpressure_agent.py`**: Baseline algoritmica "MaxPressure" che cambia fase basandosi unicamente sulla differenza (pressione) tra auto in arrivo e spazio in uscita.
- **`replay_buffer.py`**: La memoria degli agenti RL (Experience Replay) usata per campionare batch durante l'addestramento, essenziale per stabilizzare DQN e le GNN.

## `src/models/`
Contiene l'implementazione in PyTorch della vera innovazione proposta dalla tesi: l'architettura **MetaSTGAT**.
- **`metastgat.py`**: Il file principale del modello. Assembla i vari sottomoduli.
- **`stgat.py`**: L'implementazione del blocco di *Spatio-Temporal Graph Attention Network*.
- **`meta_knowledge_learner.py`**: Il modulo "Meta" che impara un set base di pesi in grado di generalizzare in fretta su nuove mappe mai viste.
- **`meta_gat.py` / `meta_lstm.py` / `state_encoder.py`**: Sotto-moduli e layer neurali custom (come l'encoder spaziale tramite GAT e quello temporale tramite LSTM) combinati all'interno della rete per comprendere l'evoluzione spaziotemporale del traffico.

## `src/utils/`
File di supporto e logistica:
- **`logger.py`**: Si occupa di salvare i log (`training_log.csv`) e scrivere nel terminale i progressi (tempo di viaggio, throughput) in maniera strutturata e uniforme per tutti gli script.
- **`metrics.py`**: Funzioni matematiche per calcolare metriche specifiche e stampare i risultati di validazione per la tesi.

---

### Stato della cartella `src/`
I file in `src/` sono tutti **estremamente coesi e utili**. Non ci sono file spazzatura, obsoleti o doppioni. L'architettura software a moduli è molto pulita e ben ingegnerizzata. L'unica cartella che richiede attenzione o pulizia è `scripts/` (vedi `descrizione_scripts.md`).
