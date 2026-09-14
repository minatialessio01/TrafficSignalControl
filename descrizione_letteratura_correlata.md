# Letteratura correlata: come gli SOTA battono MaxPressure

Sintesi di cinque metodi pubblicati che superano MaxPressure sul travel time,
letti per verificare se le scelte di design di questo progetto (stato, reward,
meta-learner) hanno un riscontro indipendente in letteratura. Utile come
materiale per il Cap. 2 (Background, taxonomy del controllo appreso) e per la
discussione critica sul meta-learner in Cap. 3/Cap. 5 (vedi
`descrizione_stato_meta_reward.md` §2, che cita la conclusione di questo
documento).

- **PressLight** (Wei et al., KDD 2019) — reward di pressione con base teorica
- **MPLight / "Toward a Thousand Lights"** (Chen et al., AAAI 2020) — pressione
  anche nello stato, non solo nella reward
- **CoLight** (Wei et al., CIKM 2019) — cooperazione tra incroci via GAT
- **AttendLight** (Oroojlooy et al., NeurIPS 2020) — modello universale via
  attention, indipendente da fasi/corsie/topologia
- **MaCAR** (Yu et al., IJCAI 2020) — comunicazione attiva + correzione del
  valore dell'azione tramite previsione del traffico

**Filo comune tra tutti e cinque**: nessuno usa una struttura "meta" che genera
pesi dinamici per un'altra rete. Tutti mettono l'informazione utile (pressione,
storia, contesto dei vicini) o direttamente nell'input di una rete a pesi
fissi (condivisi tra incroci), o la aggregano con un'attenzione che fa parte
del forward principale — mai un layer a parte che genera i pesi di un secondo
modulo. Ripreso in §6.

---

## 1. PressLight — la reward giusta conta più dello stato

**Il problema che risolvono**: una reward pesata a mano
(`w1*queue_length + w2*pressure`, la stessa famiglia di formula della reward
custom di questo progetto, pesata da `alpha`) dà risultati molto diversi al
variare dei pesi. La loro soluzione non è tarare meglio i pesi, è eliminare la
combinazione pesata: usano `r_i = -P_i`, la pressione pura, dimostrata
analiticamente (non solo empiricamente) equivalente alla massimizzazione del
throughput sotto le equazioni di evoluzione del traffico di Varaiya (2013) —
la stessa dimostrazione alla base di MaxPressure stesso, qui riusata per
giustificare la reward invece dell'azione.

**Stato**: fase corrente + veicoli su ciascuna corsia in uscita + veicoli su
**3 segmenti** di ciascuna corsia in ingresso (vicino/medio/lontano dal
semaforo) — tutto direttamente nello stato di un DQN semplice, nessun
meta-learner.

**Ablation study (Tabella 5, dati sintetici)** — il confronto più diretto per
la domanda "cosa aggiungere, stato o reward":

| Variante | Cosa aggiunge rispetto alla precedente | TT (HeavyFlat) |
|---|---|---|
| LIT | fase + veicoli in ingresso (baseline) | 233.17 |
| LIT+out | + veicoli in uscita | 201.56 |
| LIT+out+seg | + segmentazione delle corsie in ingresso | 200.28 |
| PressLight | stesso stato, reward cambiata in pressione pura | **160.48** |

Il salto da LIT+out+seg a PressLight (stesso stato!) è più grande di tutti
quelli precedenti messi insieme: il guadagno maggiore viene dalla reward, non
dallo stato. Coerente con l'esito, in questo progetto, del tentativo di
riformulare la reward custom come pressione pura (`use_pressure_reward_term`,
vedi `descrizione_stato_meta_reward.md` §5): nessun guadagno — differenza
principale, il nostro esperimento ha sostituito solo il termine throughput
mantenendo comunque il termine di equità esplicito (`alpha`), non è passato
alla pressione pura senza quel termine come fa PressLight.

**Idea non ripresa**: la segmentazione delle corsie (K=3, vicino/medio/lontano)
è un affinamento di `n_vec`/`wait_vec` che questo progetto non ha — il
`VISION_CUTOFF_M` qui è un taglio binario (dentro/fuori raggio), non una
segmentazione a gradini.

---

## 2. MPLight — la pressione va anche nello stato, non solo nella reward

Estende PressLight cambiando lo stato: invece di "veicoli per corsia",
l'osservazione diventa la pressione stessa di ciascuno dei movimenti/fasi
(zero-padding se una fase ha meno movimenti) — esattamente l'idea sperimentata
in questo progetto come `use_phase_pressure_state` (vedi
`descrizione_stato_meta_reward.md` §1): dare in input al modello la stessa
quantità che MaxPressure calcola per scegliere, non farla dedurre
indirettamente dal conteggio grezzo delle corsie.

**Prova di generalità del meccanismo (Tabella 4)**: aggiungono "+pressure" a
tre modelli di base diversi (GCN, PressLight, FRAP) e migliora sempre tutti e
tre:

| Modello | Senza pressione | Con pressione |
|---|---|---|
| GCN | 653.45 | 646.47 |
| PressLight | 654.04\* | 600.42 |
| FRAP | 512.70 | 472.51 |

\* PressLight-senza-pressure, cioè con la reward tornata a queue length invece
che pressione pura.

Testato fino a **2510 incroci reali** (Manhattan): la pressione come feature
di stato scala bene, non è un trucco valido solo su reti piccole.

---

## 3. CoLight — l'attenzione sta nel forward principale, non in un layer "meta"

Il risultato più rilevante per la domanda sull'utilità del meta-learner.
CoLight usa un GAT per la cooperazione tra incroci, ma il GAT **è la rete
principale**: prende l'embedding grezzo dell'osservazione (`h_i = Embed(o_i)`,
un singolo MLP) e fa message-passing direttamente su quello — non genera pesi
per un'altra rete.

Due dettagli tecnici verificabili indipendentemente rispetto a questo
progetto:

1. **L'incrocio include sé stesso nel proprio vicinato** ("intersection i
   itself is also included in N_i to help the agent get aware of how much
   attention should be put on its own traffic condition") — un self-loop
   nell'attenzione, esplicitamente motivato. Conferma indipendente, da un
   articolo separato, del beneficio dei self-loop misurato su MetaSTGAT in
   questo progetto (self-loop permanente in `MetaSTGAT.forward`, vedi
   `descrizione_modelli.md` §4): due percorsi diversi arrivano alla stessa
   conclusione.
2. **L'attenzione impara da sola la differenza upstream/downstream e
   arteria/strada-laterale** (Figura 7 dell'articolo), partendo solo da
   conteggi di veicoli grezzi e topologia — nessuna feature ingegnerizzata a
   mano per dire "questo vicino è più importante". Prova empirica che
   l'attenzione *può* imparare a inferire il ruolo di un vicino dalla sola
   struttura del grafo, senza feature dedicate come `pressure_diff_neighbors`
   o `traffic_asymmetry_ns_ew` di questo progetto (vedi
   `descrizione_stato_meta_reward.md` §3) — qui aggiunte esplicitamente
   perché il GAT usato non è quello di CoLight (pesi generati dal
   meta-learner, non appresi end-to-end sull'embedding grezzo).

**Rilevanza**: se in futuro si sperimenta la rimozione del meta-learner,
CoLight è il riferimento diretto per come strutturare l'alternativa — GAT
standard sull'embedding dello stato (esattamente `StandardGATLayer` in
`src/models/stgat.py`, mai allenato in questo progetto), non un GAT con pesi
generati da un meta-embedding separato.

---

## 4. AttendLight — generalità senza fissare corsie/fasi a priori

Tocca direttamente il compromesso di generalizzabilità discusso in
`descrizione_stato_meta_reward.md` §6(B): la pressione per fase, presa 1:1 da
MaxPressure, richiede di fissare `phase_lanelinks` — una struttura specifica
per schema semaforico e topologia. AttendLight risolve lo stesso problema con
due meccanismi di attenzione:

- **state-attention**: aggrega le corsie partecipanti a una fase con pesi
  appresi (non una media fissa) in una rappresentazione di dimensione fissa,
  indipendente da quante corsie/movimenti ha quella fase;
- **action-attention**: sceglie la fase successiva tra un numero variabile di
  fasi, senza assumerne un numero fisso.

Un solo modello allenato su una collezione eterogenea di intersezioni (3-way,
4-way, 1-3 corsie, 2-8 fasi) generalizza a configurazioni mai viste con un
gap contenuto (13% ATT rispetto al modello specializzato), recuperabile quasi
del tutto con **200 episodi di fine-tuning mirato** (non 100.000 da zero).

**Verifica di controllo (Appendice A.6)**: sostituendo la state-attention con
una semplice media delle corsie, il risultato peggiora sistematicamente —
l'attenzione non è un abbellimento, pesare invece di mediare misura un
vantaggio reale (resta comunque un meccanismo dentro il forward principale,
non un generatore di pesi per un secondo modulo).

**Rilevanza**: riferimento diretto se in futuro si vuole rendere il progetto
indipendente dallo schema fisso a 8 fasi/12 corsie (oggi hard-coded) — non
necessario per chiudere il gap con MaxPressure sulle topologie già testate
(4x4/5x5/6x6 condividono lo stesso schema di fasi), rilevante solo per
un'estensione a schemi diversi.

---

## 5. MaCAR — comunicazione attiva e correzione del valore dell'azione

Idea distinta e più invasiva delle altre: quando più agenti agiscono in
sincrono, il valore percepito della propria azione è distorto dal fatto che
non si conoscono le azioni degli altri finché non se ne osservano gli effetti
(un ritardo strutturale). MaCAR aggiunge una **Traffic Forecasting Network**
che prevede il traffico futuro e il valore dell'azione *dato* il traffico
previsto, e usa la differenza tra valore previsto e valore osservato per
correggere l'azione durante il training.

Il loro ablation (rimuovere la TFN, "MaCAR-noTFN") mostra un peggioramento
netto ma non totale — la rete di comunicazione da sola già porta la maggior
parte del beneficio.

**Rilevanza**: l'idea più lontana dallo stato attuale del progetto
(richiederebbe una rete di previsione separata, allenata con una loss
propria) — citata per completezza bibliografica, non proposta come prossimo
passo; utile solo se in futuro si vuole esplorare esplicitamente la
correzione del bias da azioni sincrone tra incroci vicini.

---

## 6. Sul meta-learner: cosa dice la letteratura

Messo insieme, il pattern è netto: **PressLight, MPLight, CoLight e
AttendLight — tutti e quattro i metodi che battono MaxPressure in questi
articoli — non usano una struttura che genera pesi dinamici per un altro
modulo**. Usano reti a pesi fissi (condivise tra incroci) con input ricchi
(pressione, segmenti, storia) o attenzione che fa parte del calcolo principale
(CoLight, AttendLight). L'idea "meta" di MetaSTGAT (SMK/TMK che generano i
pesi di GAT/LSTM) non ha un analogo diretto in nessuno dei cinque.

Non è una prova che il meta-learner sia sbagliato — è una prova che **non è
l'ingrediente che rende questi metodi migliori di MaxPressure**: il vantaggio
lì viene da dove va l'informazione (stato vs reward) e da come si aggrega
(attenzione vs concatenazione fissa), non da un meccanismo di generazione
dinamica dei pesi. Per la giustificazione usata in questo progetto per
mantenere comunque la struttura meta (fedeltà al paper originale + motivazione
di ricerca), vedi `descrizione_stato_meta_reward.md` §2.

**Esperimento non ancora fatto che risolverebbe la domanda per questo
progetto specifico**: `src/models/stgat.py` è già MetaSTGAT con GAT/LSTM a
pesi fissi invece che generati dal meta-learner — la stessa identica
architettura salvo la parte "meta" — mai allenato in questo progetto (solo
referenziato nel codice). Allenandolo con le stesse feature (pressione,
segmenti) date in pasto a MetaSTGAT e nelle stesse condizioni, un confronto
diretto MetaSTGAT-vs-STGAT direbbe, per questo problema specifico, se il
meta-learner aggiunge qualcosa di misurabile — invece di decidere sulla base
di cosa funziona in altri articoli con altri stati/reward. Se STGAT va uguale
o meglio, è un risultato di tesi genuino (coerente con quanto visto qui); se
MetaSTGAT lo batte comunque, la struttura meta è giustificata dai dati, non
solo dalla fedeltà al paper originale.
