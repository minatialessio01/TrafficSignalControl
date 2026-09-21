# Analisi approfondita dei risultati (19/9/2026)

Analisi dei dati grezzi dietro le tre study del Cap. 4 (sweep α, ablation,
meccanismi spaziali), oltre le medie già riportate in tesi: tendenze
generali, comportamento per-configurazione, e cause plausibili. Dati da
`results/<modello>/test_summaries/*.json` (train: 3 config, nessun seed;
test: 7 config × 5 seed, media). Per le due config da 200m si usa
`wait_max_ns_full/ew_full` (vedi `maxwait-vision-cutoff-fix` in memoria),
per le altre 5 i campi originali — invariati.

Nota metodologica: i valori di trip completion (`tc`) citati qui, quando non
esplicitamente presi dalla Tabella~\ref{tab:tc-adjusted} della
tesi, sono la percentuale **grezza** (`avg_throughput / total_vehicles`),
non corretta per i veicoli strutturalmente impossibilitati ad arrivare in
tempo. Usarli solo per confronti relativi tra modelli sulla stessa config,
mai come valore assoluto da citare al posto del $tc$ ufficiale.

---

## 1. Sweep α: il trade-off è reale, monotono, e quasi sempre robusto per-config

**Tendenza generale** (confermata dai plot `diff_pct_alpha_{train,test}.png`
e `pro_alpha_{train,test}.png`): alzando α da 0 a 0.08, il travel time medio
peggiora di poco e in modo non monotono (+10.3%, +5.1%, +9.1% rispetto a
MaxPressure in training; +4.5%, -0.8%, +1.7% in test — α=0.04 è il punto
dolce sul travel time medio), mentre $tt_{\max}$ e max wait migliorano in
modo monotono e marcato (max wait: 885→690→630s in training, 1539→1126→954s
in test man mano che α cresce da 0 a 0.08).

**Causa**: $\mathrm{MaxRedWait}_i^t$ nella ricompensa custom penalizza
esplicitamente, a ogni singolo passo di decisione, l'attesa più lunga
osservata su un incrocio, pesata da α. Aumentare α aumenta direttamente
l'incentivo a servire la corsia più in sofferenza anche a costo di
qualche secondo di travel time medio in più altrove nella rete.

**Risultato particolare 1 — l'eccezione della config di training 1**: sulle
tre config di training, $tt_{\max}$ è monotono in α su Train2 e Train3, ma
si inverte su Train1 (α=0.08: 780s, α=0.04: 675s — α=0.04 vince). Sulle
7 config di test invece la monotonicità è perfetta, 7 casi su 7, senza
eccezioni. La singola eccezione sta esattamente sull'unica delle tre
config di training con l'arteria Est-Ovest sulla riga 1 (§ambiente): un
segnale che un singolo run di training può avere idiosincrasie locali che
la valutazione a 5 seed su generalizzazione invece assorbe — coerente con
la cautela già espressa in tesi sul singolo run per preset.

**Risultato particolare 2 — α=0.00 quasi-gridlock su 5x5 flat**: con la
sola ricompensa custom ma α=0 (nessun anti-starvation), `pro_0.00` tocca
$tt_{\max}=$wait$=1800.0$s (il tetto esatto della durata dell'episodio) su
4 dei 5 seed di `config_5x5_100m_9.4k_flat`, e ancora 1800/1800 su
`config_6x6_100m_11.5k_flat` (seed 1 e 4). Un veicolo resta bloccato dallo
spawn alla fine dell'episodio, letteralmente per l'intera simulazione,
nonostante stato avanzato, maschera azioni e vincolo di visibilità restino
tutti attivi. **Causa**: rimuovere solo il termine di ricompensa, lasciando
intatta tutta l'ingegnerizzazione dell'MDP, è sufficiente da solo a far
collassare la garanzia di equità sulle griglie più grandi mai viste in
training — prova diretta che è il termine di reward, non l'osservabilità
o la maschera, il meccanismo causale specifico.

---

## 2. Studio di ablation: una garanzia letterale contro lo starvation permanente

**Il risultato più netto di tutta l'analisi**: contando ogni singola
combinazione (modello, config di test, seed) — 15 modelli × 7 config × 5
seed — in cui $tt_{\max}$ o max wait raggiungono **1750s o più** (vicino o
uguale al tetto di 1800s dell'episodio, cioè un veicolo bloccato per quasi
o tutta la simulazione):

| Modello | N° istanze ≥1750s (su 35 max) |
|---|---|
| MaxPressure | 23 |
| `paper` | 18 |
| `pro_0.00` (α=0) | 12 |
| `environment` | 9 |
| Meta-GCN 1L | 5 |
| Meta-GCN 2L | 2 |
| `pro_0.04` (α=0.04) | 2 |
| Fixed-Time | **0** |
| Pro (α=0.08, riferimento) | **0** |
| `temporal` | **0** |
| `single_dqn` | **0** |
| `vanilla_buffer` | **0** |
| Pro 2L | **0** |
| Meta-SONAR L=2 | **0** |
| Meta-SONAR L=4 | **0** |

Otto modelli non toccano **mai** un blocco totale/quasi totale, su nessuna
delle 35 istanze di test, incluse le griglie 5x5 e 6x6 mai viste in
training. Sono esattamente: Fixed-Time (round-robin cieco: ogni fase, quindi
ogni direzione, riceve comunque verde ciclicamente, quindi nessuna corsia
può restare bloccata per sempre anche se il controllo è pessimo in media) e
tutte le varianti con **α=0.08 pieno E un meccanismo spaziale capace di
sfruttarlo** (GAT o SONAR, 1 o 2 layer, L=2 o L=4, con o senza Double DQN/PER/
Huber/BPTT).

**Causa (doppia condizione necessaria)**: MaxPressure, `paper` ed
`environment` non hanno alcun termine anti-starvation. `pro_0.00`/`pro_0.04`
hanno il meccanismo giusto (GAT) ma un peso insufficiente/nullo sul
termine. Meta-GCN ha il peso pieno (α=0.08, stessa ricompensa identica del
riferimento) ma **fallisce comunque** 7 volte su 70 istanze totali (1L+2L):
prova diretta che il solo segnale di reward non basta, serve un meccanismo
capace di propagare "questa corsia/vicino è in sofferenza" fino
all'intersezione responsabile — esattamente ciò che l'attention (GAT) o la
propagazione ondulatoria iterativa (SONAR) fanno e l'aggregazione a peso
fisso di GCN no (§3, confronto meccanismi spaziali).

**Risultato particolare — `single_dqn` e `vanilla_buffer` battono il
riferimento**: su $tt_{\max}$, `single_dqn` (Double DQN disattivato) è il
migliore dell'intero studio di ablation sia in training (485s) sia in test
(839s), meglio del riferimento Pro (630s/954s). `vanilla_buffer` (PER,
Huber, soft update, warm-up, esplorazione ciclica disattivati) è il
migliore su max wait in test (804s). Deviazione singolare:
`vanilla_buffer` in test ha un picco isolato su `6x6_peak` (wait=1065s,
contro 594-837s sulle altre 6 config) — l'unica config dove il suo margine
di vantaggio si restringe visibilmente, sulla combinazione più densa e più
grande mai vista in training. **Causa plausibile per il vantaggio
single\_dqn/vanilla\_buffer** (con la cautela già in tesi: singolo run,
niente ripetizioni indipendenti): Double DQN corregge un bias di
sovrastima del valore atteso che aiuta soprattutto quando le code sono già
frequenti nel replay buffer; PER pesa di più le transizioni con TD-error
alto, che potrebbero sovra-rappresentare proprio gli eventi di attesa
estrema, distorcendo l'apprendimento verso casi rari invece che verso la
politica media — disattivarli potrebbe aver ridotto per caso, in questo
specifico run, la varianza dannosa più di quanto abbia tolto stabilità.
Da verificare con più ripetizioni indipendenti, non disponibili qui.

---

## 3. Meccanismi spaziali: l'architettura conta quanto la ricompensa

**Meta-GCN è l'unico meccanismo che non replica il comportamento di base**,
nonostante ricompensa, stato e maschera identici al riferimento:
- Trip completion sistematicamente il più basso di ogni modello RL su ogni
  singola config di test (Tabella~\ref{tab:tc-adjusted}: mai sopra
  l'89.6%, spesso sotto l'86%, unico caso ricorrente sotto il 90% oltre
  Fixed-Time).
- $tt_{\max}$/max wait a 1L addirittura **peggiori di MaxPressure** in
  training (1280s contro 955s) e sostanzialmente alla pari con MaxPressure
  anche in test (1651s contro 1626s) — l'unico modello RL di tutto il
  lavoro che non supera MaxPressure su questa metrica.
- Il secondo layer aiuta (1L→2L: max wait di test 1651→1394s) ma non
  colma il divario con GAT/SONAR (954-1067s).

**Causa**: descritta in §confronto-spaziale, un peso di aggregazione
fissato dalla sola topologia (grado del nodo/adiacenza) non può imparare a
dare più peso al segnale di un vicino specifico che sta soffrendo di
starvation — l'attention di GAT calcola invece un punteggio di
compatibilità appreso per ogni coppia di nodi, che può spostarsi
dinamicamente verso il vicino più urgente. La propagazione ondulatoria di
SONAR ottiene un effetto simile per via iterativa (L passi di
propagazione) invece che per pesi appresi diretti.

**Risultato particolare — SONAR L=2 generalizza meglio su griglie grandi ma
peggio su 4x4**: confrontando max wait per singola config di test, SONAR
L=2 batte il riferimento GAT su 5x5 flat/peak e 6x6 flat/peak (es. 6x6
flat: 885s contro 1137s) ma perde su `4x4_100m_6k_peak` (960s contro 798s)
— l'unica config di test topologicamente più vicina alla scala di
training. **Causa plausibile**: L=2 ricorrenze di propagazione danno un
campo recettivo effettivo adatto a griglie più grandi (l'informazione deve
"viaggiare" più lontano per raggiungere l'intersezione giusta), a costo di
una minore specificità locale sulla scala di training dove GAT, con il suo
punteggio di compatibilità diretto invece che propagato, resta più
preciso. Non testato: SONAR con L variabile in funzione della dimensione
della griglia.

---

## 4. Vista d'insieme: una firma comune, che si rompe in modo prevedibile

I quattro plot `diff_pct_*.png` mostrano la stessa firma per **ogni**
modello con α=0.08 pieno e meccanismo spaziale capace (GAT o SONAR,
qualunque profondità/L, con o senza le tecniche di stabilità RL):
travel time medio tra -1% e +10% rispetto a MaxPressure, $tt_{\max}$ tra
-28% e -44% rispetto a MaxPressure, in entrambi i regimi. La firma è
sorprendentemente stabile su 8 varianti diverse (5 preset di ablation + 3
profondità/architetture) — non è un risultato fragile legato a un singolo
iperparametro.

La firma si rompe esattamente in tre modi, ciascuno isolando una causa
diversa:
1. **Rimuovere/annacquare il termine di reward** (α=0.00/0.04, `paper`):
   il vantaggio su $tt_{\max}$ crolla verso 0 o quasi (α=0: -4.2% in test,
   `paper`: -7.3% in test) — causa: manca l'incentivo diretto.
2. **Rimuovere insieme visibilità, segnale di attesa e maschera**
   (`environment`): il vantaggio scende quasi a 0 (-2.2% in test) pur con
   α=0.08 pieno — causa: la ricompensa da sola non può guidare una policy
   che non riceve il segnale che dovrebbe ottimizzare.
3. **Cambiare il meccanismo spaziale con uno a peso fisso** (Meta-GCN): il
   vantaggio non solo crolla ma si inverte (+3.8% peggio di MaxPressure su
   $tt_{\max}$ in test per 1L) — causa: né la ricompensa né lo stato
   bastano senza un meccanismo capace di instradare il segnale verso
   l'intersezione giusta.

Il fatto che tre interventi concettualmente distinti (reward, MDP
engineering, architettura) producano lo stesso tipo di collasso della
stessa metrica supporta la lettura che i tre ingredienti sono
complementari, non ridondanti: rimuoverne uno qualsiasi elimina il
beneficio, indipendentemente da quale.
