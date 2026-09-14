"""
Wrapper CityFlow → MDP (Markov Decision Process).

Implementa il Reinforcement Learning environment per il controllo semaforico
multi-intersezione, come descritto in Section 3.2 dell'articolo MetaSTGAT.

MDP:
  - State  s_i^t = [n_vec (12 corsie), p_vec (8 fasi)] → dim 20
  - Action a_i^t = indice di fase (0..7)
  - Reward r_i   = -P_i  (negativo della pressione dell'intersezione)

La pressione P_i = Σ(veicoli in ingresso) - Σ(veicoli in uscita) (Fig. 2)

Fasi semaforiche (Fig. 1 del paper) – nomenclatura N/S/E/W + T/L:
  N=Nord, S=Sud, E=Est, W=Ovest; T=Through (dritto), L=Left (sinistra)
  Fase 0 – NTST : Nord dritto + Sud dritto
  Fase 1 – NLSL : Nord sinistra + Sud sinistra
  Fase 2 – NTNL : Nord dritto + Nord sinistra  (solo Nord)
  Fase 3 – STSL : Sud dritto + Sud sinistra    (solo Sud)
  Fase 4 – WTET : Ovest dritto + Est dritto
  Fase 5 – WLEL : Ovest sinistra + Est sinistra
  Fase 6 – ETEL : Est dritto + Est sinistra    (solo Est)
  Fase 7 – WTWL : Ovest dritto + Ovest sinistra (solo Ovest)

NOTA IMPORTANTE sulla svolta a destra:
  La svolta a destra NON è più sempre verde (free-right). Ogni veicolo
  che intende svoltare a destra deve aspettare la fase semaforica che lo
  autorizza, esattamente come le svolte a sinistra e il dritto.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# CityFlow è disponibile solo su Linux/WSL/Docker
try:
    import cityflow
    CITYFLOW_AVAILABLE = True
except ImportError:
    CITYFLOW_AVAILABLE = False
    print("[WARN] CityFlow non trovato. Usa setup_env.sh o Docker per installarlo.")


# ─── Costanti (dall'articolo) ──────────────────────────────────────────────────
N_PHASES = 8                  # fasi semaforiche per intersezione (Fig. 1)
                               # (NTST, NLSL, NTNL, STSL, WTET, WLEL, ETEL, WTWL)
N_LANES = 12                  # corsie per intersezione (Section 3.1)
STATE_DIM_FULL = N_LANES + N_LANES + N_PHASES  # dim stato avanzato: 32 (con wait_vec)
STATE_DIM_PAPER = N_LANES + N_PHASES            # dim stato originale paper: 20 (senza wait_vec)
STATE_DIM = STATE_DIM_FULL   # default: usa il completo; sovrascrive con STATE_DIM_PAPER se --no-wait-vec

GREEN_TIME = 10               # durata verde (s)
YELLOW_TIME = 3               # durata giallo (s) — trattato come verde (vedi nota sotto)
RED_TIME = 2                  # durata tutto-rosso reale (s)
STEP_TIME = GREEN_TIME + YELLOW_TIME + RED_TIME  # 15s per ciclo

# Fasi standard: in CityFlow le fasi si mappano a indici 0..7
# La svolta a destra è controllata dal semaforo (non più sempre verde).
ALL_PHASES = list(range(N_PHASES))

# Fase di tutto-rosso (nessun movimento abilitato), aggiunta come 9° elemento di
# "lightphases" in ciascun roadnet (vedi data/roadnet_*.json, indice = N_PHASES).
# Non è mai scelta dall'agente (lo spazio delle azioni resta 0..7): step() la
# imposta internamente per RED_TIME secondi tra un'azione e la successiva, per
# svuotare l'incrocio prima che il prossimo verde liberi un movimento conflittuale.
ALL_RED_PHASE = N_PHASES

# ─── Campo visivo (cutoff) ──────────────────────────────────────────────────
# Raggio entro cui un veicolo in avvicinamento è "rilevante" per lo stato/reward
# dell'intersezione: la distanza che un veicolo può percorrere nella finestra di
# tempo in cui la fase può ancora farlo muovere prima del prossimo cambio, cioè
# GREEN_TIME + YELLOW_TIME (il tutto-rosso, RED_TIME, non fa avanzare nessuno,
# quindi non allunga la portata utile). VEHICLE_SPEED_MS ~40 km/h, velocità
# urbana di riferimento. Se GREEN_TIME/YELLOW_TIME cambiano, questo valore si
# ricalcola da solo — non è più un numero fisso scollegato dalla dinamica reale.
VEHICLE_SPEED_MS = 11.1                                    # ~40 km/h
VISION_CUTOFF_M = (GREEN_TIME + YELLOW_TIME) * VEHICLE_SPEED_MS  # 13 * 11.1 = 144.3 m

# Mappatura delle corsie con semaforo verde per ogni fase (indici 0-11, ordinamento canonico)
GREEN_LANES_PER_PHASE = {
    0: [1, 2, 4, 5],       # NTST: Nord Dritto/Destra (1,2) + Sud Dritto/Destra (4,5)
    1: [0, 3],             # NLSL: Nord Sinistra (0) + Sud Sinistra (3)
    2: [0, 1, 2],          # NTNL: Nord Sinistra/Dritto/Destra (0,1,2)
    3: [3, 4, 5],          # STSL: Sud Sinistra/Dritto/Destra (3,4,5)
    4: [7, 8, 10, 11],     # WTET: Ovest Dritto/Destra (7,8) + Est Dritto/Destra (10,11)
    5: [6, 9],             # WLEL: Ovest Sinistra (6) + Est Sinistra (9)
    6: [9, 10, 11],        # ETEL: Est Sinistra/Dritto/Destra (9,10,11)
    7: [6, 7, 8]           # WTWL: Ovest Sinistra/Dritto/Destra (6,7,8)
}


class CityFlowEnv:
    """
    Ambiente multi-agente per il controllo semaforico con CityFlow.

    Ogni agente corrisponde a un'intersezione semaforizzata.
    Gli agenti osservano sia il proprio stato che quello dei vicini,
    per poi selezionare una fase semaforica.
    """

    def __init__(
        self,
        config_path: str,
        num_neighbors: int = 4,
        alpha: float = 0.5,
        # ── Ablation flags ──────────────────────────────────────────────────
        reward_mode: str = "custom",        # "custom" = weighted pressure; "paper" = -P_i
        use_vision_cutoff: bool = True,     # True = cutoff VISION_CUTOFF_M; False = visibilità globale
        use_wait_vec: bool = True,          # True = stato a 32 dim; False = 20 dim (no wait)
        use_action_mask: bool = True,       # True = maschera anti-starvation; False = nessuna
        use_phase_pressure_meta: bool = True,  # True = SMK include phase_pressure (meta_v3, solo SMK); False = meta_v2 (18/25 dim)
        use_pressure_reward_term: bool = False,  # True = 1o termine del reward custom = pressione (stile paper) invece di throughput
        use_phase_pressure_state: bool = False,  # True = aggiunge phase_pressure (8 dim) allo STATO principale
    ):
        """
        Args:
            config_path: percorso al file config.json di CityFlow
            num_neighbors: numero massimo di intersezioni vicine (paper: 4)
            alpha: iperparametro per la penalità dei ritardi nel reward
            reward_mode: "custom" (weighted pressure avanzata) o "paper" (-P_i originale)
            use_vision_cutoff: se True limita la visibilità a VISION_CUTOFF_M dal semaforo
            use_wait_vec: se True include wait_vec nello stato (dim=32); se False stato a 20 dim
            use_action_mask: se True abilita la maschera anti-starvation
            use_phase_pressure_meta: se True (default) il SMK include la feature
                phase_pressure (N_PHASES dim, esperimento meta_v3, 13/9/2026, vedi
                descrizione_stato_meta_reward.md) -- SMK 18->26, TMK invariato a 25 (deciso il
                13/9/2026: phase_pressure va solo nel SMK, non nel TMK -- e' un
                dato spaziale per fase, non una dinamica temporale). Se False,
                il SMK resta alla revisione meta_v2 del 12/9/2026 -- dim 18.
                Serve SOLO per poter riprendere (--resume) i checkpoint allenati
                con meta_v2 prima che phase_pressure diventasse la feature di
                default: la dimensione dei pesi di smk_learner.fc1 dipende da
                questa dimensione, quindi un checkpoint meta_v2 non si carica in
                un modello costruito con le dimensioni meta_v3 (e viceversa).
            use_pressure_reward_term: se True, nel reward_mode="custom" il primo
                termine (di norma "passed - incoming", throughput) e' sostituito da
                "outgoing - incoming_ora" (-P_i, stessa identica quantita' calcolata
                dal reward_mode="paper"), ma qui SOLO per il primo termine: il resto della
                formula custom (penalita' anti-starvation pesata da alpha,
                wasted_green_penalty, normalizzazione /100, clip [-20,5]) resta
                invariato. Esperimento 13/9/2026 (vedi descrizione_stato_meta_reward.md): vedere se
                allenare direttamente sulla pressione, invece che sul throughput,
                aiuta il modello ad avvicinarsi al comportamento di MaxPressure.
                Ignorato se reward_mode="paper" (che ha gia' -P_i come unico termine).
            use_phase_pressure_state: se True, aggiunge allo stato principale
                (`_get_observations`) un vettore di 8 valori (`N_PHASES`), uno per
                fase candidata, con la stessa formula (e la stessa funzione,
                `_compute_phase_pressure_vector`) usata dal SMK con
                use_phase_pressure_meta -- dim 32->40 (con wait_vec) o 20->28
                (senza). Rispetta `use_vision_cutoff` come tutto il resto dello
                stato (vedi descrizione_stato_meta_reward.md §1). Esperimento 13/9/2026: la stessa
                informazione, spostata dal meta-learner allo stato diretto, per
                vedere se aiuta di piu' quando la Q-network la vede direttamente
                invece che tramite pesi generati dal meta-learner.
        """
        if not CITYFLOW_AVAILABLE:
            raise RuntimeError(
                "CityFlow non installato. Esegui 'pip install cityflow' "
                "o usa Docker/WSL con setup_env.sh"
            )

        self.config_path = config_path
        self.num_neighbors = num_neighbors
        self.alpha = alpha
        # Ablation flags
        self.reward_mode = reward_mode
        self.use_vision_cutoff = use_vision_cutoff
        self.use_wait_vec = use_wait_vec
        self.use_action_mask = use_action_mask
        self.use_phase_pressure_meta = use_phase_pressure_meta
        self.use_pressure_reward_term = use_pressure_reward_term
        self.use_phase_pressure_state = use_phase_pressure_state

        # Carica la configurazione
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Carica il roadnet per estrarre la topologia del grafo
        roadnet_path = self._resolve_path(self.config.get("roadnetFile", ""))
        with open(roadnet_path, "r") as f:
            self.roadnet = json.load(f)

        # Inizializza CityFlow
        self.engine = cityflow.Engine(config_path, thread_num=1)

        # Estrai le intersezioni semaforizzate (non virtuali)
        self.intersections = self._get_signalized_intersections()
        self.n_intersections = len(self.intersections)
        self.inter_ids = [i["id"] for i in self.intersections]

        # Mappa id -> indice numerico
        self.inter_id_to_idx = {iid: idx for idx, iid in enumerate(self.inter_ids)}

        # Costruisci la lista di adiacenza (vicini di ogni intersezione)
        self.adjacency = self._build_adjacency()

        # Mappa corsie per intersezione
        # Mappa corsie per intersezione
        self.inter_lanes = self._get_lanes_per_intersection()
        
        # Distanze e lunghezze strade
        self.road_lengths = self._get_road_lengths()
        self.inter_distances = self._compute_distances()

        # Cache della pressione media per nodo, un valore per intersezione,
        # ricalcolata una sola volta per step (non per ogni singola chiamata di
        # get_spatial_meta_features) e riusata per calcolare pressure_diff_neighbors
        # di ogni nodo -- vedi descrizione_stato_meta_reward.md §2.5. Invalidata in reset().
        self._node_pressure_cache: Dict[str, float] = {}
        self._node_pressure_cache_step: Optional[int] = None

        # Movimenti (lane-link) abilitati per fase, dal roadnet — usati da
        # MaxPressureAgent per la pressione classica (Varaiya 2013), vedi
        # _build_phase_lanelinks().
        self.phase_lanelinks = self._build_phase_lanelinks()

        # Per ogni corsia in ingresso, le corsie in uscita raggiungibili dal
        # roadnet (indipendentemente da quale fase le abiliti) -- usata da
        # _compute_lane_pressure_vector per una media LOCALE (solo le uscite
        # che quella specifica corsia puo' davvero raggiungere), non globale
        # su tutte le uscite dell'incrocio. Corretto il 13/9/2026 (vedi
        # descrizione_stato_meta_reward.md), su segnalazione: la versione precedente usava
        # la stessa media (su TUTTE le corsie in uscita dell'incrocio) per
        # ogni corsia in ingresso, indipendentemente da quali uscite fossero
        # davvero raggiungibili da quella corsia.
        self.lane_reachable_outgoing = self._build_lane_reachable_outgoing()

        # Stato corrente
        self.current_step = 0
        self.current_phase = {iid: 0 for iid in self.inter_ids}
        self.consecutive_phases = {iid: 0 for iid in self.inter_ids}

        # Statistiche episodio
        self.episode_travel_times = []
        self.episode_throughput = 0

    # ─── Setup ────────────────────────────────────────────────────────────────

    def _resolve_path(self, rel_path: str) -> str:
        """Risolve un percorso relativo alla config."""
        base_dir = self.config.get("dir", "./")
        
        # 1. Prova a risolvere rispetto al CWD (come fa il motore C++ di CityFlow)
        path_cwd = os.path.join(base_dir, rel_path)
        if os.path.exists(path_cwd):
            return path_cwd
            
        # 2. Fallback: prova a risolvere rispetto alla cartella del config file
        config_dir = os.path.dirname(os.path.abspath(self.config_path))
        if not os.path.isabs(rel_path):
            full_path = os.path.join(config_dir, base_dir, rel_path)
            if os.path.exists(full_path):
                return full_path
            full_path = os.path.join(config_dir, rel_path)
        return rel_path

    def _get_signalized_intersections(self) -> List[dict]:
        """Restituisce solo le intersezioni non virtuali con semafori."""
        result = []
        for inter in self.roadnet.get("intersections", []):
            if not inter.get("virtual", True):
                tl = inter.get("trafficLight", {})
                if tl.get("lightphases"):
                    result.append(inter)
        return result

    def _build_adjacency(self) -> Dict[str, List[str]]:
        """
        Costruisce la lista di adiacenza dal roadnet.
        Un'intersezione j è vicina di i se esiste una strada diretta tra loro.
        Limita al massimo a num_neighbors vicini (i più vicini in distanza).
        """
        # Mappa id -> posizione (x, y)
        positions = {}
        for inter in self.roadnet.get("intersections", []):
            pt = inter.get("point", {})
            positions[inter["id"]] = (pt.get("x", 0), pt.get("y", 0))

        # Mappa id -> set vicini (da roads)
        raw_neighbors: Dict[str, set] = {iid: set() for iid in self.inter_ids}

        for road in self.roadnet.get("roads", []):
            src = road.get("startIntersection")
            dst = road.get("endIntersection")
            if src in raw_neighbors and dst in raw_neighbors and src != dst:
                raw_neighbors[src].add(dst)
                raw_neighbors[dst].add(src)

        # Ordina per distanza e tronca a num_neighbors
        adjacency = {}
        for iid in self.inter_ids:
            neighbors_list = list(raw_neighbors[iid] & set(self.inter_ids))
            if len(neighbors_list) > self.num_neighbors:
                px, py = positions.get(iid, (0, 0))
                neighbors_list.sort(key=lambda n: (
                    (positions.get(n, (0, 0))[0] - px) ** 2 +
                    (positions.get(n, (0, 0))[1] - py) ** 2
                ))
                neighbors_list = neighbors_list[:self.num_neighbors]
            adjacency[iid] = neighbors_list

        return adjacency

    def _get_lanes_per_intersection(self) -> Dict[str, List[str]]:
        """Mappa ogni intersezione alle sue corsie in ingresso, ordinate deterministicamente."""
        inter_lanes = {iid: [] for iid in self.inter_ids}
        
        for inter in self.roadnet.get("intersections", []):
            iid = inter.get("id")
            if iid not in inter_lanes:
                continue
            
            seen_roads = []
            for rl in inter.get("roadLinks", []):
                r = rl.get("startRoad")
                if r not in seen_roads:
                    seen_roads.append(r)
            
            # Identifica la direzione di provenienza per ogni strada
            dir_to_road = {}
            for road_id in seen_roads:
                parts = road_id.split('_')
                if len(parts) >= 6:
                    try:
                        sr, sc = int(parts[1]), int(parts[2])
                        er, ec = int(parts[4]), int(parts[5])
                        if sr > er: d = "N"
                        elif sr < er: d = "S"
                        elif sc > ec: d = "E"
                        elif sc < ec: d = "W"
                        else: d = "UNKNOWN"
                        dir_to_road[d] = road_id
                    except ValueError:
                        pass

            # Aggiunge le corsie nell'ordine canonico (N, S, W, E) e (SX=0, Dritto=1, DX=2)
            for d in ["N", "S", "W", "E"]:
                if d in dir_to_road:
                    road_id = dir_to_road[d]
                    for k in [0, 1, 2]:  # 0=Left, 1=Straight, 2=Right
                        inter_lanes[iid].append(f"{road_id}_{k}")
                else:
                    # Filler se manca un ramo all'incrocio
                    for k in [0, 1, 2]:
                        inter_lanes[iid].append(f"missing_{d}_{k}")
                    
        return inter_lanes

    def _get_road_lengths(self) -> Dict[str, float]:
        """
        Lunghezza REALE (percorribile) di ogni corsia, non la distanza grezza tra i
        punti dichiarati nel roadnet (che e' centro-incrocio a centro-incrocio).

        CityFlow tronca get_vehicle_distance() alla 'width' dell'intersezione ad
        ENTRAMBE le estremita' (un veicolo transita nell'incrocio, e quindi lascia
        la corsia, gia' 'width' metri prima del punto centrale dichiarato) — verificato
        empiricamente: su una corsia interna tra due incroci reali (width=20 ciascuno)
        con punti a distanza 167m, i veicoli non superano mai ~127m (167-20-20).
        Usare la distanza grezza (167m) qui sovrastimerebbe la corsia reale e
        renderebbe VISION_CUTOFF_M (basato su questa lunghezza) troppo restrittivo
        di quanto dovrebbe essere.
        """
        import math
        widths = {i["id"]: float(i.get("width", 0.0)) for i in self.roadnet.get("intersections", [])}
        lengths = {}
        for r in self.roadnet.get("roads", []):
            pts = r.get("points", [])
            raw_length = 0.0
            for i in range(len(pts)-1):
                raw_length += math.dist((pts[i]["x"], pts[i]["y"]), (pts[i+1]["x"], pts[i+1]["y"]))
            w_start = widths.get(r.get("startIntersection"), 0.0)
            w_end = widths.get(r.get("endIntersection"), 0.0)
            lengths[r["id"]] = max(0.0, raw_length - w_start - w_end)
        return lengths

    def _build_phase_lanelinks(self) -> Dict[str, List[List[Tuple[str, str]]]]:
        """
        Per ogni intersezione e ogni fase, la lista di coppie (corsia_in, corsia_out)
        dei movimenti (lane-link) abilitati da quella fase, letta direttamente dal
        roadnet (`roadLinks[i].laneLinks[j].startLaneIndex/endLaneIndex`).

        Usata da MaxPressureAgent per calcolare la pressione esattamente come
        l'implementazione di riferimento LibSignal (Varaiya 2013): per ogni fase,
        pressione = somma su tutti i movimenti abilitati di
        (veicoli sulla corsia di provenienza - veicoli sulla corsia di destinazione).
        Nessun cutoff di visibilita' qui: la formula classica usa la corsia intera,
        vedi get_lane_vehicle_count().
        """
        result: Dict[str, List[List[Tuple[str, str]]]] = {}
        for inter in self.roadnet.get("intersections", []):
            iid = inter.get("id")
            if iid not in self.inter_id_to_idx:
                continue

            # Coppie (corsia_in, corsia_out) per ciascun roadLink dell'intersezione
            # (un roadLink puo' avere piu' lane-link, es. una strada a 3 corsie che
            # confluisce su una strada a 2: li teniamo tutti, non solo il primo).
            pairs_per_roadlink: List[List[Tuple[str, str]]] = []
            for rl in inter.get("roadLinks", []):
                start_road = rl.get("startRoad")
                end_road = rl.get("endRoad")
                pairs = [
                    (f"{start_road}_{ll['startLaneIndex']}", f"{end_road}_{ll['endLaneIndex']}")
                    for ll in rl.get("laneLinks", [])
                ]
                pairs_per_roadlink.append(pairs)

            phase_pairs: List[List[Tuple[str, str]]] = []
            for phase in inter.get("trafficLight", {}).get("lightphases", []):
                pairs: List[Tuple[str, str]] = []
                for rl_idx in phase.get("availableRoadLinks", []):
                    if 0 <= rl_idx < len(pairs_per_roadlink):
                        pairs.extend(pairs_per_roadlink[rl_idx])
                phase_pairs.append(pairs)

            result[iid] = phase_pairs
        return result

    def _build_lane_reachable_outgoing(self) -> Dict[str, List[str]]:
        """
        Per ogni corsia in ingresso a un'intersezione, la lista delle corsie
        in uscita da quella STESSA intersezione che quella corsia può
        raggiungere — dal roadnet (`roadLinks[i].laneLinks[j]`), indipendente
        da quale fase abilita il movimento (una corsia fisica può raggiungere
        certe corsie in uscita indipendentemente da quale fase è attiva ora;
        è la stessa unione di `pairs_per_roadlink` in `_build_phase_lanelinks`,
        ma indicizzata per corsia_in invece che raggruppata per fase).

        Usata da `_compute_lane_pressure_vector` per calcolare una media LOCALE
        delle corsie in uscita raggiungibili da ciascuna corsia in ingresso,
        invece di una media GLOBALE su tutte le corsie in uscita
        dell'incrocio — corretto il 13/9/2026 (vedi descrizione_stato_meta_reward.md): la
        versione precedente usava la stessa media (su tutte le uscite
        dell'incrocio) per ogni corsia in ingresso, anche quando quella
        corsia in realtà porta solo verso un sottoinsieme specifico delle
        uscite (es. una corsia di svolta a destra non raggiunge le stesse
        corsie di una corsia dritta) — più vicino alla nozione di pressione
        per MOVIMENTO di MaxPressure (Varaiya 2013), non un aggregato
        indifferenziato per l'intero incrocio.
        """
        result: Dict[str, List[str]] = {}
        for inter in self.roadnet.get("intersections", []):
            if inter.get("id") not in self.inter_id_to_idx:
                continue
            for rl in inter.get("roadLinks", []):
                start_road = rl.get("startRoad")
                end_road = rl.get("endRoad")
                for ll in rl.get("laneLinks", []):
                    lane_in = f"{start_road}_{ll['startLaneIndex']}"
                    lane_out = f"{end_road}_{ll['endLaneIndex']}"
                    result.setdefault(lane_in, []).append(lane_out)
        return result

    def get_lane_vehicle_count(self) -> Dict[str, int]:
        """
        Conteggio veicoli per corsia, nativo del motore CityFlow, sulla corsia
        INTERA (nessun cutoff di visibilita' — a differenza di n_vec/_get_observations,
        pensati per lo stato RL). Usato dalla pressione classica di MaxPressure.
        """
        return self.engine.get_lane_vehicle_count()

    def _compute_distances(self) -> Dict[Tuple[str, str], float]:
        """Distanza euclidea normalizzata tra coppie di intersezioni vicine."""
        positions = {}
        for inter in self.roadnet.get("intersections", []):
            pt = inter.get("point", {})
            positions[inter["id"]] = (pt.get("x", 0.0), pt.get("y", 0.0))

        distances = {}
        for iid in self.inter_ids:
            for jid in self.adjacency.get(iid, []):
                xi, yi = positions.get(iid, (0, 0))
                xj, yj = positions.get(jid, (0, 0))
                dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
                distances[(iid, jid)] = dist
                distances[(jid, iid)] = dist

        # Normalizza per la distanza massima
        if distances:
            max_dist = max(distances.values()) or 1.0
            distances = {k: v / max_dist for k, v in distances.items()}

        return distances

    # ─── MDP Interface ────────────────────────────────────────────────────────

    def reset(self) -> Dict[str, np.ndarray]:
        """Resetta l'ambiente. Restituisce il dizionario degli stati iniziali."""
        self.engine.reset()
        self.current_step = 0
        self.current_phase = {iid: 0 for iid in self.inter_ids}
        self.consecutive_phases = {iid: 0 for iid in self.inter_ids}
        self.episode_travel_times = []
        self.episode_throughput = 0
        
        self.spawn_times = {}
        self.arrived_tt = []
        self.vehicle_wait_times = {}
        self.all_spawned_vehicles = set()

        # Massima attesa MAI osservata durante l'episodio su ogni intersezione,
        # separata per gruppo di corsie N/S (indici 0-5 nell'ordine canonico
        # N,S,W,E di _get_lanes_per_intersection) e W/E (indici 6-11) — usata
        # per le metriche di equita' (vedi get_direction_fairness_stats()).
        self.max_wait_ns = {iid: 0.0 for iid in self.inter_ids}
        self.max_wait_ew = {iid: 0.0 for iid in self.inter_ids}
        # Per ogni intersezione, (veicolo, corsia, timestamp) che ha prodotto il
        # record corrente — permette a get_direction_fairness_stats() di distinguere
        # uno stallo CONCLUSO (il veicolo si e' poi mosso, timestamp < fine episodio)
        # da uno ancora IN CORSO quando l'episodio e' terminato (timestamp == fine
        # episodio: il veicolo era ancora fermo li', l'attesa vera potrebbe essere
        # anche piu' lunga — non lo sappiamo, la simulazione si e' solo fermata).
        # Vedi analisi del 13/9/2026: il record piu' alto osservato su
        # config_4x4_100m_6k_peak/MaxPressure (1125s) era di questo secondo tipo.
        self._wait_ns_record = {}
        self._wait_ew_record = {}

        # Massima attesa MAI osservata da OGNI VEICOLO (non per intersezione),
        # separata N/S vs W/E in base a su quale corsia si trovava nel momento
        # dell'attesa — un campione per veicolo invece che uno per intersezione,
        # per statistiche di distribuzione con un numero di punti realistico
        # (migliaia invece di 16). Vedi get_raw_vehicle_waits().
        self.vehicle_max_wait_ns = {}
        self.vehicle_max_wait_ew = {}
        self.arrived_max_wait_ns = []
        self.arrived_max_wait_ew = []

        # Invalida la cache di pressione per nodo (vedi __init__): senza
        # questo, il primo step del nuovo episodio (current_step=0) potrebbe
        # riusare per errore la cache dell'ultimo step dell'episodio
        # precedente, che aveva anch'esso current_step=0 all'inizio.
        self._node_pressure_cache = {}
        self._node_pressure_cache_step = None

        return self._get_observations()

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],  # next_states
        Dict[str, float],        # rewards
        bool,                    # done
        dict                     # info
    ]:
        """
        Esegue un passo di simulazione.

        Args:
            actions: dict {intersection_id -> phase_index}

        Returns:
            (observations, rewards, done, info)
        """
        # Imposta i semafori per ogni intersezione
        for iid, phase in actions.items():
            if self.current_phase[iid] == phase:
                self.consecutive_phases[iid] += 1
            else:
                self.consecutive_phases[iid] = 1
            self.engine.set_tl_phase(iid, phase)
            self.current_phase[iid] = phase

        incoming_t0 = self._get_incoming_vehicles_ids()
        old_vehicles = set(self.engine.get_vehicles(include_waiting=True))

        # Esegui GREEN_TIME + YELLOW_TIME secondi sulla fase scelta dall'agente.
        # Il giallo non è modellato come stato fisico separato (CityFlow non ce
        # l'ha nativamente): per YELLOW_TIME secondi il movimento resta abilitato
        # esattamente come in verde. Scelta esplicita e accettata: il giallo
        # "vale" verde ai fini della simulazione.
        for _ in range(GREEN_TIME + YELLOW_TIME):
            self.engine.next_step()

        # Esegui RED_TIME secondi di tutto-rosso reale: nessun movimento abilitato
        # per nessuna intersezione, cosi' l'incrocio si svuota prima che il
        # prossimo step liberi un movimento potenzialmente conflittuale. Fase
        # dedicata (indice ALL_RED_PHASE), mai selezionabile dall'agente.
        for iid in actions:
            self.engine.set_tl_phase(iid, ALL_RED_PHASE)
        for _ in range(RED_TIME):
            self.engine.next_step()

        # Ripristina la fase verde scelta: il prossimo step() leggera' current_phase
        # (es. per action mask/anti-starvation) assumendo sia ancora quella verde,
        # non il tutto-rosso appena usato internamente.
        for iid, phase in actions.items():
            self.engine.set_tl_phase(iid, phase)

        self.all_spawned_vehicles.update(self.engine.get_vehicles())

        # Aggiorna il tempo di attesa dei veicoli
        try:
            speeds = self.engine.get_vehicle_speed()
            for veh_id, speed in speeds.items():
                if speed < 0.1:
                    self.vehicle_wait_times[veh_id] = self.vehicle_wait_times.get(veh_id, 0) + STEP_TIME
                else:
                    self.vehicle_wait_times[veh_id] = 0
        except Exception:
            pass

        # Aggiorna la massima attesa mai vista per intersezione, separata per
        # corsie N/S vs W/E (metriche di equita', vedi get_direction_fairness_stats()).
        try:
            lane_vehicles_fair = self.engine.get_lane_vehicles()
            for iid in self.inter_ids:
                lanes = self.inter_lanes.get(iid, [])
                for k, lane_id in enumerate(lanes[:N_LANES]):
                    if lane_id.startswith("missing_"):
                        continue
                    for veh in lane_vehicles_fair.get(lane_id, []):
                        wt = self.vehicle_wait_times.get(veh, 0.0)
                        record_time = (self.current_step + 1) * STEP_TIME
                        if k < 6:  # N (0-2) + S (3-5)
                            if wt > self.max_wait_ns[iid]:
                                self.max_wait_ns[iid] = wt
                                self._wait_ns_record[iid] = (veh, lane_id, record_time)
                            if wt > self.vehicle_max_wait_ns.get(veh, 0.0):
                                self.vehicle_max_wait_ns[veh] = wt
                        else:  # W (6-8) + E (9-11)
                            if wt > self.max_wait_ew[iid]:
                                self.max_wait_ew[iid] = wt
                                self._wait_ew_record[iid] = (veh, lane_id, record_time)
                            if wt > self.vehicle_max_wait_ew.get(veh, 0.0):
                                self.vehicle_max_wait_ew[veh] = wt
        except Exception:
            pass

        self.current_step += 1
        current_time = self.current_step * STEP_TIME
        
        # 2.2 Teleported vehicles bug: limit tt a un valore fisicamente possibile.
        # maxStep in config.json e' gia' in secondi (stessa unita' usata da _is_done(),
        # che lo confronta direttamente con current_step*STEP_TIME) — nessun veicolo
        # puo' avere un travel time piu' lungo dell'intero episodio. Bug corretto il
        # 13/9/2026: la versione precedente moltiplicava di nuovo per STEP_TIME
        # (es. 1800*15=27000 invece di 1800), rendendo il filtro 15x troppo permissivo
        # e di fatto inutile — verificato pero' che finora non ha mai lasciato passare
        # nulla sopra il vero limite (nessun tt_max osservato supera maxStep in nessuno
        # dei risultati raccolti), quindi il fix non cambia alcun numero gia' riportato,
        # solo rende il filtro genuinamente protettivo per il futuro.
        MAX_PLAUSIBLE_TT = self.config.get("maxStep", float("inf"))
        new_vehicles = set(self.engine.get_vehicles(include_waiting=True))
        
        # 2.1 Travel time offset bug: usa (current_step - 1) * STEP_TIME
        spawn_time_for_this_step = (self.current_step - 1) * STEP_TIME
        for veh in new_vehicles:
            if veh not in self.spawn_times:
                self.spawn_times[veh] = spawn_time_for_this_step
                
        arrived_vehicles = old_vehicles - new_vehicles
        for veh in arrived_vehicles:
            if veh in self.spawn_times:
                tt = current_time - self.spawn_times[veh]
                if tt <= MAX_PLAUSIBLE_TT:
                    self.arrived_tt.append(tt)
                del self.spawn_times[veh]
            # Chiude anche la distribuzione per-veicolo dell'attesa massima (vedi
            # get_raw_vehicle_waits()): 0.0 se il veicolo non ha mai aspettato su
            # quella corsia, un valore reale altrimenti. pop() invece di get()
            # per non far crescere i due dict all'infinito con veicoli ormai usciti.
            self.arrived_max_wait_ns.append(self.vehicle_max_wait_ns.pop(veh, 0.0))
            self.arrived_max_wait_ew.append(self.vehicle_max_wait_ew.pop(veh, 0.0))

        incoming_t1 = self._get_incoming_vehicles_ids()

        # Calcola stati, reward, e done
        observations = self._get_observations()
        rewards = self._compute_rewards(incoming_t0, incoming_t1)
        done = self._is_done()

        info = {
            "step": self.current_step,
            "avg_travel_time": self.get_average_travel_time(),
            "vehicles_running": self.get_throughput()
        }

        return observations, rewards, done, info

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """
        Calcola il vettore di osservazione per ogni intersezione.

        Modalità avanzata (use_wait_vec=True, use_vision_cutoff=True):
          s_i^t = [n_vec (12), wait_vec (12), p_vec (8)] — dim 32

        Modalità paper (use_wait_vec=False, use_vision_cutoff=False):
          s_i^t = [n_vec (12), p_vec (8)] — dim 20 (solo conteggio + fase)

        Se use_phase_pressure_state=True (esperimento 13/9/2026, vedi
        descrizione_stato_meta_reward.md §1): si aggiunge in coda phase_pressure (8 dim,
        una per fase candidata, stessa formula di MaxPressureAgent ma
        rispettando use_vision_cutoff) — dim 32->40 o 20->28.

        n_vec: numero di veicoli sulle corsie in ingresso
        wait_vec: max waiting time normalizzato (0.0-1.0) per corsia
        p_vec: fase corrente (one-hot encoding 8 bit)
        """
        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_distances = self.engine.get_vehicle_distance()
        observations = {}

        for iid in self.inter_ids:
            lanes = self.inter_lanes.get(iid, [])
            n_vec   = np.zeros(N_LANES, dtype=np.float32)
            wait_vec = np.zeros(N_LANES, dtype=np.float32)

            for k, lane_id in enumerate(lanes[:N_LANES]):
                if lane_id.startswith("missing_"):
                    continue
                road_id = "_".join(lane_id.split("_")[:-1])
                vehicles = lane_vehicles.get(lane_id, [])

                if self.use_vision_cutoff:
                    # Avanzato: solo veicoli entro VISION_CUTOFF_M dal semaforo
                    cutoff = max(0.0, self.road_lengths.get(road_id, 340.0) - VISION_CUTOFF_M)
                    veicoli_vicini = 0
                    max_wt = 0.0
                    for veh in vehicles:
                        dist = vehicle_distances.get(veh, 0.0)
                        if dist >= cutoff:
                            veicoli_vicini += 1
                            wt = self.vehicle_wait_times.get(veh, 0.0)
                            if wt > max_wt:
                                max_wt = wt
                    n_vec[k] = veicoli_vicini
                    # 100s come tetto di saturazione: non derivato, vedi analisi_bug.md #8.
                    wait_vec[k] = np.clip(max_wt / 100.0, 0.0, 1.0)
                else:
                    # Paper: conta tutti i veicoli sulla corsia (visibilità globale)
                    n_vec[k] = float(len(vehicles))
                    if self.use_wait_vec:  # wait_vec solo se richiesto
                        max_wt = max(
                            (self.vehicle_wait_times.get(veh, 0.0) for veh in vehicles),
                            default=0.0
                        )
                        wait_vec[k] = np.clip(max_wt / 100.0, 0.0, 1.0)

            # Normalizza il numero di veicoli. 30 e' una capacita' di corsia
            # plausibile ma non derivata a partire da lunghezza corsia reale /
            # (lunghezza veicolo + minGap) — vedi analisi_bug.md #8: con una
            # corsia da 127m, lunghezza veicolo 5m e minGap 2.5m la capacita'
            # fisica e' ~127/7.5=~17, quindi 30 e' un limite conservativo che
            # in pratica non taglia mai nulla (nessuna evidenza che sia sbagliato,
            # solo che non e' documentato/derivato esplicitamente finora).
            n_vec = np.clip(n_vec / 30.0, 0.0, 1.0)

            # p_vec: fase corrente (one-hot)
            p_vec = np.zeros(N_PHASES, dtype=np.float32)
            p_vec[self.current_phase[iid]] = 1.0

            parts = [n_vec, wait_vec, p_vec] if self.use_wait_vec else [n_vec, p_vec]
            if self.use_phase_pressure_state:
                parts.append(self._compute_phase_pressure_vector(iid, lane_vehicles, vehicle_distances))
            observations[iid] = np.concatenate(parts)

        return observations

    def _get_incoming_vehicles_ids(self) -> Dict[str, set]:
        """Restituisce per ogni intersezione il set di ID dei veicoli in ingresso (nel raggio visivo)."""
        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_distances = self.engine.get_vehicle_distance()
        incoming_ids = {iid: set() for iid in self.inter_ids}

        for iid in self.inter_ids:
            lanes = self.inter_lanes.get(iid, [])
            for l in lanes:
                if l.startswith("missing_"):
                    continue
                road_id = "_".join(l.split("_")[:-1])
                for veh in lane_vehicles.get(l, []):
                    if self.use_vision_cutoff:
                        cutoff = max(0.0, self.road_lengths.get(road_id, 340.0) - VISION_CUTOFF_M)
                        if vehicle_distances.get(veh, 0.0) >= cutoff:
                            incoming_ids[iid].add(veh)
                    else:
                        # Paper: conta tutti i veicoli sulla corsia
                        incoming_ids[iid].add(veh)
        return incoming_ids

    def _compute_rewards(self, incoming_t0: Dict[str, set] = None, incoming_t1: Dict[str, set] = None) -> Dict[str, float]:
        """
        Calcola il reward per ogni intersezione.

        reward_mode="custom" (avanzato):
          Reward = (Passed - Incoming - alpha * max_red_wait_time - wasted_green_penalty) / 100, clip [-20, 5]
          Se use_pressure_reward_term=True, "Passed - Incoming" e' sostituito da
          "Outgoing - Incoming_ora" (= -P_i, stessa quantita' del ramo "paper" sotto)
          -- vedi nota nel docstring del costruttore. Il resto della formula (termine
          anti-starvation, penalita', normalizzazione, clip) resta invariato.

        reward_mode="paper" (originale Wang et al. 2022):
          Reward = -P_i / 100  (negativo della pressione: veicoli in ingresso - veicoli in
          uscita, riscalata; vedi nota sotto)

        Nota sulla riscalatura di "paper" (/100):
          La definizione di P_i (pressione) e' quella originale, invariata. La divisione per
          100 e' una pura riparametrizzazione numerica, non una modifica dell'obiettivo: per
          una MDP scontata, moltiplicare tutti i reward per una costante positiva moltiplica
          tutti i Q-value per la stessa costante e lascia identica la policy argmax. Serve
          solo a riportare la scala del segnale nello stesso ordine di grandezza gia' usato
          dal reward "custom" (anch'esso diviso per 100 sopra). Senza questa riscalatura, il
          preset "paper" (che disabilita Huber loss, gradient clipping, Double DQN e soft
          update tutti insieme) diverge: target TD dell'ordine di decine/centinaia con MSE
          loss e nessun freno producono aggiornamenti enormi e la loss cresce invece di
          scendere (osservato empiricamente: loss 92->780 e travel time 260s->846s in 18
          episodi). Non tocchiamo i flag no_huber/no_grad_clip/no_double_dqn/no_soft_update
          (sono l'identita' dell'ablation "paper", vanno lasciati esattamente com'erano) ne'
          la definizione di P_i: solo l'unita' di misura del reward cambia.
        """
        if incoming_t0 is None or incoming_t1 is None:
            return {iid: 0.0 for iid in self.inter_ids}

        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_distances = self.engine.get_vehicle_distance()
        rewards = {}

        for iid in self.inter_ids:
            lanes = self.inter_lanes.get(iid, [])
            in_t0 = incoming_t0.get(iid, set())
            in_t1 = incoming_t1.get(iid, set())

            if self.reward_mode == "paper":
                # ── Reward originale del paper: -P_i ───────────────────────────────
                # P_i = veicoli in ingresso - veicoli in uscita (pressione)
                outgoing = self._get_outgoing_vehicles_count(iid)
                pressure = len(in_t1) - outgoing  # veicoli attuali - veicoli usciti
                rewards[iid] = -float(pressure) / 100.0  # riscalatura numerica, vedi docstring
            else:
                # ── Reward avanzata: Throughput-based multi-obiettivo ──────────────
                # Nota (analisi_bug.md #6): "passed" non distingue un veicolo
                # realmente transitato da uno rimosso dal motore per il noto bug
                # dei veicoli "teletrasportati" in gridlock (vedi il commento
                # analogo su MAX_PLAUSIBLE_TT in step()). Non e' un problema con
                # i preset attuali: questo ramo "custom" viene eseguito solo se
                # reward_mode=="custom", e in tutti e 6 i preset definiti
                # (apply_ablation_preset in train.py) reward_mode=="custom"
                # implica sempre use_vision_cutoff=True — quindi in_t0/in_t1
                # contengono gia' solo veicoli vicini al semaforo, dove la
                # teletrasportazione per gridlock e' meno plausibile. Il rischio
                # esiste solo se si combinano manualmente i flag atomici
                # --reward-mode custom --no-vision-cutoff, bypassando i preset.
                passed   = len(in_t0 - in_t1)
                incoming = len(in_t0)

                current_phase = self.current_phase[iid]
                green_lanes   = GREEN_LANES_PER_PHASE.get(current_phase, [])

                max_red_wait_time = 0.0
                for k, lane_id in enumerate(lanes[:N_LANES]):
                    if k not in green_lanes:
                        if lane_id.startswith("missing_"):
                            continue
                        road_id = "_".join(lane_id.split("_")[:-1])
                        for veh in lane_vehicles.get(lane_id, []):
                            if self.use_vision_cutoff:
                                cutoff = max(0.0, self.road_lengths.get(road_id, 340.0) - VISION_CUTOFF_M)
                                if vehicle_distances.get(veh, 0.0) < cutoff:
                                    continue
                            wt = self.vehicle_wait_times.get(veh, 0.0)
                            if wt > max_red_wait_time:
                                max_red_wait_time = wt

                # 50.0 e' scelto per essere comparabile in scala a (passed-incoming),
                # tipicamente decine di veicoli — non derivato analiticamente, vedi
                # analisi_bug.md #8.
                wasted_green_penalty = 50.0 if (passed == 0 and incoming > 0) else 0.0

                if self.use_pressure_reward_term:
                    # Primo termine sostituito da -P_i, stessa identica quantita'
                    # del ramo reward_mode="paper" sopra (outgoing - veicoli in
                    # ingresso ADESSO, len(in_t1) -- non "incoming"=len(in_t0),
                    # che e' il conteggio PRIMA dello step, usato solo per
                    # wasted_green_penalty). Vedi docstring del costruttore
                    # (use_pressure_reward_term).
                    outgoing = self._get_outgoing_vehicles_count(iid)
                    first_term = float(outgoing) - float(len(in_t1))
                else:
                    first_term = float(passed) - float(incoming)

                raw_reward = first_term - (self.alpha * max_red_wait_time) - wasted_green_penalty
                normalized_reward = raw_reward / 100.0
                rewards[iid] = max(min(normalized_reward, 5.0), -20.0)

        return rewards

    def _get_outgoing_lanes(self, inter_id: str) -> List[str]:
        """Restituisce le corsie in uscita dall'intersezione inter_id."""
        lane_vehicles = self.engine.get_lane_vehicles()
        out_lanes = []
        for road in self.roadnet.get("roads", []):
            if road.get("startIntersection") == inter_id:
                n_lanes = len(road.get("lanes", []))
                for k in range(n_lanes):
                    lid = f"{road['id']}_{k}"
                    if lid in lane_vehicles:
                        out_lanes.append(lid)
        return out_lanes

    def _get_outgoing_vehicles_count(self, inter_id: str) -> int:
        """Restituisce il totale dei veicoli in uscita (entro VISION_CUTOFF_M se
        use_vision_cutoff=True, sull'intera corsia altrimenti).

        Bug corretto il 13/9/2026: il cutoff era applicato incondizionatamente,
        ignorando self.use_vision_cutoff -- con reward_mode="paper" (usato dai
        preset "paper"/"environment", entrambi con use_vision_cutoff=False)
        questo mescolava un conteggio in ingresso a visibilita' globale
        (_get_incoming_vehicles_ids, che rispetta il flag) con uno in uscita
        sempre limitato al cutoff, gonfiando artificialmente la pressione.
        """
        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_distances = self.engine.get_vehicle_distance()
        out_lanes = self._get_outgoing_lanes(inter_id)
        count = 0
        for lid in out_lanes:
            for veh in lane_vehicles.get(lid, []):
                # CityFlow distance is from start of road segment
                if not self.use_vision_cutoff or vehicle_distances.get(veh, 0.0) <= VISION_CUTOFF_M:
                    count += 1
        return count

    def _is_done(self) -> bool:
        """Controlla se la simulazione è terminata."""
        sim_time = self.current_step * STEP_TIME
        # La simulazione termina quando viene raggiunta la durata configurata
        # (CityFlow non ha un'API diretta per questo, lo stimiamo)
        return sim_time >= self.config.get("maxStep", float("inf"))

    # ─── Meta-features (per SMK-Learner e TMK-Learner) ───────────────────────

    def _count_incoming_lane(self, lane_id: str, lane_vehicles: dict, vehicle_distances: dict) -> int:
        """Veicoli su una corsia in INGRESSO a un'intersezione. Con
        self.use_vision_cutoff=True, conta solo i veicoli vicini alla FINE
        della corsia (dist >= road_length - VISION_CUTOFF_M) -- vicini
        all'intersezione che questa corsia sta per raggiungere, stessa
        convenzione di n_vec in _get_observations(). Senza cutoff, l'intera
        corsia. Helper condiviso da _compute_lane_pressure_vector e
        _compute_phase_pressure_vector (13/9/2026, vedi descrizione_stato_meta_reward.md):
        prima ciascuna delle due aveva una propria logica di cutoff
        (una addirittura nessuna, vedi nota storica in
        _compute_phase_pressure_vector) invece di applicare consistentemente
        la stessa regola "pressione = vicino a QUESTA intersezione, sia in
        ingresso che in uscita" ovunque nel file.
        """
        if lane_id.startswith("missing_"):
            return 0
        vehicles = lane_vehicles.get(lane_id, [])
        if not self.use_vision_cutoff:
            return len(vehicles)
        road_id = "_".join(lane_id.split("_")[:-1])
        cutoff = max(0.0, self.road_lengths.get(road_id, 340.0) - VISION_CUTOFF_M)
        return sum(1 for veh in vehicles if vehicle_distances.get(veh, 0.0) >= cutoff)

    def _count_outgoing_lane(self, lane_id: str, lane_vehicles: dict, vehicle_distances: dict) -> int:
        """Veicoli su una corsia in USCITA da un'intersezione. Con
        self.use_vision_cutoff=True, conta solo i veicoli vicini all'INIZIO
        della corsia (dist <= VISION_CUTOFF_M) -- vicini all'intersezione da
        cui questa corsia parte, stessa convenzione di
        _get_outgoing_vehicles_count(). Senza cutoff, l'intera corsia. Vedi
        _count_incoming_lane per il contesto della fattorizzazione."""
        if lane_id.startswith("missing_"):
            return 0
        vehicles = lane_vehicles.get(lane_id, [])
        if not self.use_vision_cutoff:
            return len(vehicles)
        return sum(1 for veh in vehicles if vehicle_distances.get(veh, 0.0) <= VISION_CUTOFF_M)

    def _compute_lane_pressure_vector(self, inter_id: str,
                                       lane_vehicles: Optional[dict] = None,
                                       vehicle_distances: Optional[dict] = None) -> np.ndarray:
        """Vettore di pressione per corsia (N_LANES dim) per un nodo:

            lane_pressure[k] = (veicoli_in_corsia_k - media(veicoli su ogni
                                corsia in uscita RAGGIUNGIBILE da k)) / 30

        Corretto il 13/9/2026 (vedi descrizione_stato_meta_reward.md, su segnalazione): la
        media in uscita era prima GLOBALE (tutti i veicoli in uscita
        dall'incrocio / tutte le corsie in uscita dell'incrocio), la stessa
        per ogni corsia in ingresso indipendentemente da quali uscite quella
        corsia potesse davvero raggiungere. Ora è LOCALE: solo le corsie in
        uscita che `self.lane_reachable_outgoing[k]` elenca per quella
        specifica corsia in ingresso — più vicina alla nozione di pressione
        per MOVIMENTO di MaxPressure (Varaiya 2013), che è definita per
        coppia (corsia_in, corsia_out) specifica, non per un aggregato
        indifferenziato sull'intero incrocio.

        Fattorizzato da get_spatial_meta_features il 12/9/2026 perche' riusato
        anche da _get_avg_pressure_all_nodes() per calcolare la pressione
        media di TUTTI i nodi (serve a pressure_diff_neighbors). lane_vehicles/
        vehicle_distances possono essere passati gia' calcolati per evitare
        di richiamare l'engine una volta per nodo quando si itera su tutte
        le intersezioni.
        """
        if lane_vehicles is None:
            lane_vehicles = self.engine.get_lane_vehicles()
        if vehicle_distances is None:
            vehicle_distances = self.engine.get_vehicle_distance()

        lanes = self.inter_lanes.get(inter_id, [])
        lane_pressure = np.zeros(N_LANES, dtype=np.float32)

        for k, lid in enumerate(lanes[:N_LANES]):
            count = self._count_incoming_lane(lid, lane_vehicles, vehicle_distances)
            reachable_out = self.lane_reachable_outgoing.get(lid, [])
            if reachable_out:
                avg_out = float(np.mean([
                    self._count_outgoing_lane(o, lane_vehicles, vehicle_distances)
                    for o in reachable_out
                ]))
            else:
                avg_out = 0.0
            lane_pressure[k] = (count - avg_out) / 30.0
        return lane_pressure

    def _get_avg_pressure_all_nodes(self) -> Dict[str, float]:
        """Pressione media (media di _compute_lane_pressure_vector) per ogni
        intersezione, ricalcolata una sola volta per step e cachata (vedi
        self._node_pressure_cache in __init__/reset) -- usata da
        pressure_diff_neighbors in get_spatial_meta_features per non
        ricalcolare la pressione di ogni vicino ad ogni chiamata per-nodo."""
        if self._node_pressure_cache_step == self.current_step and self._node_pressure_cache:
            return self._node_pressure_cache
        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_distances = self.engine.get_vehicle_distance()
        cache = {
            iid: float(np.mean(self._compute_lane_pressure_vector(iid, lane_vehicles, vehicle_distances)))
            for iid in self.inter_ids
        }
        self._node_pressure_cache = cache
        self._node_pressure_cache_step = self.current_step
        return cache

    def _compute_phase_pressure_vector(self, inter_id: str,
                                        lane_vehicles: Optional[dict] = None,
                                        vehicle_distances: Optional[dict] = None) -> np.ndarray:
        """Vettore di pressione PER FASE (N_PHASES dim) -- la stessa identica
        quantita' che MaxPressureAgent.select_actions() usa per l'argmax
        (vedi src/agents/maxpressure_agent.py e _build_phase_lanelinks()),
        solo riscalata per una costante positiva (non altera l'ordine tra
        fasi, l'unica informazione che l'argmax usa):

            phase_pressure[k] = (Σ_{(in,out) ∈ movimenti(fase k)} count(in) - count(out)) / 30.0

        Corretto il 13/9/2026 (vedi descrizione_stato_meta_reward.md, su segnalazione):
        prima ignorava SEMPRE self.use_vision_cutoff (piena visibilita' anche
        con use_vision_cutoff=True), per dare fedelmente al meta-learner il
        "vero" valore usato da MaxPressure. Ora rispetta il cutoff come tutto
        il resto del file, usando gli stessi helper di _compute_lane_pressure_vector
        (_count_incoming_lane per le corsie di provenienza, _count_outgoing_lane
        per quelle di destinazione — sono ai lati opposti dell'incrocio, quindi
        "vicino a questo nodo" significa cose diverse in termini di distanza
        dall'inizio del segmento per l'una e per l'altra). Con
        use_vision_cutoff=False resta a piena visibilita' come prima
        (nessun cambiamento per i preset "paper"/"environment").

        Usata sia da get_spatial_meta_features (SMK, meta_v3) sia da
        _get_observations (use_phase_pressure_state, esperimento sullo
        stato) -- unica funzione, stessa formula, stesso rispetto del
        cutoff in entrambi i punti di utilizzo.
        """
        if lane_vehicles is None:
            lane_vehicles = self.engine.get_lane_vehicles()
        if vehicle_distances is None:
            vehicle_distances = self.engine.get_vehicle_distance()

        phase_pairs = self.phase_lanelinks.get(inter_id, [])[:N_PHASES]
        pressure = np.zeros(N_PHASES, dtype=np.float32)
        for k, pairs in enumerate(phase_pairs):
            pressure[k] = sum(
                self._count_incoming_lane(s, lane_vehicles, vehicle_distances)
                - self._count_outgoing_lane(e, lane_vehicles, vehicle_distances)
                for s, e in pairs
            ) / 30.0
        return pressure

    def get_spatial_meta_features(self, inter_id: str) -> np.ndarray:
        """
        Features spaziali per il SMK-Learner (Section 4.3.1, Fig. 5b), riviste
        il 12/9/2026 per rimuovere ridondanze con lo stato principale e una
        feature degenere -- vedi descrizione_stato_meta_reward.md §2.2/§2.5 per la
        motivazione completa di ciascuna:
        - lane_pressure (N_LANES): pressione approssimata per corsia,
          invariata rispetto a prima.
        - real_degree (1): frazione di vicini reali su num_neighbors --
          varia per ruolo topologico del nodo (angolo/bordo/interno).
          Sostituisce "distanze dai vicini": su una griglia a lunghezza di
          strada uniforme (tutte le nostre config lo sono) la distanza
          euclidea normalizzata tra vicini reali è quasi sempre 1.0 per
          costruzione, una feature di fatto costante — non informativa per
          un meta-learner (vedi §2.1).
        - pressure_diff_neighbors (num_neighbors): pressione media propria
          meno quella di ciascun vicino reale (0 per gli slot di padding).
        - traffic_asymmetry_ns_ew (1): pressione media sulle corsie N/S
          (indici 0-5, ordine canonico di GREEN_LANES_PER_PHASE) meno quella
          sulle corsie E/O (indici 6-11) dello stesso nodo.
        - phase_pressure (N_PHASES): AGGIUNTA il 13/9/2026 per l'esperimento
          meta_v3 (vedi descrizione_stato_meta_reward.md), SOLO nel SMK (non nel TMK, deciso
          il 13/9/2026 -- e' un dato spaziale per fase, non una dinamica
          temporale) -- la stessa formula di pressione per movimento usata
          da MaxPressureAgent (vedi _compute_phase_pressure_vector),
          rispettando use_vision_cutoff come tutto il resto del file.
          Obiettivo dell'esperimento: verificare se dare al meta-learner il
          segnale che MaxPressure usa per decidere (non solo una sua
          approssimazione ricostruibile da lane_pressure) permette al
          modello allenabile di avvicinarsi o superare il baseline.

        Returns:
            array di dim = N_LANES + 1 + num_neighbors + 1 + N_PHASES
        """
        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_distances = self.engine.get_vehicle_distance()

        lane_pressure = self._compute_lane_pressure_vector(inter_id, lane_vehicles, vehicle_distances)

        neighbors = self.adjacency.get(inter_id, [])
        real_degree = np.array(
            [len(neighbors) / max(1, self.num_neighbors)], dtype=np.float32
        )

        avg_pressure_all = self._get_avg_pressure_all_nodes()
        own_avg_pressure = avg_pressure_all.get(inter_id, 0.0)
        pressure_diff_neighbors = np.zeros(self.num_neighbors, dtype=np.float32)
        for k, nb in enumerate(neighbors[:self.num_neighbors]):
            pressure_diff_neighbors[k] = own_avg_pressure - avg_pressure_all.get(nb, own_avg_pressure)

        # Corsie 0-5 = N/S, 6-11 = E/O (ordine canonico, vedi commento in
        # reset() e GREEN_LANES_PER_PHASE in cima al file).
        traffic_asymmetry_ns_ew = np.array(
            [float(np.mean(lane_pressure[:6]) - np.mean(lane_pressure[6:]))],
            dtype=np.float32
        )

        parts = [lane_pressure, real_degree, pressure_diff_neighbors, traffic_asymmetry_ns_ew]
        if self.use_phase_pressure_meta:
            parts.append(self._compute_phase_pressure_vector(inter_id, lane_vehicles, vehicle_distances))

        return np.concatenate(parts)

    # Quanti step indietro guardare per queue_trend in get_temporal_meta_features
    # (vedi descrizione_stato_meta_reward.md §2.5).
    _TREND_STEPS_BACK = 3
    # Soglia di saturazione per phase_dwell_time, per normalizzare in [0,1].
    # Corretta il 13/9/2026 (vedi descrizione_stato_meta_reward.md, su segnalazione): con
    # use_action_mask=True, self.consecutive_phases non supera mai 2 (la
    # maschera anti-starvation forza un cambio fase al 3o step consecutivo,
    # vedi get_invalid_actions()) -- un CAP=10 (il valore precedente, mai
    # derivato da questo vincolo) squiacciava il segnale in {0, 0.1, 0.2},
    # sprecando il 90% del range [0,1]. Senza maschera invece non c'e' alcun
    # vincolo strutturale sulla durata di una fase, quindi si mantiene un
    # CAP arbitrario piu' permissivo.
    _DWELL_TIME_CAP_MASKED = 2.0
    _DWELL_TIME_CAP_UNMASKED = 10.0

    def get_temporal_meta_features(self, inter_id: str,
                                   history: Optional[List[np.ndarray]] = None,
                                   history_len: int = 5,
                                   current_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Features temporali per il TMK-Learner (Section 4.3.1), riviste il
        12/9/2026 -- vedi descrizione_stato_meta_reward.md §2.3/§2.5 per la motivazione
        completa di ciascuna:
        - queue_trend (N_LANES): n_vec(ora) − n_vec(qualche step fa), per
          corsia. Sostituisce queue_len (che ripeteva n_vec dello stato
          principale, e per di più senza rispettare use_vision_cutoff).
        - phase_dwell_time (1): da quanti step consecutivi la fase corrente
          è attiva (self.consecutive_phases, già tracciato per la maschera
          anti-starvation, mai esposto come feature finora), saturato e
          normalizzato in [0,1] (CAP=2 con maschera attiva, vedi
          _DWELL_TIME_CAP_MASKED). Rivista il 13/9/2026: NON è più "quanto
          e' rimasto aperto il verde" in senso di smaltimento coda (con la
          maschera anti-starvation il range reale è {0,1,2}, troppo corto
          per rappresentare uno smaltimento) — è invece un segnale su
          QUANTO E' VINCOLATA la prossima decisione: a 1 (fase appena
          scelta) la prossima scelta è libera su tutte le 8 fasi; a 2 la
          maschera eliminerà la fase corrente dalle opzioni valide al passo
          successivo — un vincolo strutturale imminente che la LSTM può
          imparare ad anticipare (es. "tra un istante dovrò comunque
          cambiare fase, non ha senso costruire memoria sull'assunto che
          questa prosegua"). Il valore 0 compare solo al primissimo step di
          un episodio, prima di qualunque azione.
        - queue_volatility (N_LANES): deviazione standard di n_vec per
          corsia sulla finestra recente disponibile. Sostituisce lo storico
          grezzo (ridondante con la memoria che la LSTM mantiene già da
          sola in (h,c) — vedi §2.1/§2.3).

        Nota 13/9/2026: `phase_pressure` (l'estensione di meta_v3) NON è più
        qui — solo nel SMK (get_spatial_meta_features). E' un dato spaziale
        per fase, non una dinamica temporale, e ripeterlo identico nel TMK
        non aggiungeva alcuna informazione che il TMK potesse usare in modo
        diverso dal SMK.

        Args:
            history: osservazioni PASSATE (non include lo stato corrente:
                     nei chiamanti, viene aggiornato con obs solo DOPO
                     questa chiamata — vedi train.py/test.py).
            current_state: stato corrente del nodo (obs[inter_id]), se
                     disponibile — usato come estremo "adesso" per
                     queue_trend/queue_volatility. Se assente (retrocompatibilità),
                     si usa l'ultimo elemento di history come proxy di "adesso".

        Returns:
            array di dim = 2 * N_LANES + 1
        """
        # Finestra di n_vec grezzi, dal più vecchio al più recente, includendo
        # lo stato corrente in coda se fornito -- unica fonte per trend e
        # volatilità, cosi' le due feature sono coerenti tra loro.
        window = [np.asarray(h[:N_LANES], dtype=np.float32) for h in (history or [])[-history_len:]]
        if current_state is not None:
            window.append(np.asarray(current_state[:N_LANES], dtype=np.float32))

        if window:
            current_n_vec = window[-1]
            if len(window) > self._TREND_STEPS_BACK:
                past_n_vec = window[-1 - self._TREND_STEPS_BACK]
            else:
                # Storico troppo corto per guardare _TREND_STEPS_BACK indietro:
                # usa il punto più vecchio disponibile invece di azzerare il trend.
                past_n_vec = window[0]
            queue_trend = current_n_vec - past_n_vec
        else:
            queue_trend = np.zeros(N_LANES, dtype=np.float32)

        if len(window) > 1:
            queue_volatility = np.std(np.stack(window), axis=0).astype(np.float32)
        else:
            queue_volatility = np.zeros(N_LANES, dtype=np.float32)

        dwell_cap = self._DWELL_TIME_CAP_MASKED if self.use_action_mask else self._DWELL_TIME_CAP_UNMASKED
        phase_dwell_time = np.array(
            [min(self.consecutive_phases.get(inter_id, 0), dwell_cap) / dwell_cap],
            dtype=np.float32
        )

        return np.concatenate([queue_trend, phase_dwell_time, queue_volatility])

    # ─── Utilità ──────────────────────────────────────────────────────────────

    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Restituisce la matrice di adiacenza (N x N) delle intersezioni.
        Utilizzata da PyTorch Geometric per costruire il grafo.
        """
        n = self.n_intersections
        adj = np.zeros((n, n), dtype=np.float32)
        for iid, neighbors in self.adjacency.items():
            i = self.inter_id_to_idx.get(iid, -1)
            if i < 0:
                continue
            for nb in neighbors:
                j = self.inter_id_to_idx.get(nb, -1)
                if j >= 0:
                    adj[i, j] = 1.0
        return adj

    def get_edge_index(self):
        """
        Restituisce l'edge_index (2 x E) per PyTorch Geometric.
        """
        import torch
        src_list, dst_list = [], []
        for iid, neighbors in self.adjacency.items():
            i = self.inter_id_to_idx.get(iid, -1)
            if i < 0:
                continue
            for nb in neighbors:
                j = self.inter_id_to_idx.get(nb, -1)
                if j >= 0:
                    src_list.append(i)
                    dst_list.append(j)
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        return edge_index

    def _travel_times(self, include_unfinished: bool = True) -> List[float]:
        """
        Lista dei travel time individuali, usata sia da get_average_travel_time()
        che da get_travel_time_stats() — stessa fonte per entrambe, cosi' la
        media e le statistiche di coda sono sempre coerenti tra loro (vedi nota
        del 13/9/2026 sotto).

        Con include_unfinished=True (default): include anche i veicoli ancora in
        rete al momento della chiamata, contando il tempo gia' trascorso da quando
        sono entrati (un lower bound del loro vero travel time, che sarebbe solo
        peggiore se la simulazione continuasse) — un veicolo mai arrivato e'
        esattamente il caso che si vuole vedere nel caso peggiore (max/percentili),
        non nasconderlo escludendolo dal conteggio.
        """
        times = list(self.arrived_tt)
        if include_unfinished and self.spawn_times:
            current_time = self.current_step * STEP_TIME
            times += [current_time - spawn_t for spawn_t in self.spawn_times.values()]
        return times

    def get_average_travel_time(self, include_unfinished: bool = True) -> float:
        """
        Calcola il travel time medio.

        Con include_unfinished=True (default, metrica usata per training/ranking):
        include anche i veicoli ancora in rete (vedi _travel_times()).

        Senza questo, un modello che ingolfa la rete e lascia passare solo pochi
        veicoli "fortunati" su corsie libere risulterebbe premiato (travel time
        basso calcolato su un campione piccolissimo e non rappresentativo) invece
        che penalizzato — nel caso estremo di ZERO veicoli arrivati, la versione
        "solo arrivati" restituirebbe 0.0, il punteggio migliore possibile.

        Con include_unfinished=False si ottiene la vecchia metrica "solo arrivati"
        (usa la stessa lista self.arrived_tt su cui l'engine C++ di CityFlow basa
        get_original_average_travel_time(), utile per confronti diretti col paper).
        """
        times = self._travel_times(include_unfinished)
        if not times:
            return 0.0
        return float(sum(times) / len(times))

    def get_completed_only_travel_time(self) -> float:
        """Travel time medio solo sui veicoli arrivati (vedi get_average_travel_time)."""
        return self.get_average_travel_time(include_unfinished=False)

    def get_travel_time_stats(self, include_unfinished: bool = True) -> dict:
        """
        Statistiche di coda sul travel time — utili perche' la media puo'
        nascondere casi estremi: un modello con media bassa ma coda lunga (pochi
        veicoli bloccati a lungo) e' diverso da uno con distribuzione uniforme.

        Con include_unfinished=True (default, coerente con get_average_travel_time):
        un veicolo mai arrivato entro fine episodio e' contato con il tempo gia'
        trascorso (lower bound) — se e' lui il caso peggiore in assoluto (il piu'
        delle volte lo e', essendo per definizione ancora in viaggio), e' quello il
        valore che compare in max/percentili. Correzione del 13/9/2026: prima
        veniva calcolato solo su self.arrived_tt (soli arrivati), inconsistente con
        get_average_travel_time — un veicolo mai arrivato, il candidato piu' ovvio
        per il caso peggiore, non compariva mai in tt_max.

        Returns: dict con max, std, p50, p90, p95, p99 (0.0 se non ci sono ancora
        veicoli spawnati/arrivati, es. episodio non ancora iniziato).
        """
        times = self._travel_times(include_unfinished)
        if not times:
            return {"max": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(times, dtype=np.float64)
        return {
            "max": float(arr.max()),
            "std": float(arr.std()),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def get_raw_travel_times(self, include_unfinished: bool = True) -> List[float]:
        """
        Lista completa dei travel time individuali (un valore per veicolo), non
        ridotta a statistiche aggregate — per grafici che vogliono mostrare la
        distribuzione vera (es. violin plot) invece di soli media/max/percentili.
        Stessa convenzione di get_travel_time_stats() (include_unfinished=True di
        default: i veicoli mai arrivati contano col tempo trascorso finora).
        """
        return list(self._travel_times(include_unfinished))

    def get_raw_vehicle_waits(self, include_unfinished: bool = True) -> Tuple[List[float], List[float]]:
        """
        Distribuzione per-VEICOLO (non per intersezione) della massima attesa
        consecutiva sperimentata, separata N/S vs W/E in base a su quale gruppo
        di corsie si trovava nel momento dell'attesa piu' lunga di quel veicolo.

        Alternativa a get_direction_fairness_stats() quando serve un campione con
        un numero di punti realistico per una distribuzione (uno per veicolo,
        tipicamente migliaia) invece del massimo per intersezione (uno per
        intersezione, tipicamente 16) — quest'ultimo resta la metrica ufficiale
        per il ranking tra modelli (vedi descrizione_metriche.md §5), questo e'
        pensato solo per grafici di distribuzione (violin plot).

        Con include_unfinished=True (default, stessa convenzione delle altre
        metriche): i veicoli ancora in rete contano con la loro attesa massima
        finora (lower bound, potrebbe crescere se l'episodio continuasse).

        Returns: (lista attese N/S, lista attese W/E), stessa lunghezza, un
        veicolo che non ha mai aspettato su un gruppo di corsie contribuisce 0.0.
        """
        ns = list(self.arrived_max_wait_ns)
        ew = list(self.arrived_max_wait_ew)
        if include_unfinished and self.spawn_times:
            for veh in self.spawn_times:
                ns.append(self.vehicle_max_wait_ns.get(veh, 0.0))
                ew.append(self.vehicle_max_wait_ew.get(veh, 0.0))
        return ns, ew

    def get_direction_fairness_stats(self) -> dict:
        """
        Statistiche di equita' tra corsie N/S e W/E, basate sulla massima
        attesa mai osservata per intersezione durante l'episodio
        (self.max_wait_ns/self.max_wait_ew, aggiornate ad ogni step()).

        Non presuppone la presenza di un'arteria: se il traffico e'
        uniforme i due gruppi risulteranno simili; se una delle due
        orientazioni e' sistematicamente favorita (es. un'arteria Est-Ovest,
        vedi descrizione_configurazioni.md §4bis), lo si vede da uno scarto
        marcato tra i due gruppi.

        Returns: dict con avg/max per gruppo N/S e W/E, il valore peggiore in
        assoluto osservato su una singola intersezione (worst_wait), e la
        variante "_resolved" di max_wait_ns/ew.

        Nota importante (scoperta il 13/9/2026 confrontando un record con la
        traiettoria vera nel replay): max_wait_ns/ew possono provenire da un
        veicolo ancora fermo nell'ISTANTE ESATTO in cui l'episodio finisce —
        l'attesa "vera" potrebbe essere anche piu' lunga di quella riportata,
        semplicemente non lo sappiamo perche' la simulazione si e' fermata,
        non perche' il veicolo sia stato servito. E' l'esatto analogo, per
        l'attesa, di "veicoli non arrivati" per il travel time (vedi
        get_average_travel_time): un caso CENSURATO, non un errore di calcolo.
        Va tenuto come metrica primaria (nascondere questi casi premierebbe
        chi lascia un veicolo bloccato per sempre — non farebbe mai scattare
        un nuovo record dopo l'ultimo istante osservato prima del blocco totale
        — esattamente il bias di sopravvivenza che include_unfinished=True
        evita gia' per il travel time).

        *_resolved e' invece il massimo SOLO tra gli stalli che si sono
        conclusi entro la fine dell'episodio (il veicolo si e' mosso di nuovo,
        quindi il timestamp del record e' precedente all'ultimo istante
        possibile) — un numero piu' piccolo o uguale, ma verificabile end-to-end
        in un replay come un singolo episodio di stallo con inizio e fine
        osservabili in un'unica intersezione.
        """
        ns_vals = list(self.max_wait_ns.values())
        ew_vals = list(self.max_wait_ew.values())
        if not ns_vals:
            return {"avg_wait_ns": 0.0, "max_wait_ns": 0.0, "max_wait_ns_resolved": 0.0,
                    "avg_wait_ew": 0.0, "max_wait_ew": 0.0, "max_wait_ew_resolved": 0.0,
                    "worst_wait": 0.0}
        avg_ns, max_ns = float(np.mean(ns_vals)), float(np.max(ns_vals))
        avg_ew, max_ew = float(np.mean(ew_vals)), float(np.max(ew_vals))

        episode_end = self.current_step * STEP_TIME

        def _resolved_max(values, records):
            resolved = [v for iid, v in values.items()
                       if records.get(iid) is not None and records[iid][2] < episode_end]
            return float(max(resolved)) if resolved else 0.0

        max_ns_resolved = _resolved_max(self.max_wait_ns, self._wait_ns_record)
        max_ew_resolved = _resolved_max(self.max_wait_ew, self._wait_ew_record)

        return {
            "avg_wait_ns": avg_ns, "max_wait_ns": max_ns, "max_wait_ns_resolved": max_ns_resolved,
            "avg_wait_ew": avg_ew, "max_wait_ew": max_ew, "max_wait_ew_resolved": max_ew_resolved,
            "worst_wait": max(max_ns, max_ew),
        }

    def get_invalid_actions(self) -> Dict[str, List[int]]:
        """
        Ritorna una maschera delle azioni non valide.

        Modalità avanzata (use_action_mask=True):
          Maschera la fase corrente se scelta >= 2 volte consecutive (anti-starvation).

        Modalità paper (use_action_mask=False):
          Nessun vincolo — restituisce dizionario con liste vuote.
        """
        if not self.use_action_mask:
            return {iid: [] for iid in self.inter_ids}

        invalid_actions = {}
        for iid, phase in self.current_phase.items():
            if self.consecutive_phases.get(iid, 0) >= 2:
                invalid_actions[iid] = [phase]
            else:
                invalid_actions[iid] = []
        return invalid_actions

    def get_original_average_travel_time(self) -> float:
        """
        Travel time medio nativo del motore CityFlow (solo veicoli arrivati).
        Stessa convenzione "solo arrivati" del paper originale: usare SOLO per un
        confronto diretto col numero riportato in letteratura (~430-470s), non per
        il ranking tra i propri modelli — vedi get_average_travel_time().
        """
        return self.engine.get_average_travel_time()

    def get_throughput(self) -> int:
        """Numero di veicoli che hanno completato il viaggio."""
        return len(self.all_spawned_vehicles) - int(self.engine.get_vehicle_count())

    @property
    def action_space_n(self) -> int:
        """Numero di azioni possibili per agente (= numero di fasi)."""
        return N_PHASES

    @property
    def observation_dim(self) -> int:
        """Dimensione del vettore di osservazione per agente (dipende da
        use_wait_vec e, se attivo, da use_phase_pressure_state: +N_PHASES,
        esperimento 13/9/2026, vedi descrizione_stato_meta_reward.md §1)."""
        base = STATE_DIM_FULL if self.use_wait_vec else STATE_DIM_PAPER
        return base + (N_PHASES if self.use_phase_pressure_state else 0)

    @property
    def spatial_meta_dim(self) -> int:
        """Dimensione delle feature spaziali per SMK-Learner (rivista il
        12/9/2026, vedi descrizione_stato_meta_reward.md §2.5, ed estesa il 13/9/2026 per
        meta_v3): lane_pressure (N_LANES) + real_degree (1) +
        pressure_diff_neighbors (num_neighbors) + traffic_asymmetry_ns_ew (1)
        + phase_pressure (N_PHASES) se use_phase_pressure_meta=True (default),
        altrimenti dim meta_v2 (senza phase_pressure). Deciso il 13/9/2026:
        phase_pressure va SOLO qui, non nel TMK (vedi temporal_meta_dim)."""
        return N_LANES + 1 + self.num_neighbors + 1 + (N_PHASES if self.use_phase_pressure_meta else 0)

    @property
    def temporal_meta_dim(self) -> int:
        """Dimensione delle feature temporali per TMK-Learner (rivista il
        12/9/2026, vedi descrizione_stato_meta_reward.md §2.5): queue_trend (N_LANES) +
        phase_dwell_time (1) + queue_volatility (N_LANES). Fissa a 25 --
        non dipende da use_phase_pressure_meta: dal 13/9/2026 la feature
        phase_pressure di meta_v3 va solo nel SMK (vedi spatial_meta_dim),
        mai qui."""
        return N_LANES + 1 + N_LANES
