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

        # Movimenti (lane-link) abilitati per fase, dal roadnet — usati da
        # MaxPressureAgent per la pressione classica (Varaiya 2013), vedi
        # _build_phase_lanelinks().
        self.phase_lanelinks = self._build_phase_lanelinks()

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

        self.current_step += 1
        current_time = self.current_step * STEP_TIME
        
        # 2.2 Teleported vehicles bug: limit tt to reasonable bounds
        MAX_PLAUSIBLE_TT = self.config.get("maxStep", float("inf")) * STEP_TIME
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

            if self.use_wait_vec:
                observations[iid] = np.concatenate([n_vec, wait_vec, p_vec])  # dim=32
            else:
                observations[iid] = np.concatenate([n_vec, p_vec])            # dim=20 (paper)

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
                raw_reward = float(passed) - float(incoming) - (self.alpha * max_red_wait_time) - wasted_green_penalty
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
        """Restituisce il totale dei veicoli in uscita (entro VISION_CUTOFF_M)."""
        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_distances = self.engine.get_vehicle_distance()
        out_lanes = self._get_outgoing_lanes(inter_id)
        count = 0
        for lid in out_lanes:
            for veh in lane_vehicles.get(lid, []):
                # CityFlow distance is from start of road segment
                if vehicle_distances.get(veh, 0.0) <= VISION_CUTOFF_M:
                    count += 1
        return count

    def _is_done(self) -> bool:
        """Controlla se la simulazione è terminata."""
        sim_time = self.current_step * STEP_TIME
        # La simulazione termina quando viene raggiunta la durata configurata
        # (CityFlow non ha un'API diretta per questo, lo stimiamo)
        return sim_time >= self.config.get("maxStep", float("inf"))

    # ─── Meta-features (per SMK-Learner e TMK-Learner) ───────────────────────

    def get_spatial_meta_features(self, inter_id: str) -> np.ndarray:
        """
        Features spaziali per il SMK-Learner (Section 4.3.1, Fig. 5b):
        - Pressione di ogni corsia in ingresso (normalizzata)
        - Numero di veicoli per ogni corsia (normalizzato)
        - Distanza dai vicini (normalizzata)

        Returns:
            array di dim = N_LANES * 2 + num_neighbors
        """
        lane_vehicles = self.engine.get_lane_vehicles()
        lanes = self.inter_lanes.get(inter_id, [])

        # Lane pressure (veicoli per corsia, normalizzato)
        lane_pressure = np.zeros(N_LANES, dtype=np.float32)
        n_vehicles = np.zeros(N_LANES, dtype=np.float32)
        out_count = self._get_outgoing_vehicles_count(inter_id)
        avg_out = out_count / max(1.0, len(self._get_outgoing_lanes(inter_id)))

        for k, lid in enumerate(lanes[:N_LANES]):
            count = 0
            if not lid.startswith("missing_"):
                road_id = "_".join(lid.split("_")[:-1])
                cutoff = max(0.0, self.road_lengths.get(road_id, 340.0) - VISION_CUTOFF_M)
                for veh in lane_vehicles.get(lid, []):
                    if self.engine.get_vehicle_distance().get(veh, 0.0) >= cutoff:
                        count += 1
            lane_pressure[k] = (count - avg_out) / 30.0
            n_vehicles[k] = count / 30.0

        # Distanze dai vicini (ordinate)
        neighbors = self.adjacency.get(inter_id, [])
        distances = np.zeros(self.num_neighbors, dtype=np.float32)
        for k, nb in enumerate(neighbors[:self.num_neighbors]):
            distances[k] = self.inter_distances.get((inter_id, nb), 1.0)

        return np.concatenate([lane_pressure, n_vehicles, distances])

    def get_temporal_meta_features(self, inter_id: str,
                                   history: Optional[List[np.ndarray]] = None,
                                   history_len: int = 5) -> np.ndarray:
        """
        Features temporali per il TMK-Learner (Section 4.3.1):
        - Lunghezza queue per corsia (proxy: veicoli a bassa velocità)
        - Storico degli stati (ultimi history_len timestep)

        Returns:
            array di dim = N_LANES + N_LANES * history_len
        """
        lane_vehicles = self.engine.get_lane_vehicles()
        lanes = self.inter_lanes.get(inter_id, [])

        # Queue length (approssimazione: veicoli in coda ≈ veicoli fermi)
        # In CityFlow non abbiamo accesso diretto alle velocità per corsia,
        # usiamo il numero di veicoli come proxy
        queue_len = np.zeros(N_LANES, dtype=np.float32)
        for k, lid in enumerate(lanes[:N_LANES]):
            queue_len[k] = len(lane_vehicles.get(lid, [])) / 30.0

        # Storico stati (padding con zeri se non disponibile)
        if history and len(history) > 0:
            hist_array = np.array(history[-history_len:])  # (T, STATE_DIM)
            # Padding se meno di history_len steps disponibili
            pad = history_len - len(hist_array)
            if pad > 0:
                hist_array = np.vstack([np.zeros((pad, self.observation_dim)), hist_array])
            hist_flat = hist_array[:, :N_LANES].flatten()  # solo n_vec
        else:
            hist_flat = np.zeros(N_LANES * history_len, dtype=np.float32)

        return np.concatenate([queue_len, hist_flat])

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

    def get_average_travel_time(self, include_unfinished: bool = True) -> float:
        """
        Calcola il travel time medio.

        Con include_unfinished=True (default, metrica usata per training/ranking):
        include anche i veicoli ancora in rete al momento della chiamata, contando
        il tempo gia' trascorso da quando sono entrati (un lower bound del loro vero
        travel time, che sarebbe solo peggiore se la simulazione continuasse).

        Senza questo, un modello che ingolfa la rete e lascia passare solo pochi
        veicoli "fortunati" su corsie libere risulterebbe premiato (travel time
        basso calcolato su un campione piccolissimo e non rappresentativo) invece
        che penalizzato — nel caso estremo di ZERO veicoli arrivati, la versione
        "solo arrivati" restituirebbe 0.0, il punteggio migliore possibile.

        Con include_unfinished=False si ottiene la vecchia metrica "solo arrivati"
        (usa la stessa lista self.arrived_tt su cui l'engine C++ di CityFlow basa
        get_original_average_travel_time(), utile per confronti diretti col paper).
        """
        times = list(self.arrived_tt)
        if include_unfinished and self.spawn_times:
            current_time = self.current_step * STEP_TIME
            times += [current_time - spawn_t for spawn_t in self.spawn_times.values()]
        if not times:
            return 0.0
        return float(sum(times) / len(times))

    def get_completed_only_travel_time(self) -> float:
        """Travel time medio solo sui veicoli arrivati (vedi get_average_travel_time)."""
        return self.get_average_travel_time(include_unfinished=False)

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
        """Dimensione del vettore di osservazione per agente (dipende da use_wait_vec)."""
        return STATE_DIM_FULL if self.use_wait_vec else STATE_DIM_PAPER

    @property
    def spatial_meta_dim(self) -> int:
        """Dimensione delle feature spaziali per SMK-Learner."""
        return N_LANES * 2 + self.num_neighbors

    @property
    def temporal_meta_dim(self) -> int:
        """Dimensione delle feature temporali per TMK-Learner."""
        return N_LANES + N_LANES * 5  # queue + 5 step history
