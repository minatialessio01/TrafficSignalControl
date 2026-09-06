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
STATE_DIM = N_LANES + N_LANES + N_PHASES  # dim stato: 12 (veicoli) + 12 (wait time) + 8 (fasi) = 32

GREEN_TIME = 10               # durata verde (s)
YELLOW_TIME = 3               # durata giallo (s)
RED_TIME = 2                  # durata rosso (s)
STEP_TIME = GREEN_TIME + YELLOW_TIME + RED_TIME  # 15s per ciclo

# Fasi standard: in CityFlow le fasi si mappano a indici 0..7
# La svolta a destra è controllata dal semaforo (non più sempre verde).
ALL_PHASES = list(range(N_PHASES))

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

    def __init__(self, config_path: str, num_neighbors: int = 4, alpha: float = 0.5):
        """
        Args:
            config_path: percorso al file config.json di CityFlow
            num_neighbors: numero massimo di intersezioni vicine (paper: 4)
            alpha: iperparametro per la penalità dei ritardi nel reward
        """
        if not CITYFLOW_AVAILABLE:
            raise RuntimeError(
                "CityFlow non installato. Esegui 'pip install cityflow' "
                "o usa Docker/WSL con setup_env.sh"
            )

        self.config_path = config_path
        self.num_neighbors = num_neighbors
        self.alpha = alpha

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
        import math
        lengths = {}
        for r in self.roadnet.get("roads", []):
            pts = r.get("points", [])
            length = 0.0
            for i in range(len(pts)-1):
                length += math.dist((pts[i]["x"], pts[i]["y"]), (pts[i+1]["x"], pts[i+1]["y"]))
            lengths[r["id"]] = length
        return lengths

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

        # Esegui GREEN_TIME secondi di simulazione verde
        for _ in range(GREEN_TIME):
            self.engine.next_step()

        # Esegui YELLOW_TIME secondi di simulazione giallo
        # In CityFlow non esiste 'giallo' come stato separato, lo simuliamo
        # con una fase che non permette nuove partenze (convenzionale)
        for _ in range(YELLOW_TIME + RED_TIME):
            self.engine.next_step()

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

        s_i^t = [n_vec (12), wait_vec (12), p_vec (8)] (Section 3.2 modificata)

        n_vec: numero di veicoli su ciascuna delle 12 corsie in ingresso (filtrato a 167m dal semaforo)
        wait_vec: max waiting time normalizzato (0.0 - 1.0) per corsia (filtrato a 167m dal semaforo)
        p_vec: fase corrente (one-hot encoding 8 bit)
        """
        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_distances = self.engine.get_vehicle_distance()
        observations = {}

        for iid in self.inter_ids:
            # n_vec: numero veicoli sulle corsie in ingresso
            lanes = self.inter_lanes.get(iid, [])
            # Padding/tronca a N_LANES corsie
            n_vec = np.zeros(N_LANES, dtype=np.float32)
            wait_vec = np.zeros(N_LANES, dtype=np.float32)
            
            for k, lane_id in enumerate(lanes[:N_LANES]):
                if lane_id.startswith("missing_"):
                    continue
                road_id = "_".join(lane_id.split("_")[:-1])
                cutoff = max(0.0, self.road_lengths.get(road_id, 340.0) - 167.0)
                vehicles = lane_vehicles.get(lane_id, [])
                
                veicoli_vicini = 0
                max_wt = 0.0
                
                for veh in vehicles:
                    dist = vehicle_distances.get(veh, 0.0)
                    # Filtriamo solo i veicoli vicini al semaforo
                    if dist >= cutoff:
                        veicoli_vicini += 1
                        
                        wt = self.vehicle_wait_times.get(veh, 0.0)
                        if wt > max_wt:
                            max_wt = wt
                            
                n_vec[k] = veicoli_vicini
                # Normalizza e clippa tra 0.0 e 1.0 (supponendo 100s come limite ragionevole di saturazione)
                wait_vec[k] = np.clip(max_wt / 100.0, 0.0, 1.0)

            # Normalizza il numero di veicoli
            n_vec = np.clip(n_vec / 30.0, 0.0, 1.0)

            # p_vec: fase corrente (one-hot)
            p_vec = np.zeros(N_PHASES, dtype=np.float32)
            p_vec[self.current_phase[iid]] = 1.0

            observations[iid] = np.concatenate([n_vec, wait_vec, p_vec])  # dim=32

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
                cutoff = max(0.0, self.road_lengths.get(road_id, 340.0) - 167.0)
                for veh in lane_vehicles.get(l, []):
                    if vehicle_distances.get(veh, 0.0) >= cutoff:
                        incoming_ids[iid].add(veh)
        return incoming_ids

    def _compute_rewards(self, incoming_t0: Dict[str, set] = None, incoming_t1: Dict[str, set] = None) -> Dict[str, float]:
        """
        Calcola il reward per ogni intersezione basato sul Throughput reale e i veicoli in attesa.
        Reward = Passed - Incoming - (alpha * max_red_wait_time)
        Dove:
         - Passed: Veicoli presenti al tempo t che non sono più nelle corsie in ingresso al tempo t+1.
         - Incoming: Veicoli in ingresso al tempo t.
        """
        if incoming_t0 is None or incoming_t1 is None:
            # Fallback se chiamato fuori da step (es. in init, anche se non succede)
            return {iid: 0.0 for iid in self.inter_ids}

        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_distances = self.engine.get_vehicle_distance()
        rewards = {}

        for iid in self.inter_ids:
            lanes = self.inter_lanes.get(iid, [])
            
            in_t0 = incoming_t0.get(iid, set())
            in_t1 = incoming_t1.get(iid, set())
            
            passed = len(in_t0 - in_t1)
            incoming = len(in_t0)
            
            current_phase = self.current_phase[iid]
            green_lanes = GREEN_LANES_PER_PHASE.get(current_phase, [])
            
            max_red_wait_time = 0.0
            for k, lane_id in enumerate(lanes[:N_LANES]):
                if k not in green_lanes:
                    if lane_id.startswith("missing_"):
                        continue
                    road_id = "_".join(lane_id.split("_")[:-1])
                    cutoff = max(0.0, self.road_lengths.get(road_id, 340.0) - 167.0)
                    vehicles = lane_vehicles.get(lane_id, [])
                    for veh in vehicles:
                        # Consideriamo il wait time solo per le auto vicine al semaforo
                        if vehicle_distances.get(veh, 0.0) >= cutoff:
                            wt = self.vehicle_wait_times.get(veh, 0.0)
                            if wt > max_red_wait_time:
                                max_red_wait_time = wt
            wasted_green_penalty = 0.0
            if passed == 0 and incoming > 0:
                wasted_green_penalty = 50.0  # Penalità esplicita per aver dato il verde a una corsia vuota mentre c'è traffico altrove
                            
            # Formula: Passed - Incoming - (alpha * max_red_wait_time) - wasted_green_penalty
            raw_reward = float(passed) - float(incoming) - (self.alpha * max_red_wait_time) - wasted_green_penalty
            
            # NORMALIZZAZIONE:
            # Dividiamo per 100.0 per riportare il reward in un range gestibile dalla rete (es. da -10 a +5).
            normalized_reward = raw_reward / 100.0
            
            # CLIPPING di sicurezza contro picchi anomali
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
        """Restituisce il totale dei veicoli in uscita (entro 167m)."""
        lane_vehicles = self.engine.get_lane_vehicles()
        vehicle_distances = self.engine.get_vehicle_distance()
        out_lanes = self._get_outgoing_lanes(inter_id)
        count = 0
        for lid in out_lanes:
            for veh in lane_vehicles.get(lid, []):
                # CityFlow distance is from start of road segment
                if vehicle_distances.get(veh, 0.0) <= 167.0:
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
                cutoff = max(0.0, self.road_lengths.get(road_id, 340.0) - 167.0)
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
                hist_array = np.vstack([np.zeros((pad, STATE_DIM)), hist_array])
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

    def get_average_travel_time(self) -> float:
        """
        Calcola il travel time medio includendo SOLO i veicoli arrivati a destinazione.
        """
        if not self.arrived_tt:
            return 0.0
        return float(sum(self.arrived_tt) / len(self.arrived_tt))

    def get_invalid_actions(self) -> Dict[str, List[int]]:
        """
        Ritorna una maschera delle azioni non valide (fasi scelte più di 2 volte consecutive).
        Vincolo applicato per qualunque configurazione/griglia di incroci.
        """
        invalid_actions = {}
        for iid, phase in self.current_phase.items():
            if self.consecutive_phases.get(iid, 0) >= 2:
                invalid_actions[iid] = [phase]
            else:
                invalid_actions[iid] = []
        return invalid_actions

    def get_original_average_travel_time(self) -> float:
        """Travel time medio originale di CityFlow (solo arrivati)."""
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
        """Dimensione del vettore di osservazione per agente."""
        return STATE_DIM

    @property
    def spatial_meta_dim(self) -> int:
        """Dimensione delle feature spaziali per SMK-Learner."""
        return N_LANES * 2 + self.num_neighbors

    @property
    def temporal_meta_dim(self) -> int:
        """Dimensione delle feature temporali per TMK-Learner."""
        return N_LANES + N_LANES * 5  # queue + 5 step history
