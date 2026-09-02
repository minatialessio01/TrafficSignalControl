# -*- coding: utf-8 -*-
"""
Generazione dei dataset sintetici per MetaSTGAT.

Crea una griglia NxN di intersezioni con 4 configurazioni di traffico,
fedele alla descrizione dell'articolo (Section 5.2):
  - Config 1: arrival_rate=0.388, variance=Flat (0.3)
  - Config 2: arrival_rate=0.388, variance=Peak (0.6)
  - Config 3: arrival_rate=0.416, variance=Flat (0.3)
  - Config 4: arrival_rate=0.416, variance=Peak (0.6)

Formato output: JSON compatibile con CityFlow con corretta geometria
e configurazione stradale (laneLinks completi e fasi coerenti).
"""

import json
import os
import random
import math
import argparse

# ============================================================
# COSTANTI GEOMETRICHE E DEL PAPER
# ============================================================
GRID_N = 4             # Griglia NxN (default)
ROAD_LENGTH = 300      # Lunghezza strada
LANE_WIDTH = 3.5       # Larghezza corsia (m)
LANES_PER_DIRECTION = 3  # 3 corsie per direzione (SX, dritto, DX)
SIMULATION_DURATION = 1800  # s
DELTA_T = 10           # durata minima step (green time)
INTER_WIDTH = 20.0     # semi-larghezza intersezione
MAX_SPEED = 11.11      # ~40 km/h
N_BEZIER_PTS = 11      # punti curva Bezier

# Probabilità di svolta
TURN_LEFT = 0.10
TURN_STRAIGHT = 0.60
TURN_RIGHT = 0.30

# ============================================================
# HELPER GEOMETRICI
# ============================================================
def opposite(d):
    return {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}[d]

def lane_center_offset(li):
    # offset dal centro corsia li dalla linea mediana
    # corsia 0: sx, corsia 1: dritto, corsia 2: dx
    return (LANE_WIDTH / 2.0) + LANE_WIDTH * li

def get_entry_point(cx, cy, from_dir, li):
    w = INTER_WIDTH
    off = lane_center_offset(li)
    if from_dir == 'N':     # traveling South → west side
        return (cx - off, cy + w)
    elif from_dir == 'S':   # traveling North → east side
        return (cx + off, cy - w)
    elif from_dir == 'W':   # traveling East → south side
        return (cx - w, cy - off)
    elif from_dir == 'E':   # traveling West → north side
        return (cx + w, cy + off)

def get_exit_point(cx, cy, to_dir, li):
    w = INTER_WIDTH
    off = lane_center_offset(li)
    if to_dir == 'N':    # exiting North → northbound → east side
        return (cx + off, cy + w)
    elif to_dir == 'S':  # exiting South → southbound → west side
        return (cx - off, cy - w)
    elif to_dir == 'E':  # exiting East → eastbound → south side
        return (cx + w, cy - off)
    elif to_dir == 'W':  # exiting West → westbound → north side
        return (cx - w, cy + off)

def bezier_pts(p0, p1, p2, n=N_BEZIER_PTS):
    pts = []
    for i in range(n):
        t = i / (n - 1)
        x = (1-t)**2 * p0[0] + 2*t*(1-t) * p1[0] + t**2 * p2[0]
        y = (1-t)**2 * p0[1] + 2*t*(1-t) * p1[1] + t**2 * p2[1]
        pts.append({'x': round(x, 4), 'y': round(y, 4)})
    return pts

def linear_pts(p0, p2, n=N_BEZIER_PTS):
    pts = []
    for i in range(n):
        t = i / (n - 1)
        x = p0[0] + t * (p2[0] - p0[0])
        y = p0[1] + t * (p2[1] - p0[1])
        pts.append({'x': round(x, 4), 'y': round(y, 4)})
    return pts

def turn_type_of(from_dir, to_dir):
    _map = {
        ('W','E'): 'go_straight', ('W','N'): 'turn_left',  ('W','S'): 'turn_right',
        ('E','W'): 'go_straight', ('E','S'): 'turn_left',  ('E','N'): 'turn_right',
        ('N','S'): 'go_straight', ('N','E'): 'turn_left',  ('N','W'): 'turn_right',
        ('S','N'): 'go_straight', ('S','W'): 'turn_left',  ('S','E'): 'turn_right',
    }
    return _map[(from_dir, to_dir)]

TURN_INCOMING_LANE = {'turn_left': 0, 'go_straight': 1, 'turn_right': 2}
FROM_DIR_INDEX = {'W': 0, 'S': 1, 'E': 2, 'N': 3}

def compute_lanelinks(cx, cy, from_dir, to_dir):
    turn = turn_type_of(from_dir, to_dir)
    li_in = TURN_INCOMING_LANE[turn]
    p0 = get_entry_point(cx, cy, from_dir, li_in)

    lanelinks = []
    for li_out in range(LANES_PER_DIRECTION):
        p2 = get_exit_point(cx, cy, to_dir, li_out)

        if turn == 'go_straight':
            pts = linear_pts(p0, p2)
        else:
            if from_dir in ('W', 'E'):
                p1 = (p2[0], p0[1])
            else:
                p1 = (p0[0], p2[1])
            pts = bezier_pts(p0, p1, p2)

        lanelinks.append({
            'startLaneIndex': li_in,
            'endLaneIndex': li_out,
            'points': pts
        })
    return lanelinks

# ============================================================
# ROADLINK GENERATION (Ordini Costanti per RL)
# ============================================================
ROADLINK_ORDER = [
    ('N', 'S'),  # 0: N→S go_straight
    ('N', 'E'),  # 1: N→E turn_left
    ('N', 'W'),  # 2: N→W turn_right
    ('S', 'N'),  # 3: S→N go_straight
    ('S', 'W'),  # 4: S→W turn_left
    ('S', 'E'),  # 5: S→E turn_right
    ('W', 'E'),  # 6: W→E go_straight
    ('W', 'N'),  # 7: W→N turn_left
    ('W', 'S'),  # 8: W→S turn_right
    ('E', 'W'),  # 9: E→W go_straight
    ('E', 'S'),  # 10: E→S turn_left
    ('E', 'N'),  # 11: E→N turn_right
]

PHASE_ROADLINKS = [
    [0, 2, 3, 5],    # Fase 0: NTST
    [1, 4],          # Fase 1: NLSL
    [0, 1, 2],       # Fase 2: NTNL
    [3, 4, 5],       # Fase 3: STSL
    [6, 8, 9, 11],   # Fase 4: WTET
    [7, 10],         # Fase 5: WLEL
    [9, 10, 11],     # Fase 6: ETEL
    [6, 7, 8],       # Fase 7: WTWL
]

def build_lightphases():
    return [{'time': DELTA_T, 'availableRoadLinks': indices} for indices in PHASE_ROADLINKS]

# ============================================================
# ROADNET GENERATION
# ============================================================

def get_inter_id(r, c):
    return f"intersection_{r}_{c}"

def get_road_id(from_r, from_c, to_r, to_c):
    return f"road_{from_r}_{from_c}_to_{to_r}_{to_c}"

def get_direction(from_r, from_c, to_r, to_c):
    if to_r > from_r: return 'N' # viaggia Nord (da Sud)
    if to_r < from_r: return 'S' # viaggia Sud (da Nord)
    if to_c > from_c: return 'E' # viaggia Est (da Ovest)
    if to_c < from_c: return 'W' # viaggia Ovest (da Est)
    return 'UNKNOWN'

def make_simple_roadnet(grid_n: int, road_length: int) -> dict:
    intersections = []
    roads = []
    spacing = road_length + 60
    
    # Coordinate incroci e identificazione virtuali
    # Griglia: da -1 a grid_n (bordi inclusi)
    for r in range(-1, grid_n + 1):
        for c in range(-1, grid_n + 1):
            is_virtual = (r == -1 or r == grid_n or c == -1 or c == grid_n)
            is_corner = (r == -1 or r == grid_n) and (c == -1 or c == grid_n)
            
            if is_corner:
                continue

            iid = get_inter_id(r, c)
            cx, cy = float((c + 1) * spacing), float((r + 1) * spacing)

            if is_virtual:
                # Determina connessioni strade
                if r == -1: nr, nc = 0, c
                elif r == grid_n: nr, nc = grid_n - 1, c
                elif c == -1: nr, nc = r, 0
                elif c == grid_n: nr, nc = r, grid_n - 1
                
                roads_connected = [
                    get_road_id(r, c, nr, nc),
                    get_road_id(nr, nc, r, c)
                ]
                
                intersections.append({
                    "id": iid,
                    "point": {"x": cx, "y": cy},
                    "width": 0,
                    "roads": roads_connected,
                    "roadLinks": [],
                    "trafficLight": {"lightphases": []},
                    "virtual": True
                })
            else:
                # Nodi interni
                incoming = {}
                outgoing = {}
                # Trova i vicini validi
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not ((nr == -1 or nr == grid_n) and (nc == -1 or nc == grid_n)):
                        if nr < r: # vicino a SUD
                            incoming['S'] = get_road_id(nr, nc, r, c)
                            outgoing['S'] = get_road_id(r, c, nr, nc)
                        elif nr > r: # vicino a NORD
                            incoming['N'] = get_road_id(nr, nc, r, c)
                            outgoing['N'] = get_road_id(r, c, nr, nc)
                        elif nc < c: # vicino a OVEST
                            incoming['W'] = get_road_id(nr, nc, r, c)
                            outgoing['W'] = get_road_id(r, c, nr, nc)
                        elif nc > c: # vicino a EST
                            incoming['E'] = get_road_id(nr, nc, r, c)
                            outgoing['E'] = get_road_id(r, c, nr, nc)

                roadlinks = []
                for from_dir, to_dir in ROADLINK_ORDER:
                    if from_dir in incoming and to_dir in outgoing:
                        tt = turn_type_of(from_dir, to_dir)
                        start_road = incoming[from_dir]
                        end_road = outgoing[to_dir]
                        lanelinks = compute_lanelinks(cx, cy, from_dir, to_dir)
                        
                        roadlinks.append({
                            'type': tt,
                            'startRoad': start_road,
                            'endRoad': end_road,
                            'direction': FROM_DIR_INDEX[from_dir],
                            'laneLinks': lanelinks
                        })

                intersections.append({
                    "id": iid,
                    "point": {"x": cx, "y": cy},
                    "width": INTER_WIDTH,
                    "roads": list(incoming.values()) + list(outgoing.values()),
                    "roadLinks": roadlinks,
                    "trafficLight": {"lightphases": build_lightphases()},
                    "virtual": False
                })

    # Strade
    for r in range(-1, grid_n + 1):
        for c in range(-1, grid_n + 1):
            is_corner_rc = (r == -1 or r == grid_n) and (c == -1 or c == grid_n)
            if is_corner_rc: continue

            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                is_corner_nrnc = (nr == -1 or nr == grid_n) and (nc == -1 or nc == grid_n)
                
                if -1 <= nr <= grid_n and -1 <= nc <= grid_n and not is_corner_nrnc:
                    is_virtual_rc = (r == -1 or r == grid_n or c == -1 or c == grid_n)
                    is_virtual_nrnc = (nr == -1 or nr == grid_n or nc == -1 or nc == grid_n)
                    
                    if is_virtual_rc and is_virtual_nrnc: continue
                        
                    for (fr, fc), (tr, tc) in [((r, c), (nr, nc)), ((nr, nc), (r, c))]:
                        fx, fy = float((fc + 1) * spacing), float((fr + 1) * spacing)
                        tx, ty = float((tc + 1) * spacing), float((tr + 1) * spacing)
                        roads.append({
                            "id": get_road_id(fr, fc, tr, tc),
                            "startIntersection": get_inter_id(fr, fc),
                            "endIntersection": get_inter_id(tr, tc),
                            "points": [{"x": fx, "y": fy}, {"x": tx, "y": ty}],
                            "lanes": [{"width": LANE_WIDTH, "maxSpeed": MAX_SPEED} for _ in range(LANES_PER_DIRECTION)]
                        })

    return {"intersections": intersections, "roads": roads}

# ============================================================
# FLOW GENERATION
# ============================================================

def make_flow(arrival_rate: float, variance_type: str,
              grid_n: int, road_length: int,
              duration: int = SIMULATION_DURATION, seed: int = 42) -> list:
    """Genera i flussi con DFS."""
    rng = random.Random(seed)
    flows = []
    
    sources = []
    for i in range(grid_n):
        sources.append((i, -1, 0, 1))    # W -> E
        sources.append((i, grid_n, 0, -1)) # E -> W
        sources.append((-1, i, 1, 0))    # S -> N (in coords S is lower index)
        sources.append((grid_n, i, -1, 0)) # N -> S

    rate_per_source = arrival_rate / (2.0 * grid_n)

    for start_r, start_c, dr, dc in sources:
        first_road = get_road_id(start_r, start_c, start_r+dr, start_c+dc)
        
        def dfs(r, c, c_dr, c_dc, prob, route, visited):
            if r < 0 or r >= grid_n or c < 0 or c >= grid_n:
                return [(route, prob)]
            if (r, c) in visited or len(route) > 20:
                return [(route, prob)]
            
            p_list = []
            new_visited = visited | {(r, c)}
            
            # Straight (60%)
            nr, nc = r + c_dr, c + c_dc
            p_list.extend(dfs(nr, nc, c_dr, c_dc, prob * TURN_STRAIGHT, route + [get_road_id(r, c, nr, nc)], new_visited))
            # Left (10%) - left relative to current direction
            ndr, ndc = c_dc, -c_dr
            nr, nc = r + ndr, c + ndc
            p_list.extend(dfs(nr, nc, ndr, ndc, prob * TURN_LEFT, route + [get_road_id(r, c, nr, nc)], new_visited))
            # Right (30%) - right relative to current direction
            ndr, ndc = -c_dc, c_dr
            nr, nc = r + ndr, c + ndc
            p_list.extend(dfs(nr, nc, ndr, ndc, prob * TURN_RIGHT, route + [get_road_id(r, c, nr, nc)], new_visited))
            
            return p_list

        paths = dfs(start_r+dr, start_c+dc, dr, dc, 1.0, [first_road], set())
        
        for route, prob in paths:
            if prob < 0.005: continue
            path_rate = rate_per_source * prob
            if path_rate < 1e-6: continue
            
            if variance_type == "flat": interval = 1.0 / path_rate
            else: interval = 1.0 / (path_rate * 1.5)
                
            flows.append({
                "vehicle": {
                    "length": 5.0, "width": 2.0, "maxPosAcc": 2.6, "maxNegAcc": 4.5,
                    "usualPosAcc": 2.0, "usualNegAcc": 3.5, "minGap": 2.5, "maxSpeed": MAX_SPEED, "headwayTime": 1.5
                },
                "route": route,
                "interval": float(max(0.5, interval + rng.gauss(0, interval * (0.3 if variance_type == "flat" else 0.6)))),
                "startTime": 0,
                "endTime": duration
            })

    return flows

# ============================================================
# MAIN
# ============================================================

def make_cityflow_config(roadnet_file: str, flow_file: str,
                         simulation_time: int, output_dir: str) -> dict:
    return {
        "interval": 1.0,
        "seed": 42,
        "dir": output_dir,
        "roadnetFile": roadnet_file,
        "flowFile": flow_file,
        "rlTrafficLight": True,
        "laneChange": False,
        "saveReplay": False,
        "roadnetLogFile": "roadnet.log",
        "replayLogFile": "replay.log",
        "maxStep": simulation_time
    }

def generate_all(output_dir: str = "data",
                 config_dir: str = "configs",
                 grid_n: int = GRID_N):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    roadnet = make_simple_roadnet(grid_n, ROAD_LENGTH)
    roadnet_path = os.path.join(output_dir, f"roadnet_{grid_n}x{grid_n}.json")
    with open(roadnet_path, "w", encoding='utf-8') as f:
        json.dump(roadnet, f, indent=2)
    print(f"[OK] Roadnet salvato in {roadnet_path}")

    configs_info = [
        (1, 0.388, "flat"),
        (2, 0.388, "peak"),
        (3, 0.416, "flat"),
        (4, 0.416, "peak"),
    ]

    for cfg_num, rate, variance in configs_info:
        flow = make_flow(rate, variance, grid_n, ROAD_LENGTH, SIMULATION_DURATION, seed=42 + cfg_num)
        flow_filename = f"flow_{rate:.3f}_{variance}.json"
        flow_path = os.path.join(output_dir, flow_filename)
        with open(flow_path, "w", encoding='utf-8') as f:
            json.dump(flow, f, indent=2)

        cfg = make_cityflow_config(
            roadnet_file=f"roadnet_{grid_n}x{grid_n}.json",
            flow_file=flow_filename,
            simulation_time=SIMULATION_DURATION,
            output_dir=f"{output_dir}/"
        )
        cfg_path = os.path.join(config_dir, f"synthetic_{grid_n}x{grid_n}_config{cfg_num}.json")
        with open(cfg_path, "w", encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)

        print(f"[OK] Config {cfg_num} (rate={rate}, {variance}) -> {cfg_path}")

    print(f"\nDataset sintetici generati in '{output_dir}/'")
    print(f"Configurazioni CityFlow salvate in '{config_dir}/'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera dataset sintetici NxN per MetaSTGAT")
    parser.add_argument("--output-dir", default="data", help="Directory di output per i dati")
    parser.add_argument("--config-dir", default="configs", help="Directory per i file di configurazione CityFlow")
    parser.add_argument("--grid-n", type=int, default=4, help="Dimensione griglia N (default: 4)")
    args = parser.parse_args()

    generate_all(args.output_dir, args.config_dir, args.grid_n)
