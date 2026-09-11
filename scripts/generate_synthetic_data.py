# -*- coding: utf-8 -*-
"""
Generazione parametrica dei dataset sintetici per MetaSTGAT.

Crea una griglia rettangolare MxN di intersezioni con configurazioni di traffico
calcolate "al contrario" in base a un target esatto di veicoli.
Supporta distribuzione flat, peak (3 fasi), peaks (5 fasi double-peak) e
workday (5 fasce a durata e intensita' asimmetriche, tipo giornata lavorativa:
flat_basso -> peak_basso -> flat_alto -> peak_alto -> flat_basso).
"""

import json
import os
import random
import argparse
import math

# ============================================================
# COSTANTI GEOMETRICHE E DEL PAPER
# ============================================================
LANE_WIDTH = 3.5       # Larghezza corsia (m)
LANES_PER_DIRECTION = 3  # 3 corsie per direzione (SX, dritto, DX)
SIMULATION_DURATION = 1800  # s
DELTA_T = 10           # durata minima step (green time)
INTER_WIDTH = 20.0     # semi-larghezza intersezione
MAX_SPEED = 11.11      # ~40 km/h
N_BEZIER_PTS = 11      # punti curva Bezier

# Probabilità di svolta usate dalla DFS
TURN_LEFT = 0.10
TURN_STRAIGHT = 0.60
TURN_RIGHT = 0.30

# ============================================================
# HELPER GEOMETRICI
# ============================================================
def opposite(d):
    return {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}[d]

def lane_center_offset(li):
    return (LANE_WIDTH / 2.0) + LANE_WIDTH * li

def get_entry_point(cx, cy, from_dir, li):
    w = INTER_WIDTH
    off = lane_center_offset(li)
    if from_dir == 'N': return (cx - off, cy + w)
    elif from_dir == 'S': return (cx + off, cy - w)
    elif from_dir == 'W': return (cx - w, cy - off)
    elif from_dir == 'E': return (cx + w, cy + off)

def get_exit_point(cx, cy, to_dir, li):
    w = INTER_WIDTH
    off = lane_center_offset(li)
    if to_dir == 'N': return (cx + off, cy + w)
    elif to_dir == 'S': return (cx - off, cy - w)
    elif to_dir == 'E': return (cx + w, cy - off)
    elif to_dir == 'W': return (cx - w, cy + off)

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
            if from_dir in ('W', 'E'): p1 = (p2[0], p0[1])
            else: p1 = (p0[0], p2[1])
            pts = bezier_pts(p0, p1, p2)

        lanelinks.append({
            'startLaneIndex': li_in,
            'endLaneIndex': li_out,
            'points': pts
        })
    return lanelinks

ROADLINK_ORDER = [
    ('N', 'S'), ('N', 'E'), ('N', 'W'),
    ('S', 'N'), ('S', 'W'), ('S', 'E'),
    ('W', 'E'), ('W', 'N'), ('W', 'S'),
    ('E', 'W'), ('E', 'S'), ('E', 'N'),
]

PHASE_ROADLINKS = [
    [0, 2, 3, 5], [1, 4], [0, 1, 2], [3, 4, 5],
    [6, 8, 9, 11], [7, 10], [9, 10, 11], [6, 7, 8],
]

def build_lightphases():
    phases = [{'time': DELTA_T, 'availableRoadLinks': indices} for indices in PHASE_ROADLINKS]
    # 9a fase di tutto-rosso (nessun movimento abilitato), indice = len(PHASE_ROADLINKS) = 8.
    # Mai selezionabile dall'agente: cityflow_env.py la usa internamente per RED_TIME
    # secondi tra un'azione e la successiva (vedi ALL_RED_PHASE in cityflow_env.py).
    # Necessaria in OGNI roadnet generato, altrimenti CityFlow va in out_of_range
    # non appena l'ambiente prova a impostarla.
    phases.append({'time': 2, 'availableRoadLinks': []})
    return phases

def get_inter_id(r, c): return f"intersection_{r}_{c}"
def get_road_id(from_r, from_c, to_r, to_c): return f"road_{from_r}_{from_c}_to_{to_r}_{to_c}"

def make_simple_roadnet(grid_r: int, grid_c: int, road_length: int) -> dict:
    intersections = []
    roads = []
    spacing = road_length + 60
    
    for r in range(-1, grid_r + 1):
        for c in range(-1, grid_c + 1):
            is_virtual = (r == -1 or r == grid_r or c == -1 or c == grid_c)
            is_corner = (r == -1 or r == grid_r) and (c == -1 or c == grid_c)
            
            if is_corner: continue

            iid = get_inter_id(r, c)
            cx, cy = float((c + 1) * spacing), float((r + 1) * spacing)

            if is_virtual:
                if r == -1: nr, nc = 0, c
                elif r == grid_r: nr, nc = grid_r - 1, c
                elif c == -1: nr, nc = r, 0
                elif c == grid_c: nr, nc = r, grid_c - 1
                
                roads_connected = [get_road_id(r, c, nr, nc), get_road_id(nr, nc, r, c)]
                intersections.append({
                    "id": iid, "point": {"x": cx, "y": cy}, "width": 0,
                    "roads": roads_connected, "roadLinks": [],
                    "trafficLight": {"lightphases": []}, "virtual": True
                })
            else:
                incoming, outgoing = {}, {}
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not ((nr == -1 or nr == grid_r) and (nc == -1 or nc == grid_c)):
                        if nr < r:
                            incoming['S'] = get_road_id(nr, nc, r, c)
                            outgoing['S'] = get_road_id(r, c, nr, nc)
                        elif nr > r:
                            incoming['N'] = get_road_id(nr, nc, r, c)
                            outgoing['N'] = get_road_id(r, c, nr, nc)
                        elif nc < c:
                            incoming['W'] = get_road_id(nr, nc, r, c)
                            outgoing['W'] = get_road_id(r, c, nr, nc)
                        elif nc > c:
                            incoming['E'] = get_road_id(nr, nc, r, c)
                            outgoing['E'] = get_road_id(r, c, nr, nc)

                roadlinks = []
                for from_dir, to_dir in ROADLINK_ORDER:
                    if from_dir in incoming and to_dir in outgoing:
                        tt = turn_type_of(from_dir, to_dir)
                        roadlinks.append({
                            'type': tt,
                            'startRoad': incoming[from_dir],
                            'endRoad': outgoing[to_dir],
                            'direction': FROM_DIR_INDEX[from_dir],
                            'laneLinks': compute_lanelinks(cx, cy, from_dir, to_dir)
                        })

                intersections.append({
                    "id": iid, "point": {"x": cx, "y": cy}, "width": INTER_WIDTH,
                    "roads": list(incoming.values()) + list(outgoing.values()),
                    "roadLinks": roadlinks,
                    "trafficLight": {"lightphases": build_lightphases()},
                    "virtual": False
                })

    for r in range(-1, grid_r + 1):
        for c in range(-1, grid_c + 1):
            if (r == -1 or r == grid_r) and (c == -1 or c == grid_c): continue

            for dr, dc in [(0, 1), (1, 0)]:
                nr, nc = r + dr, c + dc
                if -1 <= nr <= grid_r and -1 <= nc <= grid_c and not ((nr == -1 or nr == grid_r) and (nc == -1 or nc == grid_c)):
                    is_virtual_rc = (r == -1 or r == grid_r or c == -1 or c == grid_c)
                    is_virtual_nrnc = (nr == -1 or nr == grid_r or nc == -1 or nc == grid_c)
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
# FLOW GENERATION ALGORITHM
# ============================================================

def make_flow(target_vehicles: int, variance_type: str,
              grid_r: int, grid_c: int, duration: int = SIMULATION_DURATION, seed: int = 42,
              workday_rates: dict = None) -> list:
    rng = random.Random(seed)
    flows = []
    
    sources = []
    # Genera i bordi come sorgenti
    for r in range(grid_r):
        sources.append((r, -1, 0, 1))    # W -> E
        sources.append((r, grid_c, 0, -1)) # E -> W
    for c in range(grid_c):
        sources.append((-1, c, 1, 0))    # S -> N
        sources.append((grid_r, c, -1, 0)) # N -> S

    # Fase 1: DFS per raccogliere tutti i percorsi validi e i loro pesi
    all_paths = []
    total_weight = 0.0

    for start_r, start_c, dr, dc in sources:
        first_road = get_road_id(start_r, start_c, start_r+dr, start_c+dc)
        
        max_route_len = grid_r + grid_c + 4
        
        def dfs(r, c, c_dr, c_dc, prob, route, visited):
            if r < 0 or r >= grid_r or c < 0 or c >= grid_c:
                return [(route, prob)]
            if (r, c) in visited or len(route) > max_route_len: # Evita esplosione combinatoria
                return [(route, prob)]
            
            p_list = []
            new_visited = visited | {(r, c)}
            
            nr, nc = r + c_dr, c + c_dc
            p_list.extend(dfs(nr, nc, c_dr, c_dc, prob * TURN_STRAIGHT, route + [get_road_id(r, c, nr, nc)], new_visited))
            ndr, ndc = c_dc, -c_dr
            nr, nc = r + ndr, c + ndc
            p_list.extend(dfs(nr, nc, ndr, ndc, prob * TURN_LEFT, route + [get_road_id(r, c, nr, nc)], new_visited))
            ndr, ndc = -c_dc, c_dr
            nr, nc = r + ndr, c + ndc
            p_list.extend(dfs(nr, nc, ndr, ndc, prob * TURN_RIGHT, route + [get_road_id(r, c, nr, nc)], new_visited))
            
            return p_list

        paths = dfs(start_r+dr, start_c+dc, dr, dc, 1.0, [first_road], set())
        for route, prob in paths:
            if prob >= 0.005:
                # noise factor for route popularity
                noise_factor = rng.uniform(0.8, 1.2)
                final_prob = prob * noise_factor
                all_paths.append((route, final_prob))
                total_weight += final_prob

    # Fase 2: Allocazione proporzionale dei veicoli
    veh_params = {
        "length": 5.0, "width": 2.0, "maxPosAcc": 2.6, "maxNegAcc": 4.5,
        "usualPosAcc": 2.0, "usualNegAcc": 3.5, "minGap": 2.5, "maxSpeed": MAX_SPEED, "headwayTime": 1.5
    }

    for route, weight in all_paths:
        frac = weight / total_weight

        if variance_type == "flat":
            v_route = target_vehicles * frac
            if v_route < 1: continue
            interval = duration / v_route
            flows.append({
                "vehicle": veh_params,
                "route": route,
                "interval": float(max(1.0, interval + rng.gauss(0, interval * 0.1))),
                "startTime": 0,
                "endTime": duration
            })

        elif variance_type == "peak":
            # Low (25%) -> Peak (50%) -> Low (25%). Confini a 1/3 e 2/3 della
            # durata (generalizzato: prima erano assoluti 600/1200, validi
            # solo per duration=1800).
            v_route = target_vehicles * frac
            if v_route < 1: continue
            t1, t2 = duration / 3.0, 2.0 * duration / 3.0
            v_bg1 = v_route * 0.25
            v_pk  = v_route * 0.50
            v_bg2 = v_route * 0.25

            if v_bg1 > 0.5:
                interval_bg1 = t1 / v_bg1
                flows.append({"vehicle": veh_params, "route": route, "interval": float(max(1.0, interval_bg1 + rng.gauss(0, interval_bg1*0.1))), "startTime": 0, "endTime": round(t1)})
            if v_pk > 0.5:
                interval_pk = (t2 - t1) / v_pk
                flows.append({"vehicle": veh_params, "route": route, "interval": float(max(1.0, interval_pk + rng.gauss(0, interval_pk*0.1))), "startTime": round(t1), "endTime": round(t2)})
            if v_bg2 > 0.5:
                interval_bg2 = (duration - t2) / v_bg2
                flows.append({"vehicle": veh_params, "route": route, "interval": float(max(1.0, interval_bg2 + rng.gauss(0, interval_bg2*0.1))), "startTime": round(t2), "endTime": duration})

        elif variance_type == "peaks":
            # 5 Fasi: Low(9%) -> Peak1(36.5%) -> Low(9%) -> Peak2(36.5%) -> Low(9%)
            # Ogni fascia dura duration/5 secondi (generalizzato: prima erano
            # blocchi assoluti da 360s, validi solo per duration=1800).
            v_route = target_vehicles * frac
            if v_route < 1: continue
            w_low = 1.0 / 11.0
            w_pk = 4.0 / 11.0
            bounds = [round(duration * k / 5.0) for k in range(6)]
            weights_list = [w_low, w_pk, w_low, w_pk, w_low]

            for i, w in enumerate(weights_list):
                v_seg = v_route * w
                if v_seg > 0.5:
                    start, end = bounds[i], bounds[i + 1]
                    seg_dur = end - start
                    interval = seg_dur / v_seg
                    flows.append({
                        "vehicle": veh_params, "route": route,
                        "interval": float(max(1.0, interval + rng.gauss(0, interval*0.1))),
                        "startTime": start, "endTime": end
                    })

        elif variance_type == "workday":
            # 5 fasce asimmetriche che mappano una giornata lavorativa di 24h
            # sulla durata della simulazione (default 3600s -> 150s/ora):
            #   00-07 flat_basso, 07-09 peak_basso, 09-16 flat_alto,
            #   16-19 peak_alto, 19-24 flat_basso.
            # Ogni intensita' e' un "rate" espresso in veicoli equivalenti per
            # 1800s (stessa unita' delle altre config), convertito in veicoli
            # assoluti per la fascia in base alla sua durata reale.
            r = workday_rates
            day_fracs = [
                (0/24, 7/24,  r["basso"]),
                (7/24, 9/24,  r["peak_basso"]),
                (9/24, 16/24, r["alto"]),
                (16/24, 19/24, r["peak_alto"]),
                (19/24, 24/24, r["basso"]),
            ]
            for f_start, f_end, rate in day_fracs:
                seg_start, seg_end = f_start * duration, f_end * duration
                seg_dur = seg_end - seg_start
                v_seg = rate * (seg_dur / 1800.0) * frac
                if v_seg > 0.5:
                    interval = seg_dur / v_seg
                    flows.append({
                        "vehicle": veh_params, "route": route,
                        "interval": float(max(1.0, interval + rng.gauss(0, interval*0.1))),
                        "startTime": round(seg_start), "endTime": round(seg_end)
                    })

    return flows

# ============================================================
# MAIN
# ============================================================

def make_cityflow_config(roadnet_file: str, flow_file: str, simulation_time: int, output_dir: str) -> dict:
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera dataset sintetici MxN per MetaSTGAT con target veicoli.")
    parser.add_argument("--output-dir", default="data", help="Directory per i dati")
    parser.add_argument("--config-dir", default="configs", help="Directory per le config")
    parser.add_argument("--grid", type=str, required=True, help="Dimensione griglia (es. 4x4, 3x5)")
    parser.add_argument("--road-length", type=int, default=200, help="Lunghezza delle strade in metri (parametro fisico: la distanza reale tra incroci e' road-length + 60)")
    parser.add_argument("--road-length-label", type=str, default=None,
                        help="Etichetta da usare nei nomi file al posto di --road-length "
                             "(es. --road-length 107 --road-length-label 100: nome 'pulito' "
                             "per una rete la cui distanza reale, 107+60=167m, e' calibrata "
                             "per l'onda verde). Default: uguale a --road-length.")
    parser.add_argument("--duration", type=int, default=SIMULATION_DURATION,
                        help=f"Durata simulazione in secondi (default {SIMULATION_DURATION}; "
                             "usare 3600 per --variance workday).")
    parser.add_argument("--vehicles", type=str, default=None,
                        help="Quantità di veicoli esatta o in k (es. 8000, 8k, 1.3k). "
                             "Obbligatorio per flat/peak/peaks, ignorato per workday.")
    parser.add_argument("--variance", type=str, required=True, choices=["flat", "peak", "peaks", "workday"],
                        help="Distribuzione: flat (costante), peak (3 fasi), peaks (5 fasi double-peak), "
                             "workday (5 fasce a durata/intensita' asimmetriche, tipo giornata lavorativa).")
    parser.add_argument("--rate-basso", type=float, default=2160,
                        help="[solo --variance workday] intensita' 'bassa' (veicoli equiv./1800s) = 1.2 veic./s. Default 2160.")
    parser.add_argument("--rate-peak-basso", type=float, default=4500,
                        help="[solo --variance workday] intensita' 'picco basso' (veicoli equiv./1800s) = 2.5 veic./s. Default 4500.")
    parser.add_argument("--rate-alto", type=float, default=6300,
                        help="[solo --variance workday] intensita' 'alta' (veicoli equiv./1800s) = 3.5 veic./s, "
                             "la stessa densita' sostenuta di config_4x4_100m_6k_flat (validata: MaxPressure regge "
                             "l'intero episodio a questo ritmo). Default 6300.")
    parser.add_argument("--rate-peak-alto", type=float, default=7560,
                        help="[solo --variance workday] intensita' 'picco alto' (veicoli equiv./1800s) = 4.2 veic./s "
                             "(+20% su 'alto', entro il range gia' validato da config_4x4_100m_6k_peak). Default 7560.")
    args = parser.parse_args()

    # Parse grid
    try:
        parts = args.grid.lower().split('x')
        grid_r, grid_c = int(parts[0]), int(parts[1])
    except:
        parser.error("Formato --grid non valido. Usa MxN (es. 4x4 o 3x5).")

    if args.variance != "workday" and args.vehicles is None:
        parser.error("--vehicles e' obbligatorio per --variance flat/peak/peaks.")

    # Parse vehicles
    target_vehicles = None
    if args.vehicles is not None:
        v_str = args.vehicles.lower()
        if v_str.endswith('k'):
            target_vehicles = int(float(v_str[:-1]) * 1000)
        else:
            target_vehicles = int(v_str)

    road_length_label = args.road_length_label if args.road_length_label is not None else str(args.road_length)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.config_dir, exist_ok=True)

    # 1. Genera Roadnet (la geometria usa sempre la distanza FISICA reale,
    #    l'etichetta influisce solo sui nomi dei file)
    roadnet = make_simple_roadnet(grid_r, grid_c, args.road_length)
    roadnet_filename = f"roadnet_{grid_r}x{grid_c}_{road_length_label}m.json"
    with open(os.path.join(args.output_dir, roadnet_filename), "w", encoding='utf-8') as f:
        json.dump(roadnet, f, indent=2)

    # 2. Genera Flow
    workday_rates = None
    if args.variance == "workday":
        workday_rates = {
            "basso": args.rate_basso,
            "peak_basso": args.rate_peak_basso,
            "alto": args.rate_alto,
            "peak_alto": args.rate_peak_alto,
        }
    flow = make_flow(target_vehicles, args.variance, grid_r, grid_c, args.duration, seed=42,
                      workday_rates=workday_rates)

    # Configurazione del nome in base alla varianza
    if args.variance == "workday":
        flow_filename = f"flow_{grid_r}x{grid_c}_train.json"
    else:
        # Formatta k_val con una cifra decimale o precisa se intero, per il nome del file
        k_val_str = f"{target_vehicles / 1000.0:g}k"
        name_variance = args.variance  # "flat" | "peak" | "peaks"
        flow_filename = f"flow_{grid_r}x{grid_c}_{k_val_str}_{name_variance}.json"

    with open(os.path.join(args.output_dir, flow_filename), "w", encoding='utf-8') as f:
        json.dump(flow, f, indent=2)

    # 3. Genera Config
    cfg = make_cityflow_config(roadnet_filename, flow_filename, args.duration, "data/")
    if args.variance == "workday":
        cfg_filename = f"config_{grid_r}x{grid_c}_{road_length_label}m_train.json"
    else:
        cfg_filename = f"config_{grid_r}x{grid_c}_{road_length_label}m_{k_val_str}_{name_variance}.json"

    with open(os.path.join(args.config_dir, cfg_filename), "w", encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)

    print(f"[OK] Generazione completata con successo:")
    print(f"  - Roadnet: {os.path.join(args.output_dir, roadnet_filename)}")
    print(f"  - Flow:    {os.path.join(args.output_dir, flow_filename)}")
    print(f"  - Config:  {os.path.join(args.config_dir, cfg_filename)}")

    # Conferma conteggio
    actual_vehicles = 0
    for item in flow:
        s, e, i = item.get("startTime", 0), item.get("endTime", args.duration), item.get("interval", 1)
        if i > 0: actual_vehicles += math.ceil((e - s) / i)
    target_str = str(target_vehicles) if target_vehicles is not None else "workday (vedi --rate-*)"
    print(f"  - Veicoli effettivi generati: {actual_vehicles} (Target: {target_str})")
