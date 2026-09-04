import json
import random

def load_routes(template_file):
    with open(template_file, 'r') as f:
        data = json.load(f)
    
    # Raccogli tutte le route uniche
    routes = []
    seen = set()
    for item in data:
        r_tuple = tuple(item['route'])
        if r_tuple not in seen:
            seen.add(r_tuple)
            routes.append(item['route'])
    return routes

def create_vehicle():
    return {
        "length": 5.0,
        "width": 2.0,
        "maxPosAcc": 2.6,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.0,
        "usualNegAcc": 3.5,
        "minGap": 2.5,
        "maxSpeed": 11.11,
        "headwayTime": 1.5
    }

def generate_flat_flow(routes, total_vehicles=10000, duration=1800):
    flow = []
    vehicles_per_route = total_vehicles / len(routes)
    
    for route in routes:
        # Moltiplichiamo per un piccolo random per variare un po' il traffico su ogni strada
        # mantenendo la media
        factor = random.uniform(0.8, 1.2)
        vpr = vehicles_per_route * factor
        
        interval = duration / vpr if vpr > 0 else duration
        
        flow.append({
            "vehicle": create_vehicle(),
            "route": route,
            "interval": interval,
            "startTime": 0,
            "endTime": duration
        })
    return flow

def generate_double_peak_flow(routes, total_vehicles=10000, duration=1800):
    flow = []
    # 5 segmenti da 360 secondi: Low, Peak1, Low, Peak2, Low
    segments = [
        (0, 360, 0.5),     # Low (0.5x volume)
        (360, 720, 2.0),   # Peak1 (2.0x volume)
        (720, 1080, 0.5),  # Low
        (1080, 1440, 2.0), # Peak2
        (1440, 1800, 0.5)  # Low
    ]
    
    # Adjust weights so average is 1.0
    adjusted_segments = [(start, end, w / 1.1) for start, end, w in segments]
    
    vehicles_per_route = total_vehicles / len(routes)
    
    for route in routes:
        factor = random.uniform(0.8, 1.2)
        vpr = vehicles_per_route * factor
        
        for start, end, weight in adjusted_segments:
            seg_duration = end - start
            seg_vehicles = (vpr / duration) * seg_duration * weight
            interval = seg_duration / seg_vehicles if seg_vehicles > 0 else seg_duration
            
            flow.append({
                "vehicle": create_vehicle(),
                "route": route,
                "interval": interval,
                "startTime": start,
                "endTime": end
            })
    return flow

def main():
    routes = load_routes('../data/flow_0.388_flat.json')
    print(f"Loaded {len(routes)} distinct routes.")
    
    flat_flow = generate_flat_flow(routes, 6000, 1800)
    with open('../data/flow_10k_flat.json', 'w') as f:
        json.dump(flat_flow, f, indent=2)
    print("Generated flow_10k_flat.json")
    
    peak_flow = generate_double_peak_flow(routes, 6000, 1800)
    with open('../data/flow_10k_double_peak.json', 'w') as f:
        json.dump(peak_flow, f, indent=2)
    print("Generated flow_10k_double_peak.json")

    # Genera le configurazioni
    config_flat = {
      "interval": 1.0,
      "seed": 42,
      "dir": "data/",
      "roadnetFile": "roadnet_4x4_200m.json",
      "flowFile": "flow_10k_flat.json",
      "rlTrafficLight": True,
      "laneChange": False,
      "saveReplay": False,
      "roadnetLogFile": "roadnet.log",
      "replayLogFile": "replay.log",
      "maxStep": 1800
    }
    
    config_peak = config_flat.copy()
    config_peak["flowFile"] = "flow_10k_double_peak.json"
    
    with open('../configs/synthetic_4x4_200m_stress_flat.json', 'w') as f:
        json.dump(config_flat, f, indent=2)
        
    with open('../configs/synthetic_4x4_200m_stress_peak.json', 'w') as f:
        json.dump(config_peak, f, indent=2)
        
    print("Generated config files.")

if __name__ == "__main__":
    main()
