import json, math, os

DATA = "/workspace/data"
CONFIGS = "/workspace/configs"
EPISODE_DURATION = 1800.0

CONFIG_FILES = [
    "config_4x4_100m_train1", "config_4x4_100m_train2", "config_4x4_100m_train3",
    "config_4x4_100m_6k_peak", "config_4x4_100m_6k_peak2", "config_4x4_100m_6k_peak3",
    "config_4x4_100m_6k_peak4", "config_4x4_100m_6k_peak5",
    "config_4x4_200m_6k_flat", "config_4x4_200m_6k_flat2", "config_4x4_200m_6k_flat3",
    "config_4x4_200m_6k_flat4", "config_4x4_200m_6k_flat5",
    "config_4x4_200m_6k_peak", "config_4x4_200m_6k_peak2", "config_4x4_200m_6k_peak3",
    "config_4x4_200m_6k_peak4", "config_4x4_200m_6k_peak5",
    "config_5x5_100m_9.4k_flat", "config_5x5_100m_9.4k_flat2", "config_5x5_100m_9.4k_flat3",
    "config_5x5_100m_9.4k_flat4", "config_5x5_100m_9.4k_flat5",
    "config_5x5_100m_9.4k_peak", "config_5x5_100m_9.4k_peak2", "config_5x5_100m_9.4k_peak3",
    "config_5x5_100m_9.4k_peak4", "config_5x5_100m_9.4k_peak5",
    "config_6x6_100m_11.5k_flat", "config_6x6_100m_11.5k_flat2", "config_6x6_100m_11.5k_flat3",
    "config_6x6_100m_11.5k_flat4", "config_6x6_100m_11.5k_flat5",
    "config_6x6_100m_11.5k_peak", "config_6x6_100m_11.5k_peak2", "config_6x6_100m_11.5k_peak3",
    "config_6x6_100m_11.5k_peak4", "config_6x6_100m_11.5k_peak5",
]

_roadnet_cache = {}


def load_roadnet(path):
    if path in _roadnet_cache:
        return _roadnet_cache[path]
    with open(path) as f:
        rn = json.load(f)
    road_time = {}
    for road in rn["roads"]:
        p0, p1 = road["points"][0], road["points"][-1]
        length = math.dist((p0["x"], p0["y"]), (p1["x"], p1["y"]))
        max_speed = road["lanes"][0]["maxSpeed"] if road["lanes"] else 11.11
        road_time[road["id"]] = length / max_speed
    _roadnet_cache[path] = road_time
    return road_time


def process_config(cfg_name):
    with open(f"{CONFIGS}/{cfg_name}.json") as f:
        cfg = json.load(f)
    root = "/workspace"
    roadnet_path = f"{root}/{cfg['dir']}{cfg['roadnetFile']}"
    flow_path = f"{root}/{cfg['dir']}{cfg['flowFile']}"

    road_time = load_roadnet(roadnet_path)
    with open(flow_path) as f:
        flows = json.load(f)

    total_vehicles = 0
    impossible = 0
    for entry in flows:
        route = entry["route"]
        ff_time = sum(road_time[r] for r in route)
        start, end, interval = entry["startTime"], entry["endTime"], entry["interval"]
        n = int(math.floor((end - start) / interval)) + 1
        total_vehicles += n
        for k in range(n):
            spawn = start + k * interval
            if spawn + ff_time > EPISODE_DURATION:
                impossible += 1
    return total_vehicles, impossible, roadnet_path, flow_path


if __name__ == "__main__":
    print(f"{'config':40s} {'total_vehicles':>15s} {'impossible':>10s} {'pct_impossible':>15s}")
    results = {}
    for cfg_name in CONFIG_FILES:
        try:
            total, impossible, rn_path, fl_path = process_config(cfg_name)
            pct = 100.0 * impossible / total if total else 0.0
            print(f"{cfg_name:40s} {total:15d} {impossible:10d} {pct:14.2f}%")
            results[cfg_name] = {"total_vehicles": total, "impossible": impossible}
        except Exception as e:
            print(f"{cfg_name:40s} ERROR: {e}")
            results[cfg_name] = {"error": str(e)}

    with open("/workspace/scratch_out/tc_adjustment.json", "w") as f:
        json.dump(results, f, indent=2)
