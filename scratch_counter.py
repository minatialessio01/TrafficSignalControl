import json
import os
import glob
import math

data_dir = r"c:\Users\user\Documents\Antigravity\CodiceTesi\data"

print("--- ACTUAL FLOW VEHICLE COUNTS ---")
for f in glob.glob(os.path.join(data_dir, "flow_*.json")):
    try:
        with open(f, 'r') as file:
            data = json.load(file)
            total_vehicles = 0
            for item in data:
                start = item.get("startTime", 0)
                end = item.get("endTime", 1800)
                interval = item.get("interval", 1)
                if interval > 0:
                    total_vehicles += math.ceil((end - start) / interval)
            
            # Formattazione in k
            k_val = total_vehicles / 1000.0
            print(f"{os.path.basename(f)}: ~{k_val:.1f}k ({total_vehicles} vehicles)")
    except Exception as e:
        print(f"Error reading {f}: {e}")
