import glob
import json
import os
import re

configs = glob.glob("configs/config_*.json")

# Gestiamo i file flow già processati per non rinominarli o sovrascriverli male
processed_flows = set()

for config_path in configs:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
        
    old_flow = cfg["flowFile"]
    roadnet = cfg["roadnetFile"]
    
    # Esempio roadnet: roadnet_4x4_200m.json -> estraiamo 4x4
    grid_match = re.search(r'roadnet_(\d+x\d+)_', roadnet)
    if not grid_match:
        continue
    grid_str = grid_match.group(1)
    
    # Esempio old_flow: flow_2k_flat.json
    # Vogliamo farlo diventare flow_4x4_2k_flat.json
    if f"_{grid_str}_" not in old_flow:
        new_flow = old_flow.replace("flow_", f"flow_{grid_str}_")
    else:
        new_flow = old_flow # già corretto
        
    if new_flow != old_flow:
        old_flow_path = os.path.join("data", old_flow)
        new_flow_path = os.path.join("data", new_flow)
        
        # Se esiste il file vecchio, lo rinominiamo (se non l'abbiamo già rinominato per un altro config)
        if os.path.exists(old_flow_path):
            os.rename(old_flow_path, new_flow_path)
            print(f"Rinominato: {old_flow} -> {new_flow}")
            
        cfg["flowFile"] = new_flow
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"Aggiornato config: {config_path}")
