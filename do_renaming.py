import os
import glob
import json

data_dir = r"c:\Users\user\Documents\Antigravity\CodiceTesi\data"
configs_dir = r"c:\Users\user\Documents\Antigravity\CodiceTesi\configs"

flows_mapping = {
    "flow_0.416_flat.json": "flow_1.3k_flat.json",
    "flow_0.416_peak.json": "flow_1.9k_peak.json",
    "flow_6k_flat.json": "flow_6.2k_flat.json",
    "flow_6k_double_peak.json": "flow_7.3k_peak.json",
    "flow_5x5_6k_flat.json": "flow_6.3k_flat.json"
}

configs_mapping = {
    "synthetic_4x4_200m_config3.json": "config_4x4_200m_1.3k_flat.json",
    "synthetic_4x4_200m_config4.json": "config_4x4_200m_1.9k_peak.json",
    "synthetic_4x4_200m_stress_flat.json": "config_4x4_200m_6.2k_flat.json",
    "synthetic_4x4_200m_stress_peak.json": "config_4x4_200m_7.3k_peak.json",
    "synthetic_5x5_200m_config3.json": "config_5x5_200m_1.3k_flat.json",
    "synthetic_5x5_200m_config4.json": "config_5x5_200m_1.9k_peak.json",
    "synthetic_5x5_200m_stress_flat.json": "config_5x5_200m_6.3k_flat.json"
}

# 1. Rename flow files
print("Renaming flow files...")
for old_name, new_name in flows_mapping.items():
    old_path = os.path.join(data_dir, old_name)
    new_path = os.path.join(data_dir, new_name)
    if os.path.exists(old_path):
        os.rename(old_path, new_path)
        print(f"Renamed {old_name} -> {new_name}")

# 2. Rename config files and update internal contents
print("\nRenaming and updating config files...")
for old_name, new_name in configs_mapping.items():
    old_path = os.path.join(configs_dir, old_name)
    new_path = os.path.join(configs_dir, new_name)
    
    if os.path.exists(old_path):
        with open(old_path, 'r') as f:
            cfg = json.load(f)
        
        # Aggiorna il puntatore al flow file se esiste nel mapping
        old_flow = cfg.get("flowFile")
        if old_flow in flows_mapping:
            cfg["flowFile"] = flows_mapping[old_flow]
        
        # Salva col nuovo nome
        with open(new_path, 'w') as f:
            json.dump(cfg, f, indent=2)
            
        # Rimuovi il file vecchio
        os.remove(old_path)
        print(f"Renamed and updated {old_name} -> {new_name}")
