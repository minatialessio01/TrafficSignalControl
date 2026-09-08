import os
import glob

replacements = {
    "flow_0.416_flat.json": "flow_1.3k_flat.json",
    "flow_0.416_peak.json": "flow_1.9k_peak.json",
    "flow_6k_flat.json": "flow_6.2k_flat.json",
    "flow_6k_double_peak.json": "flow_7.3k_peak.json",
    "flow_5x5_6k_flat.json": "flow_6.3k_flat.json",
    "synthetic_4x4_200m_config3.json": "config_4x4_200m_1.3k_flat.json",
    "synthetic_4x4_200m_config4.json": "config_4x4_200m_1.9k_peak.json",
    "synthetic_4x4_200m_stress_flat.json": "config_4x4_200m_6.2k_flat.json",
    "synthetic_4x4_200m_stress_peak.json": "config_4x4_200m_7.3k_peak.json",
    "synthetic_5x5_200m_config3.json": "config_5x5_200m_1.3k_flat.json",
    "synthetic_5x5_200m_config4.json": "config_5x5_200m_1.9k_peak.json",
    "synthetic_5x5_200m_stress_flat.json": "config_5x5_200m_6.3k_flat.json",
    "flow_10k_flat.json": "flow_10k_flat.json", # (was used in generate_stress, just mapping if it's there)
    "flow_10k_double_peak.json": "flow_10k_double_peak.json",
    # Aggiorna anche i nomi parziali
    "config1.json": "config_4x4_200m_1.3k_flat.json", # solo per gli esempi d'uso negli help
}

files_to_check = glob.glob("scripts/*.py") + glob.glob("*.md")

for file_path in files_to_check:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    for old_str, new_str in replacements.items():
        if old_str in content:
            content = content.replace(old_str, new_str)
            modified = True
            
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated {file_path}")
