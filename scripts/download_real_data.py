"""
Download dei dataset reali (Hangzhou e Jinan) dal repository CoLight.

I dataset provengono da: https://github.com/wingsweihua/colight
Sono gli stessi dataset usati nella letteratura TSC e citati nell'articolo MetaSTGAT.

Formato: roadnet.json + flow.json (compatibile CityFlow)
"""

import os
import json
import argparse
import urllib.request
import urllib.error

# ─── URL dei dataset dal repository CoLight ────────────────────────────────────
# Repository: https://github.com/wingsweihua/colight
# I dataset sono presenti nella cartella data/
COLIGHT_BASE = "https://raw.githubusercontent.com/wingsweihua/colight/master"

DATASETS = {
    "hangzhou": {
        "intersections": 16,
        "description": "Gudang block, Hangzhou, China (4x4 grid, ~16 intersections)",
        "roadnet": f"{COLIGHT_BASE}/data/Hangzhou/roadnet.json",
        "flow": f"{COLIGHT_BASE}/data/Hangzhou/flow.json",
        "output_dir": "data/hangzhou",
        "config_file": "configs/hangzhou_config.json",
        "simulation_time": 3600
    },
    "jinan": {
        "intersections": 12,
        "description": "Dongfeng block, Jinan, China (3x4 grid, ~12 intersections)",
        "roadnet": f"{COLIGHT_BASE}/data/Jinan/roadnet.json",
        "flow": f"{COLIGHT_BASE}/data/Jinan/flow.json",
        "output_dir": "data/jinan",
        "config_file": "configs/jinan_config.json",
        "simulation_time": 3600
    }
}

# URL alternativi (LibSignal / altri repo pubblici)
ALTERNATIVE_URLS = {
    "hangzhou": {
        "roadnet": "https://raw.githubusercontent.com/DaRL-LibSignal/LibSignal/master/data/CityFlow/Hangzhou/roadnet.json",
        "flow": "https://raw.githubusercontent.com/DaRL-LibSignal/LibSignal/master/data/CityFlow/Hangzhou/flow.json"
    },
    "jinan": {
        "roadnet": "https://raw.githubusercontent.com/DaRL-LibSignal/LibSignal/master/data/CityFlow/Jinan/roadnet.json",
        "flow": "https://raw.githubusercontent.com/DaRL-LibSignal/LibSignal/master/data/CityFlow/Jinan/flow.json"
    }
}


def download_file(url: str, dest_path: str, timeout: int = 30) -> bool:
    """Scarica un file da URL. Restituisce True se ha successo."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        print(f"  Scaricando {os.path.basename(dest_path)} da {url[:80]}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
             open(dest_path, "wb") as f:
            f.write(resp.read())
        print(f"  [OK] Salvato in {dest_path}")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        print(f"  [WARN] Fallito: {e}")
        return False


def make_cityflow_config(roadnet_file: str, flow_file: str,
                         simulation_time: int) -> dict:
    """Genera il file config.json per CityFlow."""
    return {
        "interval": 1.0,
        "seed": 42,
        "dir": "./",
        "roadnetFile": roadnet_file,
        "flowFile": flow_file,
        "rlTrafficLight": True,
        "laneChange": False,
        "saveReplay": False,
        "roadnetLogFile": "roadnet.log",
        "replayLogFile": "replay.log"
    }


def download_dataset(name: str, info: dict) -> bool:
    """Scarica roadnet e flow per un dataset. Prova URL principale poi alternativi."""
    print(f"\n=== Dataset: {name.upper()} ===")
    print(f"  {info['description']}")

    os.makedirs(info["output_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(info["config_file"]), exist_ok=True)

    roadnet_dest = os.path.join(info["output_dir"], "roadnet.json")
    flow_dest = os.path.join(info["output_dir"], "flow.json")

    # Prova URL principale
    roadnet_ok = download_file(info["roadnet"], roadnet_dest)
    flow_ok = download_file(info["flow"], flow_dest)

    # Se fallisce, prova gli URL alternativi
    if not roadnet_ok and name in ALTERNATIVE_URLS:
        print(f"  Provo URL alternativo per roadnet...")
        roadnet_ok = download_file(ALTERNATIVE_URLS[name]["roadnet"], roadnet_dest)

    if not flow_ok and name in ALTERNATIVE_URLS:
        print(f"  Provo URL alternativo per flow...")
        flow_ok = download_file(ALTERNATIVE_URLS[name]["flow"], flow_dest)

    if roadnet_ok and flow_ok:
        # Genera la config CityFlow
        cfg = make_cityflow_config(
            roadnet_file=os.path.abspath(roadnet_dest),
            flow_file=os.path.abspath(flow_dest),
            simulation_time=info["simulation_time"]
        )
        with open(info["config_file"], "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"  [OK] Config salvata in {info['config_file']}")
        return True
    else:
        print(f"\n  [ERROR] Impossibile scaricare il dataset {name}.")
        print(f"  Download manuale: https://github.com/wingsweihua/colight/tree/master/data/{name.capitalize()}/")
        print(f"  Metti roadnet.json e flow.json in '{info['output_dir']}/'")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Scarica i dataset reali di Hangzhou e Jinan per MetaSTGAT"
    )
    parser.add_argument("--datasets", nargs="+", default=["hangzhou", "jinan"],
                        choices=["hangzhou", "jinan"],
                        help="Dataset da scaricare (default: entrambi)")
    args = parser.parse_args()

    print("=== Download dataset reali per MetaSTGAT ===\n")
    print("Fonte principale: CoLight repo (wingsweihua/colight)")
    print("Fonte alternativa: LibSignal (DaRL-LibSignal/LibSignal)")

    success = {}
    for name in args.datasets:
        if name in DATASETS:
            success[name] = download_dataset(name, DATASETS[name])

    print("\n=== Riepilogo ===")
    for name, ok in success.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {name}")

    if all(success.values()):
        print("\nTutti i dataset scaricati con successo!")
        print("Puoi ora eseguire il training con i dataset reali:")
        print("  python scripts/train.py --config configs/hangzhou_config.json --model MetaSTGAT")
    else:
        print("\nAlcuni dataset non sono stati scaricati automaticamente.")
        print("Vedi le istruzioni sopra per il download manuale.")


if __name__ == "__main__":
    main()
