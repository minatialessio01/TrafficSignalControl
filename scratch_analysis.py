import cityflow
import json
import glob
import os
import random

def main():
    configs = glob.glob("configs/*.json")
    base_out_dir = "analisi_configurazioni"
    os.makedirs(base_out_dir, exist_ok=True)
    
    # Rimuoviamo eventuali file temporanei vecchi
    configs = [c for c in configs if "temp_analysis" not in c]

    for config_path in configs:
        config_name = os.path.basename(config_path).replace(".json", "")
        
        out_dir = os.path.join(base_out_dir, config_name)
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. Creiamo un config temporaneo per forzare l'output nella cartella giusta
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        cfg["saveReplay"] = True
        # Poiché il dir base è "data/", per risalire e andare in analisi_configurazioni usiamo "../"
        cfg["roadnetLogFile"] = f"../{out_dir}/roadnet.json".replace("\\", "/")
        cfg["replayLogFile"] = f"../{out_dir}/replay.txt".replace("\\", "/")
        
        temp_cfg = "configs/temp_analysis.json"
        with open(temp_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
            
        # 2. Otteniamo la lista degli incroci dal roadnet per poter mandare azioni
        roadnet_file = os.path.join(cfg.get("dir", "data/"), cfg["roadnetFile"])
        with open(roadnet_file, "r", encoding="utf-8") as f:
            rn = json.load(f)
            
        intersections = [inter["id"] for inter in rn["intersections"] if not inter["virtual"]]
        
        # 3. Avviamo CityFlow
        print(f"Avvio simulazione random per: {config_name}")
        try:
            eng = cityflow.Engine(temp_cfg, thread_num=1)
            
            max_steps = cfg.get("maxStep", 1800)
            
            for step in range(max_steps):
                # Cambiamo fase semaforica ogni 10 secondi in modo random
                if step % 10 == 0:
                    for inter in intersections:
                        # Assumiamo 8 fasi come generato dallo script (0 to 7)
                        random_phase = random.randint(0, 7)
                        eng.set_tl_phase(inter, random_phase)
                
                eng.next_step()
                
            print(f"  -> Completato. Replay salvato in {out_dir}\n")
            
            # Necessario per forzare il flush del file replay
            del eng 
            
        except Exception as e:
            print(f"  -> ERRORE durante la simulazione di {config_name}: {e}\n")

    if os.path.exists(temp_cfg):
        os.remove(temp_cfg)
        
    print("Analisi batch completata!")

if __name__ == "__main__":
    main()
