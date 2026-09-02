import argparse
import subprocess
import sys
import os
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="Script unificato per eseguire Training e subito dopo il Test/Valutazione.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Argomenti principali per l'orchestratore
    parser.add_argument("--config", type=str, required=True, 
                        help="Percorso al file config.json di CityFlow")
    parser.add_argument("--model", type=str, default="MetaSTGAT", 
                        choices=["MetaSTGAT", "STGAT", "FixedTime"],
                        help="Modello da usare")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Numero totale di episodi di training")
    parser.add_argument("--output", type=str, default="auto", 
                        help="Directory di output. Se 'auto', usa un timestamp.")
    
    # Raccoglie tutti gli altri argomenti opzionali (es. --lr, --batch-size) e li passa al train
    args, unknown = parser.parse_known_args()

    # Determiniamo la cartella di output
    output_dir = args.output
    if output_dir == "auto":
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("results", f"{args.model.lower()}_{ts}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  AVVIO PIPELINE: {args.model}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")

    # =========================================================================
    # 1. TRAINING
    # =========================================================================
    train_cmd = [
        sys.executable, "scripts/train.py",
        "--config", args.config,
        "--model", args.model,
        "--episodes", str(args.episodes),
        "--output", output_dir
    ]
    # Aggiungi eventuali altri flag (--lr, --batch-size, ecc)
    train_cmd.extend(unknown)

    print("\n>>> FASE 1: TRAINING <<<")
    print("Comando:", " ".join(train_cmd))
    
    res = subprocess.run(train_cmd)
    
    if res.returncode != 0:
        print("\n[Errore] Il training è fallito o è stato interrotto manualmente (es. Ctrl+C).")
        print("La pipeline di test non verrà eseguita.")
        sys.exit(res.returncode)

    # =========================================================================
    # 2. TEST & REPLAY
    # =========================================================================
    test_cmd = [
        sys.executable, "scripts/test.py",
        "--config", args.config,
        "--model", args.model,
        "--output", output_dir
    ]
    
    # Se il modello non è FixedTime (che non ha pesi), cerchiamo il miglior checkpoint
    if args.model != "FixedTime":
        best_ckpt = os.path.join(output_dir, "best_model.pt")
        if not os.path.exists(best_ckpt):
            print(f"\n[Errore] Checkpoint non trovato in {best_ckpt}.")
            print("Il training non ha salvato il modello (forse è crashato?). Impossibile avviare il test.")
            sys.exit(1)
        test_cmd.extend(["--checkpoint", best_ckpt])
        
    print("\n>>> FASE 2: TEST (con salvataggio Replay) <<<")
    print("Comando:", " ".join(test_cmd))
    
    res = subprocess.run(test_cmd)
    
    if res.returncode != 0:
        print("\n[Errore] Il test è fallito.")
        sys.exit(res.returncode)

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETATA CON SUCCESSO")
    print(f"  Tutti i log, il miglior modello e il replay sono in: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
