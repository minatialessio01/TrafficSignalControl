"""
Punto di ingresso unico della pipeline sperimentale della tesi.

Esegue in-process (nessun subprocess: si puo' mettere un breakpoint e debuggare
normalmente) l'allenamento (`train.py`) e la valutazione (`test.py`) di:
  - UN singolo modello, comodo per il debug            → --model
  - PIU' modelli in sequenza (incl. "all")              → --models
e alla fine chiama `plot_results.py` per generare grafici e tabelle di confronto
(solo per un run multi-modello: con un singolo --model il confronto non avrebbe
comunque nulla con cui confrontarsi).

Ogni modello allenabile viene allenato su UNA sola configurazione (config_train):
a fine training, `train.py` sceglie da solo tra l'ultimo modello e il best
model (valutati su una config di validazione) e salva il vincitore come
`selected_model.pth`, che `test.py` usa automaticamente per la fase di
generalizzazione sulle config di test.

Uso:
  # Debug di un solo modello (utile per mettere breakpoint in train.py/test.py)
  python scripts/main.py --model metastgat_pro

  # Tutta la pipeline in un click: 6 modelli allenabili + FixedTime + MaxPressure,
  # allenati sulla config di training, validati e testati sulle config di
  # generalizzazione, grafici e tabelle finali in results/plots/
  python scripts/main.py --models all

  # Un sottoinsieme di modelli
  python scripts/main.py --models metastgat_pro fixedtime maxpressure
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime

import torch

# Root del progetto e cartella scripts/ nel path (per "import train"/"import test"
# e per gli import interni di src/ che train.py/test.py fanno a loro volta)
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SCRIPTS_DIR)

import train as train_mod
import test as test_mod


MODEL_REGISTRY = {
    "metastgat_pro": {
        "model": "MetaSTGAT",
        "ablation": "pro",
        "trainable": True
    },
    "metastgat_paper": {
        "model": "MetaSTGAT",
        "ablation": "paper",
        "trainable": True
    },
    "ablation_environment": {
        "model": "MetaSTGAT",
        "ablation": "environment",
        "trainable": True
    },
    "ablation_temporal": {
        "model": "MetaSTGAT",
        "ablation": "temporal",
        "trainable": True
    },
    "ablation_rl_core": {
        "model": "MetaSTGAT",
        "ablation": "rl_core",
        "trainable": True
    },
    "ablation_replay_stability": {
        "model": "MetaSTGAT",
        "ablation": "replay_stability",
        "trainable": True
    },
    "fixedtime": {
        "model": "FixedTime",
        "ablation": None,
        "trainable": False
    },
    "maxpressure": {
        "model": "MaxPressure",
        "ablation": None,
        "trainable": False
    },
}

# Suddivisione train / validation / test (vedi descrizione_configurazioni.md):
#   - train:      l'unica config su cui si allena (flusso "giornata lavorativa")
#   - validation: usata da train.py (--select-best-config) per scegliere tra
#                 final_model.pth e best_model.pt a fine training
#   - test:       tutte le altre config, usate solo per la valutazione di
#                 generalizzazione (mai viste in training/selezione)
DEFAULT_TRAIN_CONFIGS = [
    "configs/config_4x4_100m_train1.json",
    "configs/config_4x4_100m_train2.json",
    "configs/config_4x4_100m_train3.json",
]
DEFAULT_VALIDATION_CONFIG = "configs/config_4x4_100m_6k_flat.json"
DEFAULT_TEST_CONFIGS = [
    "configs/config_4x4_100m_6k_peak.json",
    "configs/config_4x4_200m_6k_flat.json",
    "configs/config_4x4_200m_6k_peak.json",
    "configs/config_5x5_100m_9.4k_flat.json",
    "configs/config_5x5_100m_9.4k_peak.json",
    "configs/config_6x6_100m_11.5k_flat.json",
    "configs/config_6x6_100m_11.5k_peak.json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline sperimentale Tesi — punto di ingresso unico (train + test + plot)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None, choices=list(MODEL_REGISTRY.keys()),
                        help="Esegui UN solo modello (comodo per il debug). "
                             "Se specificato, ha precedenza su --models.")
    parser.add_argument("--models", nargs="+", default=["all"],
                        help="Modelli da eseguire in sequenza. 'all' oppure "
                             "una lista, es. 'metastgat_pro fixedtime maxpressure'.")
    parser.add_argument("--train-configs", nargs="+", default=DEFAULT_TRAIN_CONFIGS,
                        help="Config CityFlow di training. Se piu' di una, train.py cicla "
                             "una variante per episodio (stesso roadnet/densita', seed diverso)")
    parser.add_argument("--validation-config", default=DEFAULT_VALIDATION_CONFIG,
                        help="Config CityFlow di validazione (train.py --select-best-config): "
                             "usata a fine training per scegliere tra final_model.pth e best_model.pt")
    parser.add_argument("--test-configs", nargs="+", default=DEFAULT_TEST_CONFIGS,
                        help="Config CityFlow per il test di generalizzazione")
    parser.add_argument("--episodes-per-config", type=int, default=50,
                        help="Numero di episodi di training (episodi da 3600s: "
                             "50x3600s = stesso tempo simulato totale dei precedenti 100x1800s)")
    parser.add_argument("--test-episodes", type=int, default=1,
                        help="Numero di episodi di valutazione per singola config di test. "
                             "Default 1: a epsilon=0, con flow e seed CityFlow fissi e "
                             "laneChange disattivato, la simulazione e' completamente "
                             "deterministica — ripetere l'episodio non introduce varianza "
                             "da mediare, produce solo lo stesso identico risultato N volte. "
                             "Alza il valore solo se in futuro si introduce una vera fonte di "
                             "variabilita' (es. piu' realizzazioni di flow con seed diversi).")
    parser.add_argument("--results-dir", default="results",
                        help="Directory principale dove salvare tutti gli output")
    parser.add_argument("--plot", action="store_true",
                        help="Forza la generazione dei grafici comparativi anche con --model singolo")
    parser.add_argument("--no-plot", action="store_true",
                        help="Salta la generazione dei grafici comparativi anche con --models multipli")
    return parser.parse_args()


def _run_with_argv(fn, argv):
    """Esegue fn() (una parse_args()-like) simulando sys.argv, poi lo ripristina."""
    old_argv = sys.argv
    sys.argv = argv
    try:
        return fn()
    finally:
        sys.argv = old_argv


def train_one_model(model_id, info, train_configs, episodes_per_config, output_dir, validation_config):
    """Allena un modello (in-process) usando train.py. Se train_configs contiene piu'
    di una config, train.py cicla una variante per episodio. Restituisce True/False."""
    argv = [
        "train.py",
        "--config", *train_configs,
        "--model", info["model"],
        "--episodes", str(episodes_per_config),
        "--output-dir", output_dir,
        "--select-best-config", validation_config,
    ]
    if info["ablation"]:
        argv += ["--ablation", info["ablation"]]

    args = _run_with_argv(train_mod.parse_args, argv)
    train_mod.run_training(args)
    return True


def test_one_model(model_id, info, config, output_dir, n_eval):
    """Valuta un modello (in-process) su una config, usando test.py. Restituisce
    il summary dict prodotto da evaluate()."""
    argv = [
        "test.py",
        "--config", config,
        "--model", info["model"],
        "--model-id", model_id,
        "--output-dir", output_dir,
        "--n-eval", str(n_eval),
    ]
    if info["ablation"]:
        argv += ["--ablation", info["ablation"]]

    args = _run_with_argv(test_mod.parse_args, argv)
    return test_mod.evaluate(args)


def run_pipeline(model_ids, args):
    """Allena (se addestrabile) e testa ogni modello in model_ids. Ritorna
    l'experiment_summary dict, nello stesso formato che produceva run_experiment.py."""
    os.makedirs(args.results_dir, exist_ok=True)

    experiment_summary = {
        "timestamp": datetime.now().isoformat(),
        "train_configs": args.train_configs,
        "validation_config": args.validation_config,
        "test_configs": args.test_configs,
        "episodes_per_config": args.episodes_per_config,
        "test_episodes": args.test_episodes,
        "models": model_ids,
        "status": {},
        "checkpoints": {},
    }

    start_time = time.time()

    for model_id in model_ids:
        print(f"\n{'='*80}")
        print(f"=== INIZIO PIPELINE PER MODELLO: {model_id} ===")
        print(f"{'='*80}")

        info = MODEL_REGISTRY[model_id]
        output_dir = os.path.join(args.results_dir, model_id)
        os.makedirs(output_dir, exist_ok=True)

        model_status = "ok"

        # --- TRAINING ---
        if info["trainable"]:
            print(f"\n--- 1. TRAINING ({model_id}) ---")
            try:
                train_one_model(model_id, info, args.train_configs,
                                 args.episodes_per_config, output_dir,
                                 args.validation_config)
                experiment_summary["checkpoints"][model_id] = os.path.join(output_dir, "selected_model.pth")
            except Exception:
                traceback.print_exc()
                model_status = "error_train"
                experiment_summary["status"][model_id] = model_status
                continue  # Salta al prossimo modello
        else:
            print(f"\n--- 1. TRAINING ({model_id}) --- SALTATO (non addestrabile)")

        # --- TEST ---
        # Testiamo sia sulle config di training (in-distribution: quanto va bene
        # sui dati che ha visto/su cui e' calibrato un baseline) sia su quelle di
        # generalizzazione -- prassi introdotta il 13/9/2026 perche' compare_models.py
        # confronta i modelli su entrambe le categorie, non solo sulla generalizzazione.
        print(f"\n--- 2. TEST ({model_id}) ---")
        all_test_configs = args.train_configs + args.test_configs
        for config in all_test_configs:
            print(f"\n>>> Test su config: {config}")
            try:
                test_one_model(model_id, info, config, output_dir, args.test_episodes)
            except Exception:
                traceback.print_exc()
                model_status = "error_test"

        experiment_summary["status"][model_id] = model_status
        print(f"\n=== FINE PIPELINE PER MODELLO: {model_id} (Stato: {model_status}) ===")

        # Libera la memoria GPU tra un modello e l'altro: essendo tutto in-process
        # (a differenza dei subprocess separati di prima), senza questo passaggio
        # la memoria CUDA di piu' modelli allenati in sequenza potrebbe accumularsi.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = os.path.join(args.results_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(experiment_summary, f, indent=4)

    print(f"\n{'='*80}")
    print(f"ESPERIMENTI COMPLETATI in {time.time() - start_time:.1f} s")
    print(f"Summary salvato in: {summary_path}")
    print(f"{'='*80}")

    return experiment_summary


def generate_plots(results_dir):
    """Chiama plot_results.py in-process."""
    import plot_results as plot_mod
    argv = ["plot_results.py", "--results-dir", results_dir]
    plot_args = _run_with_argv(plot_mod.parse_args, argv)
    if not os.path.isdir(plot_args.results_dir):
        print(f"[AVVISO] '{plot_args.results_dir}' non trovata: grafici saltati.")
        return
    print("\n--- GENERAZIONE GRAFICI ---")
    training_logs = plot_mod.load_training_logs(plot_args.results_dir)
    training_configs = plot_mod.load_training_configs(plot_args.results_dir)
    test_summaries = plot_mod.load_test_summaries(plot_args.results_dir)
    test_episodes = plot_mod.load_test_episodes(plot_args.results_dir)

    if training_logs:
        plot_mod.plot_training_curves(training_logs, plot_args.plots_dir, training_configs=training_configs)
        print("[Plotting] Generate curve di training (Loss e TT)")
    if test_episodes:
        plot_mod.plot_phase_percentages(test_episodes, plot_args.plots_dir)
        print("[Plotting] Generati grafici percentuali fasi")
    if test_summaries:
        plot_mod.plot_test_bars(test_summaries, plot_args.plots_dir)
        plot_mod.generate_tables(test_summaries, plot_args.plots_dir)
        print("[Plotting] Generati barplot e tabelle comparative")
    print(f"[Plotting] Tutti i grafici sono stati salvati in {plot_args.plots_dir}")


def main():
    args = parse_args()

    if args.model:
        model_ids = [args.model]
        should_plot = args.plot  # con un solo modello, plotta solo se richiesto esplicitamente
    else:
        if "all" in args.models:
            model_ids = list(MODEL_REGISTRY.keys())
        else:
            model_ids = [m for m in args.models if m in MODEL_REGISTRY]
            invalid = [m for m in args.models if m not in MODEL_REGISTRY]
            if invalid:
                print(f"[ERRORE] Modelli non validi: {invalid}")
                print(f"Modelli disponibili: {list(MODEL_REGISTRY.keys())}")
                sys.exit(1)
        should_plot = not args.no_plot  # con piu' modelli, plotta di default

    run_pipeline(model_ids, args)

    if should_plot:
        generate_plots(args.results_dir)
    else:
        print(f"\n[Plotting] Saltato. Per generarli: python scripts/plot_results.py --results-dir {args.results_dir}")


if __name__ == "__main__":
    main()
