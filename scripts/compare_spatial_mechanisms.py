"""
Confronto tra meccanismi spaziali: GAT vs GCN vs SONAR (seconda parte della
tesi, vedi `istruzioni seconda parte.md` e `descrizione_gcn_sonar.md`).

Registro SEPARATO da `MODEL_REGISTRY` di `main.py` (decisione D5 di
`istruzioni seconda parte.md`): questo confronto risponde a una domanda di
ricerca diversa ("quale meccanismo spaziale e' migliore, a parita' di resto
dell'architettura") da quella dello studio di ablation ("quali componenti del
framework Pro contano"), e tutti i modelli qui usano lo stesso preset
`pro` (nessuna combinazione con gli altri preset di ablation, per evitare
un'esplosione combinatoria di 20+ modelli poco leggibile in tesi).

Uso:
    # Un solo modello (debug)
    python scripts/compare_spatial_mechanisms.py --model metastgat_2l

    # Tutti i 5 modelli della tabella principale (raccomandazione D5)
    python scripts/compare_spatial_mechanisms.py --models all

    # Includi anche il secondo valore di L per SONAR (6 modelli totali)
    python scripts/compare_spatial_mechanisms.py --models all --include-sonar-l4
"""

import argparse
import os
import sys
from datetime import datetime

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _ROOT_DIR)
sys.path.insert(0, _SCRIPTS_DIR)

import train as train_mod
import test as test_mod
from main import (
    DEFAULT_TRAIN_CONFIGS, DEFAULT_VALIDATION_CONFIG, DEFAULT_TEST_CONFIGS,
    _run_with_argv,
)


# Tabella principale a inizio `istruzioni seconda parte.md`. MetaSTGAT-1L
# (Pro) non e' ripetuto qui: e' gia' `results/metastgat_pro_0.5`, allenato
# nella prima parte del progetto -- il confronto va fatto contro quel
# checkpoint, non riallenandolo una seconda volta.
SPATIAL_MODEL_REGISTRY = {
    "metastgat_2l": {
        "model": "MetaSTGAT",
        "extra_args": ["--num-layers", "2"],
    },
    "metastgnn_1l": {
        "model": "MetaSTGNN",
        "extra_args": ["--num-layers", "1"],
    },
    "metastgnn_2l": {
        "model": "MetaSTGNN",
        "extra_args": ["--num-layers", "2"],
    },
    "metastsonar_l2": {
        "model": "MetaSTSONAR",
        "extra_args": ["--sonar-recurrences", "2"],
    },
    # Incluso solo con --include-sonar-l4 (D1: secondo valore di L
    # raccomandato se si vuole mostrare il punto di forza di SONAR --
    # propagazione piu' lontana a costo di parametri costante).
    "metastsonar_l4": {
        "model": "MetaSTSONAR",
        "extra_args": ["--sonar-recurrences", "4"],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Confronto GAT vs GCN vs SONAR, a parita' di ablation (preset 'pro')",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=None, choices=list(SPATIAL_MODEL_REGISTRY.keys()))
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--include-sonar-l4", action="store_true",
                        help="Include anche metastsonar_l4 nel run con --models all "
                             "(6 modelli invece di 5, D1/D5).")
    parser.add_argument("--train-configs", nargs="+", default=DEFAULT_TRAIN_CONFIGS)
    parser.add_argument("--validation-config", default=DEFAULT_VALIDATION_CONFIG)
    parser.add_argument("--test-configs", nargs="+", default=DEFAULT_TEST_CONFIGS)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=1)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _resolve_model_ids(args):
    if args.model:
        return [args.model]
    if args.models == ["all"]:
        ids = ["metastgnn_1l", "metastgat_2l", "metastgnn_2l", "metastsonar_l2"]
        if args.include_sonar_l4:
            ids.append("metastsonar_l4")
        return ids
    return args.models


def train_one(model_id, info, args, output_dir):
    argv = [
        "train.py",
        "--config", *args.train_configs,
        "--model", info["model"],
        "--episodes", str(args.episodes),
        "--output-dir", output_dir,
        "--select-best-config", args.validation_config,
        "--ablation", "pro",
        "--seed", str(args.seed),
        *info["extra_args"],
    ]
    t_args = _run_with_argv(train_mod.parse_args, argv)
    train_mod.run_training(t_args)


def test_one(model_id, info, args, config, output_dir):
    argv = [
        "test.py",
        "--config", config,
        "--model", info["model"],
        "--model-id", model_id,
        "--output-dir", output_dir,
        "--n-eval", str(args.test_episodes),
        "--ablation", "pro",
        *info["extra_args"],
    ]
    e_args = _run_with_argv(test_mod.parse_args, argv)
    return test_mod.evaluate(e_args)


def main():
    args = parse_args()
    model_ids = _resolve_model_ids(args)
    all_configs = list(args.train_configs) + list(args.test_configs)

    print(f"[Confronto meccanismi spaziali] Modelli: {model_ids}")
    print(f"[Confronto meccanismi spaziali] Config (train+test, {len(all_configs)} totali): "
          f"{[os.path.basename(c) for c in all_configs]}")

    summary = {"timestamp": datetime.now().isoformat(), "models": model_ids, "status": {}}

    for model_id in model_ids:
        info = SPATIAL_MODEL_REGISTRY[model_id]
        output_dir = os.path.join(args.results_dir, model_id)
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*80}\n=== {model_id} ({info['model']} {' '.join(info['extra_args'])}) ===\n{'='*80}")
        try:
            print(f"\n--- Training ---")
            train_one(model_id, info, args, output_dir)

            print(f"\n--- Test (su tutte le {len(all_configs)} config, train+test) ---")
            for config in all_configs:
                test_one(model_id, info, args, config, output_dir)

            summary["status"][model_id] = "ok"
        except Exception as e:
            print(f"[ERRORE] {model_id}: {e}")
            summary["status"][model_id] = f"errore: {e}"

    print(f"\n[Confronto meccanismi spaziali] Riepilogo: {summary['status']}")
    print("Usa scripts/compare_models.py --model-ids <...> per i grafici comparativi "
          "(stessa convenzione della prima parte del progetto).")


if __name__ == "__main__":
    main()
