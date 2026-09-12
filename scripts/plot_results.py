"""
Generazione automatica di grafici e tabelle per i risultati dell'esperimento.
Crea i grafici richiesti per la tesi a partire dai log salvati in `results/`.
"""

import argparse
import glob
import json
import os
import sys
import pandas as pd
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Stile coerente con matplotlib/seaborn
    plt.style.use('seaborn-v0_8-darkgrid')
except ImportError:
    print("[ERRORE] matplotlib o seaborn non trovati. Installa: pip install matplotlib seaborn pandas")
    import sys
    sys.exit(1)


# Colori coerenti per modello
# Palette categorica validata (CVD-safe, ordine fisso): stesso ordine di MODEL_REGISTRY
# in scripts/main.py, cosi' un modello ha sempre lo stesso colore in ogni grafico.
MODEL_COLORS = {
    "metastgat_pro":             "#2a78d6",  # blue
    "metastgat_paper":           "#eb6834",  # orange
    "ablation_environment":      "#1baf7a",  # aqua
    "ablation_temporal":         "#eda100",  # yellow
    "ablation_rl_core":          "#e87ba4",  # magenta
    "ablation_replay_stability": "#008300",  # green
    "fixedtime":                 "#4a3aa7",  # violet
    "maxpressure":               "#e34948",  # red
}

# Modelli da mostrare sempre nei grafici comparativi primari (es. barre 4 modelli)
MAIN_MODELS = ["metastgat_pro", "metastgat_paper", "fixedtime", "maxpressure"]


def parse_args():
    parser = argparse.ArgumentParser(description="Plotting risultati Tesi")
    parser.add_argument("--results-dir", default="results",
                        help="Directory contenente i risultati (es. results/)")
    parser.add_argument("--plots-dir", default="results/plots",
                        help="Directory di output per grafici e tabelle")
    return parser.parse_args()


def load_training_logs(results_dir):
    """Restituisce dict {model_id: DataFrame} per il training_log.csv"""
    training_data = {}
    for model_id in os.listdir(results_dir):
        path = os.path.join(results_dir, model_id, "training_log.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            training_data[model_id] = df
    return training_data


def load_training_configs(results_dir):
    """Restituisce dict {model_id: training_config_dict} leggendo training_config.json."""
    configs = {}
    for model_id in os.listdir(results_dir):
        path = os.path.join(results_dir, model_id, "training_config.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                configs[model_id] = json.load(f)
    return configs


def load_test_summaries(results_dir):
    """Restituisce dict {config_basename: {model_id: summary_dict}}

    Dal riordino del 12/9/2026 (vedi results/REORGANIZATION.md) i
    test_summary_*.json vivono in <model_dir>/test_summaries/; si cercano li'
    e, per compatibilita' con una cartella modello testata di fresco (test.py
    scrive ancora piatto in model_dir/) o non ancora riordinata, anche
    direttamente in model_dir/.
    """
    summaries = {}
    for model_id in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_id)
        if not os.path.isdir(model_dir):
            continue
        pattern = "test_summary_*.json"
        summary_files = (glob.glob(os.path.join(model_dir, "test_summaries", pattern))
                          + glob.glob(os.path.join(model_dir, pattern)))
        for summary_file in summary_files:
            with open(summary_file, "r") as f:
                data = json.load(f)
            config_name = data["config_basename"]
            if config_name not in summaries:
                summaries[config_name] = {}
            summaries[config_name][model_id] = data
    return summaries


def load_test_episodes(results_dir):
    """Restituisce dict {model_id: {config_basename: DataFrame}}

    Dal riordino del 12/9/2026 (vedi results/REORGANIZATION.md) i
    test_<config>.csv vivono in <model_dir>/test_csv/; si cercano li' e,
    per compatibilita' con una cartella non ancora riordinata, anche
    direttamente in model_dir/.
    """
    test_data = {}
    for model_id in os.listdir(results_dir):
        model_dir = os.path.join(results_dir, model_id)
        if not os.path.isdir(model_dir):
            continue
        test_data[model_id] = {}
        test_files = (glob.glob(os.path.join(model_dir, "test_csv", "test_*.csv"))
                      + glob.glob(os.path.join(model_dir, "test_*.csv")))
        for test_file in test_files:
            # Salta se è un log generico non test_<config>
            if "training" in test_file:
                continue
            config_name = os.path.basename(test_file).replace("test_", "").replace(".csv", "")
            df = pd.read_csv(test_file)
            test_data[model_id][config_name] = df
    return test_data


def plot_training_curves(training_logs, plots_dir, training_configs=None, window=5):
    """Plotta Loss e Travel Time in addestramento con media mobile."""

    os.makedirs(plots_dir, exist_ok=True)
    training_configs = training_configs or {}

    for model_id, df in training_logs.items():
        if len(df) == 0:
            continue

        color = MODEL_COLORS.get(model_id, "tab:blue")

        # Loss Plot
        plt.figure(figsize=(10, 5))
        # Se 'avg_loss' è 0 ovunque (es. primi episodi di warm-up, buffer non ancora pieno)
        plt.plot(df['episode'], df['avg_loss'], alpha=0.3, color=color, label="Loss grezza")
        plt.plot(df['episode'], df['avg_loss'].rolling(window=window, min_periods=1).mean(),
                 color=color, linewidth=2, label=f"Media mobile (w={window})")
        plt.title(f"Andamento Loss in Training - {model_id}")
        plt.xlabel("Episodio")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"loss_{model_id}.png"), dpi=150)
        plt.close()

        # Travel Time Plot
        plt.figure(figsize=(10, 5))
        plt.plot(df['episode'], df['travel_time'], alpha=0.3, color=color, label="TT grezzo")
        plt.plot(df['episode'], df['travel_time'].rolling(window=window, min_periods=1).mean(),
                 color=color, linewidth=2, label=f"Media mobile (w={window})")
        plt.title(f"Andamento Travel Time in Training - {model_id}")
        plt.xlabel("Episodio")
        plt.ylabel("Travel Time (s)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"travel_time_training_{model_id}.png"), dpi=150)
        plt.close()


def plot_phase_percentages(test_episodes, plots_dir):
    """
    Plotta la distribuzione media delle fasi in test per ogni modello
    (media tra tutte le configurazioni).
    """
    os.makedirs(plots_dir, exist_ok=True)
    
    for model_id, configs_dict in test_episodes.items():
        if not configs_dict:
            continue
            
        # Concatena tutti i df dei test per fare una media unica del modello
        all_dfs = []
        for cfg, df in configs_dict.items():
            all_dfs.append(df)
        if not all_dfs:
            continue
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Cerca le colonne phase_pct_*
        phase_cols = [c for c in combined_df.columns if c.startswith("phase_pct_")]
        if not phase_cols:
            continue
            
        avg_phases = combined_df[phase_cols].mean()
        
        # Estrai gli indici delle fasi
        phases = [int(c.split("_")[-1]) for c in phase_cols]
        values = avg_phases.values
        
        plt.figure(figsize=(10, 5))
        color = MODEL_COLORS.get(model_id, "steelblue")
        bars = plt.bar(phases, values, color=color, edgecolor="black")
        
        plt.title(f"Distribuzione media fasi - {model_id}")
        plt.xlabel("Fase")
        plt.ylabel("Percentuale di Scelta (%)")
        plt.xticks(phases)
        
        # Aggiungi etichette percentuali
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha='center', va='bottom', fontsize=9)
                     
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"phase_pct_{model_id}.png"), dpi=150)
        plt.close()


def plot_test_bars(test_summaries, plots_dir):
    """
    Genera barplot affiancati (Travel Time e Throughput) per ogni config di test.
    Confronta SOLO i modelli principali (MAIN_MODELS): Pro, Paper, FixedTime,
    MaxPressure. Le ablation non compaiono qui per non sovraffollare il grafico
    (restano visibili nelle tabelle generate da generate_tables()).
    """
    os.makedirs(plots_dir, exist_ok=True)

    for config_name, models_data in test_summaries.items():
        models = [m for m in MAIN_MODELS if m in models_data]
        if not models:
            continue

        tts = [models_data[m]["avg_travel_time"] for m in models]
        tps = [models_data[m]["avg_throughput"] for m in models]
        colors = [MODEL_COLORS.get(m, "gray") for m in models]
        # Veicoli totali spawnati dalla config (uguale per tutti i modelli, prendi il primo disponibile)
        total_vehicles = next(
            (models_data[m]["total_vehicles"] for m in models if "total_vehicles" in models_data[m]),
            None
        )
        
        # --- TRAVEL TIME ---
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, tts, color=colors, edgecolor="black")
        plt.title(f"Average Travel Time (Lower is Better) - {config_name}")
        plt.ylabel("Travel Time (s)")
        plt.xticks(rotation=45, ha="right")
        
        # Etichette
        for bar, val in zip(bars, tts):
            plt.text(bar.get_x() + bar.get_width()/2, val + max(tts)*0.01,
                     f"{val:.1f}", ha='center', va='bottom', fontweight='bold')
                     
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"bars_tt_{config_name}.png"), dpi=150)
        plt.close()
        
        # --- THROUGHPUT ---
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, tps, color=colors, edgecolor="black")
        plt.title(f"Average Throughput (Higher is Better) - {config_name}")
        plt.ylabel("Throughput (veicoli)")
        plt.xticks(rotation=45, ha="right")

        if total_vehicles is not None:
            ax = plt.gca()
            plt.axhline(total_vehicles, color="black", linestyle="--", linewidth=1.2)
            # Testo ancorato sopra la linea (non sopra, altrimenti il tratteggio la attraversa)
            ax.text(0.99, total_vehicles, f" Veicoli totali spawnati ({total_vehicles}) ",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=9,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))

        # Margine superiore esplicito: la linea dei veicoli totali (se presente) e' il
        # tetto concettuale del grafico, quindi il limite dell'asse si calcola su di lei
        # (non sull'altezza delle barre) cosi' resta sempre spazio per la sua etichetta
        # e le barre/etichette non sfiorano mai il titolo.
        top_reference = total_vehicles if total_vehicles is not None else max(tps)
        label_gap = top_reference * 0.02

        # Etichette: normalmente sopra la barra; se una barra e' cosi' vicina alla
        # linea dei veicoli totali che l'etichetta la toccherebbe, la mettiamo invece
        # dentro la barra (testo bianco) per non sovrapporla mai alla linea.
        for bar, val in zip(bars, tps):
            if total_vehicles is not None and val + 2 * label_gap > total_vehicles:
                label_y, va, color = val - label_gap, 'top', 'white'
            else:
                label_y, va, color = val + label_gap, 'bottom', 'black'
            plt.text(bar.get_x() + bar.get_width()/2, label_y, f"{int(val)}",
                     ha='center', va=va, fontweight='bold', color=color)

        plt.ylim(0, top_reference * 1.12)
                     
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"bars_tp_{config_name}.png"), dpi=150)
        plt.close()


def _bold_best_latex(df, fmt, best="min"):
    """
    Restituisce il LaTeX di df con il valore migliore di ogni colonna (min o max,
    ignorando i NaN) evidenziato in \\textbf{}. df deve contenere valori numerici
    (o NaN); il rendering testuale usa `fmt` (es. "%.2f").
    """
    str_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)
    for col in df.columns:
        series = df[col]
        best_val = series.min(skipna=True) if best == "min" else series.max(skipna=True)
        for idx in df.index:
            val = series.loc[idx]
            if pd.isna(val):
                str_df.at[idx, col] = "-"
            else:
                text = fmt % val
                if pd.notna(best_val) and val == best_val:
                    text = f"\\textbf{{{text}}}"
                str_df.at[idx, col] = text
    return str_df.to_latex(escape=False)


def generate_tables(test_summaries, plots_dir):
    """
    Genera file CSV e LaTeX con i risultati tabulari (TT e Throughput).
    Righe = Modelli, Colonne = Config di test.

    - Travel Time: valore assoluto in secondi (piu' basso e' meglio).
    - Throughput: percentuale di veicoli completati rispetto al totale spawnato
      dalla config (piu' alto e' meglio) — richiede "total_vehicles" nel summary
      (scritto da test.py); se assente per un run piu' vecchio, la cella resta NaN.

    Nei .tex il valore migliore di ogni colonna (config) e' in grassetto. I .csv
    restano numeri puri (senza markup), per un eventuale riuso dei dati.
    """
    os.makedirs(plots_dir, exist_ok=True)

    if not test_summaries:
        return

    configs = sorted(list(test_summaries.keys()))

    # Raccogli tutti i modelli visti
    all_models = set()
    for cfg in configs:
        all_models.update(test_summaries[cfg].keys())

    all_models = sorted(list(all_models))

    def sort_key(m):
        if m == "metastgat_pro": return 0
        if m == "metastgat_paper": return 1
        if m.startswith("ablation_"): return 2
        if m == "fixedtime": return 8
        if m == "maxpressure": return 9
        return 5
    all_models.sort(key=sort_key)

    # Crea i dataframe
    df_tt = pd.DataFrame(index=all_models, columns=configs, dtype=float)
    df_tp = pd.DataFrame(index=all_models, columns=configs, dtype=float)

    for cfg in configs:
        for m in all_models:
            if m in test_summaries[cfg]:
                summary = test_summaries[cfg][m]
                df_tt.at[m, cfg] = summary["avg_travel_time"]
                total_vehicles = summary.get("total_vehicles")
                if total_vehicles:
                    df_tp.at[m, cfg] = summary["avg_throughput"] / total_vehicles * 100.0
                else:
                    df_tp.at[m, cfg] = np.nan
            else:
                df_tt.at[m, cfg] = np.nan
                df_tp.at[m, cfg] = np.nan

    # Salva CSV (numeri puri, senza grassetto: TT in secondi, Throughput in % veicoli completati)
    df_tt.to_csv(os.path.join(plots_dir, "table_travel_time.csv"))
    df_tp.to_csv(os.path.join(plots_dir, "table_throughput.csv"))

    # Salva LaTeX con il migliore di ogni colonna in grassetto
    with open(os.path.join(plots_dir, "table_travel_time.tex"), "w") as f:
        f.write(_bold_best_latex(df_tt, "%.2f", best="min"))

    with open(os.path.join(plots_dir, "table_throughput.tex"), "w") as f:
        f.write(_bold_best_latex(df_tp, "%.1f%%", best="max"))

    print(f"[Plotting] Tabelle salvate in {plots_dir}/table_*.csv e .tex (migliore per colonna in grassetto nel .tex)")


def main():
    args = parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"[ERRORE] Directory dei risultati '{args.results_dir}' non trovata.")
        sys.exit(1)
        
    print(f"Analisi dei risultati in: {args.results_dir}")
    
    training_logs = load_training_logs(args.results_dir)
    training_configs = load_training_configs(args.results_dir)
    test_summaries = load_test_summaries(args.results_dir)
    test_episodes = load_test_episodes(args.results_dir)

    if training_logs:
        plot_training_curves(training_logs, args.plots_dir, training_configs=training_configs)
        print(f"[Plotting] Generate curve di training (Loss e TT)")
        
    if test_episodes:
        plot_phase_percentages(test_episodes, args.plots_dir)
        print(f"[Plotting] Generati grafici percentuali fasi")
        
    if test_summaries:
        plot_test_bars(test_summaries, args.plots_dir)
        print(f"[Plotting] Generati barplot comparativi di test")
        generate_tables(test_summaries, args.plots_dir)
        
    print(f"[Plotting] Tutti i grafici sono stati salvati in {args.plots_dir}")


if __name__ == "__main__":
    main()
