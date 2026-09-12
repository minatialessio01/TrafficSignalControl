"""
Confronto multi-modello, un grafico per configurazione (sia di training che di test).

Sostituisce compare_pro_vs_maxpressure.py: quello aggregava le 7 config di test in
un unico boxplot per metrica, che comprimeva l'informazione config-per-config in una
distribuzione poco leggibile con solo 7 punti. Qui invece ogni configurazione (le 3
varianti di training + le 7 di test) ha il proprio grafico, con un confronto diretto
e immediato tra i modelli su quella singola configurazione.

Per ogni config, due pannelli affiancati (stessa unita' di misura, i secondi, quindi
nessun problema di scale diverse su uno stesso asse):
  - Travel Time: TT medio e TT massimo, una barra per modello in ciascun gruppo.
  - Attesa massima direzionale: N/S e W/E, una barra per modello in ciascun gruppo.

Vedi descrizione_metriche.md per la definizione precisa di ogni metrica.

Uso:
    python scripts/compare_models.py
    python scripts/compare_models.py --model-ids metastgat_pro metastgat_pro_0.2 maxpressure

Richiede che results/<model_id>/test_summary_<config>.json esista gia' (prodotto da
scripts/test.py) per ogni combinazione modello/config che si vuole confrontare -- le
combinazioni mancanti vengono saltate con un avviso, non fanno fallire lo script.
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERRORE] matplotlib non trovato. Installa con: pip install matplotlib")
    sys.exit(1)


# Palette per modello (coerente con plot_results.MODEL_COLORS dove i model_id
# coincidono; estesa per le varianti che non esistevano ancora li', es. sweep su
# alpha). Fallback per model_id non elencati: ciclo di colori CVD-safe di riserva.
#
# Scelta delle tinte (12/9/2026): validata con lo script della skill "dataviz"
# (scripts/validate_palette.py, checks CVD/contrasto/OKLCH su OKLab) sui gruppi
# di modelli che realisticamente compaiono INSIEME in uno stesso grafico (non
# sulle 12+ tinte tutte insieme: oltre 3-4 serie contemporanee nessun ordine
# di tinte supera il check "all-pairs", limite intrinseco documentato dalla
# skill stessa) -- i gruppi verificati con --pairs all (il piu' severo, perche'
# le barre sono riordinate per TT ad ogni config: due modelli qualsiasi
# possono finire adiacenti, quindi "adiacente nell'ordine dato" non basta):
#   - "main" (pro_0.5, pro_0.2, maxpressure, fixedtime, paper): PASS
#   - self-loop (pro_0.2, *_self_loop, maxpressure): PASS
#   - ablation study (paper, pro_0.5, le 4 ablation): PASS
# Residuo noto, non risolvibile senza sacrificare uno di questi tre gruppi:
# pro_0.2 (viola) e fixedtime (violetto) restano moderatamente vicini se
# comparissero insieme in un futuro grafico non ancora esistente oggi --
# mitigato dalle etichette numeriche sempre presenti sopra ogni barra e dalla
# legenda testuale (la "relief rule" della skill per i casi di sola-tinta
# insufficiente).
MODEL_COLORS = {
    "metastgat_pro":                "#2a78d6",  # blue (alias legacy, vedi metastgat_pro_0.5)
    "metastgat_pro_0.5":            "#2a78d6",  # blue -- alpha di default (0.5)
    "metastgat_pro_0.2":            "#8e44ad",  # purple -- sweep alpha=0.2 (compare in tutti e 3 i gruppi sopra)
    "metastgat_pro_0.1":            "#c2185b",  # rose -- sweep alpha=0.1
    "metastgat_pro_0.0":            "#1b9e77",  # teal -- sweep alpha=0.0 (nessuna penalita' anti-starvation)
    "metastgat_paper":              "#eb6834",  # orange
    "ablation_environment":         "#1baf7a",  # aqua
    "ablation_temporal":            "#c7ad1a",  # senape -- ritinta dal giallo originale (troppo vicino ad arancio/aqua in gruppo)
    "ablation_rl_core":             "#b8508f",  # magenta-viola -- ritinta (il magenta originale era troppo vicino ad arancio)
    "ablation_replay_stability":    "#0f7a5c",  # verde petrolio -- ritinta (il verde puro originale era troppo vicino ad arancio sotto CVD)
    "fixedtime":                    "#4a3aa7",  # violetto
    "maxpressure":                  "#b23a56",  # cremisi -- ritinta dal rosso originale (troppo vicino ad arancio/rosa)
    "metastgat_pro_0.2_self_loop":  "#1f6b40",  # verde foresta -- esperimenti self-loop su Meta-GAT (non applicati in via definitiva)
    "metastgat_pro_0.0_self_loop":  "#1f6b40",  # stessa tinta: "esperimento self-loop", a prescindere dall'alpha di base
}
_FALLBACK_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

# Durata fissa di un episodio in questo progetto (maxStep=1800, interval=1.0s
# in ogni configs/*.json, vedi descrizione_configurazioni.md): hardcoded qui
# per disegnare una riga di riferimento nei grafici (un veicolo/attesa che la
# tocca ha "consumato" l'intero episodio, non e' un valore arbitrario tra tanti).
EPISODE_DURATION_S = 1800


def _color_for(model_id, idx):
    if model_id in MODEL_COLORS:
        return MODEL_COLORS[model_id]
    return _FALLBACK_COLORS[idx % len(_FALLBACK_COLORS)]


# (basename, etichetta breve, ruolo) -- vedi descrizione_configurazioni.md
TRAIN_CONFIGS = [
    ("config_4x4_100m_train1", "Train 1", "training"),
    ("config_4x4_100m_train2", "Train 2", "training"),
    ("config_4x4_100m_train3", "Train 3", "training"),
]
TEST_CONFIGS = [
    ("config_4x4_100m_6k_peak",    "4x4/100m peak",  "test"),
    ("config_4x4_200m_6k_flat",    "4x4/200m flat",  "test"),
    ("config_4x4_200m_6k_peak",    "4x4/200m peak",  "test"),
    ("config_5x5_100m_9.4k_flat",  "5x5 flat",       "test"),
    ("config_5x5_100m_9.4k_peak",  "5x5 peak",       "test"),
    ("config_6x6_100m_11.5k_flat", "6x6 flat",       "test"),
    ("config_6x6_100m_11.5k_peak", "6x6 peak",       "test"),
]
ALL_CONFIGS = TRAIN_CONFIGS + TEST_CONFIGS

DEFAULT_MODEL_IDS = ["metastgat_pro_0.5", "metastgat_pro_0.2", "maxpressure"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results",
                        help="Directory contenente results/<model_id>/test_summary_*.json")
    parser.add_argument("--model-ids", nargs="+", default=DEFAULT_MODEL_IDS,
                        help="model_id da confrontare (cartelle sotto results-dir). "
                             "Un model_id senza alcun risultato disponibile viene "
                             "escluso silenziosamente da tutti i grafici.")
    parser.add_argument("--output-dir", default=None,
                        help="Cartella per i PNG, uno per config "
                             "(default: <results-dir>/plots/compare_per_config)")
    return parser.parse_args()


def load_all_summaries(results_dir, model_ids):
    """{model_id: {config_basename: summary_dict}}, solo per le combinazioni presenti.

    Dal riordino del 12/9/2026 ogni cartella modello tiene i test_summary_*.json
    dentro una sottocartella test_summaries/ (vedi results/REORGANIZATION.md);
    per compatibilita' con eventuali cartelle non ancora riordinate (es. un
    modello testato di fresco con test.py, che scrive ancora piatto in
    model_dir/), si cerca prima li' e poi, come fallback, direttamente in
    model_dir/.
    """
    data = {mid: {} for mid in model_ids}
    for mid in model_ids:
        model_dir = os.path.join(results_dir, mid)
        for cfg, _label, _role in ALL_CONFIGS:
            fname = f"test_summary_{cfg}.json"
            path = os.path.join(model_dir, "test_summaries", fname)
            if not os.path.exists(path):
                path = os.path.join(model_dir, fname)
            if os.path.exists(path):
                with open(path) as f:
                    data[mid][cfg] = json.load(f)
    return data


def print_table(data, model_ids):
    metrics = [
        ("avg_travel_time", "TT medio (s)"),
        ("tt_max", "TT massimo (s)"),
        ("wait_max_ns", "Attesa max N/S (s)"),
        ("wait_max_ns_resolved", "  ...solo stalli conclusi (s)"),
        ("wait_max_ew", "Attesa max W/E (s)"),
        ("wait_max_ew_resolved", "  ...solo stalli conclusi (s)"),
    ]
    present_models = [m for m in model_ids if data.get(m)]
    if not present_models:
        print("[ATTENZIONE] Nessun model_id ha risultati disponibili.")
        return

    for section_name, configs in [("CONFIG DI TRAINING", TRAIN_CONFIGS), ("CONFIG DI TEST", TEST_CONFIGS)]:
        print(f"\n{'='*100}\n  {section_name}\n{'='*100}")
        for cfg, label, _role in configs:
            models_here = [m for m in present_models if cfg in data[m]]
            if not models_here:
                continue
            print(f"\n  {label}  ({cfg})")
            for key, mlabel in metrics:
                vals = {m: data[m][cfg].get(key) for m in models_here if data[m][cfg].get(key) is not None}
                if not vals:
                    print(f"    {mlabel:<22}  dato mancante per tutti i modelli")
                    continue
                best = min(vals, key=vals.get)
                parts = "  ".join(
                    f"{m}={v:>7.1f}{'*' if m == best else ' '}" for m, v in vals.items()
                )
                print(f"    {mlabel:<22}  {parts}   (* = migliore)")


def _balanced_legend_ncol(n, max_col=4):
    """Numero di colonne per una legenda a n voci, righe il piu' possibile
    bilanciate (mai una riga quasi vuota tipo 4+1, o piena+corta tipo 4+2 se
    3+3 e' possibile) entro un massimo di max_col colonne per riga.
    Es.: n=6 -> 3 (righe 3+3, non 4+2); n=5 -> 3 (righe 3+2); n=8 -> 4 (4+4)."""
    if n <= max_col:
        return max(n, 1)
    nrows = -(-n // max_col)  # ceil
    return -(-n // nrows)     # ceil(n / nrows)


def _row_major_legend_order(handles, labels, ncol):
    """matplotlib riempie una legenda multi-colonna per COLONNE (dall'alto in
    basso, poi colonna successiva), il che disperde un ordine gia' significativo
    (qui: dal modello peggiore al migliore) in un ordine di lettura innaturale.
    Questo reindicizza handles/labels in modo che il fill per-colonne di
    matplotlib produca a schermo un ordine per RIGHE (sinistra->destra,
    alto->basso, l'ordine di lettura naturale) -- trucco standard: si calcola
    la griglia (nrows x ncol) letta per righe e la si "trasporne" prima di
    passarla a legend(), riempiendo con voci vuote trasparenti le celle in
    eccesso (n non multiplo di ncol)."""
    n = len(handles)
    if ncol <= 1 or n <= ncol:
        return handles, labels
    nrows = -(-n // ncol)  # ceil
    pad = nrows * ncol - n
    padded_h = list(handles) + [plt.Line2D([], [], alpha=0)] * pad
    padded_l = list(labels) + [""] * pad
    new_h, new_l = [], []
    for c in range(ncol):
        for r in range(nrows):
            idx = r * ncol + c
            new_h.append(padded_h[idx])
            new_l.append(padded_l[idx])
    return new_h, new_l


def make_config_plot(cfg, label, role, data, model_ids, output_dir):
    present_models = [m for m in model_ids if cfg in data.get(m, {})]
    if not present_models:
        return False

    # Ordine dei modelli nel grafico: dal peggiore al migliore per TT medio
    # su QUESTA config (non un ordine fisso globale — il ranking puo' cambiare
    # da una config all'altra). "Peggiore" = TT medio piu' alto. Solo per i
    # grafici: print_table() mantiene l'ordine di --model-ids cosi' come dato.
    present_models = sorted(
        present_models,
        key=lambda m: data[m][cfg].get("avg_travel_time", float("inf")),
        reverse=True,
    )

    # Scelta deliberata (13/9/2026): sia tt_max che wait_max_ns/ew includono SEMPRE
    # i veicoli che non sono ancora arrivati/non si sono ancora rimossi dallo stallo
    # entro fine episodio, contati col caso peggiore osservato (tempo trascorso finora
    # per tt_max, attesa accumulata finora per wait_max) — nessuna distinzione
    # risolto/censurato qui: se il caso peggiore in assoluto e' un veicolo mai arrivato,
    # e' quello il numero che deve comparire nel grafico, altrimenti un modello che
    # blocca un veicolo per sempre risulterebbe premiato invece che penalizzato.
    # (tt_max lo fa gia' di default in CityFlowEnv.get_travel_time_stats(); wait_max_ns/ew
    # lo fa gia' sempre, non ha mai avuto una variante "solo arrivati" — la variante
    # "_resolved" resta disponibile in test_summary_*.json/CSV per chi la vuole ispezionare
    # ma non e' usata qui.)
    tt_medio = [data[m][cfg].get("avg_travel_time", np.nan) for m in present_models]
    tt_max = [data[m][cfg].get("tt_max", np.nan) for m in present_models]
    wait_ns = [data[m][cfg].get("wait_max_ns", np.nan) for m in present_models]
    wait_ew = [data[m][cfg].get("wait_max_ew", np.nan) for m in present_models]
    colors = [_color_for(m, i) for i, m in enumerate(present_models)]

    n = len(present_models)
    width = min(0.8 / n, 0.3)

    fig, (ax_tt, ax_wait) = plt.subplots(1, 2, figsize=(12, 5.5))
    # Titolo a due righe: prima riga solo "Train"/"Test" (convenzione standard
    # in letteratura ML per distinguere le due famiglie di config), sottotitolo
    # con la config precisa come prima -- l'etichetta descrittiva (label, es.
    # "Train 1") resta solo nella tabella testuale di print_table().
    role_title = {"training": "Train", "test": "Test"}.get(role, role.capitalize())
    fig.suptitle(f"{role_title}\n{cfg}", fontsize=13, fontweight="bold")

    def _grouped_bars(ax, group_values, group_names, ylabel, title):
        group_pos = np.arange(len(group_names))
        for i, (model, color) in enumerate(zip(present_models, colors)):
            offsets = group_pos + (i - (n - 1) / 2) * width
            values = [group_values[g][i] for g in range(len(group_names))]
            bars = ax.bar(offsets, values, width=width, color=color, edgecolor="black",
                          linewidth=0.6, label=model)
            for x, v in zip(offsets, values):
                if not np.isnan(v):
                    ax.text(x, v + max([vv for vv in sum(group_values, []) if not np.isnan(vv)], default=1) * 0.015,
                            f"{v:.0f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(group_pos)
        ax.set_xticklabels(group_names)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        # Riga di riferimento: durata di un intero episodio (1800s, fissa per
        # questo progetto -- vedi EPISODE_DURATION_S). Un valore che la tocca
        # ha "consumato" tutto l'episodio (veicolo mai arrivato / stallo mai
        # risolto), non e' un secondo come un altro sull'asse. Estende l'asse Y
        # se serve, cosi' la riga e la sua etichetta sono sempre visibili anche
        # quando tutte le barre restano molto piu' basse.
        all_vals = [v for vv in group_values for v in vv if not np.isnan(v)]
        top = max(all_vals + [EPISODE_DURATION_S]) * 1.08
        ax.set_ylim(0, top)
        ax.axhline(EPISODE_DURATION_S, color="black", linestyle=":", linewidth=1.1, zorder=0.5)
        ax.text(ax.get_xlim()[1], EPISODE_DURATION_S, f" durata episodio ({EPISODE_DURATION_S}s) ",
                ha="right", va="bottom", fontsize=7.5, color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))

    _grouped_bars(ax_tt, [tt_medio, tt_max], ["TT medio", "TT massimo"],
                  "Secondi (s)", "Travel Time")
    _grouped_bars(ax_wait, [wait_ns, wait_ew], ["Attesa N/S", "Attesa W/E"],
                  "Secondi (s)", "Attesa massima direzionale")

    handles, labels_ = ax_tt.get_legend_handles_labels()
    ncol = _balanced_legend_ncol(n, max_col=4)
    handles, labels_ = _row_major_legend_order(handles, labels_, ncol)
    fig.legend(handles, labels_, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout(rect=[0, 0.04, 1, 0.93])
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"compare_{cfg}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.results_dir, "plots", "compare_per_config")

    data = load_all_summaries(args.results_dir, args.model_ids)
    available = {m: len(cfgs) for m, cfgs in data.items()}
    print(f"[INFO] Config disponibili per modello: {available}  (su {len(ALL_CONFIGS)} totali)")
    missing_models = [m for m, n in available.items() if n == 0]
    if missing_models:
        print(f"[INFO] Nessun risultato trovato per: {missing_models} -- esclusi dai grafici.")

    print_table(data, args.model_ids)

    print(f"\n[INFO] Generazione grafici in: {output_dir}")
    n_saved = 0
    for cfg, label, role in ALL_CONFIGS:
        out = make_config_plot(cfg, label, role, data, args.model_ids, output_dir)
        if out:
            print(f"  [OK] {cfg} -> {out}")
            n_saved += 1
        else:
            print(f"  [SALTATO] {cfg}: nessun modello ha risultati per questa config.")
    print(f"\n[OK] {n_saved}/{len(ALL_CONFIGS)} grafici generati in {output_dir}")


if __name__ == "__main__":
    main()
