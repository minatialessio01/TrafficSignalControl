"""
Estende overlay_all_groups.py in due direzioni:

1. Oltre alla media sulle 10 config, genera anche la media SOLO sulle 3
   config di training e SOLO sulle 7 config di test, per ciascuno dei 4
   gruppi -- tre varianti tenute nella stessa cartella (compare_overlay_3stat),
   distinte per suffisso nel nome file (_all10 / _train / _test).

2. Genera anche una versione PER CONFIGURAZIONE SINGOLA (non mediata) dello
   stesso stile di grafico, una cartella per gruppo, un file per config
   (10 file ciascuna) -- nomi di cartella distinti sia dalle cartelle "vecchie"
   (compare_pro_alpha_v3 ecc., stile a barre affiancate) sia dalla cartella
   di aggregati (compare_overlay_3stat).

Stile del grafico invariato rispetto alla versione approvata: tre fasce
annidate (TT medio < attesa massima a un incrocio < TT massimo), colore
unico blu scuro->chiaro a piena saturazione, legenda sotto l'asse x, nessuna
delle due frasi esplicative in cima (solo, per le versioni per-config, un
titolo minimo con il nome della config -- altrimenti i file sarebbero
indistinguibili aprendoli senza vedere il nome del file).
"""
import json
import os
import sys

sys.path.insert(0, "/workspace/scripts")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from compare_models import ALL_CONFIGS, TRAIN_CONFIGS, TEST_CONFIGS

RESULTS_DIR = "/workspace/results"
PLOTS_DIR = "/workspace/results/plots"
AGG_DIR = os.path.join(PLOTS_DIR, "compare_3stat")

SHADES = ("#08306b", "#2171b5", "#6baed6")  # scuro (TT medio) -> chiaro (TT massimo)

LABELS = {
    "fixedtime": "Fixed-Time",
    "maxpressure": "MaxPressure",
    "ablation_paper": "Paper",
    "ablation_temporal": "Temporal",
    "ablation_single_dqn": "Single DQN",
    "ablation_environment": "Environment",
    "ablation_vanilla_buffer": "Vanilla buffer",
    "pro_0.08": "Pro (α=0.08)",
    "pro_0.04": "Pro (α=0.04)",
    "pro_0.00": "Pro (α=0.00)",
    "metastgcn_1l_pro_0.08": "MetaSTGCN 1L",
    "metastgcn_2l_pro_0.08": "MetaSTGCN 2L",
    "pro_2l": "MetaSTGAT 2L",
    "metastsonar_l4_pro_0.08": "MetaSTSONAR L4",
    "metastsonar_l2_pro_0.08": "MetaSTSONAR L2",
}

# (group_key, model_ids, nome cartella per-config -- distinto sia dalle
# cartelle "vecchie" a barre affiancate sia da compare_overlay_3stat)
#
# 17/9/2026 (2): "official" non ha piu' senso ora che soft-wait-scale e' lo
# standard per ogni modello di questo lavoro: eliminata la distinzione,
# tutte le cartelle risultati sono state rinominate togliendo sia
# "_official" sia il suffisso "_softwait" ormai ridondante (vedi
# results/<nome>/ senza suffissi). Gruppo "150ep" resta rimosso (dati
# sorgente cancellati in precedenza).
GROUPS = [
    ("pro_alpha", [
        "fixedtime", "pro_0.08", "pro_0.04",
        "pro_0.00", "maxpressure",
    ], "compare_pro_alpha"),
    ("ablation", [
        "fixedtime", "ablation_paper", "ablation_temporal",
        "pro_0.08", "ablation_single_dqn",
        "ablation_environment", "ablation_vanilla_buffer", "maxpressure",
    ], "compare_ablation"),
    ("architectures", [
        "fixedtime", "metastgcn_1l_pro_0.08", "metastgcn_2l_pro_0.08",
        "pro_0.08", "pro_2l",
        "metastsonar_l4_pro_0.08", "metastsonar_l2_pro_0.08", "maxpressure",
    ], "compare_architectures"),
]

ALL_CFG_NAMES = [c for c, _l, _r in ALL_CONFIGS]
TRAIN_CFG_NAMES = [c for c, _l, _r in TRAIN_CONFIGS]
TEST_CFG_NAMES = [c for c, _l, _r in TEST_CONFIGS]
CFG_LABELS = {c: l for c, l, _r in ALL_CONFIGS}


SEED_SUFFIXES = ["", "2", "3", "4", "5"]

# 19/9/2026: le config con corsie da 260m ("200m") avevano max wait
# sottostimato dal vincolo di visibilita' (144,3m dall'incrocio) -- vedi
# memoria "maxwait-vision-cutoff-fix". Per queste due usiamo i nuovi campi
# wait_max_ns_full/ew_full (piena corsia, ri-testati il 19/9), per le altre
# 5 config di test e le 3 di training i vecchi campi restano corretti.
FULL_LANE_CFGS = {"config_4x4_200m_6k_flat", "config_4x4_200m_6k_peak"}


def load_all():
    """{model_id: {config: (tt, wait_worst, ttmax)}} per tutti i modelli usati in qualunque gruppo.
    Per le config di test, media sui seed disponibili (stessa convenzione di
    compare_models.py), cosi' i numeri combaciano con le tabelle del Cap. 4."""
    all_models = sorted({m for _k, ids, _f in GROUPS for m in ids})
    data = {m: {} for m in all_models}
    for m in all_models:
        for cfg in ALL_CFG_NAMES:
            suffixes = SEED_SUFFIXES if cfg in TEST_CFG_NAMES else [""]
            ns_key, ew_key = (("wait_max_ns_full", "wait_max_ew_full") if cfg in FULL_LANE_CFGS
                               else ("wait_max_ns", "wait_max_ew"))
            tts, waits, ttmaxs = [], [], []
            for suf in suffixes:
                path = os.path.join(RESULTS_DIR, m, "test_summaries", f"test_summary_{cfg}{suf}.json")
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    d = json.load(f)
                tts.append(d["avg_travel_time"])
                waits.append(max(d[ns_key], d[ew_key]))
                ttmaxs.append(d["tt_max"])
            if not tts:
                continue
            data[m][cfg] = (np.mean(tts), np.mean(waits), np.mean(ttmaxs))
    return data


DATA = load_all()


def draw_chart(model_ids, values, out_path, title=None):
    """values: {model_id: (tt, wait, ttmax)}"""
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(model_ids))
    width = 0.6
    ymax = max(v[2] for v in values.values())

    for i, m in enumerate(model_ids):
        tt, wait, ttmax = values[m]
        if not (tt < wait < ttmax):
            print(f"  [ATTENZIONE] ordine non rispettato per {m} in {out_path}: "
                  f"tt={tt:.1f} wait={wait:.1f} ttmax={ttmax:.1f} (disegnato comunque)")

        ax.bar(x[i], ttmax, width=width, color=SHADES[2], edgecolor=SHADES[2], linewidth=0.8, zorder=2)
        ax.bar(x[i], wait, width=width, color=SHADES[1], zorder=3)
        ax.bar(x[i], tt, width=width, color=SHADES[0], zorder=4)

        ax.text(x[i], ttmax + ymax * 0.012, f"{ttmax:.0f}", ha="center", va="bottom",
                fontsize=8, color=SHADES[0], fontweight="bold", zorder=5)
        ax.text(x[i], (tt + wait) / 2, f"{wait:.0f}", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold", zorder=5)
        ax.text(x[i], tt / 2, f"{tt:.0f}", ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold", zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(m, m) for m in model_ids], fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Secondi (s)")
    if title:
        ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.set_ylim(0, ymax * 1.15)

    legend_elems = [
        Patch(facecolor=SHADES[0], label="TT medio"),
        Patch(facecolor=SHADES[1], label="Max wait"),
        Patch(facecolor=SHADES[2], label="TT massimo"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def mean_over(model_ids, cfg_names):
    out = {}
    for m in model_ids:
        rows = [DATA[m][c] for c in cfg_names if c in DATA[m]]
        out[m] = tuple(np.mean([r[k] for r in rows]) for k in range(3))
    return out


# ── 1. Aggregati: tutte e 10, solo train, solo test -- stessa cartella,
#    nomi distinti ────────────────────────────────────────────────────────
os.makedirs(AGG_DIR, exist_ok=True)
for group_key, model_ids, _folder in GROUPS:
    for suffix, cfg_subset in [("all10", ALL_CFG_NAMES), ("train", TRAIN_CFG_NAMES), ("test", TEST_CFG_NAMES)]:
        values = mean_over(model_ids, cfg_subset)
        out_path = os.path.join(AGG_DIR, f"{group_key}_{suffix}.png")
        draw_chart(model_ids, values, out_path)
        print(f"[OK] {out_path}")

# ── 2. Per-config, una cartella per gruppo, un file per config ──────────────
for group_key, model_ids, folder in GROUPS:
    out_dir = os.path.join(PLOTS_DIR, folder)
    for cfg in ALL_CFG_NAMES:
        values = {m: DATA[m][cfg] for m in model_ids if cfg in DATA[m]}
        if len(values) < len(model_ids):
            missing = [m for m in model_ids if cfg not in DATA[m]]
            print(f"  [SALTATO PARZIALE] {group_key}/{cfg}: mancano {missing}")
        if not values:
            continue
        out_path = os.path.join(out_dir, f"compare_{cfg}.png")
        draw_chart(list(values.keys()), values, out_path, title=CFG_LABELS.get(cfg, cfg))
    print(f"[OK] {out_dir} (10 config)")

print("=== FATTO ===")
