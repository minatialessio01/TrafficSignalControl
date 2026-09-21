"""
Grafici a barre divergenti: variazione percentuale di TT medio e TT massimo
rispetto a MaxPressure (riferimento a 0%), calcolata per singola
configurazione (e seed, in test) e poi mediata -- stesso metodo dei numeri
già discussi in chat, non media-poi-percentuale.

18/9/2026: aggiunto Fixed-Time a ogni confronto (prima riga, sopra Pro),
e nell'etichetta dell'asse x il range assoluto (min-max in secondi) dei
valori grezzi di MaxPressure su quelle stesse config/seed, cosi' si vede a
colpo d'occhio che uno stesso scarto percentuale vale molti piu' secondi in
assoluto su TT massimo che su TT medio (scale molto diverse: centinaia di
secondi contro oltre mille).
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "/workspace/results"
OUT_DIR = "/workspace/results/plots/differenza_percentuale"

TRAIN_CFGS = ["config_4x4_100m_train1", "config_4x4_100m_train2", "config_4x4_100m_train3"]
TEST_CFGS = [
    "config_4x4_100m_6k_peak", "config_4x4_200m_6k_flat", "config_4x4_200m_6k_peak",
    "config_5x5_100m_9.4k_flat", "config_5x5_100m_9.4k_peak",
    "config_6x6_100m_11.5k_flat", "config_6x6_100m_11.5k_peak",
]
SEED_SUFFIXES = ["", "2", "3", "4", "5"]

COLOR_TT = "#2a78d6"      # blu, TT medio
COLOR_TTMAX = "#c9a227"   # oro scuro, TT massimo


def load(model, cfg, suffix=""):
    p = f"{RESULTS}/{model}/test_summaries/test_summary_{cfg}{suffix}.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def pct_diff(model, cfgs, use_seeds):
    suffixes = SEED_SUFFIXES if use_seeds else [""]
    tt_pcts, ttmax_pcts = [], []
    for cfg in cfgs:
        for suf in suffixes:
            dm = load(model, cfg, suf)
            dref = load("maxpressure", cfg, suf)
            if dm is None or dref is None:
                continue
            tt_pcts.append((dm["avg_travel_time"] - dref["avg_travel_time"]) / dref["avg_travel_time"] * 100.0)
            ttmax_pcts.append((dm["tt_max"] - dref["tt_max"]) / dref["tt_max"] * 100.0)
    return statistics.mean(tt_pcts), statistics.mean(ttmax_pcts)


def maxpressure_range(cfgs, use_seeds):
    """Min-max assoluto (secondi) dei valori grezzi di MaxPressure su
    queste config/seed -- non una media, il range in cui cadono sempre."""
    suffixes = SEED_SUFFIXES if use_seeds else [""]
    tts, ttmaxs = [], []
    for cfg in cfgs:
        for suf in suffixes:
            d = load("maxpressure", cfg, suf)
            if d is None:
                continue
            tts.append(d["avg_travel_time"])
            ttmaxs.append(d["tt_max"])
    return (min(tts), max(tts)), (min(ttmaxs), max(ttmaxs))


def draw(title, labels, models, cfgs, use_seeds, out_path):
    tt_vals, ttmax_vals = [], []
    for m in models:
        tt, ttmax = pct_diff(m, cfgs, use_seeds)
        tt_vals.append(tt)
        ttmax_vals.append(ttmax)

    n = len(labels)
    y = np.arange(n)
    bar_h = 0.32

    fig_h = 1.3 + 0.62 * n
    fig, ax = plt.subplots(figsize=(9, fig_h))

    # TT medio sopra, TT massimo sotto, entro la stessa fascia di categoria
    b1 = ax.barh(y + bar_h / 2 + 0.03, tt_vals, height=bar_h, color=COLOR_TT, zorder=3, label="TT medio")
    b2 = ax.barh(y - bar_h / 2 - 0.03, ttmax_vals, height=bar_h, color=COLOR_TTMAX, zorder=3, label="TT massimo")

    ax.axvline(0, color="#444444", linewidth=1.1, zorder=2)
    xmin, xmax = ax.get_xlim()
    span = max(abs(xmin), abs(xmax))
    pad = span * 0.16
    ax.set_xlim(-span - pad if xmin < 0 else 0, span + pad)

    for bars, vals in ((b1, tt_vals), (b2, ttmax_vals)):
        for rect, v in zip(bars, vals):
            offset = span * 0.015
            x = rect.get_width()
            ha = "left" if x >= 0 else "right"
            lx = x + offset if x >= 0 else x - offset
            ax.text(lx, rect.get_y() + rect.get_height() / 2, f"{v:+.1f}%",
                     va="center", ha=ha, fontsize=9, color="#222222", zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()

    (mn_tt, mx_tt), (mn_ttmax, mx_ttmax) = maxpressure_range(cfgs, use_seeds)
    ax.set_xlabel(
        "Variazione percentuale rispetto a MaxPressure\n"
        f"(MaxPressure: TT medio {mn_tt:.0f}–{mx_tt:.0f}s  ·  "
        f"TT massimo {mn_ttmax:.0f}–{mx_ttmax:.0f}s — stesso % vale molti più secondi su TT massimo)",
        fontsize=8.5
    )
    ax.set_title(title, fontsize=11, pad=12)

    ax.text(0.0, 1.02, "← migliore di MaxPressure", transform=ax.transAxes,
            fontsize=8.5, color="#666666", ha="left", va="bottom", style="italic")
    ax.text(1.0, 1.02, "peggiore di MaxPressure →", transform=ax.transAxes,
            fontsize=8.5, color="#666666", ha="right", va="bottom", style="italic")

    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.legend(handles=[b1, b2], loc="lower center", ncol=2, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_path}")


# ── Sweep alpha ──────────────────────────────────────────────────────────
ALPHA_MODELS = ["fixedtime", "pro_0.08", "pro_0.04", "pro_0.00"]
ALPHA_LABELS = ["Fixed-Time", "Pro (α=0.08)", "Pro (α=0.04)", "Pro (α=0.00)"]

draw("Sweep $\\alpha$ — config di training", ALPHA_LABELS, ALPHA_MODELS, TRAIN_CFGS, False,
     f"{OUT_DIR}/diff_pct_alpha_train.png")
draw("Sweep $\\alpha$ — config di test", ALPHA_LABELS, ALPHA_MODELS, TEST_CFGS, True,
     f"{OUT_DIR}/diff_pct_alpha_test.png")

# ── Ablation ─────────────────────────────────────────────────────────────
ABL_MODELS = ["fixedtime", "pro_0.08", "ablation_paper", "ablation_environment",
              "ablation_temporal", "ablation_single_dqn", "ablation_vanilla_buffer"]
ABL_LABELS = ["Fixed-Time", "Pro (riferimento)", "paper", "environment", "temporal", "single_dqn", "vanilla_buffer"]

draw("Studio di ablation — config di training", ABL_LABELS, ABL_MODELS, TRAIN_CFGS, False,
     f"{OUT_DIR}/diff_pct_ablation_train.png")
draw("Studio di ablation — config di test", ABL_LABELS, ABL_MODELS, TEST_CFGS, True,
     f"{OUT_DIR}/diff_pct_ablation_test.png")

# ── Meccanismi spaziali ──────────────────────────────────────────────────
ARCH_MODELS = ["fixedtime", "pro_0.08", "pro_2l",
               "metastgcn_1l_pro_0.08", "metastgcn_2l_pro_0.08",
               "metastsonar_l2_pro_0.08", "metastsonar_l4_pro_0.08"]
ARCH_LABELS = ["Fixed-Time", "MetaSTGAT 1L (rif.)", "MetaSTGAT 2L", "MetaSTGCN 1L", "MetaSTGCN 2L", "MetaSTSONAR L=2", "MetaSTSONAR L=4"]

draw("Meccanismi spaziali — config di training", ARCH_LABELS, ARCH_MODELS, TRAIN_CFGS, False,
     f"{OUT_DIR}/diff_pct_architetture_train.png")
draw("Meccanismi spaziali — config di test", ARCH_LABELS, ARCH_MODELS, TEST_CFGS, True,
     f"{OUT_DIR}/diff_pct_architetture_test.png")

print("=== FATTO ===")
