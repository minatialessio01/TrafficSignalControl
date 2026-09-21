import json, os, statistics

RESULTS = "/workspace/results"

MODELS = [
    ("Fixed-Time", "fixedtime"),
    ("MaxPressure", "maxpressure"),
    ("Pro ($\\alpha$=0.08)", "pro_0.08"),
    ("Pro ($\\alpha$=0.04)", "pro_0.04"),
    ("Pro ($\\alpha$=0.00)", "pro_0.00"),
    ("Meta-GAT 2L", "pro_2l"),
    ("Ablation: paper", "ablation_paper"),
    ("Ablation: temporal", "ablation_temporal"),
    ("Ablation: single\\_dqn", "ablation_single_dqn"),
    ("Ablation: environment", "ablation_environment"),
    ("Ablation: vanilla\\_buffer", "ablation_vanilla_buffer"),
    ("Meta-GCN 1L", "metastgcn_1l_pro_0.08"),
    ("Meta-GCN 2L", "metastgcn_2l_pro_0.08"),
    ("Meta-SONAR L=2", "metastsonar_l2_pro_0.08"),
    ("Meta-SONAR L=4", "metastsonar_l4_pro_0.08"),
]

TRAIN_CFGS = [("Train 1", "config_4x4_100m_train1"), ("Train 2", "config_4x4_100m_train2"), ("Train 3", "config_4x4_100m_train3")]
TEST_CFGS = [
    ("4x4 Pk", "config_4x4_100m_6k_peak"),
    ("4x4-200 F", "config_4x4_200m_6k_flat"),
    ("4x4-200 Pk", "config_4x4_200m_6k_peak"),
    ("5x5 F", "config_5x5_100m_9.4k_flat"),
    ("5x5 Pk", "config_5x5_100m_9.4k_peak"),
    ("6x6 F", "config_6x6_100m_11.5k_flat"),
    ("6x6 Pk", "config_6x6_100m_11.5k_peak"),
]
FULL_LANE = {"config_4x4_200m_6k_flat", "config_4x4_200m_6k_peak"}
SEEDS = ["", "2", "3", "4", "5"]


def load(model, cfg, suf=""):
    p = f"{RESULTS}/{model}/test_summaries/test_summary_{cfg}{suf}.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def cell_train(model, cfg, metric):
    d = load(model, cfg, "")
    if d is None:
        return None
    if metric == "tt":
        return d["avg_travel_time"], None
    if metric == "ttmax":
        return d["tt_max"], None
    if metric == "wait":
        ns_k, ew_k = ("wait_max_ns_full", "wait_max_ew_full") if cfg in FULL_LANE else ("wait_max_ns", "wait_max_ew")
        return max(d[ns_k], d[ew_k]), None


def cell_test(model, cfg, metric):
    vals = []
    for suf in SEEDS:
        d = load(model, cfg, suf)
        if d is None:
            continue
        if metric == "tt":
            vals.append(d["avg_travel_time"])
        elif metric == "ttmax":
            vals.append(d["tt_max"])
        elif metric == "wait":
            ns_k, ew_k = ("wait_max_ns_full", "wait_max_ew_full") if cfg in FULL_LANE else ("wait_max_ns", "wait_max_ew")
            vals.append(max(d[ns_k], d[ew_k]))
    if not vals:
        return None
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def build_table(metric, caption, label):
    all_cfgs = TRAIN_CFGS + TEST_CFGS
    lines = []
    lines.append(r"\begin{sidewaystable}")
    lines.append(r"  \centering")
    lines.append(r"  \tiny")
    ncols = len(all_cfgs)
    lines.append(r"  \begin{tabular}{@{}l" + "r" * ncols + r"@{}}")
    lines.append(r"    \toprule")
    header = "    Modello & " + " & ".join(lbl for lbl, _c in all_cfgs) + r" \\"
    lines.append(header)
    lines.append(r"    \midrule")
    for mlabel, mid in MODELS:
        row = [mlabel]
        for lbl, cfg in TRAIN_CFGS:
            r = cell_train(mid, cfg, metric)
            row.append(f"{r[0]:.1f}" if r else "--")
        for lbl, cfg in TEST_CFGS:
            r = cell_test(mid, cfg, metric)
            if r is None:
                row.append("--")
            else:
                mean, std = r
                row.append(f"{mean:.1f} {{\\tiny$\\pm${std:.1f}}}")
        lines.append("    " + " & ".join(row) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{sidewaystable}")
    return "\n".join(lines)


tt_table = build_table(
    "tt",
    "Travel time medio $\\overline{tt}$ (s) per ogni modello e ogni configurazione "
    "(Tabella~\\ref{tab:split} per la corrispondenza tra nomi abbreviati e "
    "configurazione completa). Sulle configurazioni di test, media $\\pm$ "
    "deviazione standard su 5 seed diversi (1 solo seed per Meta-GCN e "
    "Meta-SONAR sulle config non-200m, deviazione standard non riportata in "
    "quel caso).",
    "tab:ttmedio-full",
)

ttmax_table = build_table(
    "ttmax",
    "Travel time massimo $tt_{\\max}$ (s) per ogni modello e ogni "
    "configurazione (Tabella~\\ref{tab:split} per la corrispondenza tra nomi "
    "abbreviati e configurazione completa). Sulle configurazioni di test, "
    "media $\\pm$ deviazione standard su 5 seed diversi (1 solo seed per "
    "Meta-GCN e Meta-SONAR sulle config non-200m, deviazione standard non "
    "riportata in quel caso).",
    "tab:ttmax-full",
)

wait_table = build_table(
    "wait",
    "Max wait $\\text{wait}_{\\max}$ (s) per ogni modello e ogni "
    "configurazione (Tabella~\\ref{tab:split} per la corrispondenza tra nomi "
    "abbreviati e configurazione completa). Sulle configurazioni di test, "
    "media $\\pm$ deviazione standard su 5 seed diversi (1 solo seed per "
    "Meta-GCN e Meta-SONAR sulle config non-200m, deviazione standard non "
    "riportata in quel caso). Sulle config \\texttt{4x4-200}, valori a piena "
    "corsia (vedi §\\ref{sec:metriche}).",
    "tab:wait-full",
)

with open("/scratch/three_tables_output.tex", "w") as f:
    f.write(tt_table + "\n\n" + ttmax_table + "\n\n" + wait_table + "\n")

print(tt_table)
print()
print(ttmax_table)
print()
print(wait_table)
