import json, os, statistics

RESULTS = "/workspace/results"
OUT_DIR = "/workspace/scratch_out"
os.makedirs(OUT_DIR, exist_ok=True)

# model label -> results dir (post hard-update migration; ablation_no_soft_update
# is an explicit PLACEHOLDER for pro_0.08 until the real pro_0.08_no_soft run finishes)
MODELS = [
    ("Fixed-Time", "fixedtime"),
    ("MaxPressure", "maxpressure"),
    ("Pro (alpha=0.08)", "pro_0.08_no_soft"),
    ("Pro (alpha=0.04)", "pro_0.04_no_soft"),
    ("Pro (alpha=0.00)", "pro_0.00_no_soft"),
    ("MetaSTGAT 2L", "pro_2l_no_soft"),
    ("Ablation: paper", "ablation_paper"),
    ("Ablation: temporal", "ablation_temporal_no_soft"),
    ("Ablation: single_dqn", "ablation_single_dqn_no_soft"),
    ("Ablation: environment", "ablation_environment_no_soft"),
    ("Ablation: vanilla_buffer", "ablation_vanilla_buffer"),
    ("MetaSTGCN 1L", "metastgcn_1l_pro_0.08_no_soft"),
    ("MetaSTGCN 2L", "metastgcn_2l_pro_0.08_no_soft"),
    ("MetaSTSONAR L=2", "metastsonar_l2_pro_0.08_no_soft"),
    ("MetaSTSONAR L=4", "metastsonar_l4_pro_0.08_no_soft"),
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

# Numero di veicoli strutturalmente impossibilitati ad arrivare in tempo anche
# in condizioni di flusso libero (nessun semaforo, velocita' massima costante
# lungo il percorso), calcolato una volta per configurazione con
# scripts/compute_tc_adjustment.py -- e' una proprieta' dello scenario di
# traffico, non del modello, quindi va bene riusarlo per ogni modello.
with open(f"{OUT_DIR}/tc_adjustment.json") as _f:
    TC_ADJUSTMENT = json.load(_f)

MISSING = []  # (model, cfg, seed) with no file found -- reported at the end


def load(model, cfg, suf=""):
    p = f"{RESULTS}/{model}/test_summaries/test_summary_{cfg}{suf}.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def metric_value(d, cfg, metric, full_cfg_key=None):
    if metric == "tt":
        return d["avg_travel_time"]
    if metric == "ttmax":
        return d["tt_max"]
    if metric == "wait":
        ns_k, ew_k = ("wait_max_ns_full", "wait_max_ew_full") if cfg in FULL_LANE else ("wait_max_ns", "wait_max_ew")
        return max(d[ns_k], d[ew_k])
    if metric == "tc_plain":
        return 100.0 * d["avg_throughput"] / d["total_vehicles"]
    if metric == "tc_adjusted":
        adj = TC_ADJUSTMENT.get(full_cfg_key)
        if adj is None or "impossible" not in adj:
            raise KeyError(f"tc_adjustment.json non ha una voce valida per {full_cfg_key}")
        # denom: veicoli EFFETTIVAMENTE entrati in simulazione secondo test.py
        # (d["total_vehicles"]), non il numero teorico schedulato dal file di
        # flusso (adj["total_vehicles"], puo' essere piu' alto se qualche
        # veicolo non e' mai entrato per congestione al punto di ingresso) --
        # da questo si sottraggono i veicoli strutturalmente impossibilitati
        # secondo il calcolo a flusso libero. Verificato contro i vecchi
        # numeri di fixedtime (84.9% atteso, 84.8% ottenuto con questa
        # formula, 79.4% con quella basata sul totale teorico -- sbagliata).
        denom = d["total_vehicles"] - adj["impossible"]
        return 100.0 * d["avg_throughput"] / denom
    raise ValueError(metric)


def scenario_values(mid, cfg, metric, seeded):
    vals = []
    seeds = SEEDS if seeded else [""]
    for suf in seeds:
        d = load(mid, cfg, suf)
        if d is None:
            if seeded:
                MISSING.append((mid, cfg + suf))
            continue
        vals.append(metric_value(d, cfg, metric, full_cfg_key=cfg + suf))
    return vals


def scenario_mean_std(mid, cfg, metric, seeded):
    vals = scenario_values(mid, cfg, metric, seeded)
    if not vals:
        return None
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


# ---------------------------------------------------------------------------
# 1) Full per-config detail tables (tt, ttmax, wait, tc_plain)
# ---------------------------------------------------------------------------

def build_full_table(metric, caption, label, fmt="{:.1f}"):
    all_cfgs = TRAIN_CFGS + TEST_CFGS
    lines = [r"\begin{sidewaystable}", r"  \centering", r"  \tiny"]
    lines.append(r"  \begin{tabular}{@{}l" + "r" * len(all_cfgs) + r"@{}}")
    lines.append(r"    \toprule")
    lines.append("    Modello & " + " & ".join(lbl for lbl, _c in all_cfgs) + r" \\")
    lines.append(r"    \midrule")
    for mlabel, mid in MODELS:
        row = [mlabel]
        for lbl, cfg in TRAIN_CFGS:
            r = scenario_mean_std(mid, cfg, metric, seeded=False)
            row.append(fmt.format(r[0]) if r else "--")
        for lbl, cfg in TEST_CFGS:
            r = scenario_mean_std(mid, cfg, metric, seeded=True)
            if r is None:
                row.append("--")
            else:
                mean, std = r
                row.append(f"{fmt.format(mean)} {{\\tiny$\\pm${std:.1f}}}")
        lines.append("    " + " & ".join(row) + r" \\")
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(f"  \\caption{{{caption}}}")
    lines.append(f"  \\label{{{label}}}")
    lines.append(r"\end{sidewaystable}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2) Aggregate (train-avg / test-avg) numbers per model, for the summary tables
#    train-avg = unweighted mean over the 3 training scenarios (single seed each)
#    test-avg  = unweighted mean over the 7 test SCENARIO MEANS (each already
#                averaged over its 5 seeds) -- matches the existing tables'
#                "mediati sulle 3/7 configurazioni" convention.
# ---------------------------------------------------------------------------

def aggregate(mid, metric):
    train_vals = []
    for _lbl, cfg in TRAIN_CFGS:
        r = scenario_mean_std(mid, cfg, metric, seeded=False)
        if r:
            train_vals.append(r[0])
    test_scenario_means = []
    for _lbl, cfg in TEST_CFGS:
        r = scenario_mean_std(mid, cfg, metric, seeded=True)
        if r:
            test_scenario_means.append(r[0])
    train_avg = statistics.mean(train_vals) if train_vals else None
    test_avg = statistics.mean(test_scenario_means) if test_scenario_means else None
    return train_avg, test_avg


def per_config_diffpct_vs(mid, ref_mid, metric, cfgs, seeded):
    """Mean of per-config percentage differences vs a reference model (e.g. MaxPressure),
    the 'media delle variazioni per configurazione' convention chosen for this chapter."""
    diffs = []
    for _lbl, cfg in cfgs:
        a = scenario_mean_std(mid, cfg, metric, seeded)
        b = scenario_mean_std(ref_mid, cfg, metric, seeded)
        if a and b and b[0] != 0:
            diffs.append(100.0 * (a[0] - b[0]) / b[0])
    return statistics.mean(diffs) if diffs else None


ALL_METRICS = ["tt", "ttmax", "wait", "tc_adjusted"]

if __name__ == "__main__":
    tt_table = build_full_table("tt", "Travel time medio (s).", "tab:ttmedio-full")
    ttmax_table = build_full_table("ttmax", "Travel time massimo (s).", "tab:ttmax-full")
    wait_table = build_full_table("wait", "Max wait (s).", "tab:wait-full")
    tc_table = build_full_table("tc_adjusted", "Trip completion $tc$ corretta (%), esclude i veicoli strutturalmente impossibilitati ad arrivare in tempo.", "tab:tc-adjusted", fmt="{:.1f}")

    with open(f"{OUT_DIR}/full_tables.tex", "w") as f:
        f.write("\n\n".join([tt_table, ttmax_table, wait_table, tc_table]))

    # Aggregate summary CSV: model,metric,train_avg,test_avg,diffpct_train_vs_maxpressure,diffpct_test_vs_maxpressure
    with open(f"{OUT_DIR}/aggregates.csv", "w") as f:
        f.write("model,metric,train_avg,test_avg,diffpct_train_vs_mp,diffpct_test_vs_mp\n")
        for mlabel, mid in MODELS:
            for metric in ALL_METRICS:
                train_avg, test_avg = aggregate(mid, metric)
                dp_train = per_config_diffpct_vs(mid, "maxpressure", metric, TRAIN_CFGS, seeded=False)
                dp_test = per_config_diffpct_vs(mid, "maxpressure", metric, TEST_CFGS, seeded=True)
                f.write(f'"{mlabel}",{metric},'
                        f'{train_avg if train_avg is not None else ""},'
                        f'{test_avg if test_avg is not None else ""},'
                        f'{dp_train if dp_train is not None else ""},'
                        f'{dp_test if dp_test is not None else ""}\n')

    with open(f"{OUT_DIR}/missing.txt", "w") as f:
        if MISSING:
            for mid, cfg in MISSING:
                f.write(f"{mid}\t{cfg}\n")
        else:
            f.write("(none)\n")

    print("=== AGGREGATES (train_avg / test_avg) ===")
    for mlabel, mid in MODELS:
        parts = []
        for metric in ALL_METRICS:
            ta, te = aggregate(mid, metric)
            ta_s = f"{ta:.1f}" if ta is not None else "NA"
            te_s = f"{te:.1f}" if te is not None else "NA"
            parts.append(f"{metric}: train={ta_s} test={te_s}")
        print(f"{mlabel:28s} ({mid:32s}) " + " | ".join(parts))

    print(f"\nMissing files: {len(MISSING)} (see {OUT_DIR}/missing.txt)")
    print(f"Full tables written to {OUT_DIR}/full_tables.tex")
    print(f"Aggregates CSV written to {OUT_DIR}/aggregates.csv")
