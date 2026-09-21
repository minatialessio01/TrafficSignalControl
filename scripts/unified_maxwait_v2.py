import json
import os
import statistics

RESULTS = "/workspace/results"

TRAIN_CFGS = ["config_4x4_100m_train1", "config_4x4_100m_train2", "config_4x4_100m_train3"]
# I 5 non-200m restano con la vecchia definizione (verificato equivalente).
TEST_CFGS_UNAFFECTED = [
    "config_4x4_100m_6k_peak",
    "config_5x5_100m_9.4k_flat", "config_5x5_100m_9.4k_peak",
    "config_6x6_100m_11.5k_flat", "config_6x6_100m_11.5k_peak",
]
# I 2 200m usano ora wait_max_ns_full/ew_full (piena corsia, appena rigenerati).
TEST_CFGS_200M = ["config_4x4_200m_6k_flat", "config_4x4_200m_6k_peak"]
SEEDS = ["", "2", "3", "4", "5"]


def load(model, cfg, suf=""):
    p = f"{RESULTS}/{model}/test_summaries/test_summary_{cfg}{suf}.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def unified_maxwait_train(model):
    vals = []
    for cfg in TRAIN_CFGS:
        d = load(model, cfg, "")
        if d is None:
            continue
        vals.append(max(d["wait_max_ns"], d["wait_max_ew"]))
    return statistics.mean(vals) if vals else None


def unified_maxwait_test(model):
    # Media per-config-poi-tra-config (non un pool di tutte le istanze
    # seed x config): con modelli a copertura seed disomogenea (es.
    # Meta-GCN/Meta-SONAR, 1 solo seed su 5 delle 7 config di test) un pool
    # unico peserebbe le config a piu' seed sproporzionatamente di piu' --
    # bug scoperto il 19/9/2026, vedi analisi_risultati.md.
    per_config_means = []
    for cfg in TEST_CFGS_UNAFFECTED:
        vals = []
        for suf in SEEDS:
            d = load(model, cfg, suf)
            if d is None:
                continue
            vals.append(max(d["wait_max_ns"], d["wait_max_ew"]))
        if vals:
            per_config_means.append(statistics.mean(vals))
    for cfg in TEST_CFGS_200M:
        vals = []
        for suf in SEEDS:
            d = load(model, cfg, suf)
            if d is None:
                continue
            if "wait_max_ns_full" not in d:
                raise RuntimeError(f"MISSING _full fields for {model}/{cfg}{suf}")
            vals.append(max(d["wait_max_ns_full"], d["wait_max_ew_full"]))
        if vals:
            per_config_means.append(statistics.mean(vals))
    return statistics.mean(per_config_means) if per_config_means else None


STUDIES = {
    "ALPHA": ["maxpressure", "pro_0.08", "pro_0.04", "pro_0.00"],
    "ABLATION": ["fixedtime", "maxpressure", "ablation_paper", "ablation_temporal",
                 "pro_0.08", "ablation_single_dqn", "ablation_environment",
                 "ablation_vanilla_buffer"],
    "ARCHITECTURES": ["fixedtime", "maxpressure", "metastgcn_1l_pro_0.08",
                       "metastgcn_2l_pro_0.08", "pro_0.08", "pro_2l",
                       "metastsonar_l4_pro_0.08", "metastsonar_l2_pro_0.08"],
}

for name, models in STUDIES.items():
    print(f"\n=== {name} ===")
    for m in models:
        tr = unified_maxwait_train(m)
        te = unified_maxwait_test(m)
        if tr is not None and te is not None:
            print(f"  {m:30s} train={tr:8.1f}  test={te:8.1f}")
        else:
            print(f"  {m:30s} MANCA")
