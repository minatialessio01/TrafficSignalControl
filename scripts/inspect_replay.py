# -*- coding: utf-8 -*-
"""
Genera un replay con una politica semplice (nessun training/checkpoint) per una
data config CityFlow — serve per ispezionare visivamente/numericamente quanti
veicoli vengono immessi in una rete nuova, prima di allenare qualunque modello
su di essa (vedi descrizione_configurazioni.md).

Due politiche disponibili (--policy):
  - maxpressure (default): MaxPressureAgent (Varaiya 2013, via CityFlowEnv.
    phase_lanelinks/get_lane_vehicle_count) — un controllore reale seppur
    semplice, quindi il throughput che produce è un segnale ragionevole di
    quanto la rete regga il traffico assegnato.
  - random: ogni intersezione sceglie una fase a caso ad ogni step. Stressa
    la rete al massimo ma NON è un baseline onesto (nessun vero controllore
    sarebbe così inefficiente) — utile solo come stress-test visivo estremo.

Uso:
  python scripts/inspect_replay.py --config configs/config_4x4_100m_train.json \\
      --output-dir "analisi_configurazioni2/config_4x4_100m_train" --policy maxpressure
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.cityflow_env import CityFlowEnv, N_PHASES
from src.agents.maxpressure_agent import MaxPressureAgent


def main():
    parser = argparse.ArgumentParser(description="Replay con politica semplice per ispezione config.")
    parser.add_argument("--config", required=True, help="Config CityFlow da ispezionare")
    parser.add_argument("--output-dir", required=True, help="Cartella di output (replay.txt, roadnet_log.json, summary.json)")
    parser.add_argument("--policy", choices=["maxpressure", "random"], default="maxpressure",
                        help="Politica di controllo semaforico da usare per il replay (default: maxpressure)")
    parser.add_argument("--seed", type=int, default=42, help="Seed (usato solo da --policy random)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    # ── Prepara config CityFlow con saveReplay attivo (stessa logica di test.py) ──
    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg["saveReplay"] = True
    base_dir = cfg.get("dir", "./")
    rel_out_dir = os.path.relpath(args.output_dir, base_dir)
    cfg["replayLogFile"] = os.path.join(rel_out_dir, "replay.txt").replace("\\", "/")
    cfg["roadnetLogFile"] = os.path.join(rel_out_dir, "roadnet.log").replace("\\", "/")
    temp_config_path = os.path.join(args.output_dir, f"temp_inspect_config_{config_basename}.json")
    with open(temp_config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  INSPECT REPLAY ({args.policy}): {config_basename}")
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    env = CityFlowEnv(temp_config_path)
    rng = random.Random(args.seed)
    agent = MaxPressureAgent(n_phases=N_PHASES) if args.policy == "maxpressure" else None

    env.reset()
    done = False
    while not done:
        if args.policy == "maxpressure":
            actions = agent.select_actions(env.inter_ids, env=env)
        else:
            actions = {iid: rng.randint(0, N_PHASES - 1) for iid in env.inter_ids}
        _, _, done, _ = env.step(actions)

    tt = env.get_average_travel_time()
    tt_completed_only = env.get_completed_only_travel_time()
    tp = env.get_throughput()
    total_vehicles = len(env.all_spawned_vehicles)

    print(f"  Travel Time medio:  {tt:.2f} s  (solo arrivati: {tt_completed_only:.2f} s)")
    print(f"  Throughput:         {tp} / {total_vehicles} veicoli ({100.0 * tp / max(1, total_vehicles):.1f}%)")

    summary = {
        "config": args.config,
        "config_basename": config_basename,
        "policy": args.policy,
        "seed": args.seed if args.policy == "random" else None,
        "avg_travel_time": round(tt, 4),
        "avg_travel_time_completed_only": round(tt_completed_only, 4),
        "throughput": tp,
        "total_vehicles": total_vehicles,
        "throughput_pct": round(100.0 * tp / max(1, total_vehicles), 2),
    }
    with open(os.path.join(args.output_dir, f"summary_{config_basename}.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # ── Pulizia ──────────────────────────────────────────────────────────────
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    temp_roadnet_log = os.path.join(args.output_dir, "roadnet.log")
    if os.path.exists(temp_roadnet_log):
        os.rename(temp_roadnet_log, os.path.join(args.output_dir, "roadnet_log.json"))

    print(f"  Salvato: {args.output_dir}/replay.txt + roadnet_log.json + summary_{config_basename}.json\n")


if __name__ == "__main__":
    main()
