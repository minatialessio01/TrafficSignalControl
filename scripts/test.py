"""
Script di test e valutazione per MetaSTGAT / STGAT / FixedTime.

Carica un modello addestrato e lo valuta per N episodi,
calcolando le metriche finali (travel time, throughput) come da Section 5.4:
  "the average value of the last ten tests as the final result"

Uso:
  python scripts/test.py \\
      --config configs/synthetic_4x4_config1.json \\
      --checkpoint results/metastgat_config1/best_model.pt \\
      --model MetaSTGAT

Output:
  - Stampa a schermo le metriche per ogni episodio
  - Salva i risultati in results/<run>/test_results.json
"""

import argparse
import json
import os
import sys
import collections
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.environment.cityflow_env import CityFlowEnv
from src.models.metastgat import MetaSTGAT
from src.models.stgat import STGAT
from src.agents.dqn_agent import DQNAgent
from src.agents.fixedtime_agent import FixedTimeAgent
from src.agents.maxpressure_agent import MaxPressureAgent


# Stessi default del training
DEFAULTS = {
    "hidden_dim":    64,
    "num_heads":     4,
    "num_neighbors": 4,
    "history_len":   5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valutazione MetaSTGAT / STGAT / FixedTime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True,
                        help="File config CityFlow")
    parser.add_argument("--model", default="MetaSTGAT",
                        choices=["MetaSTGAT", "STGAT", "FixedTime"])
    parser.add_argument("--checkpoint", default=None,
                        help="Percorso del checkpoint .pt (non serve per FixedTime)")
    parser.add_argument("--n-eval", type=int, default=10,
                        help="Numero di episodi di valutazione (paper: 10)")
    parser.add_argument("--output", default=None,
                        help="Directory output per i risultati")
    parser.add_argument("--hidden-dim",    type=int, default=DEFAULTS["hidden_dim"])
    parser.add_argument("--num-heads",     type=int, default=DEFAULTS["num_heads"])
    parser.add_argument("--num-neighbors", type=int, default=DEFAULTS["num_neighbors"])
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def evaluate(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"\n=== Valutazione {args.model} ===")
    print(f"Config: {args.config}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")

    # ── Ambiente ───────────────────────────────────────────────────────────
    # Impostiamo saveReplay a False per i primi N-1 episodi
    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg["saveReplay"] = False
    
    base_dir = cfg.get("dir", "./")
    output_dir = args.output or os.path.dirname(args.checkpoint or "results/test")
    os.makedirs(output_dir, exist_ok=True)
    
    # Calcola il percorso relativo da base_dir (es. data/) a output_dir
    rel_out_dir = os.path.relpath(output_dir, base_dir)
    cfg["replayLogFile"] = os.path.join(rel_out_dir, "replay.txt").replace("\\", "/")
    cfg["roadnetLogFile"] = os.path.join(rel_out_dir, "roadnet.log").replace("\\", "/")
    
    temp_config_path = os.path.join(output_dir, "temp_test_config.json")
    with open(temp_config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Rimosso il salvataggio del "vero roadnet json" di configurazione,
    # perché al frontend serve il log pre-renderizzato generato dall'engine (roadnet.log).

    env = CityFlowEnv(temp_config_path, num_neighbors=args.num_neighbors)
    edge_index = env.get_edge_index().to(device)

    # Funzione helper per ricaricare l'ambiente con saveReplay=True
    def reload_env_for_replay():
        nonlocal env
        print("  -> Riattivazione saveReplay per l'ultimo episodio...")
        cfg["saveReplay"] = True
        with open(temp_config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        # Forza il garbage collector a chiudere il vecchio engine C++
        del env 
        import gc
        gc.collect()
        return CityFlowEnv(temp_config_path, num_neighbors=args.num_neighbors)

    travel_times = []
    throughputs = []
    
    phase_counts = collections.Counter()

    # ── FixedTime ──────────────────────────────────────────────────────────
    if args.model == "FixedTime":
        agent = FixedTimeAgent(n_phases=env.action_space_n)

        for ep in range(1, args.n_eval + 1):
            if ep == args.n_eval:
                env = reload_env_for_replay()
            
            obs = env.reset()
            agent.reset(env.inter_ids)
            done = False
            step = 0
            while not done:
                actions = agent.select_actions(env.inter_ids, current_step=step)
                for p in actions.values():
                    phase_counts[p] += 1
                obs, _, done, _ = env.step(actions)
                step += 1

            tt = env.get_average_travel_time()
            tp = env.get_throughput()
            travel_times.append(tt)
            throughputs.append(tp)
            print(f"  Ep {ep:>3}/{args.n_eval}: TT={tt:.2f}s | TP={tp}")

    else:
        # ── DQN Modelli ─────────────────────────────────────────────────────
        if args.model == "MetaSTGAT":
            model = MetaSTGAT(
                state_dim=env.observation_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                n_actions=env.action_space_n,
                spatial_meta_dim=env.spatial_meta_dim,
                temporal_meta_dim=env.temporal_meta_dim,
                meta_hidden_dim=args.hidden_dim,
            )
        else:
            model = STGAT(
                state_dim=env.observation_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                n_actions=env.action_space_n,
            )

        agent = DQNAgent(model=model, n_intersections=env.n_intersections, device=device)

        if args.checkpoint:
            agent.load_checkpoint(args.checkpoint)
        else:
            print("[WARN] Nessun checkpoint specificato. Uso il modello non addestrato.")

        # Epsilon = 0 per greedy puro durante test
        agent.epsilon = 0.0
        agent.model.eval()

        for ep in range(1, args.n_eval + 1):
            if ep == args.n_eval:
                env = reload_env_for_replay()
            
            obs = env.reset()
            agent.reset_hidden()
            state_history = {iid: [] for iid in env.inter_ids}
            done = False

            while not done:
                spatial_meta = None
                temporal_meta = None
                if args.model == "MetaSTGAT":
                    spatial_meta = {
                        iid: env.get_spatial_meta_features(iid)
                        for iid in env.inter_ids
                    }
                    temporal_meta = {
                        iid: env.get_temporal_meta_features(
                            iid, history=state_history[iid],
                            history_len=DEFAULTS["history_len"]
                        )
                        for iid in env.inter_ids
                    }

                with torch.no_grad():
                    actions = agent.select_actions(
                        states=obs,
                        edge_index=edge_index,
                        inter_ids=env.inter_ids,
                        inter_id_to_idx=env.inter_id_to_idx,
                        spatial_meta=spatial_meta,
                        temporal_meta=temporal_meta,
                        invalid_actions=env.get_invalid_actions()
                    )

                for p in actions.values():
                    phase_counts[p] += 1
                obs, _, done, _ = env.step(actions)

                for iid in env.inter_ids:
                    state_history[iid].append(obs[iid])
                    if len(state_history[iid]) > DEFAULTS["history_len"]:
                        state_history[iid].pop(0)

            tt = env.get_average_travel_time()
            tp = env.get_throughput()
            travel_times.append(tt)
            throughputs.append(tp)
            print(f"  Ep {ep:>3}/{args.n_eval}: TT={tt:.2f}s | TP={tp}")

    # ── Risultati finali ────────────────────────────────────────────────────
    avg_tt = float(np.mean(travel_times))
    avg_tp = float(np.mean(throughputs))

    print(f"\n{'─'*50}")
    print(f"  RISULTATI FINALI ({args.n_eval} episodi)")
    print(f"  Travel Time medio:  {avg_tt:.2f} s")
    print(f"  Throughput medio:   {avg_tp:.1f} veicoli")
    print(f"{'─'*50}\n")

    results = {
        "model": args.model,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "n_eval": args.n_eval,
        "avg_travel_time": round(avg_tt, 4),
        "avg_throughput": round(avg_tp, 1),
        "travel_times": [round(t, 4) for t in travel_times],
        "throughputs": throughputs,
    }
    
    # ── Baseline Evaluation per Confronto Grafico ────────────────────────────
    baseline_results = {args.model: (avg_tt, avg_tp)}
    
    # Disabilitiamo saveReplay per i test delle baseline
    cfg["saveReplay"] = False
    with open(temp_config_path, "w") as f:
        json.dump(cfg, f, indent=2)
        
    if args.model != "FixedTime":
        print("\n[Baseline] Valutazione FixedTime...")
        ft_agent = FixedTimeAgent(n_phases=env.action_space_n)
        ft_env = CityFlowEnv(temp_config_path, num_neighbors=args.num_neighbors)
        ft_tt_list, ft_tp_list = [], []
        for ep in range(1, args.n_eval + 1):
            obs = ft_env.reset()
            ft_agent.reset(ft_env.inter_ids)
            done = False
            step = 0
            while not done:
                actions = ft_agent.select_actions(ft_env.inter_ids, current_step=step)
                obs, _, done, _ = ft_env.step(actions)
                step += 1
            ft_tt_list.append(ft_env.get_average_travel_time())
            ft_tp_list.append(ft_env.get_throughput())
            print(f"  Ep {ep:>3}/{args.n_eval}: TT={ft_tt_list[-1]:.2f}s | TP={ft_tp_list[-1]}")
        baseline_results["FixedTime"] = (float(np.mean(ft_tt_list)), float(np.mean(ft_tp_list)))
        results["baseline_fixedtime_tt"] = round(baseline_results["FixedTime"][0], 4)
        results["baseline_fixedtime_tp"] = round(baseline_results["FixedTime"][1], 1)
        del ft_env
        import gc; gc.collect()

    if args.model != "MaxPressure":
        print("\n[Baseline] Valutazione MaxPressure...")
        mp_agent = MaxPressureAgent(n_phases=env.action_space_n)
        mp_env = CityFlowEnv(temp_config_path, num_neighbors=args.num_neighbors)
        mp_tt_list, mp_tp_list = [], []
        for ep in range(1, args.n_eval + 1):
            obs = mp_env.reset()
            mp_agent.reset(mp_env.inter_ids)
            done = False
            while not done:
                actions = mp_agent.select_actions(states=obs, inter_ids=mp_env.inter_ids)
                obs, _, done, _ = mp_env.step(actions)
            mp_tt_list.append(mp_env.get_average_travel_time())
            mp_tp_list.append(mp_env.get_throughput())
            print(f"  Ep {ep:>3}/{args.n_eval}: TT={mp_tt_list[-1]:.2f}s | TP={mp_tp_list[-1]}")
        baseline_results["MaxPressure"] = (float(np.mean(mp_tt_list)), float(np.mean(mp_tp_list)))
        results["baseline_maxpressure_tt"] = round(baseline_results["MaxPressure"][0], 4)
        results["baseline_maxpressure_tp"] = round(baseline_results["MaxPressure"][1], 1)
        del mp_env
        import gc; gc.collect()

    # ── Salva i risultati e Grafici ──────────────────────────────────────────
    output_dir = args.output or os.path.dirname(args.checkpoint or "results/test")
    os.makedirs(output_dir, exist_ok=True)
    
    if len(baseline_results) > 1:
        plot_baseline_comparison(baseline_results, output_dir, os.path.basename(args.config))

    # Salva i risultati
    output_dir = args.output or os.path.dirname(args.checkpoint or "results/test")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"test_results_{args.model.lower()}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
        
    # Salva il plot delle fasi
    if sum(phase_counts.values()) > 0:
        total_steps = sum(phase_counts.values())
        phases = sorted(phase_counts.keys())
        percentages = [phase_counts[p] / total_steps * 100 for p in phases]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(phases, percentages, color='skyblue', edgecolor='black')
        plt.xlabel('Fase Semaforica (Indice)')
        plt.ylabel('Percentuale di Scelta (%)')
        plt.title(f'Distribuzione Fasi - {args.model}\nConfig: {os.path.basename(args.config)}')
        plt.xticks(phases)
        
        for bar, perc in zip(bars, percentages):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{perc:.1f}%', ha='center', va='bottom', fontsize=10)
            
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "phase_percentages.png")
        plt.savefig(plot_path)
        print(f"  Plot fasi salvato in: {plot_path}")
    print(f"  Risultati salvati in: {results_path}")

    # Pulisce i file temporanei
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    # Rinomina il log renderizzato della roadnet per il frontend
    temp_roadnet_log = os.path.join(output_dir, "roadnet.log")
    if os.path.exists(temp_roadnet_log):
        os.rename(temp_roadnet_log, os.path.join(output_dir, "roadnet_log.json"))

    return results



def plot_baseline_comparison(results_dict, output_dir, config_name=None):
    """
    results_dict: dict { "ModelName": (avg_tt, avg_tp) }
    """
    models = list(results_dict.keys())
    tts = [results_dict[m][0] for m in models]
    tps = [results_dict[m][1] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    title = "Confronto Modelli — Valutazione Finale"
    if config_name:
        title += f"\nConfig: {config_name}"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    colors = ['steelblue', 'mediumseagreen', 'orchid', 'coral', 'gold']
    
    # Travel Time Plot
    bars1 = ax1.bar(models, tts, color=colors[:len(models)], edgecolor='black')
    ax1.set_ylabel('Average Travel Time (s)')
    ax1.set_title('Travel Time ↓ (lower is better)')
    ax1.grid(axis='y', linestyle='-', alpha=0.3)
    # Add values on top
    for bar, val in zip(bars1, tts):
        ax1.text(bar.get_x() + bar.get_width()/2, val + (max(tts)*0.02), f'{val:.1f}s', 
                 ha='center', va='bottom', fontweight='bold')
    ax1.set_ylim(0, max(tts) * 1.15)

    # Throughput Plot
    bars2 = ax2.bar(models, tps, color=colors[:len(models)], edgecolor='black')
    ax2.set_ylabel('Throughput (vehicles)')
    ax2.set_title('Throughput ↑ (higher is better)')
    ax2.grid(axis='y', linestyle='-', alpha=0.3)
    # Add values on top
    for bar, val in zip(bars2, tps):
        ax2.text(bar.get_x() + bar.get_width()/2, val + (max(tps)*0.02), f'{int(val)}', 
                 ha='center', va='bottom', fontweight='bold')
    ax2.set_ylim(0, max(tps) * 1.15)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "baseline_comparison.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Grafico di confronto salvato in: {plot_path}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
