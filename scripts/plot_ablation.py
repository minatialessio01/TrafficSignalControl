import json
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot Ablation Study Results")
    parser.add_argument("--input", default="results/ablation/ablation_results.json", help="Path to ablation JSON")
    parser.add_argument("--output", default="results/ablation", help="Directory to save plots")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} non trovato.")
        return

    with open(args.input, "r") as f:
        results = json.load(f)

    models = [r["model"] for r in results]
    avg_tt = [r["avg_travel_time"] for r in results]
    best_tt = [r["best_travel_time"] for r in results]
    avg_tp = [r["avg_throughput"] for r in results]

    os.makedirs(args.output, exist_ok=True)

    # 1. Travel Time Plot (Bar Chart)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(models))
    width = 0.35

    ax.bar([i - width/2 for i in x], avg_tt, width, label='Avg Travel Time', color='#ff7f0e')
    ax.bar([i + width/2 for i in x], best_tt, width, label='Best Travel Time', color='#2ca02c')

    ax.set_ylabel('Travel Time (s)')
    ax.set_title('Ablation Study: Travel Time (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    tt_path = os.path.join(args.output, "ablation_travel_time.png")
    plt.savefig(tt_path)
    print(f"Grafico Travel Time salvato in {tt_path}")
    plt.close()

    # 2. Throughput Plot (Bar Chart)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.bar(x, avg_tp, width=0.5, color='#1f77b4')

    ax.set_ylabel('Throughput (vehicles)')
    ax.set_title('Ablation Study: Throughput (Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    tp_path = os.path.join(args.output, "ablation_throughput.png")
    plt.savefig(tp_path)
    print(f"Grafico Throughput salvato in {tp_path}")
    plt.close()

if __name__ == "__main__":
    main()
