"""
Script di training principale per MetaSTGAT.

Implementa l'Algorithm 1 dell'articolo con supporto a:
  - Checkpoint automatico ogni 10 episodi
  - Resume da checkpoint (--resume)
  - Stop a episodio specifico (--stop-at)
  - Interruzione pulita con Ctrl+C (salva automaticamente)
  - Tutti i modelli: MetaSTGAT, STGAT, FixedTime

Uso:
  # Training standard
  python scripts/train.py --config configs/synthetic_4x4_config1.json --model MetaSTGAT

  # Fermarsi all'episodio 50
  python scripts/train.py --config configs/synthetic_4x4_config1.json --stop-at 50

  # Riprendere da un checkpoint
  python scripts/train.py --config configs/synthetic_4x4_config1.json \\
      --resume results/metastgat_config1/checkpoint_ep0050.pt

  # Premi Ctrl+C in qualsiasi momento per interrompere — viene salvato un checkpoint
"""

import argparse
import csv
import json
import os
import sys
import time

# Aggiunge la root del progetto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.environment.cityflow_env import CityFlowEnv
from src.models.metastgat import MetaSTGAT
from src.models.stgat import STGAT
from src.agents.dqn_agent import DQNAgent
from src.agents.fixedtime_agent import FixedTimeAgent
from src.utils.logger import TrainingLogger
from src.utils.metrics import EpisodeMetrics, RunningMetrics


# ─── Iperparametri (dall'articolo, Section 5.1) ───────────────────────────────
DEFAULTS = {
    "episodes":            100,    # numero totale di episodi (default test: 100)
    "updates_per_ep":      100,    # aggiornamenti per episodio
    "batch_size":          20,     # batch size (paper: 20)
    "gamma":               0.85,   # discount factor (paper: 0.85)
    "lr":                  1e-3,   # learning rate (non spec. → 1e-3)
    "buffer_size":         2_400,  # replay buffer (modificato per limitarlo a 2400 come richiesto)
    "min_buffer_size":     1_000,  # minimum size prima del training (paper: 1,000)
    "hidden_dim":          64,     # hidden dim (non spec. → 64)
    "num_heads":           4,      # teste attenzione (Fig. 10b → 4)
    "num_neighbors":       4,      # vicini (Fig. 10a → 4)
    "epsilon_start":       0.9,    # epsilon iniziale (non spec. → 0.9)
    "epsilon_end":         0.01,   # epsilon finale (non spec. → 0.01)
    "epsilon_decay":       0.7985, # decay per episodio (raggiunge 0.01 in 20 episodi)
    "history_len":         5,      # lunghezza storico TMK (non spec. → 5)
    "checkpoint_interval": 10,     # salva ogni N episodi (non spec. → 10)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training MetaSTGAT per Traffic Signal Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Configurazione obbligatoria ─────────────────────────────────────────
    parser.add_argument("--config", type=str, default="configs/benchmark_config.json",
                        help="Path al file config.json di CityFlow")
    parser.add_argument("--model", default="MetaSTGAT",
                        choices=["MetaSTGAT", "STGAT", "FixedTime"],
                        help="Modello da allenare")
    parser.add_argument("--output", default="auto",
                        help="Directory di output per checkpoint e log. "
                             "Default 'auto': crea una cartella con timestamp in results/")

    # ── Controllo training ──────────────────────────────────────────────────
    parser.add_argument("--episodes", type=int, default=DEFAULTS["episodes"],
                        help="Numero totale di episodi di training")
    parser.add_argument("--stop-at", type=int, default=None,
                        metavar="N",
                        help="Interrompi training dopo l'episodio N "
                             "(es. --stop-at 50). Salva un checkpoint.")
    parser.add_argument("--resume", default=None,
                        metavar="CHECKPOINT",
                        help="Percorso del checkpoint da cui riprendere "
                             "(es. results/run/checkpoint_ep0050.pt)")

    # ── Iperparametri ───────────────────────────────────────────────────────
    parser.add_argument("--batch-size",    type=int,   default=DEFAULTS["batch_size"])
    parser.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    parser.add_argument("--gamma",         type=float, default=DEFAULTS["gamma"])
    parser.add_argument("--hidden-dim",    type=int,   default=DEFAULTS["hidden_dim"])
    parser.add_argument("--num-heads",     type=int,   default=DEFAULTS["num_heads"])
    parser.add_argument("--num-neighbors", type=int,   default=DEFAULTS["num_neighbors"])
    parser.add_argument("--epsilon-start", type=float, default=DEFAULTS["epsilon_start"])
    parser.add_argument("--epsilon-end",   type=float, default=DEFAULTS["epsilon_end"])
    parser.add_argument("--epsilon-decay", type=float, default=DEFAULTS["epsilon_decay"])

    # ── Misc ────────────────────────────────────────────────────────────────
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--device",  default=None,
                        help="Dispositivo PyTorch (es. cpu, cuda:0). "
                             "Auto-rilevato se non specificato.")
    parser.add_argument("--verbose", action="store_true",
                        help="Output dettagliato per ogni step")

    return parser.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def format_eta(seconds: float) -> str:
    """Formatta un numero di secondi come stringa leggibile (es. '1h 23m 45s')."""
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h {m:02d}m"





def generate_comparison_plots(args, env, rl_agent, edge_index, logger):
    """
    Genera due grafici al termine del training:
      1. training_curves.png  — Travel Time, Throughput e Loss vs Episodio
      2. baseline_comparison.png — Bar chart MetaSTGAT vs FixedTime vs MaxPressure

    I grafici vengono salvati nella stessa cartella del training log.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # backend non-interattivo (nessuna finestra)
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib non trovato. Installa con: pip install matplotlib")
        return

    print(f"\n{'='*60}")
    print("  GENERAZIONE GRAFICI")
    print(f"{'='*60}")

    # ── Lettura CSV ────────────────────────────────────────────────────────────
    csv_episodes, csv_tt, csv_tp, csv_loss = [], [], [], []
    try:
        with open(logger.csv_path, newline="") as f:
            for row in csv.DictReader(f):
                csv_episodes.append(float(row["episode"]))
                csv_tt.append(float(row["travel_time"]))
                csv_tp.append(float(row["throughput"]))
                csv_loss.append(float(row.get("avg_loss", 0)))
    except Exception as exc:
        print(f"[Plot] Impossibile leggere il CSV: {exc}")
        return

    def moving_avg(data, w=10):
        """Media mobile centrata a destra (causal)."""
        out = []
        for i in range(len(data)):
            out.append(float(np.mean(data[max(0, i - w + 1): i + 1])))
        return out

    # ── Grafico 1: curve di training ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Training Progress — {args.model}\nConfig: {os.path.basename(args.config)}", fontsize=14, fontweight="bold")

    # Travel Time
    ax = axes[0]
    ax.plot(csv_episodes, csv_tt, alpha=0.25, color="steelblue", linewidth=0.8, label="raw")
    ax.plot(csv_episodes, moving_avg(csv_tt), color="steelblue", linewidth=2.0, label="MA(10)")
    ax.set_ylabel("Travel Time (s)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Throughput
    ax = axes[1]
    ax.plot(csv_episodes, csv_tp, alpha=0.25, color="forestgreen", linewidth=0.8)
    ax.plot(csv_episodes, moving_avg(csv_tp), color="forestgreen", linewidth=2.0, label="MA(10)")
    ax.set_ylabel("Throughput (vehicles)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Loss (escludi episodi con loss=0, ovvero prima che il buffer sia pieno)
    ax = axes[2]
    ep_loss = [(e, l) for e, l in zip(csv_episodes, csv_loss) if l > 1e-9]
    if ep_loss:
        ep_x, loss_y = zip(*ep_loss)
        ax.plot(ep_x, loss_y, alpha=0.25, color="crimson", linewidth=0.8)
        ax.plot(ep_x, moving_avg(list(loss_y)), color="crimson", linewidth=2.0, label="MA(10)")
        ax.legend(loc="upper right", fontsize=9)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Episode")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(args.output, "training_curves.png")
    plt.savefig(curves_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Curve di training  → {curves_path}")
    print("[Plot] (La valutazione delle baseline è stata delegata allo script test.py)")


def build_model(args, env: CityFlowEnv) -> torch.nn.Module:
    """Costruisce il modello in base all'argomento --model."""
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
    elif args.model == "STGAT":
        model = STGAT(
            state_dim=env.observation_dim,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            n_actions=env.action_space_n,
        )
    else:
        raise ValueError(f"Modello sconosciuto: {args.model}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {args.model} | Parametri: {n_params:,}")
    return model


def run_training(args):
    """Loop principale di training."""
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Cartella di output auto-timestampata se non specificata ────────────────
    if args.output == "auto":
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/{args.model.lower()}_{ts}"

    os.makedirs(args.output, exist_ok=True)
    
    # ── Salva configurazione e iperparametri ──────────────────────────────────
    config_out = os.path.join(args.output, "training_config.json")
    save_data = vars(args).copy()
    save_data["buffer_size"] = DEFAULTS["buffer_size"]
    with open(config_out, "w") as f:
        json.dump(save_data, f, indent=4)

    print(f"[Config] Device: {device}")
    print(f"[Config] Modello: {args.model}")
    print(f"[Config] Config CityFlow: {args.config}")
    print(f"[Config] Output: {args.output}")

    # ── FixedTime: nessun training ──────────────────────────────────────────
    if args.model == "FixedTime":
        run_fixedtime(args, device)
        return

    # ── Inizializza ambiente ────────────────────────────────────────────────
    print("\n[Env] Preparazione configurazione CityFlow...")
    with open(args.config, 'r') as f:
        custom_config = json.load(f)
    
    # DISABILITA IL REPLAY DURANTE IL TRAINING PER RISPARMIARE SPAZIO
    custom_config["saveReplay"] = False
    
    base_dir = custom_config.get("dir", "./")
    
    # Calcola il percorso relativo da base_dir (es. data/) a args.output
    rel_out_dir = os.path.relpath(args.output, base_dir)
    
    # Non modifichiamo "dir", roadnetFile e flowFile per non confondere il motore C++
    custom_config["replayLogFile"] = os.path.join(rel_out_dir, "replay.txt").replace("\\", "/")
    custom_config["roadnetLogFile"] = os.path.join(rel_out_dir, "roadnet.log").replace("\\", "/")
    
    custom_config_path = os.path.join(args.output, "cityflow_config.json")
    with open(custom_config_path, 'w') as f:
        json.dump(custom_config, f, indent=4)

    print("[Env] Inizializzazione CityFlow...")
    env = CityFlowEnv(custom_config_path, num_neighbors=args.num_neighbors)
    edge_index = env.get_edge_index().to(device)

    print(f"[Env] Intersezioni: {env.n_intersections}")
    print(f"[Env] State dim: {env.observation_dim}")
    print(f"[Env] Actions: {env.action_space_n}")

    # ── Costruisci modello e agente ─────────────────────────────────────────
    model = build_model(args, env)
    agent = DQNAgent(
        model=model,
        n_intersections=env.n_intersections,
        n_actions=env.action_space_n,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=DEFAULTS["buffer_size"],
        batch_size=args.batch_size,
        device=device,
    )

    # ── Logger ─────────────────────────────────────────────────────────────
    run_name = f"{args.model}_{os.path.basename(args.config).replace('.json', '')}"
    logger = TrainingLogger(args.output, run_name, resume=bool(args.resume))
    running_metrics = RunningMetrics(window=10)

    # ── Resume da checkpoint ────────────────────────────────────────────────
    start_episode = 0
    if args.resume:
        start_episode = agent.load_checkpoint(args.resume)
        print(f"[Resume] Ripresa dall'episodio {start_episode + 1}")

    # ── Calcola l'episodio finale ───────────────────────────────────────────
    end_episode = args.stop_at if args.stop_at else args.episodes
    end_episode = min(end_episode, args.episodes)

    if start_episode >= end_episode:
        print(f"[!] Training già completato fino all'episodio {start_episode}.")
        print(f"    L'episodio di stop ({end_episode}) è già stato raggiunto.")
        return

    print(f"\n{'='*60}")
    print(f"  TRAINING: episodi {start_episode + 1} → {end_episode}")
    print(f"  Ctrl+C per interrompere e salvare checkpoint")
    print(f"{'='*60}\n")

    training_start_time = time.time()  # per il calcolo dell'ETA

    def run_episode():
        obs = env.reset()
        agent.reset_hidden()
        ep_metrics = EpisodeMetrics()
        state_history = {iid: [] for iid in env.inter_ids}

        n_steps = 0
        done = False

        while not done:
            # Feature meta per MetaSTGAT
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

            # Seleziona azioni (epsilon-greedy)
            actions = agent.select_actions(
                states=obs,
                edge_index=edge_index,
                inter_ids=env.inter_ids,
                inter_id_to_idx=env.inter_id_to_idx,
                spatial_meta=spatial_meta,
                temporal_meta=temporal_meta,
            )

            # Esegui step nell'ambiente
            next_obs, rewards, done, info = env.step(actions)

            # Memorizza transizioni nel replay buffer
            agent.store_transitions(
                states=obs,
                actions=actions,
                rewards=rewards,
                next_states=next_obs,
                inter_ids=env.inter_ids,
                spatial_meta=spatial_meta,
                temporal_meta=temporal_meta,
            )

            # Aggiorna storico per TMK
            for iid in env.inter_ids:
                state_history[iid].append(obs[iid])
                if len(state_history[iid]) > DEFAULTS["history_len"]:
                    state_history[iid].pop(0)

            # Metriche passo
            total_step_reward = sum(rewards.values())
            ep_metrics.update(
                travel_time=info["avg_travel_time"],
                throughput=info["vehicles_running"],
                reward=total_step_reward
            )

            obs = next_obs
            n_steps += 1

            if args.verbose and n_steps % 10 == 0:
                print(f"    Step {n_steps}: TT={info['avg_travel_time']:.1f}s")

        # ── Chiusura Traiettoria ───────────────────────────────────────────────
        agent.replay_buffer.end_episode()
        return ep_metrics


    # === FASE DI WARM-UP ===
    if start_episode == 0:
        print("\n=== FASE DI WARM-UP (10 episodi casuali per riempire il buffer) ===")
        agent.epsilon = 1.0
        for w_ep in range(1, 11):
            run_episode()
            print(f"  Warm-up Ep {w_ep}/10 completato. (Buffer size: {len(agent.replay_buffer)})")
        agent.epsilon = args.epsilon_start
        print("=== FINE WARM-UP ===\n")

    # ── Loop di training principale (Algorithm 1) ────────────────────────────
    for episode in range(start_episode + 1, end_episode + 1):

        # ── Raccolta dati dell'episodio (Algorithm 1, line 4-8) ───────────
        ep_metrics = run_episode()
        
        # Salva l'epsilon attuale prima che agent.update lo modifichi per il prossimo episodio
        epsilon_used = getattr(agent, 'epsilon', 0.0)

        # ── Aggiornamento modello (Algorithm 1, line 9-14) ─────────────────
        loss = agent.update(
            edge_index=edge_index,
            n_updates=DEFAULTS["updates_per_ep"],
            min_buffer_size=DEFAULTS["min_buffer_size"]
        )
        ep_metrics.losses.append(loss or 0.0)

        # ── Metriche finali ────────────────────────────────────────────────
        travel_time = ep_metrics.final_travel_time
        throughput = ep_metrics.final_throughput
        running_metrics.add_episode(travel_time, throughput)

        # ── Salva il best model ────────────────────────────────────────────
        is_best = agent.save_best(
            path=logger.best_path,
            episode=episode,
            travel_time=travel_time
        )

        # ── Checkpoint periodico ───────────────────────────────────────────
        if logger.should_save_checkpoint(episode):
            agent.save_checkpoint(
                path=logger.checkpoint_path(episode),
                episode=episode,
                travel_time=travel_time,
                extra_info={
                    "running_avg_travel_time": running_metrics.last_n_avg_travel_time,
                    "epsilon": epsilon_used,
                }
            )

        # ── Log e stampa ───────────────────────────────────────────────────
        logger.log_episode(
            episode=episode,
            travel_time=travel_time,
            throughput=throughput,
            total_reward=ep_metrics.total_reward,
            avg_loss=ep_metrics.avg_loss,
            epsilon=epsilon_used,
            buffer_size=len(agent.replay_buffer)
        )

        # ── Copia Live per CityFlow Frontend ───────────────────────────────
        try:
            import shutil
            base_dir = env.config.get("dir", "./")
            
            # CityFlow engine lavora rispetto al CWD, quindi path_cwd è corretto.
            replay_path = os.path.join(base_dir, env.config.get("replayLogFile", "replay.txt"))
            roadnet_log_path = os.path.join(base_dir, env.config.get("roadnetLogFile", "roadnet.log"))
            
            # Crea la cartella frontend se non esiste
            frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CityFlow", "frontend")
            os.makedirs(frontend_dir, exist_ok=True)
            
            if os.path.exists(replay_path):
                shutil.copy(replay_path, os.path.join(frontend_dir, "replay.txt"))
            if os.path.exists(roadnet_log_path):
                # Il frontend web di default cerca "roadnet.json" per il log pre-renderizzato
                shutil.copy(roadnet_log_path, os.path.join(frontend_dir, "roadnet.json"))
        except Exception as e:
            pass
        # ETA: media mobile del tempo per episodio × episodi rimanenti
        elapsed = time.time() - training_start_time
        episodes_done = episode - start_episode
        avg_ep_time = elapsed / episodes_done
        remaining_eps = end_episode - episode
        eta_str = format_eta(avg_ep_time * remaining_eps) if remaining_eps > 0 else "done"

        logger.print_episode(
            episode=episode,
            total_episodes=args.episodes,
            travel_time=travel_time,
            throughput=throughput,
            loss=loss,
            epsilon=epsilon_used,
            total_reward=ep_metrics.total_reward,
            is_best=is_best,
            eta_str=eta_str
        )

        # ── Episodio Random Periodico ──────────────────────────────────────
        if episode % 10 == 0 and episode < args.episodes:
            print(f"\n[!] Esecuzione di 1 episodio random (senza aggiornamento pesi) per esplorazione...")
            old_epsilon = agent.epsilon
            agent.epsilon = 1.0
            run_episode()
            agent.epsilon = old_epsilon
            print(f"[!] Episodio random completato. Epsilon ripristinato a {agent.epsilon:.4f}\n")

        # ── Interruzione per Ctrl+C ────────────────────────────────────────
        if logger.interrupted:
            print(f"\n[!] Salvataggio checkpoint d'emergenza all'episodio {episode}...")
            agent.save_checkpoint(
                path=logger.checkpoint_path(episode),
                episode=episode,
                travel_time=travel_time,
                extra_info={"interrupted": True}
            )
            print(f"    Riprendi con: --resume {logger.checkpoint_path(episode)}")
            break

        # ── Stop a episodio specificato ────────────────────────────────────
        if args.stop_at and episode >= args.stop_at:
            print(f"\n[!] Raggiunto l'episodio di stop ({args.stop_at}).")
            agent.save_checkpoint(
                path=logger.checkpoint_path(episode),
                episode=episode,
                travel_time=travel_time,
                extra_info={"stop_at": args.stop_at}
            )
            print(f"    Riprendi con: --resume {logger.checkpoint_path(episode)}")
            break

    # ── Fine training ──────────────────────────────────────────────────────
    logger.close()
    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETATO")
    print(f"  Episodi completati: {episode}")
    print(f"  Best travel time: {running_metrics.best_travel_time:.2f}s")
    print(f"  Last 10 avg travel time: {running_metrics.last_n_avg_travel_time:.2f}s")
    print(f"  Best throughput: {running_metrics.best_throughput}")
    print(f"  Modello migliore salvato in: {logger.best_path}")
    print(f"  Log salvato in: {logger.csv_path}")
    print(f"{'='*60}\n")

    # ── Grafici di confronto (solo se training completato, non interrotto) ──
    if not logger.interrupted:
        generate_comparison_plots(args, env, agent, edge_index, logger)

    # ── Pulizia file temporanei (config engine) ───────────────
    try:
        temp_config = os.path.join(args.output, "cityflow_config.json")
        if os.path.exists(temp_config):
            os.remove(temp_config)
        
        # Rinominiamo roadnet.log in roadnet_log.json per chiarezza (è quello richiesto dal frontend)
        temp_roadnet_log = os.path.join(args.output, "roadnet.log")
        if os.path.exists(temp_roadnet_log):
            os.rename(temp_roadnet_log, os.path.join(args.output, "roadnet_log.json"))
    except Exception as e:
        print(f"  [WARN] Errore durante la pulizia dei file temporanei: {e}")


def run_fixedtime(args, device):
    """Valuta il baseline FixedTime senza training."""
    print("\n[FixedTime] Valutazione baseline FixedTime...")
    env = CityFlowEnv(args.config, num_neighbors=args.num_neighbors)
    agent = FixedTimeAgent(n_phases=env.action_space_n)

    n_eval = min(args.episodes, 10)  # bastano 10 episodi per FixedTime
    travel_times = []

    for ep in range(1, n_eval + 1):
        obs = env.reset()
        agent.reset(env.inter_ids)
        done = False
        step = 0
        while not done:
            actions = agent.select_actions(env.inter_ids, current_step=step)
            obs, rewards, done, info = env.step(actions)
            step += 1

        tt = env.get_average_travel_time()
        travel_times.append(tt)
        print(f"  Ep {ep}/{n_eval}: TT={tt:.2f}s | TP={env.get_throughput()}")

    import statistics
    avg_tt = statistics.mean(travel_times)
    print(f"\n[FixedTime] Average Travel Time: {avg_tt:.2f}s")

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "fixedtime_results.json"), "w") as f:
        import json
        json.dump({"avg_travel_time": avg_tt, "travel_times": travel_times}, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
