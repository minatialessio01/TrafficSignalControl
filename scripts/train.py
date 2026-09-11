"""
Script di training principale per MetaSTGAT.

Implementa l'Algorithm 1 dell'articolo con supporto a:
  - Checkpoint automatico ogni 10 episodi
  - Resume da checkpoint (--resume)
  - Stop a episodio specifico (--stop-at)
  - Interruzione pulita con Ctrl+C (salva automaticamente)
  - Tutti i modelli: MetaSTGAT, STGAT, FixedTime
  - Flag di ablation per disabilitare singoli componenti del modello avanzato
  - Selezione automatica finale tra ultimo modello e best model, valutati su
    --select-best-config (config di validazione)

Training su una singola configurazione, warmup ed esplorazione:
  warmup_episodes episodi casuali non loggati (riempiono il buffer, non
  allenano nulla) -> episodi di training con epsilon che decade da
  epsilon_start a epsilon_end entro eps_fraction degli episodi totali -> ogni
  10 episodi, 1 episodio casuale non loggato (esplorazione ciclica).
  Warmup ed esplorazione ciclica sono disattivati da --no-warmup e
  --no-cyclic-exploration (attivi di default nei preset "paper" e
  "replay_stability", che replicano la procedura originale del paper senza
  queste aggiunte).

Uso:
  # Training standard (modello avanzato completo)
  python scripts/train.py --config configs/config_4x4_100m_train.json

  # Con selezione automatica finale su una config di validazione: a fine
  # training confronta final_model.pth col best_model.pt (valutati su
  # --select-best-config) e salva il vincitore come selected_model.pth
  # (usato in automatico da scripts/test.py).
  python scripts/train.py \\
      --config configs/config_4x4_100m_train.json --episodes 200 \\
      --select-best-config configs/config_4x4_100m_6k_flat.json \\
      --output-dir results/metastgat_pro

  # Training con preset ablation (Proposta 1 – Sottosistemi Funzionali)
  python scripts/train.py --config configs/config_4x4_100m_train.json --ablation environment
  python scripts/train.py --config configs/config_4x4_100m_train.json --ablation temporal
  python scripts/train.py --config configs/config_4x4_100m_train.json --ablation rl_core
  python scripts/train.py --config configs/config_4x4_100m_train.json --ablation replay_stability
  python scripts/train.py --config configs/config_4x4_100m_train.json --ablation paper

  # Output in cartella specifica
  python scripts/train.py --config configs/config_4x4_100m_train.json --output-dir results/metastgat_pro

  # Fermarsi all'episodio 50
  python scripts/train.py --config configs/config_4x4_100m_train.json --stop-at 50

  # Premi Ctrl+C in qualsiasi momento per interrompere — viene salvato un checkpoint
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Optional

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
    "episodes":            50,     # numero totale di episodi
    "updates_per_ep":      100,    # aggiornamenti per episodio (tornato a 100: config di training di
                                    # nuovo a 1800s, come il valore originale del paper)
    "batch_size":          20,     # batch size (paper: 20)
    "gamma":               0.85,   # discount factor (paper: 0.85)
    "lr":                  1e-3,   # learning rate (non spec. → 1e-3)
    "buffer_size":         4_800,  # replay buffer (~40 episodi di storia con la config da 1800s)
    "min_buffer_size":     1_000,  # minimum size prima del training (paper: 1,000)
    "hidden_dim":          64,     # hidden dim (non spec. → 64)
    "num_heads":           4,      # teste attenzione (Fig. 10b → 4)
    "num_neighbors":       4,      # vicini (Fig. 10a → 4)
    "epsilon_start":       0.9,    # epsilon iniziale (non spec. → 0.9)
    "epsilon_end":         0.01,   # epsilon finale (non spec. → 0.01)
    "epsilon_decay":       0.7985, # decay per episodio (raggiunge 0.01 in 20 episodi)
    "history_len":         5,      # lunghezza storico TMK (non spec. → 5)
    "checkpoint_interval": 10,     # salva ogni N episodi (non spec. → 10)
    # ── Warmup ed epsilon decay ────────────────────────────────────────────
    "warmup_episodes":     10,     # episodi random non loggati prima del training (riempiono il buffer)
    "eps_fraction":        0.6,    # l'agente raggiunge epsilon_end al 60% degli episodi totali
}


def _eps_decay_for(n_episodes: int, fraction: float, eps_start: float, eps_end: float) -> float:
    """Tasso di decadimento epsilon per raggiungere eps_end a `fraction` degli episodi."""
    target_ep = max(1, int(fraction * n_episodes))
    return (eps_end / eps_start) ** (1.0 / target_ep)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training MetaSTGAT per Traffic Signal Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Configurazione obbligatoria ─────────────────────────────────────────
    parser.add_argument("--config", type=str, default="configs/benchmark_config.json",
                        help="Path al file config.json di CityFlow")
    # [LIBSIGNAL ADDITION: Aggiunto CoLight alle scelte]
    parser.add_argument("--model", default="MetaSTGAT",
                        choices=["MetaSTGAT", "STGAT", "FixedTime", "CoLight"],
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

    # ── Selezione finale del modello (config di validazione) ──────────────────
    parser.add_argument("--select-best-config", default="configs/config_4x4_100m_6k_flat.json",
                        help="Config di validazione usata a fine training per scegliere tra "
                             "final_model.pth e best_model.pt.")
    parser.add_argument("--select-best-episodes", type=int, default=1,
                        help="Numero di episodi di valutazione (epsilon=0) per la selezione finale. "
                             "Default 1: stessa ragione di --n-eval in test.py — a epsilon=0, con "
                             "flow/seed CityFlow fissi, ripetere l'episodio non aggiunge varianza "
                             "da mediare, raddoppia solo il tempo (la selezione valuta 2 candidati).")
    parser.add_argument("--no-select-best", action="store_true",
                        help="Disattiva la selezione automatica finale tra final_model e best_model.")

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

    # ── Ablation – Preset ────────────────────────────────────────────────────
    parser.add_argument(
        "--ablation",
        default="pro",
        choices=["pro", "paper", "environment", "temporal", "rl_core", "replay_stability"],
        help="Preset ablation (imposta automaticamente i flag --no-*). "
             "pro=modello avanzato completo, paper=replica originale, "
             "environment=senza reward+visibilità+wait+mask, temporal=senza BPTT, "
             "rl_core=senza Double DQN, replay_stability=senza PER/Huber/grad-clip/warmup"
    )

    # ── Ablation – Flag Atomici (possono sovrascrivere il preset) ───────────
    parser.add_argument("--reward-mode", default=None,
                        choices=["custom", "paper"],
                        help="Formula reward: custom (weighted pressure) o paper (-P_i). "
                             "Se non specificato, usa il default del preset.")
    parser.add_argument("--no-vision-cutoff",     action="store_true",
                        help="Rimuove il cutoff dal campo visivo (VISION_CUTOFF_M, ~144m) (visibilità globale come nel paper)")
    parser.add_argument("--no-wait-vec",          action="store_true",
                        help="Stato a 20 dim senza wait_vec (solo n_vec + p_vec)")
    parser.add_argument("--no-action-mask",       action="store_true",
                        help="Rimuove la maschera anti-starvation (consente stessa fase ≥3 volte)")
    parser.add_argument("--no-bptt",              action="store_true",
                        help="Training single-step senza BPTT e senza burn-in (L=1)")
    parser.add_argument("--no-double-dqn",        action="store_true",
                        help="Usa DQN standard invece di Double DQN")
    parser.add_argument("--no-per",               action="store_true",
                        help="Buffer FIFO flat da 10k, campionamento uniforme, no IS weights")
    parser.add_argument("--no-huber",             action="store_true",
                        help="Usa MSE loss invece di Huber Loss")
    parser.add_argument("--no-soft-update",       action="store_true",
                        help="Hard copy del target network invece di Polyak averaging")
    parser.add_argument("--no-grad-clip",         action="store_true",
                        help="Disabilita gradient clipping")
    parser.add_argument("--no-warmup",            action="store_true",
                        help="Salta i 10 episodi di warm-up iniziali")
    parser.add_argument("--no-cyclic-exploration", action="store_true",
                        help="Salta l'episodio random periodico ogni 10 ep")
    parser.add_argument("--no-tanh-meta",         action="store_true",
                        help="Rimuove la Tanh finale dai meta-learner SMK/TMK")

    # ── Output directory esplicita ───────────────────────────────────────────
    parser.add_argument("--output-dir", default=None,
                        help="Cartella di output per checkpoint e log (sovrascrive --output). "
                             "Se non specificato, usa --output (default: auto-timestamp).")

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


def apply_ablation_preset(args):
    """
    Applica il preset --ablation impostando i flag --no-* corretti.
    I flag esplicitamente specificati dall'utente hanno la precedenza sul preset.

    Mappa preset (Proposta 1 – Sottosistemi Funzionali):
      pro              → nessun flag (modello avanzato completo)
      paper            → tutti i --no-* + reward-mode paper
      environment      → --reward-mode paper + --no-vision-cutoff + --no-wait-vec + --no-action-mask
      temporal         → --no-bptt
      rl_core          → --no-double-dqn
      replay_stability → --no-per + --no-huber + --no-soft-update + --no-grad-clip
                         + --no-warmup + --no-cyclic-exploration + --no-tanh-meta
    """
    preset = args.ablation

    def _set_if_not_explicit(attr, value):
        """Imposta l'attributo solo se l'utente non lo ha già specificato."""
        # store_true args partono da False; se sono True l'utente li ha esplicitamente settati.
        # reward_mode parte da None; se è None l'utente non l'ha specificato.
        if attr == "reward_mode":
            if getattr(args, attr) is None:
                setattr(args, attr, value)
        else:
            if not getattr(args, attr, False):
                setattr(args, attr, value)

    if preset == "pro":
        # Nessun flag da impostare — già tutti False di default
        if args.reward_mode is None:
            args.reward_mode = "custom"

    elif preset == "paper":
        if args.reward_mode is None:
            args.reward_mode = "paper"
        _set_if_not_explicit("no_vision_cutoff",      True)
        _set_if_not_explicit("no_wait_vec",           True)
        _set_if_not_explicit("no_action_mask",        True)
        _set_if_not_explicit("no_bptt",               True)
        _set_if_not_explicit("no_double_dqn",         True)
        _set_if_not_explicit("no_per",                True)
        _set_if_not_explicit("no_huber",              True)
        _set_if_not_explicit("no_soft_update",        True)
        _set_if_not_explicit("no_grad_clip",          True)
        _set_if_not_explicit("no_warmup",             True)
        _set_if_not_explicit("no_cyclic_exploration", True)
        _set_if_not_explicit("no_tanh_meta",          True)

    elif preset == "environment":
        # Ablation 1: spegne M1 (reward), M2 (cutoff campo visivo), M3 (wait_vec), M4 (action mask)
        if args.reward_mode is None:
            args.reward_mode = "paper"
        _set_if_not_explicit("no_vision_cutoff", True)
        _set_if_not_explicit("no_wait_vec",      True)
        _set_if_not_explicit("no_action_mask",   True)

    elif preset == "temporal":
        # Ablation 2: spegne M5 (BPTT + burn-in)
        _set_if_not_explicit("no_bptt", True)
        if args.reward_mode is None:
            args.reward_mode = "custom"

    elif preset == "rl_core":
        # Ablation 3: spegne M6 (Double DQN)
        _set_if_not_explicit("no_double_dqn", True)
        if args.reward_mode is None:
            args.reward_mode = "custom"

    elif preset == "replay_stability":
        # Ablation 4: spegne M7 (PER+IS), M8 (Huber+grad+soft), M9 (warmup+cyclic), M10 (Tanh)
        _set_if_not_explicit("no_per",                True)
        _set_if_not_explicit("no_huber",              True)
        _set_if_not_explicit("no_soft_update",        True)
        _set_if_not_explicit("no_grad_clip",          True)
        _set_if_not_explicit("no_warmup",             True)
        _set_if_not_explicit("no_cyclic_exploration", True)
        _set_if_not_explicit("no_tanh_meta",          True)
        if args.reward_mode is None:
            args.reward_mode = "custom"

    # Normalizza i nomi con trattino → underscore (argparse converte automaticamente,
    # ma li stampiamo per debug)
    print(f"[Ablation] preset='{preset}' | reward_mode={args.reward_mode} | "
          f"no_vision_cutoff={args.no_vision_cutoff} | no_wait_vec={args.no_wait_vec} | "
          f"no_action_mask={args.no_action_mask} | no_bptt={args.no_bptt} | "
          f"no_double_dqn={args.no_double_dqn} | no_per={args.no_per} | "
          f"no_huber={args.no_huber} | no_soft_update={args.no_soft_update} | "
          f"no_grad_clip={args.no_grad_clip} | no_warmup={args.no_warmup} | "
          f"no_cyclic_exploration={args.no_cyclic_exploration} | no_tanh_meta={args.no_tanh_meta}")


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

    # ── Titolo: solo il nome della config di training ────────────────────────
    config_name = os.path.basename(args.config)
    title_config_line = f"Config: {config_name}"

    # ── Grafico 1: curve di training ──────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Training Progress — {args.model}\n{title_config_line}", fontsize=14, fontweight="bold")

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


def _prepare_cityflow_env(config_path: str, args, tmp_name: str = "cityflow_config.json") -> CityFlowEnv:
    """
    Prepara il file di config temporaneo (replay/roadnet log rediretti in args.output)
    e costruisce il CityFlowEnv corrispondente, con i flag di ablation correnti.
    """
    with open(config_path, 'r') as f:
        custom_config = json.load(f)

    # DISABILITA IL REPLAY DURANTE IL TRAINING PER RISPARMIARE SPAZIO
    custom_config["saveReplay"] = False
    custom_config["seed"] = args.seed

    base_dir = custom_config.get("dir", "./")
    rel_out_dir = os.path.relpath(args.output, base_dir)
    custom_config["replayLogFile"] = os.path.join(rel_out_dir, "replay.txt").replace("\\", "/")
    custom_config["roadnetLogFile"] = os.path.join(rel_out_dir, "roadnet.log").replace("\\", "/")

    custom_config_path = os.path.join(args.output, tmp_name)
    with open(custom_config_path, 'w') as f:
        json.dump(custom_config, f, indent=4)

    env = CityFlowEnv(
        custom_config_path,
        num_neighbors=args.num_neighbors,
        reward_mode=args.reward_mode,
        use_vision_cutoff=not args.no_vision_cutoff,
        use_wait_vec=not args.no_wait_vec,
        use_action_mask=not args.no_action_mask,
    )
    return env


def run_final_selection(args, final_model_path: str, best_model_path: str, device) -> Optional[dict]:
    """
    Confronta a fine training final_model.pth (ultimo episodio in assoluto) e
    best_model.pt (miglior travel time di tutto il training) su
    --select-best-episodes episodi di --select-best-config, e salva il
    vincitore come selected_model.pth.

    scripts/test.py cerca selected_model.pth con priorita' massima, quindi il
    risultato di questa funzione e' gia' "pronto" per essere testato senza
    ulteriori argomenti.
    """
    import shutil

    if args.no_select_best:
        print("[Select] Selezione finale disattivata (--no-select-best).")
        return None

    if args.model not in ("MetaSTGAT", "STGAT"):
        print(f"[Select] Selezione finale non supportata per --model {args.model}. Salto.")
        return None

    selected_path = os.path.join(args.output, "selected_model.pth")

    if not os.path.exists(best_model_path):
        print("[Select] Nessun best_model.pt trovato per questo stage: uso final_model.pth.")
        shutil.copy(final_model_path, selected_path)
        return {"selected": "final_model", "reason": "no_best_model", "selected_path": selected_path}

    print(f"\n{'='*60}")
    print(f"  SELEZIONE MODELLO FINALE")
    print(f"  {args.select_best_episodes} episodi di valutazione su "
          f"{os.path.basename(args.select_best_config)}")
    print(f"{'='*60}")

    from scripts.test import evaluate  # import locale: evita import circolare a livello di modulo

    candidates = {"final_model": final_model_path, "best_model": best_model_path}
    scores = {}
    for name, ckpt in candidates.items():
        eval_args = argparse.Namespace(
            config=args.select_best_config,
            model=args.model,
            model_id=f"selection_{name}",
            checkpoint=ckpt,
            output_dir=os.path.join(args.output, "selection_eval", name),
            ablation=args.ablation,
            reward_mode=args.reward_mode,
            no_vision_cutoff=args.no_vision_cutoff,
            no_wait_vec=args.no_wait_vec,
            no_action_mask=args.no_action_mask,
            no_tanh_meta=args.no_tanh_meta,
            n_eval=args.select_best_episodes,
            hidden_dim=args.hidden_dim,
            num_heads=args.num_heads,
            num_neighbors=args.num_neighbors,
            device=str(device),
        )
        try:
            summary = evaluate(eval_args)
            scores[name] = summary["avg_travel_time"]
            print(f"  [Select] {name:<12} TT medio = {scores[name]:.2f}s  ({ckpt})")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Select] Errore durante la valutazione di '{name}': {e}")

    if not scores:
        print("[Select] Nessuna valutazione riuscita: uso final_model.pth come fallback.")
        shutil.copy(final_model_path, selected_path)
        return {"selected": "final_model", "reason": "evaluation_failed", "selected_path": selected_path}

    winner = min(scores, key=scores.get)
    winner_path = candidates[winner]
    shutil.copy(winner_path, selected_path)

    result = {
        "final_model_tt": scores.get("final_model"),
        "best_model_tt": scores.get("best_model"),
        "selected": winner,
        "selected_source": winner_path,
        "selected_path": selected_path,
        "eval_config": args.select_best_config,
        "n_eval_episodes": args.select_best_episodes,
    }
    with open(os.path.join(args.output, "selection_summary.json"), "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n[Select] Vincitore: {winner} (TT={scores[winner]:.2f}s) → salvato in {selected_path}")
    print(f"{'='*60}\n")
    return result


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
            use_tanh_meta=not args.no_tanh_meta,
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
    # ── Applica preset ablation (prima di qualsiasi altra cosa) ───────────────
    apply_ablation_preset(args)

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ── Cartella di output: --output-dir > --output > auto-timestamp ──────────
    if args.output_dir:
        args.output = args.output_dir
    elif args.output == "auto":
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/{args.ablation}_{ts}"

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

    # ── Inizializza ambiente ──────────────────────────────────────────────────
    print("\n[Env] Preparazione configurazione CityFlow...")
    env = _prepare_cityflow_env(args.config, args, tmp_name="cityflow_config.json")
    edge_index = env.get_edge_index().to(device)

    print(f"[Env] Intersezioni: {env.n_intersections}")
    print(f"[Env] State dim: {env.observation_dim}")
    print(f"[Env] Actions: {env.action_space_n}")

    # ── Episodi totali (singola configurazione) ─────────────────────────────
    total_episodes = args.episodes

    # ── Costruisci modello e agente ─────────────────────────────────────────
    # [LIBSIGNAL ADDITION: Gestione specifica per l'agente CoLight]
    if args.model == "CoLight":
        from src.agents.colight_agent import CoLightWrapperAgent
        agent = CoLightWrapperAgent(
            n_intersections=env.n_intersections,
            n_actions=env.action_space_n,
            lr=args.lr,
            device=device
        )
    else:
        model = build_model(args, env)

        # Epsilon decay: raggiunge epsilon_end a eps_fraction degli episodi totali.
        eps_decay = _eps_decay_for(args.episodes, DEFAULTS["eps_fraction"],
                                    args.epsilon_start, args.epsilon_end)

        # Dimensione buffer: paper usa 10k flat; il nostro avanzato usa 2400 seq
        buffer_sz = 10_000 if args.no_per else DEFAULTS["buffer_size"]
        agent = DQNAgent(
            model=model,
            n_intersections=env.n_intersections,
            n_actions=env.action_space_n,
            lr=args.lr,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
            epsilon_decay=eps_decay,
            buffer_size=buffer_sz,
            batch_size=args.batch_size,
            device=device,
            # ── Ablation flags ──────────────────────────────────────────────
            use_double_dqn=not args.no_double_dqn,
            use_per=not args.no_per,
            use_huber=not args.no_huber,
            use_soft_update=not args.no_soft_update,
            use_grad_clip=not args.no_grad_clip,
            use_bptt=not args.no_bptt,
            use_tanh_meta=not args.no_tanh_meta,
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
    end_episode = args.stop_at if args.stop_at else total_episodes
    end_episode = min(end_episode, total_episodes)

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
                invalid_actions=env.get_invalid_actions()
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
        agent.replay_buffer.end_episode(seq_len=agent.seq_len)
        return ep_metrics


    # === FASE DI WARM-UP ===
    n_warmup = DEFAULTS["warmup_episodes"]
    if start_episode == 0 and not args.no_warmup:
        print(f"\n=== FASE DI WARM-UP ({n_warmup} episodi casuali per riempire il buffer) ===")
        agent.epsilon = 1.0
        for w_ep in range(1, n_warmup + 1):
            run_episode()
            print(f"  Warm-up Ep {w_ep}/{n_warmup} completato. (Buffer size: {len(agent.replay_buffer)})")
        agent.epsilon = args.epsilon_start
        print("=== FINE WARM-UP ===\n")
    elif start_episode == 0 and args.no_warmup:
        print("[Ablation] Warm-up disabilitato (--no-warmup).")

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

        # (Rimossa qui la vecchia copia "live" del replay nel frontend CityFlow:
        # durante il training saveReplay e' sempre False, quindi CityFlow non
        # scrive mai quei file — il blocco non ha mai fatto nulla, vedi
        # analisi_bug.md #7. Per ispezionare un replay vero usare scripts/test.py
        # o scripts/inspect_replay.py con saveReplay attivo.)

        # ETA: media mobile del tempo per episodio × episodi rimanenti
        elapsed = time.time() - training_start_time
        episodes_done = episode - start_episode
        avg_ep_time = elapsed / episodes_done
        remaining_eps = end_episode - episode
        eta_str = format_eta(avg_ep_time * remaining_eps) if remaining_eps > 0 else "done"

        logger.print_episode(
            episode=episode,
            total_episodes=total_episodes,
            travel_time=travel_time,
            throughput=throughput,
            loss=loss,
            epsilon=epsilon_used,
            total_reward=ep_metrics.total_reward,
            is_best=is_best,
            eta_str=eta_str
        )

        # ── Episodio Random Periodico ──────────────────────────────────────
        if not args.no_cyclic_exploration and episode % 10 == 0 and episode < total_episodes:
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

    # Salva final_model.pth (checkpoint dell'ultimo episodio — usato come primario per il test)
    final_model_path = os.path.join(args.output, "final_model.pth")
    agent.save_checkpoint(
        path=final_model_path,
        episode=episode,
        travel_time=travel_time,
        extra_info={"ablation": args.ablation, "is_final": True}
    )

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETATO")
    print(f"  Episodi completati: {episode}")
    print(f"  Best travel time: {running_metrics.best_travel_time:.2f}s")
    print(f"  Last 10 avg travel time: {running_metrics.last_n_avg_travel_time:.2f}s")
    print(f"  Best throughput: {running_metrics.best_throughput}")
    print(f"  Modello finale salvato in: {final_model_path}")
    print(f"  Modello best salvato in:   {logger.best_path}")
    print(f"  Log salvato in: {logger.csv_path}")
    print(f"{'='*60}\n")

    # ── Grafici di confronto (solo se training completato, non interrotto) ──
    if not logger.interrupted:
        generate_comparison_plots(args, env, agent, edge_index, logger)

    # ── Selezione finale: final_model vs best_model dell'ultimo stage ─────────
    # (solo se il training e' arrivato in fondo, non se e' stato interrotto)
    if not logger.interrupted:
        run_final_selection(
            args,
            final_model_path=final_model_path,
            best_model_path=logger.best_path,
            device=device,
        )

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
