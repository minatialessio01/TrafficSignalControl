"""
Ablation Study per MetaSTGAT.

Allena in sequenza i modelli dell'ablation study (Section 5.5.2):
  - GAT-only:  solo GAT spaziale (no LSTM, no meta)
  - STGAT:     LSTM + GAT (no meta)
  - MetaGAT:   LSTM + Meta-GAT (meta solo su GAT)
  - MetaLSTM:  Meta-LSTM + GAT (meta solo su LSTM)
  - MetaSTGAT: Meta-LSTM + Meta-GAT (modello completo)

Produce un report comparativo con travel time e throughput
per tutti i modelli, confrontabile con le Fig. 7-8 dell'articolo.

Uso:
  python scripts/ablation.py \\
      --config configs/synthetic_4x4_config2.json \\
      --episodes 100 \\
      --output results/ablation
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.environment.cityflow_env import CityFlowEnv
from src.models.metastgat import MetaSTGAT
from src.models.stgat import STGAT
from src.agents.dqn_agent import DQNAgent
from src.utils.logger import TrainingLogger
from src.utils.metrics import RunningMetrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation Study MetaSTGAT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--output", default="results/ablation")
    parser.add_argument("--hidden-dim",    type=int, default=64)
    parser.add_argument("--num-heads",     type=int, default=4)
    parser.add_argument("--num-neighbors", type=int, default=4)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


class GATOnly(STGAT):
    """
    Variante GAT-only: solo il modulo spaziale (CS), senza LSTM.
    L'hidden state LSTM è sempre zero → degrada a puro GAT.
    """
    def forward(self, states, edge_index, h_prev=None, c_prev=None):
        N = states.size(0)
        device = states.device
        if h_prev is None:
            h_prev = torch.zeros(N, self.hidden_dim, device=device)
        if c_prev is None:
            c_prev = torch.zeros(N, self.hidden_dim, device=device)

        _, e_j = self.encoder(states)

        # Salta LSTM: x_i = zeros
        x_i = torch.zeros_like(e_j)
        c_t = c_prev

        # Solo modulo CS (spaziale)
        z_cs = self.cs_gat(query=e_j, key=e_j, value=e_j, edge_index=edge_index)
        z_cst = torch.zeros_like(z_cs)

        z = torch.cat([z_cst, z_cs], dim=-1)
        q_values = self.q_head(z)
        return q_values, x_i, c_t


class MetaGATModel(nn.Module):
    """
    MetaGAT: LSTM standard + Meta-GAT (meta solo sul GAT).
    """
    def __init__(self, state_dim, hidden_dim, num_heads, n_actions,
                 spatial_meta_dim, temporal_meta_dim, meta_hidden_dim):
        super().__init__()
        # Usa MetaSTGAT ma con Meta-LSTM sostituita da LSTM standard
        self.full = MetaSTGAT(
            state_dim=state_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, n_actions=n_actions,
            spatial_meta_dim=spatial_meta_dim,
            temporal_meta_dim=temporal_meta_dim,
            meta_hidden_dim=meta_hidden_dim
        )
        # Sostituisce Meta-LSTM con LSTM standard
        self.std_lstm = nn.LSTMCell(hidden_dim, hidden_dim)

    def forward(self, states, edge_index, smf, tmf, h_prev=None, c_prev=None):
        N = states.size(0)
        device = states.device
        if h_prev is None:
            h_prev = torch.zeros(N, self.full.hidden_dim, device=device)
        if c_prev is None:
            c_prev = torch.zeros(N, self.full.hidden_dim, device=device)

        e_i, e_j = self.full.encoder(states)
        smk = self.full.smk_learner(smf)
        tmk = self.full.tmk_learner(tmf)

        # LSTM standard invece di Meta-LSTM
        x_i, c_t = self.std_lstm(e_i, (h_prev, c_prev))

        # Meta-GAT CST
        z_cst = self.full.meta_gat_cst(e_j, x_i, x_i, edge_index, tmk)
        # Meta-GAT CS
        z_cs = self.full.meta_gat_cs(e_j, e_j, e_j, edge_index, smk)

        z = torch.cat([z_cst, z_cs], dim=-1)
        q_values = self.full.q_head(z)
        return q_values, x_i, c_t

    def init_hidden(self, n, device=None):
        return self.full.init_hidden(n, device)

    @property
    def hidden_dim(self):
        return self.full.hidden_dim

    @property
    def smk_learner(self):
        return self.full.smk_learner

    @property
    def tmk_learner(self):
        return self.full.tmk_learner


class MetaLSTMModel(nn.Module):
    """
    MetaLSTM: Meta-LSTM + GAT standard (meta solo sull'LSTM).
    """
    def __init__(self, state_dim, hidden_dim, num_heads, n_actions,
                 spatial_meta_dim, temporal_meta_dim, meta_hidden_dim):
        super().__init__()
        self.full = MetaSTGAT(
            state_dim=state_dim, hidden_dim=hidden_dim,
            num_heads=num_heads, n_actions=n_actions,
            spatial_meta_dim=spatial_meta_dim,
            temporal_meta_dim=temporal_meta_dim,
            meta_hidden_dim=meta_hidden_dim
        )
        # GAT standard per CS e CST
        from src.models.stgat import StandardGATLayer
        self.std_gat_cst = StandardGATLayer(hidden_dim, num_heads)
        self.std_gat_cs = StandardGATLayer(hidden_dim, num_heads)

    def forward(self, states, edge_index, smf, tmf, h_prev=None, c_prev=None):
        N = states.size(0)
        device = states.device
        if h_prev is None:
            h_prev = torch.zeros(N, self.full.hidden_dim, device=device)
        if c_prev is None:
            c_prev = torch.zeros(N, self.full.hidden_dim, device=device)

        e_i, e_j = self.full.encoder(states)
        tmk = self.full.tmk_learner(tmf)

        # Meta-LSTM
        x_i, c_t = self.full.meta_lstm(e_i, h_prev, c_prev, tmk)

        # GAT standard (no meta)
        z_cst = self.std_gat_cst(e_j, x_i, x_i, edge_index)
        z_cs = self.std_gat_cs(e_j, e_j, e_j, edge_index)

        z = torch.cat([z_cst, z_cs], dim=-1)
        q_values = self.full.q_head(z)
        return q_values, x_i, c_t

    def init_hidden(self, n, device=None):
        return self.full.init_hidden(n, device)

    @property
    def hidden_dim(self):
        return self.full.hidden_dim

    @property
    def smk_learner(self):
        return self.full.smk_learner

    @property
    def tmk_learner(self):
        return self.full.tmk_learner


def train_variant(model_name: str, model: nn.Module, env: CityFlowEnv,
                  edge_index: torch.Tensor, device: torch.device,
                  episodes: int, output_dir: str,
                  is_meta: bool = False,
                  epsilon_decay: float = 0.995) -> dict:
    """Allena una variante del modello e restituisce le metriche finali."""
    print(f"\n{'─'*50}")
    print(f"  Training: {model_name} ({episodes} episodi)")
    print(f"{'─'*50}")

    agent = DQNAgent(
        model=model,
        n_intersections=env.n_intersections,
        device=device,
        epsilon_decay=epsilon_decay
    )
    # Forza il flag is_meta in base al tipo
    agent.is_meta = is_meta

    running = RunningMetrics(window=10)
    history_len = 5
    def run_episode():
        obs = env.reset()
        agent.reset_hidden()
        state_history = {iid: [] for iid in env.inter_ids}
        done = False

        while not done:
            spatial_meta = None
            temporal_meta = None
            if is_meta:
                spatial_meta = {iid: env.get_spatial_meta_features(iid) for iid in env.inter_ids}
                temporal_meta = {iid: env.get_temporal_meta_features(iid, state_history[iid], history_len) for iid in env.inter_ids}

            actions = agent.select_actions(
                states=obs, edge_index=edge_index,
                inter_ids=env.inter_ids, inter_id_to_idx=env.inter_id_to_idx,
                spatial_meta=spatial_meta, temporal_meta=temporal_meta,
                invalid_actions=env.get_invalid_actions()
            )
            next_obs, rewards, done, info = env.step(actions)
            agent.store_transitions(
                obs, actions, rewards, next_obs, env.inter_ids,
                spatial_meta, temporal_meta
            )
            for iid in env.inter_ids:
                state_history[iid].append(obs[iid])
                if len(state_history[iid]) > history_len:
                    state_history[iid].pop(0)
            obs = next_obs
        
        agent.replay_buffer.end_episode()
            
    print("\n=== FASE DI WARM-UP (10 episodi casuali per riempire il buffer) ===")
    old_eps = agent.epsilon
    agent.epsilon = 1.0
    for w_ep in range(1, 11):
        run_episode()
    agent.epsilon = old_eps
    print("=== FINE WARM-UP ===\n")

    for episode in range(1, episodes + 1):
        run_episode()

        agent.update(edge_index, n_updates=100, min_buffer_size=100)
        tt = env.get_average_travel_time()
        tp = env.get_throughput()
        running.add_episode(tt, tp)

        if episode % 10 == 0 or episode == episodes:
            print(f"    Ep {episode:>4}: TT={tt:.1f}s | TP={tp} | "
                  f"ε={agent.epsilon:.4f}")

        # ── Episodio Random Periodico ──────────────────────────────────────
        if episode % 10 == 0 and episode < episodes:
            print(f"    [!] Esecuzione di 1 episodio random per esplorazione...")
            old_epsilon = agent.epsilon
            agent.epsilon = 1.0
            run_episode()
            agent.epsilon = old_epsilon
            print(f"    [!] Episodio random completato.")

    result = {
        "model": model_name,
        "avg_travel_time": round(running.last_n_avg_travel_time, 4),
        "avg_throughput": round(running.last_n_avg_throughput, 1),
        "best_travel_time": round(running.best_travel_time, 4),
    }

    # Salva il modello
    torch.save(model.state_dict(),
               os.path.join(output_dir, f"{model_name.lower()}_final.pt"))

    return result


def run_ablation(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    env = CityFlowEnv(args.config, num_neighbors=args.num_neighbors)
    edge_index = env.get_edge_index().to(device)

    os.makedirs(args.output, exist_ok=True)

    H = args.hidden_dim
    NH = args.num_heads
    NA = env.action_space_n
    SD = env.observation_dim
    SMD = env.spatial_meta_dim
    TMD = env.temporal_meta_dim

    variants = [
        ("GAT-only",   GATOnly(SD, H, NH, NA), False),
        ("STGAT",      STGAT(SD, H, NH, NA), False),
        ("MetaGAT",    MetaGATModel(SD, H, NH, NA, SMD, TMD, H), True),
        ("MetaLSTM",   MetaLSTMModel(SD, H, NH, NA, SMD, TMD, H), True),
        ("MetaSTGAT",  MetaSTGAT(SD, H, NH, NA, SMD, TMD, H), True),
    ]

    results = []
    for name, model, is_meta in variants:
        model = model.to(device)
        result = train_variant(
            name, model, env, edge_index, device,
            args.episodes, args.output, is_meta,
            epsilon_decay=args.epsilon_decay
        )
        results.append(result)
        print(f"  [{name}] TT={result['avg_travel_time']}s | TP={result['avg_throughput']}")

    # Stampa tabella comparativa
    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY — Risultati finali ({args.episodes} episodi)")
    print(f"  Config: {args.config}")
    print(f"{'='*60}")
    print(f"  {'Modello':<14} {'Avg TT (s)':>12} {'Best TT (s)':>12} {'Avg TP':>10}")
    print(f"  {'─'*50}")
    for r in results:
        print(f"  {r['model']:<14} {r['avg_travel_time']:>12.2f} "
              f"{r['best_travel_time']:>12.2f} {r['avg_throughput']:>10.1f}")
    print(f"{'='*60}\n")

    # Salva risultati
    out_path = os.path.join(args.output, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Risultati salvati in: {out_path}")


if __name__ == "__main__":
    args = parse_args()
    run_ablation(args)
