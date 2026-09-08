import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import collections

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.cityflow_env import CityFlowEnv
from src.models.metastgat import MetaSTGAT
from src.agents.dqn_agent import DQNAgent
from src.utils.logger import TrainingLogger
from src.utils.metrics import EpisodeMetrics, RunningMetrics

# Import evaluate from test.py for Phase 4
from scripts.test import evaluate

# ─── Iperparametri (dall'articolo, Section 5.1) ───────────────────────────────
DEFAULTS = {
    "updates_per_ep":      100,
    "batch_size":          20,
    "gamma":               0.85,
    "lr":                  1e-3,
    "buffer_size":         2_400,
    "min_buffer_size":     1_000,
    "hidden_dim":          64,
    "num_heads":           4,
    "num_neighbors":       4,
    "epsilon_start":       0.9,
    "epsilon_end":         0.01,
    "history_len":         5,
}

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h {m:02d}m"

def run_generalization_experiment():
    print("=== AVVIO ESPERIMENTO DI GENERALIZZAZIONE ===")
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config3_path = "configs/config_4x4_200m_1.3k_flat.json"
    config4_path = "configs/config_4x4_200m_1.9k_peak.json"
    stress_path = "configs/config_4x4_200m_6.2k_flat.json"
    
    from datetime import datetime
    output_dir = "results/generalization_20260907_103450"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")

    # ==========================================
    # PHASE 1: Addestramento su Config 3
    # ==========================================
    print("\n" + "="*50)
    print(" PHASE 1: Addestramento su Config 3 (50 Episodi)")
    print("="*50)
    
    def setup_env(config_path):
        with open(config_path, 'r') as f:
            custom_config = json.load(f)
        custom_config["saveReplay"] = False
        custom_config["seed"] = seed
        base_dir = custom_config.get("dir", "./")
        rel_out_dir = os.path.relpath(output_dir, base_dir)
        custom_config["replayLogFile"] = os.path.join(rel_out_dir, "replay.txt").replace("\\", "/")
        custom_config["roadnetLogFile"] = os.path.join(rel_out_dir, "roadnet.log").replace("\\", "/")
        custom_config_path = os.path.join(output_dir, "cityflow_config_temp.json")
        with open(custom_config_path, 'w') as f:
            json.dump(custom_config, f, indent=4)
        return CityFlowEnv(custom_config_path, num_neighbors=DEFAULTS["num_neighbors"])

    env = setup_env(config3_path)
    edge_index = env.get_edge_index().to(device)

    model = MetaSTGAT(
        state_dim=env.observation_dim,
        hidden_dim=DEFAULTS["hidden_dim"],
        num_heads=DEFAULTS["num_heads"],
        n_actions=env.action_space_n,
        spatial_meta_dim=env.spatial_meta_dim,
        temporal_meta_dim=env.temporal_meta_dim,
        meta_hidden_dim=DEFAULTS["hidden_dim"],
    ).to(device)

    eps_target_ep_ph1 = int(0.6 * 50)
    eps_decay_ph1 = (DEFAULTS["epsilon_end"] / DEFAULTS["epsilon_start"]) ** (1.0 / eps_target_ep_ph1)

    agent = DQNAgent(
        model=model,
        n_intersections=env.n_intersections,
        n_actions=env.action_space_n,
        lr=DEFAULTS["lr"],
        gamma=DEFAULTS["gamma"],
        epsilon_start=DEFAULTS["epsilon_start"],
        epsilon_end=DEFAULTS["epsilon_end"],
        epsilon_decay=eps_decay_ph1,
        buffer_size=DEFAULTS["buffer_size"],
        batch_size=DEFAULTS["batch_size"],
        device=device,
    )

    logger = TrainingLogger(output_dir, "generalization_ph1", resume=False)
    running_metrics = RunningMetrics(window=10)

    def run_episode(agent, env, edge_index):
        obs = env.reset()
        agent.reset_hidden()
        ep_metrics = EpisodeMetrics()
        state_history = {iid: [] for iid in env.inter_ids}
        done = False
        
        while not done:
            spatial_meta = {iid: env.get_spatial_meta_features(iid) for iid in env.inter_ids}
            temporal_meta = {
                iid: env.get_temporal_meta_features(iid, history=state_history[iid], history_len=DEFAULTS["history_len"])
                for iid in env.inter_ids
            }
            
            actions = agent.select_actions(
                states=obs,
                edge_index=edge_index,
                inter_ids=env.inter_ids,
                inter_id_to_idx=env.inter_id_to_idx,
                spatial_meta=spatial_meta,
                temporal_meta=temporal_meta,
                invalid_actions=env.get_invalid_actions()
            )
            
            next_obs, rewards, done, info = env.step(actions)
            
            agent.store_transitions(
                states=obs, actions=actions, rewards=rewards,
                next_states=next_obs, inter_ids=env.inter_ids,
                spatial_meta=spatial_meta, temporal_meta=temporal_meta,
            )
            
            for iid in env.inter_ids:
                state_history[iid].append(obs[iid])
                if len(state_history[iid]) > DEFAULTS["history_len"]:
                    state_history[iid].pop(0)
                    
            total_step_reward = sum(rewards.values())
            ep_metrics.update(travel_time=info["avg_travel_time"], throughput=info["vehicles_running"], reward=total_step_reward)
            obs = next_obs
            
        agent.replay_buffer.end_episode()
        return ep_metrics

    # Siccome stiamo riprendendo il run saltando la Fase 1, carichiamo il modello addestrato finora.
    ckpt_path = "results/generalization_20260907_103450/best_model.pt"
    if os.path.exists(ckpt_path):
        print(f"\n[INFO] Caricamento modello pre-addestrato da {ckpt_path}")
        agent.load_checkpoint(ckpt_path)
    else:
        print(f"\n[WARN] Checkpoint {ckpt_path} non trovato! L'agente partirà da zero.")

    # ==========================================
    # PHASE 2: Esplorazione su Config 4 (10 Episodi)
    # ==========================================
    print("\n" + "="*50)
    print(" PHASE 2: Esplorazione su Config 4 (10 Episodi - epsilon=1.0, NO TRAINING)")
    print("="*50)
    
    # Rilasciamo l'ambiente vecchio e carichiamo config 4
    del env
    env = setup_env(config4_path)
    edge_index = env.get_edge_index().to(device)
    
    agent.epsilon = 1.0
    for ep in range(1, 11):
        ep_metrics = run_episode(agent, env, edge_index)
        print(f"  Esplorazione Ep {ep}/10 | Buffer size: {len(agent.replay_buffer)} | TT: {ep_metrics.final_travel_time:.2f}s")
        # NESSUN AGGIORNAMENTO PESI (agent.update() non viene chiamato)

    # ==========================================
    # PHASE 3: Addestramento su Config 4 (26 Episodi)
    # ==========================================
    print("\n" + "="*50)
    print(" PHASE 3: Addestramento su Config 4 (26 Episodi, Fast Epsilon Decay)")
    print("="*50)
    
    agent.epsilon = DEFAULTS["epsilon_start"]
    # Fast decay: 30% di 26 episodi
    eps_target_ep_ph3 = int(0.3 * 26) 
    eps_decay_ph3 = (DEFAULTS["epsilon_end"] / DEFAULTS["epsilon_start"]) ** (1.0 / eps_target_ep_ph3)
    agent.epsilon_decay = eps_decay_ph3
    print(f"  Nuovo epsilon decay calcolato: {eps_decay_ph3:.4f} (raggiungerà {DEFAULTS['epsilon_end']} in {eps_target_ep_ph3} episodi)")

    logger_ph3 = TrainingLogger(output_dir, "generalization_ph3", resume=False)
    running_metrics_ph3 = RunningMetrics(window=10)
    
    for ep in range(1, 27):
        ep_metrics = run_episode(agent, env, edge_index)
        eps_used = getattr(agent, 'epsilon', 0.0)
        
        loss = agent.update(edge_index=edge_index, n_updates=DEFAULTS["updates_per_ep"], min_buffer_size=DEFAULTS["min_buffer_size"])
        ep_metrics.losses.append(loss or 0.0)
        
        running_metrics_ph3.add_episode(ep_metrics.final_travel_time, ep_metrics.final_throughput)
        # Sovrascriviamo il best model se migliora durante config4
        agent.save_best(logger_ph3.best_path, ep, ep_metrics.final_travel_time)
        
        logger_ph3.log_episode(ep, ep_metrics.final_travel_time, ep_metrics.final_throughput, ep_metrics.total_reward, ep_metrics.avg_loss, eps_used, len(agent.replay_buffer))
        logger_ph3.print_episode(ep, 26, ep_metrics.final_travel_time, ep_metrics.final_throughput, loss, eps_used, ep_metrics.total_reward, False, "N/A")

    # Salva il modello finale unito
    final_checkpoint_path = os.path.join(output_dir, "final_generalized_model.pt")
    agent.save_checkpoint(final_checkpoint_path, episode=26, travel_time=running_metrics_ph3.last_n_avg_travel_time)
    print(f"\nPhase 3 completata. Modello salvato in {final_checkpoint_path}")

    # Pulisci file temporaneo config
    temp_config_path = os.path.join(output_dir, "cityflow_config_temp.json")
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

    # ==========================================
    # PHASE 4: Test e Valutazione sulle 3 config
    # ==========================================
    print("\n" + "="*50)
    print(" PHASE 4: Test e Valutazione Modello Generalizzato")
    print("="*50)
    
    class TestArgs:
        def __init__(self, config, output, checkpoint):
            self.config = config
            self.model = "MetaSTGAT"
            self.checkpoint = checkpoint
            self.n_eval = 10
            self.output = output
            self.hidden_dim = DEFAULTS["hidden_dim"]
            self.num_heads = DEFAULTS["num_heads"]
            self.num_neighbors = DEFAULTS["num_neighbors"]
            self.device = None

    test_configs = [
        ("Config 3", config3_path, "eval_config3"),
        ("Config 4", config4_path, "eval_config4"),
        ("Stress Flat", stress_path, "eval_stress_flat")
    ]

    for name, c_path, out_sub in test_configs:
        print(f"\n---> Esecuzione test su: {name}")
        c_out_dir = os.path.join(output_dir, out_sub)
        os.makedirs(c_out_dir, exist_ok=True)
        
        args_eval = TestArgs(c_path, c_out_dir, final_checkpoint_path)
        
        try:
            evaluate(args_eval)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Errore durante la valutazione di {name}: {e}")

    print("\n=== ESPERIMENTO DI GENERALIZZAZIONE COMPLETATO ===")
    print(f"Tutti i risultati e i grafici sono salvati in: {output_dir}")

if __name__ == "__main__":
    run_generalization_experiment()
