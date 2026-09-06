"""
Wrapper per l'agente CoLight di LibSignal, in modo da poterlo testare
con la nostra pipeline e il nostro ambiente CityFlowEnv.
"""

import copy
import random
from typing import Dict, List, Optional
import numpy as np
import torch

from .replay_buffer import ReplayBuffer

import sys
sys.path.insert(0, '/opt/LibSignal')

try:
    from agent.colight_pytorch_agent import ColightNet, CoLightAgent
except Exception as e:
    import traceback
    traceback.print_exc()
    # Fallback/placeholder se LibSignal non è nel PYTHONPATH o mancano dipendenze
    ColightNet = None
    print(f"[WARN] LibSignal (ColightNet) non caricato: {e}")

class CoLightBuffer(ReplayBuffer):
    def end_episode(self, seq_len: int = 1):
        super().end_episode(seq_len)

class CoLightWrapperAgent:
    def __init__(self, n_intersections, n_actions=8, lr=1e-3, device=None, **kwargs):
        self.n_intersections = n_intersections
        self.n_actions = n_actions
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 16
        
        # Buffer classico senza sequenze (CoLight non usa LSTM)
        self.replay_buffer = CoLightBuffer(capacity=5000, alpha=0.6)
        
        # Inizializziamo il modello se disponibile
        if ColightNet is not None:
            class DummyActionSpace:
                def __init__(self, n):
                    self.n = n
                    
            colight_config = {
                'action_space': DummyActionSpace(self.n_actions),
                'NODE_EMB_DIM': [128, 128],
                'N_LAYERS': 1,
                'INPUT_DIM': [128],
                'NODE_LAYER_DIMS_EACH_HEAD': [16],
                'OUTPUT_DIM': [128],
                'NUM_HEADS': [5],
                'OUTPUT_LAYERS': []
            }
            # L'input dim per CityFlowEnv (state dim) nel nostro setup è 32
            self.model = ColightNet(input_dim=32, **colight_config).to(self.device)
            self.target_model = copy.deepcopy(self.model)
            self.target_model.eval()
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.9, eps=1e-7)
        else:
            self.model = None
            self.target_model = None
            self.optimizer = None

        self.epsilon = 0.9

    @torch.no_grad()
    def select_actions(self,
                       states: Dict[str, np.ndarray],
                       edge_index: torch.Tensor,
                       inter_ids: List[str],
                       inter_id_to_idx: Dict[str, int],
                       spatial_meta: Optional[Dict[str, np.ndarray]] = None,
                       temporal_meta: Optional[Dict[str, np.ndarray]] = None,
                       invalid_actions: Optional[Dict[str, List[int]]] = None
                       ) -> Dict[str, int]:
        
        N = len(inter_ids)
        if self.model is None:
            return {iid: np.random.randint(self.n_actions) for iid in inter_ids}

        state_tensor = torch.tensor(
            np.stack([states[iid] for iid in inter_ids]),
            dtype=torch.float32, device=self.device
        )
        
        # Forward pass (train=False)
        q_values = self.model(state_tensor, edge_index, train=False)

        actions = {}
        for idx, iid in enumerate(inter_ids):
            invalids = invalid_actions.get(iid, []) if invalid_actions else []
            valid_list = [a for a in range(self.n_actions) if a not in invalids]
            if not valid_list:
                valid_list = list(range(self.n_actions))
                invalids = []
            
            if random.random() < self.epsilon:
                actions[iid] = random.choice(valid_list)
            else:
                q_vals = q_values[idx].clone()
                for inv in invalids:
                    q_vals[inv] = -float('inf')
                actions[iid] = int(q_vals.argmax().item())

        return actions

    def store_transitions(self,
                          states: Dict[str, np.ndarray],
                          actions: Dict[str, int],
                          rewards: Dict[str, float],
                          next_states: Dict[str, np.ndarray],
                          inter_ids: List[str],
                          spatial_meta=None,
                          temporal_meta=None):
        
        state_arr = np.stack([states[iid] for iid in inter_ids])
        action_arr = np.array([actions[iid] for iid in inter_ids])
        reward_arr = np.array([rewards[iid] for iid in inter_ids])
        next_state_arr = np.stack([next_states[iid] for iid in inter_ids])

        self.replay_buffer.push(
            state=state_arr,
            action=action_arr,
            reward=reward_arr,
            next_state=next_state_arr,
            spatial_meta=None,
            temporal_meta=None
        )

    def update(self,
               edge_index: torch.Tensor,
               n_updates: int = 100,
               min_buffer_size: int = 1000
               ) -> Optional[float]:
        
        if self.model is None or len(self.replay_buffer) < 16:
            return None

        total_loss = 0.0
        self.model.train()
        
        for _ in range(n_updates):
            # Usiamo seq_len=1 perché non usiamo LSTM, ma ReplayBuffer potrebbe restituire BxLxNxD
            batch, keys, is_weights = self.replay_buffer.sample_sequences(self.batch_size, seq_len=1)
            is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
            tensors = self.replay_buffer.to_tensors_seq(batch, self.device)

            # Forma attesa: (B, L, N, D), noi consideriamo L=1, per cui (B, 1, N, D)
            st = tensors["states"][:, 0].reshape(-1, tensors["states"].shape[-1])
            at = tensors["actions"][:, 0].reshape(-1)
            rt = tensors["rewards"][:, 0].reshape(-1)
            nst = tensors["next_states"][:, 0].reshape(-1, tensors["next_states"].shape[-1])
            
            B = tensors["states"].size(0)
            N = self.n_intersections

            # Crea batch_edge_index replicato (block-diagonal) per B grafi
            batch_edge_index = torch.cat(
                [edge_index + i * N for i in range(B)], dim=1
            ).to(self.device)

            q_preds = self.model(st, batch_edge_index, train=True)
            q_values = q_preds.gather(1, at.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                q_targets_next = self.target_model(nst, batch_edge_index, train=False).max(1)[0]
                q_targets = rt + 0.85 * q_targets_next

            td_errors = (q_targets - q_values)
            loss = (is_weights_t.repeat_interleave(N) * (td_errors ** 2)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Aggiorna pesi PER
            errors_per_sequence = td_errors.abs().view(B, N).mean(dim=1).detach().cpu().numpy()
            self.replay_buffer.update_priorities(keys, errors_per_sequence)

            total_loss += loss.item()
            
            # Soft update target model
            tau = 0.01
            for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        return total_loss / n_updates

    def reset_hidden(self):
        pass
    
    def save_best(self, path, episode, travel_time):
        return False
        
    def save_checkpoint(self, path, episode, travel_time, extra_info):
        pass
