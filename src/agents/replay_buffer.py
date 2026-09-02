"""
Experience Replay Buffer Sequenziale per MetaSTGAT.

Memorizza snapshot globali del grafo per ogni time-step.
Supporta il campionamento di traiettorie continue di lunghezza L
per l'addestramento della rete con BPTT (Backpropagation Through Time).
"""

import random
from collections import deque
from typing import List, Optional

import numpy as np
import torch


class Transition:
    """Snapshot globale del grafo al tempo t."""
    __slots__ = ["state", "action", "reward", "next_state",
                 "spatial_meta", "temporal_meta"]

    def __init__(self,
                 state: np.ndarray,      # (N, state_dim)
                 action: np.ndarray,     # (N,)
                 reward: np.ndarray,     # (N,)
                 next_state: np.ndarray, # (N, state_dim)
                 spatial_meta: Optional[np.ndarray] = None,   # (N, smk_dim)
                 temporal_meta: Optional[np.ndarray] = None): # (N, tmk_dim)
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.spatial_meta = spatial_meta
        self.temporal_meta = temporal_meta


class ReplayBuffer:
    """
    Prioritized Experience Replay Buffer Sequenziale.
    Mantiene il conteggio totale delle transizioni per rispettare la capacity.
    Utilizza PER (Prioritized Experience Replay) proporzionale basato sugli errori TD.
    """

    def __init__(self, capacity: int = 3000, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_increment = (1.0 - beta_start) / beta_frames
        
        self.episodes: deque = deque()
        self.current_episode: List[Transition] = []
        self.total_transitions = 0
        
        self.next_ep_id = 0
        self.ep_dict = {}  # ep_id -> List[Transition]
        self.seq_priorities = {}  # (ep_id, start_idx) -> priority
        self.max_priority = 1.0

    def push(self,
             state: np.ndarray,
             action: np.ndarray,
             reward: np.ndarray,
             next_state: np.ndarray,
             spatial_meta: Optional[np.ndarray] = None,
             temporal_meta: Optional[np.ndarray] = None):
        """Aggiunge una transizione al buffer dell'episodio corrente."""
        t = Transition(state, action, reward, next_state, spatial_meta, temporal_meta)
        self.current_episode.append(t)
        self.total_transitions += 1

    def end_episode(self, seq_len: int = 8):
        """Richiamato alla fine di un episodio per chiudere la traiettoria corrente."""
        if len(self.current_episode) > 0:
            ep_id = self.next_ep_id
            self.next_ep_id += 1
            
            self.episodes.append({'id': ep_id, 'transitions': self.current_episode})
            self.ep_dict[ep_id] = self.current_episode
            
            # Aggiunge le sequenze valide alle priorità
            ep_len = len(self.current_episode)
            for start_idx in range(ep_len - seq_len + 1):
                self.seq_priorities[(ep_id, start_idx)] = self.max_priority
                
            self.current_episode = []
            
        # Evict old episodes se superiamo la capacity
        while self.total_transitions > self.capacity and len(self.episodes) > 1:
            removed_ep = self.episodes.popleft()
            rem_id = removed_ep['id']
            rem_len = len(removed_ep['transitions'])
            self.total_transitions -= rem_len
            
            # Rimuovi dalla memoria
            for start_idx in range(rem_len - seq_len + 1):
                self.seq_priorities.pop((rem_id, start_idx), None)
            self.ep_dict.pop(rem_id, None)

    def sample_sequences(self, batch_size: int, seq_len: int):
        """
        Campiona batch_size sequenze continue di lunghezza seq_len usando PER.
        Returns:
            batch: List[List[Transition]]
            keys: List[Tuple[int, int]] identificatori delle sequenze per update_priorities
            weights: np.ndarray pesi di Importance Sampling
        """
        valid_keys = list(self.seq_priorities.keys())
        if len(valid_keys) == 0:
            raise ValueError(f"Nessuna sequenza valida di lunghezza {seq_len} nel buffer.")
            
        priorities = np.array([self.seq_priorities[k] for k in valid_keys], dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(valid_keys), batch_size, p=probs, replace=True)
        batch = []
        weights = []
        selected_keys = []
        
        N = len(valid_keys)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for idx in indices:
            key = valid_keys[idx]
            ep_id, start_idx = key
            
            ep_trans = self.ep_dict[ep_id]
            batch.append(ep_trans[start_idx : start_idx + seq_len])
            
            prob = probs[idx]
            weight = (N * prob) ** (-self.beta)
            weights.append(weight)
            selected_keys.append(key)
            
        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()
        
        return batch, selected_keys, weights

    def update_priorities(self, keys, errors):
        """
        Aggiorna le priorità delle sequenze campionate.
        errors: array di TD errors (magnitudine) per ogni sequenza
        """
        for key, error in zip(keys, errors):
            # error = |TD_error| + epsilon (per evitare probabilità 0)
            priority = float(error) + 1e-5
            if key in self.seq_priorities:
                self.seq_priorities[key] = priority
            self.max_priority = max(self.max_priority, priority)

    def to_tensors_seq(self,
                       batch: List[List[Transition]],
                       device: torch.device) -> dict:
        """
        Converte un batch di sequenze in tensori PyTorch di forma (B, L, N, ...).
        """
        B = len(batch)
        L = len(batch[0])
        
        states_b, actions_b, rewards_b, next_states_b = [], [], [], []
        sm_b, tm_b = [], []
        
        has_meta = batch[0][0].spatial_meta is not None
        
        for seq in batch:
            states_b.append(np.stack([t.state for t in seq]))           # (L, N, state_dim)
            actions_b.append(np.stack([t.action for t in seq]))         # (L, N)
            rewards_b.append(np.stack([t.reward for t in seq]))         # (L, N)
            next_states_b.append(np.stack([t.next_state for t in seq])) # (L, N, state_dim)
            if has_meta:
                sm_b.append(np.stack([t.spatial_meta for t in seq]))
                tm_b.append(np.stack([t.temporal_meta for t in seq]))

        states = torch.tensor(np.stack(states_b), dtype=torch.float32, device=device) # (B, L, N, state_dim)
        actions = torch.tensor(np.stack(actions_b), dtype=torch.long, device=device)
        rewards = torch.tensor(np.stack(rewards_b), dtype=torch.float32, device=device)
        next_states = torch.tensor(np.stack(next_states_b), dtype=torch.float32, device=device)
        
        result = {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
        }
        
        if has_meta:
            result["spatial_meta"] = torch.tensor(np.stack(sm_b), dtype=torch.float32, device=device)
            result["temporal_meta"] = torch.tensor(np.stack(tm_b), dtype=torch.float32, device=device)
            
        return result

    def __len__(self) -> int:
        return self.total_transitions

    def is_ready(self, min_size: int, seq_len: int) -> bool:
        """Controlla se il buffer ha almeno min_size campioni e almeno un ep. di len seq_len."""
        if self.total_transitions < min_size:
            return False
        if len(self.seq_priorities) > 0:
            return True
        return False

    def state_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "episodes": list(self.episodes),
            "current_episode": self.current_episode,
            "total_transitions": self.total_transitions,
            "next_ep_id": self.next_ep_id,
            "seq_priorities": {f"{k[0]}_{k[1]}": v for k, v in self.seq_priorities.items()},
            "max_priority": self.max_priority
        }

    def load_state_dict(self, state: dict):
        self.capacity = state["capacity"]
        self.episodes = deque(state["episodes"])
        self.current_episode = state["current_episode"]
        self.total_transitions = state["total_transitions"]
        self.next_ep_id = state.get("next_ep_id", 0)
        
        self.ep_dict = {ep['id']: ep['transitions'] for ep in self.episodes}
        
        sp = state.get("seq_priorities", {})
        self.seq_priorities = {}
        for k, v in sp.items():
            parts = k.split('_')
            self.seq_priorities[(int(parts[0]), int(parts[1]))] = v
            
        self.max_priority = state.get("max_priority", 1.0)
