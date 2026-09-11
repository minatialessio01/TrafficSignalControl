"""
DQN Agent per MetaSTGAT / STGAT.

Implementa l'Algorithm 1 dell'articolo con:
  - Epsilon-greedy exploration
  - Experience replay (buffer 10,000, batch 20)
  - Target network (soft update)
  - Ottimizzatore RMSprop (come da paper, Section 5.1)
  - Checkpoint save/load per riprendere il training

Parametri dell'articolo (Section 5.1):
  - batch_size: 20
  - gamma (discount factor): 0.85
  - buffer_size: 10,000
  - sampling_size: 1,000
  - episodes: 200, ogni episodio addestrato 100 volte
  - ottimizzatore: RMSprop

Parametri non specificati (scelti da letteratura standard DQN):
  - epsilon: 0.9 → 0.01, decay = 0.995
  - lr: 1e-3
  - target_update_freq: 100 step (soft update τ=0.01)
"""

import os
import copy
import json
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .replay_buffer import ReplayBuffer
from ..models.metastgat import MetaSTGAT
from ..models.stgat import STGAT


class DQNAgent:
    """
    Agente DQN multi-intersezione per il controllo semaforico.

    Supporta sia MetaSTGAT che STGAT come modello di Q-network.
    Ogni intersezione condivide lo stesso modello (parameter sharing),
    ma ha i propri stati nascosti LSTM.

    Args:
        model:            istanza di MetaSTGAT o STGAT
        n_intersections:  numero di intersezioni
        n_actions:        numero di azioni/fasi (default: 8)
        lr:               learning rate (default: 1e-3)
        gamma:            discount factor (default: 0.85)
        epsilon_start:    epsilon iniziale per greedy (default: 0.9)
        epsilon_end:      epsilon finale (default: 0.01)
        epsilon_decay:    fattore di decay per episodio (default: 0.995)
        buffer_size:      dimensione replay buffer (default: 10,000)
        batch_size:       batch size per update (default: 20)
        target_update_tau: tau per soft update target network (default: 0.01)
        target_update_freq: frequenza aggiornamento target (in step, default: 100)
        device:           dispositivo PyTorch
    """

    def __init__(self,
                 model: nn.Module,
                 n_intersections: int,
                 n_actions: int = 8,
                 lr: float = 1e-3,
                 gamma: float = 0.85,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.7985,  # ε: 0.9→0.01 in 20 episodi
                 buffer_size: int = 2400,
                 batch_size: int = 16,
                 seq_len: int = 4,
                 burn_in: int = 2,
                 target_update_tau: float = 0.01,
                 target_update_freq: int = 100,
                 device: Optional[torch.device] = None,
                 # ── Ablation flags ───────────────────────────────────────────
                 use_double_dqn: bool = True,   # False = DQN standard (target max)
                 use_per: bool = True,           # False = buffer flat, IS weights = 1.0
                 use_huber: bool = True,         # False = MSE loss
                 use_soft_update: bool = True,   # False = hard copy del target network
                 use_grad_clip: bool = True,     # False = no gradient clipping
                 use_bptt: bool = True,          # False = single-step L=1, burn_in=0
                 use_tanh_meta: bool = True,     # False = no Tanh sui meta-learner
                 ):

        self.n_intersections = n_intersections
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        # Ablation: se no_bptt, forza single-step
        self.use_bptt = use_bptt
        self.seq_len  = seq_len if use_bptt else 1
        self.burn_in  = burn_in if use_bptt else 0
        self.target_update_tau = target_update_tau
        self.target_update_freq = target_update_freq
        self.is_meta = isinstance(model, MetaSTGAT)
        # Ablation flags (usati in update e _soft_update_target)
        self.use_double_dqn = use_double_dqn
        self.use_per        = use_per
        self.use_huber      = use_huber
        self.use_soft_update = use_soft_update
        self.use_grad_clip  = use_grad_clip

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Modello online e target network
        self.model = model.to(self.device)
        self.target_model = copy.deepcopy(model).to(self.device)
        self.target_model.eval()

        # Ottimizzatore RMSprop (come da paper, Section 5.1)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Stato nascosto LSTM per ogni intersezione (durante l'episode)
        self.h_state: Optional[torch.Tensor] = None
        self.c_state: Optional[torch.Tensor] = None

        # Contatori
        self.total_steps = 0
        self.total_episodes = 0
        self.best_travel_time = float("inf")

    # ─── Selezione azione ─────────────────────────────────────────────────────

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
        """
        Seleziona le azioni per tutte le intersezioni con epsilon-greedy.

        Args:
            states:        {inter_id -> state array (state_dim,)}
            edge_index:    (2, E) tensor sul device corretto
            inter_ids:     lista ordinata degli ID intersezione
            inter_id_to_idx: mappa id -> indice
            spatial_meta:  {inter_id -> spatial_meta_features} (MetaSTGAT)
            temporal_meta: {inter_id -> temporal_meta_features} (MetaSTGAT)
            invalid_actions: maschera {inter_id -> list_of_invalid_phases}

        Returns:
            {inter_id -> phase_index}
        """
        N = len(inter_ids)

        # Corto circuito epsilon=1.0 (warm-up / esplorazione ciclica): ogni
        # intersezione sceglierebbe comunque un'azione casuale (vedi il ciclo
        # epsilon-greedy sotto: random.random() < 1.0 e' sempre vero), quindi
        # il forward pass della rete (il costo dominante per step su CPU) e'
        # puro spreco — i suoi risultati verrebbero scartati. Saltarlo non
        # altera nulla: lo stato nascosto LSTM non serve (durante il warm-up
        # non c'e' alcun aggiornamento via BPTT — il replay buffer salva solo
        # stato/azione/reward, l'hidden state viene ricalcolato col burn-in al
        # momento del training vero) e ogni episodio riparte comunque da
        # reset_hidden() la prossima volta.
        if self.epsilon >= 1.0:
            actions = {}
            for iid in inter_ids:
                invalids = invalid_actions.get(iid, []) if invalid_actions else []
                valid_list = [a for a in range(self.n_actions) if a not in invalids] or list(range(self.n_actions))
                actions[iid] = random.choice(valid_list)
            return actions

        # Costruisci il tensore degli stati: (N, state_dim)
        state_tensor = torch.tensor(
            np.stack([states[iid] for iid in inter_ids]),
            dtype=torch.float32, device=self.device
        )

        # Inizializza h solo al primo step dell'episodio (quando reset_hidden() ha posto h=None).
        # Negli step successivi h porta avanti la memoria accumulata dall'LSTM.
        # Nel training, la mancanza del vero h iniziale nelle sequenze del buffer
        # viene compensata dal burn-in (primi burn_in step senza gradiente).
        if self.h_state is None or self.h_state.size(0) != N:
            self.h_state, self.c_state = self.model.init_hidden(N, self.device)

        # Forward pass
        if self.is_meta:
            smf = torch.tensor(
                np.stack([spatial_meta[iid] for iid in inter_ids]),
                dtype=torch.float32, device=self.device
            ) if spatial_meta else torch.zeros(N, self.model.smk_learner.fc1.in_features, device=self.device)

            tmf = torch.tensor(
                np.stack([temporal_meta[iid] for iid in inter_ids]),
                dtype=torch.float32, device=self.device
            ) if temporal_meta else torch.zeros(N, self.model.tmk_learner.fc1.in_features, device=self.device)

            q_values, self.h_state, self.c_state = self.model(
                state_tensor, edge_index, smf, tmf,
                self.h_state, self.c_state
            )
        else:
            q_values, self.h_state, self.c_state = self.model(
                state_tensor, edge_index,
                self.h_state, self.c_state
            )

        # Epsilon-greedy
        actions = {}
        for idx, iid in enumerate(inter_ids):
            invalids = invalid_actions.get(iid, []) if invalid_actions else []
            valid_list = [a for a in range(self.n_actions) if a not in invalids]
            
            # Se tutte le azioni sono invalide (non dovrebbe succedere), ripiega su fallback
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

    def reset_hidden(self):
        """Resetta gli stati nascosti LSTM (chiamato all'inizio di ogni episodio).
        Impostare h=None forza l'inizializzazione a zero al primo step del nuovo episodio.
        """
        self.h_state = None
        self.c_state = None

    # ─── Training ─────────────────────────────────────────────────────────────

    def store_transitions(self,
                          states: Dict[str, np.ndarray],
                          actions: Dict[str, int],
                          rewards: Dict[str, float],
                          next_states: Dict[str, np.ndarray],
                          inter_ids: List[str],
                          spatial_meta: Optional[Dict[str, np.ndarray]] = None,
                          temporal_meta: Optional[Dict[str, np.ndarray]] = None):
        """
        Memorizza l'intero stato del grafo nel replay buffer.
        """
        state_arr = np.stack([states[iid] for iid in inter_ids])
        action_arr = np.array([actions[iid] for iid in inter_ids])
        reward_arr = np.array([rewards[iid] for iid in inter_ids])
        next_state_arr = np.stack([next_states[iid] for iid in inter_ids])

        sm_arr = np.stack([spatial_meta[iid] for iid in inter_ids]) if spatial_meta else None
        tm_arr = np.stack([temporal_meta[iid] for iid in inter_ids]) if temporal_meta else None

        self.replay_buffer.push(
            state=state_arr,
            action=action_arr,
            reward=reward_arr,
            next_state=next_state_arr,
            spatial_meta=sm_arr,
            temporal_meta=tm_arr
        )

    def update(self,
               edge_index: torch.Tensor,
               n_updates: int = 100,
               min_buffer_size: int = 1000
               ) -> Optional[float]:
        """
        Aggiorna il modello con n_updates passi di gradient descent.
        (Algorithm 1, line 9-13: ogni episodio viene addestrato 100 volte)

        Args:
            edge_index:      (2, E) topologia del grafo
            n_updates:       numero di update per episodio (paper: 100)
            min_buffer_size: non inizia il training prima di avere almeno
                             min_buffer_size transizioni nel buffer

        Returns:
            loss media dell'episodio, o None se buffer insufficiente
        """

        # Non allenare finche' il buffer non ha almeno min_buffer_size transizioni
        # (e comunque almeno una sequenza valida di lunghezza seq_len).
        if not self.replay_buffer.is_ready(min_buffer_size, self.seq_len):
            return None

        total_loss = 0.0
        self.model.train()

        for _ in range(n_updates):
            # Campiona batch: con PER usa priorità TD + IS weights; senza PER IS weights = 1.0
            batch, keys, is_weights = self.replay_buffer.sample_sequences(self.batch_size, self.seq_len)
            if self.use_per:
                is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
            else:
                # Campionamento uniforme: tutti i pesi uguali a 1.0 (nessuna correzione IS)
                is_weights_t = torch.ones(len(batch), dtype=torch.float32, device=self.device)
            
            tensors = self.replay_buffer.to_tensors_seq(batch, self.device)

            B = tensors["states"].size(0)
            L = tensors["states"].size(1)
            N = self.n_intersections

            states_b = tensors["states"]         # (B, L, N, state_dim)
            actions_b = tensors["actions"]        # (B, L, N)
            rewards_b = tensors["rewards"]        # (B, L, N)
            next_states_b = tensors["next_states"]  # (B, L, N, state_dim)
            
            smf_b = tensors.get("spatial_meta")
            tmf_b = tensors.get("temporal_meta")

            # Crea batch_edge_index replicato (block-diagonal) per B grafi
            batch_edge_index = torch.cat(
                [edge_index + i * N for i in range(B)], dim=1
            ).to(self.device)

            # Inizializza hidden states LSTM a zero (inizio sequenza)
            h_b = torch.zeros(B * N, self.model.hidden_dim, device=self.device)
            c_b = torch.zeros(B * N, self.model.hidden_dim, device=self.device)

            # Target network sui next states (Double DQN: valutazione Q)
            h_target = torch.zeros(B * N, self.model.hidden_dim, device=self.device)
            c_target = torch.zeros(B * N, self.model.hidden_dim, device=self.device)

            # Rete online sui next states (Double DQN: selezione azione)
            # Separata da h_b perché viene applicata a nst, non a st
            h_online_next = torch.zeros(B * N, self.model.hidden_dim, device=self.device)
            c_online_next = torch.zeros(B * N, self.model.hidden_dim, device=self.device)

            q_preds_valid = []
            q_targets_valid = []

            # ── Unrolling BPTT ────────────────────────────────────────────────
            for t in range(L):
                st = states_b[:, t].reshape(B * N, -1)
                at = actions_b[:, t].reshape(-1)
                rt = rewards_b[:, t].reshape(-1)
                nst = next_states_b[:, t].reshape(B * N, -1)
                
                if self.is_meta:
                    sm_t = smf_b[:, t].reshape(B * N, -1) if smf_b is not None else torch.zeros(B * N, self.model.smk_learner.fc1.in_features, device=self.device)
                    tm_t = tmf_b[:, t].reshape(B * N, -1) if tmf_b is not None else torch.zeros(B * N, self.model.tmk_learner.fc1.in_features, device=self.device)
                
                # Burn-in phase (No gradienti)
                if t < self.burn_in:
                    with torch.no_grad():
                        if self.is_meta:
                            _, h_b, c_b = self.model(st, batch_edge_index, sm_t, tm_t, h_b, c_b)
                            _, h_target, c_target = self.target_model(nst, batch_edge_index, sm_t, tm_t, h_target, c_target)
                            _, h_online_next, c_online_next = self.model(nst, batch_edge_index, sm_t, tm_t, h_online_next, c_online_next)
                        else:
                            _, h_b, c_b = self.model(st, batch_edge_index, h_b, c_b)
                            _, h_target, c_target = self.target_model(nst, batch_edge_index, h_target, c_target)
                            _, h_online_next, c_online_next = self.model(nst, batch_edge_index, h_online_next, c_online_next)
                else:
                    # Training phase — Double DQN (van Hasselt et al., AAAI 2016)
                    # Rete online con gradienti (sui current states st)
                    if self.is_meta:
                        q_pred, h_b, c_b = self.model(st, batch_edge_index, sm_t, tm_t, h_b, c_b)
                        with torch.no_grad():
                            # Rete online sui next states: seleziona l'azione migliore
                            q_online_next, h_online_next, c_online_next = self.model(
                                nst, batch_edge_index, sm_t, tm_t, h_online_next, c_online_next)
                            # Target network valuta l'azione selezionata dalla rete online
                            q_next, h_target, c_target = self.target_model(
                                nst, batch_edge_index, sm_t, tm_t, h_target, c_target)
                    else:
                        q_pred, h_b, c_b = self.model(st, batch_edge_index, h_b, c_b)
                        with torch.no_grad():
                            q_online_next, h_online_next, c_online_next = self.model(
                                nst, batch_edge_index, h_online_next, c_online_next)
                            q_next, h_target, c_target = self.target_model(
                                nst, batch_edge_index, h_target, c_target)

                    # Double DQN vs DQN standard (controllato da use_double_dqn)
                    if self.use_double_dqn:
                        # Double DQN (van Hasselt et al., 2016): riduce bias di sovrastima
                        best_next_actions = q_online_next.argmax(dim=-1, keepdim=True)  # (B*N, 1)
                        q_next_value = q_next.gather(1, best_next_actions).squeeze(-1)  # (B*N,)
                    else:
                        # DQN standard: target = r + gamma * max_a' Q_target(s', a')
                        q_next_value = q_next.max(dim=-1).values  # (B*N,)

                    q_target = rt + self.gamma * q_next_value
                    q_action = q_pred.gather(1, at.unsqueeze(-1)).squeeze(-1)

                    # Rimodella a (B, N)
                    q_preds_valid.append(q_action.view(B, N))
                    q_targets_valid.append(q_target.detach().view(B, N))
            
            # (T_valid, B, N)
            q_preds_all = torch.stack(q_preds_valid, dim=0)
            q_targets_all = torch.stack(q_targets_valid, dim=0)
            
            # Trasponi a (B, T_valid, N)
            q_preds_all = q_preds_all.transpose(0, 1)
            q_targets_all = q_targets_all.transpose(0, 1)
            
            # Calcola l'errore TD medio per sequenza per aggiornare le priorità
            td_errors = torch.abs(q_preds_all - q_targets_all)
            seq_errors = torch.mean(td_errors, dim=(1, 2)) # (B,)
            self.replay_buffer.update_priorities(keys, seq_errors.detach().cpu().numpy())
            
            # Huber Loss vs MSE (controllato da use_huber)
            if self.use_huber:
                loss_per_element = nn.functional.smooth_l1_loss(q_preds_all, q_targets_all, reduction='none')
            else:
                loss_per_element = nn.functional.mse_loss(q_preds_all, q_targets_all, reduction='none')
            loss_per_seq = torch.mean(loss_per_element, dim=(1, 2))  # (B,)

            # IS weights: 1.0 costante se no_per (campionamento uniforme)
            loss = torch.mean(loss_per_seq * is_weights_t)

            # Gradient descent
            self.optimizer.zero_grad()
            loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            self.total_steps += 1

            # ── Aggiornamento target network (soft update) ────────────────────
            if self.total_steps % self.target_update_freq == 0:
                self._update_target()

        self.model.eval()

        self.total_episodes += 1
        
        # Decay semplice di epsilon dopo l'episodio di allenamento
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return total_loss / n_updates

    def _update_target(self):
        """Aggiorna il target network: soft (Polyak) o hard copy in base a use_soft_update."""
        if self.use_soft_update:
            tau = self.target_update_tau
            for online_p, target_p in zip(
                self.model.parameters(), self.target_model.parameters()
            ):
                target_p.data.copy_(tau * online_p.data + (1 - tau) * target_p.data)
        else:
            # Hard update: copia diretta (come da paper originale)
            self.target_model.load_state_dict(self.model.state_dict())

    # ─── Checkpoint save/load ─────────────────────────────────────────────────

    def save_checkpoint(self, path: str, episode: int,
                        travel_time: float, extra_info: dict = None):
        """
        Salva un checkpoint completo del training.

        Il checkpoint contiene tutto il necessario per riprendere il training
        dall'episodio `episode`, incluso il replay buffer: senza di esso un
        `--resume` ripartirebbe con un buffer vuoto (bug corretto il 2026-09-11,
        vedi analisi_bug.md #1), rendendo inefficaci sia la diversita' del
        campionamento PER sia il gate min_buffer_size.

        Args:
            path:        percorso del file .pt
            episode:     numero episodio corrente
            travel_time: travel time medio dell'episodio (per log)
            extra_info:  dizionario opzionale con informazioni aggiuntive
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            # Stato del modello
            "model_state_dict": self.model.state_dict(),
            "target_model_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),

            # Stato del training
            "episode": episode,
            "total_steps": self.total_steps,
            "epsilon": self.epsilon,
            "best_travel_time": self.best_travel_time,
            "travel_time": travel_time,

            # Replay buffer (necessario per un resume corretto, vedi analisi_bug.md #1)
            "replay_buffer_state": self.replay_buffer.state_dict(),

            # Tipo di modello
            "model_class": type(self.model).__name__,

            # Info extra
            "extra_info": extra_info or {},
        }

        torch.save(checkpoint, path)
        print(f"  [CKPT] Checkpoint salvato in '{path}' (episodio {episode})")

    def load_checkpoint(self, path: str) -> int:
        """
        Carica un checkpoint e ripristina lo stato del training.

        Args:
            path: percorso del file .pt

        Returns:
            episode: numero dell'episodio da cui riprendere
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint non trovato: '{path}'")

        # weights_only=False: dal checkpoint carichiamo anche il replay buffer
        # (oggetti Transition, non solo tensori) — sicuro perche' i nostri stessi
        # checkpoint, mai file di terzi non fidati.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.total_steps = checkpoint.get("total_steps", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon)
        self.best_travel_time = checkpoint.get("best_travel_time", float("inf"))

        # Ripristina il replay buffer se presente (checkpoint pre-fix: bug #1 in
        # analisi_bug.md, non l'avevano mai salvato — restano compatibili, il
        # buffer parte semplicemente vuoto come accadeva prima della correzione).
        buffer_state = checkpoint.get("replay_buffer_state")
        if buffer_state is not None:
            self.replay_buffer.load_state_dict(buffer_state)
            print(f"         Replay buffer ripristinato: {len(self.replay_buffer)} transizioni")
        else:
            print("         [WARN] Checkpoint senza replay buffer salvato (pre-fix): riparte vuoto.")

        episode = checkpoint.get("episode", 0)
        print(f"  [CKPT] Checkpoint caricato da '{path}'")
        print(f"         Ripresa da episodio {episode + 1}")
        print(f"         Epsilon corrente: {self.epsilon:.4f}")
        print(f"         Best travel time: {self.best_travel_time:.2f}s")

        return episode

    def save_best(self, path: str, episode: int, travel_time: float):
        """Salva il modello migliore se travel_time è migliorato."""
        if travel_time < self.best_travel_time:
            self.best_travel_time = travel_time
            self.save_checkpoint(path, episode, travel_time,
                                 extra_info={"is_best": True})
            return True
        return False

    def get_training_state(self) -> dict:
        """Restituisce lo stato corrente del training (per logging)."""
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "epsilon": round(self.epsilon, 6),
            "best_travel_time": round(self.best_travel_time, 2),
            "buffer_size": len(self.replay_buffer),
        }
