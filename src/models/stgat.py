"""
STGAT: Spatial-Temporal Graph Attention Network (baseline senza meta-learning).

Implementa la Section 4.2 dell'articolo (Equazioni 5-12).

Architettura:
  1. Dual State Encoder → e_i (temporal branch), e_j (spatial branch)
  2. Current Spatial-Temporal Module:
       LSTM(e_i) → x_i           (Eq. 5)
       Q=e_j, K=x_i, V=x_i
       z_CST = Attention(Q, K, V)  (Eq. 6-10)
  3. Current Spatial Module:
       Q=K=V=e_j
       z_CS = Attention(Q, K, V)   (Eq. 11)
  4. Q-value prediction:
       q̃(o_i^t) = Dense(Concat(z_CST, z_CS))  (Eq. 12)

La differenza rispetto a MetaSTGAT:
  - STGAT usa i pesi fissi standard di LSTM e GAT
  - MetaSTGAT usa i pesi dinamici generati dai meta-learner
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_encoder import DualStateEncoder


class StandardGATLayer(nn.Module):
    """
    Layer GAT con pesi fissi (senza meta-learning).
    Usato nel baseline STGAT.

    Implementa l'attention delle Eq. 7-9.
    """

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_size = hidden_dim // num_heads
        # Scala corretta: √(head_size), non √(hidden_dim) (Vaswani et al., 2017)
        self.scale = math.sqrt(self.head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,     # (N, d1)
                key: torch.Tensor,       # (N, d1)
                value: torch.Tensor,     # (N, d1)
                edge_index: torch.Tensor  # (2, E)
                ) -> torch.Tensor:
        """
        Multi-head scaled dot-product attention su grafo (Eq. 7-9).

        Returns: (N, d1)
        """
        N = query.size(0)
        src, dst = edge_index[0], edge_index[1]

        head_size = self.hidden_dim // self.num_heads
        attn_heads = []

        for h in range(self.num_heads):
            start = h * head_size
            end = start + head_size

            Q_h = query[dst][:, start:end]   # (E, head_size)
            K_h = key[src][:, start:end]      # (E, head_size)
            V_h = value[src][:, start:end]    # (E, head_size)

            # Scaled dot-product (Eq. 7)
            dot = (Q_h * K_h).sum(dim=-1) / self.scale  # (E,)

            # Softmax normalizzato per nodo (Eq. 8)
            attn = self._edge_softmax(dot, dst, N)       # (E,)
            attn = self.dropout(attn)

            # Aggregazione pesata (Eq. 9)
            weighted_V = attn.unsqueeze(-1) * V_h        # (E, head_size)
            out_h = torch.zeros(N, head_size, device=query.device)
            out_h.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_V), weighted_V)
            attn_heads.append(out_h)

        # Concatena le teste (standard multi-head): ogni testa contribuisce con head_size
        # dimensioni distinte → output totale = num_heads * head_size = hidden_dim
        out = torch.cat(attn_heads, dim=-1)  # (N, hidden_dim)
        return out

    @staticmethod
    def _edge_softmax(scores, dst, num_nodes):
        max_scores = torch.zeros(num_nodes, device=scores.device)
        max_scores.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_scores[dst])
        sum_exp = torch.zeros(num_nodes, device=scores.device)
        sum_exp.scatter_add_(0, dst, exp_scores)
        return exp_scores / (sum_exp[dst] + 1e-9)


class STGAT(nn.Module):
    """
    Spatial-Temporal Graph Attention Network (baseline).

    Architettura:
      DualEncoder → LSTM → [CST Module + CS Module] → DQN head

    Args:
        state_dim:   dimensione stato input (default: 32)
        hidden_dim:  dimensione nascosta (default: 64, non specificata nel paper)
        num_heads:   teste di attenzione (default: 4, da Fig. 10b)
        n_actions:   numero di azioni/fasi (default: 8)
        dropout:     dropout (default: 0.0, non specificato nel paper)
    """

    def __init__(self,
                 state_dim: int = 32,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 n_actions: int = 8,
                 dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.n_actions = n_actions

        # 1. Dual encoder (Eq. 3-4)
        self.encoder = DualStateEncoder(state_dim, hidden_dim)

        # 2. LSTM temporale (Eq. 5)
        # Usa LSTM standard con pesi fissi (baseline senza meta)
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        # 3. CST Module: Attention(Q=e_j, K=x_i, V=x_i) (Eq. 6-10)
        self.cst_gat = StandardGATLayer(hidden_dim, num_heads, dropout)

        # 4. CS Module: Attention(Q=e_j, K=e_j, V=e_j) (Eq. 11)
        self.cs_gat = StandardGATLayer(hidden_dim, num_heads, dropout)

        # 5. DQN head: Dense(Concat(z_CST, z_CS)) → Q-values (Eq. 12)
        self.q_head = nn.Linear(hidden_dim * 2, n_actions)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                states: torch.Tensor,        # (N, state_dim)
                edge_index: torch.Tensor,    # (2, E)
                h_prev: torch.Tensor = None, # (N, hidden_dim)
                c_prev: torch.Tensor = None  # (N, hidden_dim)
                ):
        """
        Forward pass del modello STGAT.

        Args:
            states:     (N, state_dim) — stato corrente di ogni nodo
            edge_index: (2, E)         — topologia del grafo
            h_prev:     (N, D_h)       — hidden state LSTM precedente
            c_prev:     (N, D_h)       — cell state LSTM precedente

        Returns:
            q_values: (N, n_actions)  — Q-values per ogni nodo
            h_t:      (N, D_h)        — nuovo hidden state LSTM
            c_t:      (N, D_h)        — nuovo cell state LSTM
        """
        N = states.size(0)
        device = states.device

        # Inizializza hidden state se non fornito
        if h_prev is None:
            h_prev = torch.zeros(N, self.hidden_dim, device=device)
        if c_prev is None:
            c_prev = torch.zeros(N, self.hidden_dim, device=device)

        # ── 1. Dual State Encoding (Eq. 3-4) ──────────────────────────────────
        e_i, e_j = self.encoder(states)  # (N, D_h) ciascuno

        # ── 2. LSTM temporale (Eq. 5) ─────────────────────────────────────────
        x_i, c_t = self.lstm(e_i, (h_prev, c_prev))  # (N, D_h), (N, D_h)

        # ── 3. Current Spatial-Temporal Module (Eq. 6-10) ─────────────────────
        # Q = e_j, K = x_i, V = x_i
        z_cst = self.cst_gat(
            query=e_j,
            key=x_i,
            value=x_i,
            edge_index=edge_index
        )  # (N, D_h)

        # ── 4. Current Spatial Module (Eq. 11) ────────────────────────────────
        # Q = K = V = e_j
        z_cs = self.cs_gat(
            query=e_j,
            key=e_j,
            value=e_j,
            edge_index=edge_index
        )  # (N, D_h)

        # ── 5. Q-value prediction (Eq. 12) ────────────────────────────────────
        z = torch.cat([z_cst, z_cs], dim=-1)  # (N, 2*D_h)
        q_values = self.q_head(z)              # (N, n_actions)

        return q_values, x_i, c_t

    def init_hidden(self, num_nodes: int, device: torch.device = None):
        """Inizializza gli stati nascosti a zero."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(num_nodes, self.hidden_dim, device=device)
        c = torch.zeros(num_nodes, self.hidden_dim, device=device)
        return h, c
