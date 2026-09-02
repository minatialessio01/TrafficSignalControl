"""
MetaSTGAT: Meta-learning Spatial-Temporal Graph Attention Network.

Implementa il modello completo descritto nell'articolo (Section 4.3, Fig. 5).

Architettura completa:
  1. Dual State Encoder → e_i (temporal), e_j (spatial)
  2. Meta-knowledge learners:
       SMK(i) = SMKLearner(spatial_features_i)    [feature spaziali]
       TMK(i) = TMKLearner(temporal_features_i)   [feature temporali]
  3. Meta-LSTM: x_i^t = MetaLSTM(e_i, h_{t-1} | MetaDense3(TMK(i)))
  4. Meta-GAT (CST module):
       Q=e_j, K=x_i, V=x_i, meta=TMK(i)
       z_CST = MetaGAT(Q, K, V, edge_index, TMK(i))
  5. Meta-GAT (CS module):
       Q=K=V=e_j, meta=SMK(i)
       z_CS = MetaGAT(Q, K, V, edge_index, SMK(i))
  6. Q-value: q̃(o_i^t) = Dense(Concat(z_CST, z_CS))

Confronto con STGAT baseline:
  - STGAT: LSTM e GAT con pesi fissi
  - MetaSTGAT: LSTM e GAT con pesi generati dinamicamente dai meta-learner
"""

import torch
import torch.nn as nn

from .state_encoder import DualStateEncoder
from .meta_knowledge_learner import SpatialMetaKnowledgeLearner, TemporalMetaKnowledgeLearner
from .meta_gat import MetaGATLayer
from .meta_lstm import MetaLSTM


class MetaSTGAT(nn.Module):
    """
    MetaSTGAT: modello completo con meta-learning.

    Args:
        state_dim:      dimensione stato input (d0=20)
        hidden_dim:     dimensione nascosta (d1=D_h, default 64)
        num_heads:      teste attenzione (H, default 4)
        n_actions:      numero di azioni/fasi (default 8)
        spatial_meta_dim:  dimensione feature spaziali per SMK-Learner
        temporal_meta_dim: dimensione feature temporali per TMK-Learner
        meta_hidden_dim:   dimensione nascosta dei meta-learner (default 64)
        dropout:           dropout (default 0.0)
    """

    def __init__(self,
                 state_dim: int = 20,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 n_actions: int = 8,
                 spatial_meta_dim: int = 28,    # N_LANES*2 + num_neighbors = 28
                 temporal_meta_dim: int = 72,   # N_LANES + N_LANES*5 = 72
                 meta_hidden_dim: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.n_actions = n_actions

        # ── 1. Dual State Encoder (Eq. 3-4) ───────────────────────────────────
        self.encoder = DualStateEncoder(state_dim, hidden_dim)

        # ── 2. Meta-Knowledge Learners (Section 4.3.1) ────────────────────────
        # SMK-Learner: feature spaziali → embedding spaziale
        self.smk_learner = SpatialMetaKnowledgeLearner(
            spatial_dim=spatial_meta_dim,
            hidden_dim=meta_hidden_dim,
            output_dim=meta_hidden_dim
        )
        # TMK-Learner: feature temporali → embedding temporale
        self.tmk_learner = TemporalMetaKnowledgeLearner(
            temporal_dim=temporal_meta_dim,
            hidden_dim=meta_hidden_dim,
            output_dim=meta_hidden_dim
        )

        # ── 3. Meta-LSTM (Section 4.3.3, Eq. 20-21) ───────────────────────────
        self.meta_lstm = MetaLSTM(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            meta_dim=meta_hidden_dim
        )

        # ── 4. Meta-GAT CST Module (Section 4.3.2, Eq. 16-17) ────────────────
        # Q=e_j, K=x_i, V=x_i; meta=TMK(i)
        self.meta_gat_cst = MetaGATLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            meta_dim=meta_hidden_dim,
            dropout=dropout
        )

        # ── 5. Meta-GAT CS Module (Section 4.3.2, Eq. 14-15) ─────────────────
        # Q=K=V=e_j; meta=SMK(i)
        self.meta_gat_cs = MetaGATLayer(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            meta_dim=meta_hidden_dim,
            dropout=dropout
        )

        # ── 6. DQN Head (Eq. 12) ──────────────────────────────────────────────
        self.q_head = nn.Linear(hidden_dim * 2, n_actions)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                states: torch.Tensor,              # (N, state_dim)
                edge_index: torch.Tensor,          # (2, E)
                spatial_meta_feat: torch.Tensor,   # (N, spatial_meta_dim)
                temporal_meta_feat: torch.Tensor,  # (N, temporal_meta_dim)
                h_prev: torch.Tensor = None,       # (N, D_h)
                c_prev: torch.Tensor = None        # (N, D_h)
                ):
        """
        Forward pass di MetaSTGAT.

        Args:
            states:             (N, state_dim)         — stato corrente
            edge_index:         (2, E)                 — topologia grafo
            spatial_meta_feat:  (N, spatial_meta_dim)  — feature spaziali raw
            temporal_meta_feat: (N, temporal_meta_dim) — feature temporali raw
            h_prev:             (N, D_h)               — hidden LSTM precedente
            c_prev:             (N, D_h)               — cell LSTM precedente

        Returns:
            q_values: (N, n_actions)
            h_t:      (N, D_h)
            c_t:      (N, D_h)
        """
        N = states.size(0)
        device = states.device

        # Inizializza hidden state se non fornito
        if h_prev is None:
            h_prev = torch.zeros(N, self.hidden_dim, device=device)
        if c_prev is None:
            c_prev = torch.zeros(N, self.hidden_dim, device=device)

        # ── 1. Dual State Encoding (Eq. 3-4) ──────────────────────────────────
        e_i, e_j = self.encoder(states)   # (N, D_h) ciascuno

        # ── 2. Meta-knowledge learning (Section 4.3.1) ────────────────────────
        smk = self.smk_learner(spatial_meta_feat)    # (N, meta_hidden_dim)
        tmk = self.tmk_learner(temporal_meta_feat)   # (N, meta_hidden_dim)

        # ── 3. Meta-LSTM (Eq. 20-21) ─────────────────────────────────────────
        x_i, c_t = self.meta_lstm(
            e_i=e_i,
            h_prev=h_prev,
            c_prev=c_prev,
            tmk_embedding=tmk
        )  # x_i: (N, D_h), c_t: (N, D_h)

        # ── 4. Meta-GAT CST Module (Eq. 16-17) ───────────────────────────────
        # Q = e_j, K = x_i, V = x_i; meta = TMK(i)
        z_cst = self.meta_gat_cst(
            query=e_j,
            key=x_i,
            value=x_i,
            edge_index=edge_index,
            meta_embedding=tmk
        )  # (N, D_h)

        # ── 5. Meta-GAT CS Module (Eq. 14-15) ────────────────────────────────
        # Q = K = V = e_j; meta = SMK(i)
        z_cs = self.meta_gat_cs(
            query=e_j,
            key=e_j,
            value=e_j,
            edge_index=edge_index,
            meta_embedding=smk
        )  # (N, D_h)

        # ── 6. Q-value prediction (Eq. 12) ────────────────────────────────────
        z = torch.cat([z_cst, z_cs], dim=-1)  # (N, 2*D_h)
        q_values = self.q_head(z)              # (N, n_actions)

        return q_values, x_i, c_t

    def init_hidden(self, num_nodes: int, device: torch.device = None):
        """Inizializza gli stati nascosti LSTM a zero."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(num_nodes, self.hidden_dim, device=device)
        c = torch.zeros(num_nodes, self.hidden_dim, device=device)
        return h, c
