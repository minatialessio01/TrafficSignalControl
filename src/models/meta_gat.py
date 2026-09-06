"""
Meta Graph Attention Network (Meta-GAT).

Implementa la Section 4.3.2 dell'articolo (Equazioni 14-17):

  Meta-GAT (Spatial Module):
    W_j, b_j = MetaDense(1)(SMK(i))          # Eq. 14
    φ(Q_i, K_{i,u}) = [W_j · (Q_i · K_{i,u}) + b_j] / √D_h   # Eq. 15

  Meta-GAT (Spatial-Temporal Module):
    W_k, b_k = MetaDense(2)(TMK(i))          # Eq. 16
    φ(Q_i, K_{i,u}) = [W_k · (Q_i · K_{i,u}) + b_k] / √D_h   # Eq. 17

Il meccanismo di attenzione è multi-head (Eq. 7-9 per STGAT, esteso con
i pesi dinamici del meta-learner per MetaSTGAT).

I pesi W_j, b_j (e W_k, b_k) vengono generati a ogni forward pass dal
meta-learner (non sono parametri fissi del modello), implementando il
principio di "weight generation" tipico del meta-learning.

Architettura della MetaDense:
  - Input: meta-knowledge embedding (dim = meta_hidden_dim)
  - Layer 1: Linear(meta_hidden_dim, D_h) con ReLU
  - Output peso W: Linear(D_h, D_h * hidden_dim)
  - Output bias b: Linear(D_h, 1)
  (2 layer, come indicato nell'articolo: "two-layer fully connected network")
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaDense(nn.Module):
    """
    Rete MetaDense per la generazione dinamica dei pesi.

    Prende in input un meta-embedding e produce W e b da usare
    come parametri dinamici dell'attention mechanism.

    W ∈ R^{D_h × d1},  b ∈ R   (Eq. 14/16)
    """

    def __init__(self, meta_dim: int, hidden_dim: int, out_w_dim: int):
        """
        Args:
            meta_dim:    dimensione input (embedding meta-knowledge)
            hidden_dim:  D_h (dimensione hidden attention)
            out_w_dim:   d1  (dimensione output attention, tipicamente = hidden_dim)
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_w_dim = out_w_dim

        # Layer intermedio
        self.fc1 = nn.Linear(meta_dim, hidden_dim)
        self.activation = nn.ReLU()

        # Genera il peso W: output è hidden_dim × out_w_dim (appiattito)
        self.fc_weight = nn.Linear(hidden_dim, hidden_dim * out_w_dim)
        # Genera il bias b: scalare
        self.fc_bias = nn.Linear(hidden_dim, 1)

    def forward(self, meta_embedding: torch.Tensor):
        """
        Args:
            meta_embedding: (batch, meta_dim)

        Returns:
            W: (batch, hidden_dim, out_w_dim)
            b: (batch, 1)
        """
        h = self.activation(self.fc1(meta_embedding))           # (B, D_h)
        W_flat = self.fc_weight(h)                               # (B, D_h * d1)
        W = W_flat.view(-1, self.hidden_dim, self.out_w_dim)    # (B, D_h, d1)
        b = self.fc_bias(h)                                      # (B, 1)
        return W, b


class MetaGATLayer(nn.Module):
    """
    Singolo layer di Meta-GAT con multi-head attention.

    Implementa le Eq. 14-17 dell'articolo MetaSTGAT.

    Funzionamento:
      1. Il meta-learner produce W_j, b_j (o W_k, b_k) dal meta-embedding
      2. Questi pesi scalano il dot-product tra query Q e key K dei vicini
      3. L'attention score risultante è normalizzato con softmax
      4. Il valore atteso è la somma pesata dei valori V dei vicini

    Args:
        hidden_dim:  d1 = D_h (dimensione nascosta)
        num_heads:   H (numero teste multi-head, paper usa 4)
        meta_dim:    dimensione dell'embedding meta-knowledge in input
        dropout:     dropout sull'attention (non specificato, usiamo 0.0)
    """

    def __init__(self,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 meta_dim: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # d_h = D_h / H (es. 64 / 4 = 16)
        head_size = hidden_dim // num_heads
        self.head_size = head_size
        self.dropout = dropout
        self.scale = math.sqrt(head_size)

        # Generatore di pesi dinamici (MetaDense, Eq. 14/16):
        #   Meta-embedding → W_h per ogni testa (H × hs × hs) + b_h per testa (H)
        #   Primo strato FC con ReLU, poi due head lineari per W e b
        #   ("two-layer fully connected network", Section 4.3.1)
        self.meta_fc1 = nn.Linear(meta_dim, hidden_dim)
        self.meta_fc_W = nn.Linear(hidden_dim, num_heads * head_size * head_size)
        self.meta_fc_b = nn.Linear(hidden_dim, num_heads)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,          # Q_i: (N, d1)
                key: torch.Tensor,             # K: (N, d1)
                value: torch.Tensor,           # V: (N, d1)
                edge_index: torch.Tensor,      # (2, E)
                meta_embedding: torch.Tensor,  # SMK(i) o TMK(i): (N, meta_dim)
                ) -> torch.Tensor:
        """
        Args:
            query:          (N, d1)
            key:            (N, d1)
            value:          (N, d1)
            edge_index:     (2, E)
            meta_embedding: (N, meta_dim)

        Returns:
            out: (N, d1) — aggregazione spaziale pesata dai vicini
        """
        N = query.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Genera i pesi dinamici per ogni nodo e per ogni testa (Eq. 14 o 16)
        # m → W: (N, H, hs, hs)  e  b: (N, H)
        m = F.relu(self.meta_fc1(meta_embedding))                          # (N, D_h)
        W_flat = self.meta_fc_W(m)                                          # (N, H*hs*hs)
        W = W_flat.view(N, self.num_heads, self.head_size, self.head_size)  # (N, H, hs, hs)
        b = self.meta_fc_b(m)                                               # (N, H)

        # ── Multi-head attention con pesi dinamici (Eq. 15/17) ─────────────
        attn_heads = []

        for h in range(self.num_heads):
            start = h * self.head_size
            end = start + self.head_size

            Q_h = query[dst, start:end]   # (E, head_size)
            K_h = key[src, start:end]     # (E, head_size)
            V_h = value[src, start:end]   # (E, head_size)

            # Trasforma la query con la matrice dinamica W_h (Eq. 15/17)
            W_h = W[dst, h, :, :]                                        # (E, hs, hs)
            Q_h_trans = (W_h @ Q_h.unsqueeze(-1)).squeeze(-1)            # (E, hs)

            # Attention score: dot-product scalato + bias scalare per testa
            dot = (Q_h_trans * K_h).sum(dim=-1) + b[dst, h]             # (E,)
            dot_scaled = dot / self.scale                                  # (E,)

            # Softmax per nodo destinazione (Eq. 8)
            attn = self._edge_softmax(dot_scaled, dst, N)                 # (E,)
            attn = self.dropout_layer(attn)

            # Aggregazione pesata dei valori V (Eq. 9)
            weighted_V = attn.unsqueeze(-1) * V_h                        # (E, hs)
            out_h = torch.zeros(N, self.head_size, device=query.device)
            out_h.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_V), weighted_V)
            attn_heads.append(out_h)

        # Concatena le teste: (N, H * hs) = (N, d1)
        out = torch.cat(attn_heads, dim=-1)
        return out

    @staticmethod
    def _edge_softmax(scores: torch.Tensor,
                      dst: torch.Tensor,
                      num_nodes: int) -> torch.Tensor:
        """
        Softmax per arco, normalizzato per ogni nodo destinazione.

        Equivale a: α_{i,u} = exp(φ_{i,u}) / Σ_{u'∈N_i} exp(φ_{i,u'})
        """
        max_scores = torch.zeros(num_nodes, device=scores.device)
        max_scores.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
        scores_shifted = scores - max_scores[dst]

        exp_scores = torch.exp(scores_shifted)

        sum_exp = torch.zeros(num_nodes, device=scores.device)
        sum_exp.scatter_add_(0, dst, exp_scores)

        return exp_scores / (sum_exp[dst] + 1e-9)
