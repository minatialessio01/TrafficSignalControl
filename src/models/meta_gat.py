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
        self.head_dim = hidden_dim  # D_h (usiamo full dim per ogni testa)
        self.dropout = dropout
        self.scale = math.sqrt(self.head_dim)

        # MetaDense per generare W e b dal meta-embedding (Eq. 14 o 16)
        self.meta_dense = MetaDense(meta_dim, hidden_dim, hidden_dim)

        # Proiezioni Q, K, V (opzionali: nell'articolo non sono menzionate
        # proiezioni esplicite; usiamo proiezioni identità per H teste)
        # Nota: il paper usa Q=e_j, K=x_i, V=x_i direttamente senza proiezione
        # aggiuntiva — i pesi dinamici sono solo nel dot-product scaling

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,          # Q_i: (N, d1)  — e_j o e_j
                key: torch.Tensor,             # K: (N, d1)    — x_i o e_j
                value: torch.Tensor,           # V: (N, d1)    — x_i o e_j
                edge_index: torch.Tensor,      # (2, E)
                meta_embedding: torch.Tensor,  # SMK(i) o TMK(i): (N, meta_dim)
                ) -> torch.Tensor:
        """
        Args:
            query:          (N, d1)  — rappresentazione query per ogni nodo
            key:            (N, d1)  — rappresentazione key per ogni nodo
            value:          (N, d1)  — rappresentazione value per ogni nodo
            edge_index:     (2, E)   — archi del grafo [src, dst]
            meta_embedding: (N, meta_dim) — embedding meta-knowledge per ogni nodo

        Returns:
            out: (N, d1) — aggregazione spaziale pesata dai vicini
        """
        N = query.size(0)
        E = edge_index.size(1)
        src, dst = edge_index[0], edge_index[1]  # dst ← src

        # Genera i pesi dinamici per ogni nodo (Eq. 14 o 16)
        W, b = self.meta_dense(meta_embedding)   # W: (N, D_h, d1), b: (N, 1)

        # ── Attention score (Eq. 15 o 17) ─────────────────────────────────────
        # Per ogni arco (src → dst):
        #   φ(Q_{dst}, K_{src}) = [W_{dst} · (Q_{dst} · K_{src}) + b_{dst}] / √D_h

        Q_dst = query[dst]   # (E, d1)  — query del nodo destinazione
        K_src = key[src]     # (E, d1)  — key del nodo sorgente (vicino)

        # Dot-product scalare tra Q e K (senza proiezione separata per testa
        # come nel paper originale, che usa la forma in Eq. 7)
        # Aggregiamo le H teste come media (Eq. 9: 1/H Σ_h)
        head_size = self.hidden_dim // self.num_heads
        attn_heads = []

        for h in range(self.num_heads):
            start = h * head_size
            end = start + head_size

            Q_h = Q_dst[:, start:end]  # (E, head_size)
            K_h = K_src[:, start:end]  # (E, head_size)

            # Applica i pesi dinamici del meta-learner (Eq. 15/17)
            # W_dst è (N, D_h, d1) → prendiamo la proiezione per questa testa
            W_h = W[dst, start:end, start:end]   # (E, head_size, head_size)
            
            # Trasformiamo la query con la matrice dinamica W_h
            Q_h_trans = (W_h @ Q_h.unsqueeze(-1)).squeeze(-1)  # (E, head_size)
            
            # Dot-product tra la query trasformata e la key
            dot = (Q_h_trans * K_h).sum(dim=-1)  # (E,)
            
            # Scalatura e bias
            dot_scaled = dot + b[dst].squeeze(-1)  # (E,)
            dot_scaled = dot_scaled / self.scale  # / √D_h
            
            attn_heads.append(dot_scaled)

        # Media delle teste (Eq. 9: 1/H Σ_h)
        attn_score = torch.stack(attn_heads, dim=-1).mean(dim=-1)  # (E,)

        # ── Softmax per ogni nodo destinazione ────────────────────────────────
        # (Eq. 8: exp(φ) / Σ_{u'∈N_i} exp(φ))
        attn_weight = self._edge_softmax(attn_score, dst, N)  # (E,)
        attn_weight = self.dropout_layer(attn_weight)

        # ── Aggregazione dei valori (Eq. 9) ────────────────────────────────────
        V_src = value[src]  # (E, d1)
        weighted_V = attn_weight.unsqueeze(-1) * V_src  # (E, d1)

        # Somma per ogni nodo dst
        out = torch.zeros(N, self.hidden_dim, device=query.device)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_V), weighted_V)

        return out  # (N, d1) = z_CST o z_CS

    @staticmethod
    def _edge_softmax(scores: torch.Tensor,
                      dst: torch.Tensor,
                      num_nodes: int) -> torch.Tensor:
        """
        Softmax per arco, normalizzato per ogni nodo destinazione.

        Equivale a: α_{i,u} = exp(φ_{i,u}) / Σ_{u'∈N_i} exp(φ_{i,u'})
        """
        # Sottrai il massimo per stabilità numerica
        max_scores = torch.zeros(num_nodes, device=scores.device)
        max_scores.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
        scores_shifted = scores - max_scores[dst]

        exp_scores = torch.exp(scores_shifted)

        # Somma per ogni nodo dst
        sum_exp = torch.zeros(num_nodes, device=scores.device)
        sum_exp.scatter_add_(0, dst, exp_scores)

        # Normalizza
        return exp_scores / (sum_exp[dst] + 1e-9)
