"""
MetaSTSONAR: Meta-learning Spatial-Temporal SONAR.

Variante di MetaSTGAT (metastgat.py) dove il meccanismo di attenzione grafica
(Meta-GAT) e' sostituito dalla propagazione a onda di Meta-SONAR
(meta_sonar.py) -- vedi `istruzioni seconda parte.md` §4.

A differenza di MetaSTGAT/MetaSTGNN, qui non esiste un parametro `num_layers`:
la profondita' della propagazione spaziale e' governata da `n_recurrences`
(L), il numero di passi di discretizzazione eseguiti *dentro* un singolo
MetaSONARLayer a pesi condivisi tra le ricorrenze (non layer indipendenti
impilati) -- e' il motivo per cui il confronto con GAT/GCN nella tesi e' "a
parita' di L", non "a parita' di numero di layer" (istruzioni seconda
parte.md §0, tabella a inizio file).

Tutto il resto (Dual State Encoder, SMK/TMK-Learner, Meta-LSTM) e' identico a
MetaSTGAT.
"""

import torch
import torch.nn as nn

from .state_encoder import DualStateEncoder
from .meta_knowledge_learner import SpatialMetaKnowledgeLearner, TemporalMetaKnowledgeLearner
from .meta_sonar import MetaSONARLayer
from .meta_lstm import MetaLSTM


class MetaSTSONAR(nn.Module):
    """
    MetaSTSONAR: MetaSTGAT con Meta-GAT sostituito da Meta-SONAR.

    Args: identici a MetaSTGAT per la parte condivisa; `num_heads` e'
    accettato solo per uniformita' di interfaccia con `build_model()` e
    ignorato da MetaSONARLayer. `n_recurrences` (L) e `step_size` (h) sono i
    due nuovi iperparametri specifici di SONAR (istruzioni seconda parte.md
    §4.1 punto 4).
    """

    def __init__(self,
                 state_dim: int = 32,
                 hidden_dim: int = 64,
                 num_heads: int = 4,            # ignorato da MetaSONARLayer, solo compatibilita'
                 n_actions: int = 8,
                 spatial_meta_dim: int = 28,
                 temporal_meta_dim: int = 72,
                 meta_hidden_dim: int = 64,
                 dropout: float = 0.0,
                 use_tanh_meta: bool = True,
                 n_recurrences: int = 2,        # L, raccomandazione D1: 2 (eventuale secondo valore 4)
                 step_size: float = 0.1,        # h, raccomandazione D1/§4.1: parti da 0.1
                 use_dissipation: bool = True,
                 use_forcing: bool = True):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.n_recurrences = n_recurrences
        self.step_size = step_size

        self.encoder = DualStateEncoder(state_dim, hidden_dim)

        self.smk_learner = SpatialMetaKnowledgeLearner(
            spatial_dim=spatial_meta_dim,
            hidden_dim=meta_hidden_dim,
            output_dim=meta_hidden_dim,
            use_tanh_output=use_tanh_meta
        )
        self.tmk_learner = TemporalMetaKnowledgeLearner(
            temporal_dim=temporal_meta_dim,
            hidden_dim=meta_hidden_dim,
            output_dim=meta_hidden_dim,
            use_tanh_output=use_tanh_meta
        )

        self.meta_lstm = MetaLSTM(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            meta_dim=meta_hidden_dim
        )

        # Meta-SONAR CST e CS: un solo blocco ciascuno (L ricorrenze interne
        # a pesi condivisi, non layer impilati indipendenti -- vedi docstring
        # del modulo).
        self.meta_sonar_cst = MetaSONARLayer(
            hidden_dim=hidden_dim, num_heads=num_heads, meta_dim=meta_hidden_dim,
            n_recurrences=n_recurrences, step_size=step_size,
            use_dissipation=use_dissipation, use_forcing=use_forcing, dropout=dropout
        )
        self.meta_sonar_cs = MetaSONARLayer(
            hidden_dim=hidden_dim, num_heads=num_heads, meta_dim=meta_hidden_dim,
            n_recurrences=n_recurrences, step_size=step_size,
            use_dissipation=use_dissipation, use_forcing=use_forcing, dropout=dropout
        )

        self.q_head = nn.Linear(hidden_dim * 2, n_actions)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                states: torch.Tensor,
                edge_index: torch.Tensor,
                spatial_meta_feat: torch.Tensor,
                temporal_meta_feat: torch.Tensor,
                h_prev: torch.Tensor = None,
                c_prev: torch.Tensor = None
                ):
        """Stessa firma/contratto di MetaSTGAT.forward()."""
        N = states.size(0)
        device = states.device

        if h_prev is None:
            h_prev = torch.zeros(N, self.hidden_dim, device=device)
        if c_prev is None:
            c_prev = torch.zeros(N, self.hidden_dim, device=device)

        e_i, e_j = self.encoder(states)

        smk = self.smk_learner(spatial_meta_feat)
        tmk = self.tmk_learner(temporal_meta_feat)

        x_i, c_t = self.meta_lstm(
            e_i=e_i, h_prev=h_prev, c_prev=c_prev, tmk_embedding=tmk
        )

        # CST: query=e_j (sorgente di V^0), value=x_i (posizione iniziale X^0), meta=TMK(i)
        z_cst = self.meta_sonar_cst(
            query=e_j, key=x_i, value=x_i,
            edge_index=edge_index, meta_embedding=tmk
        )

        # CS: query=key=value=e_j, meta=SMK(i)
        z_cs = self.meta_sonar_cs(
            query=e_j, key=e_j, value=e_j,
            edge_index=edge_index, meta_embedding=smk
        )

        z = torch.cat([z_cst, z_cs], dim=-1)
        q_values = self.q_head(z)

        return q_values, x_i, c_t

    def init_hidden(self, num_nodes: int, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(num_nodes, self.hidden_dim, device=device)
        c = torch.zeros(num_nodes, self.hidden_dim, device=device)
        return h, c
