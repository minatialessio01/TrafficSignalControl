"""
MetaSTGNN: Meta-learning Spatial-Temporal Graph Convolutional Network.

Variante di MetaSTGAT (metastgat.py) dove il meccanismo di attenzione grafica
(Meta-GAT) e' sostituito dalla regola di aggregazione a normalizzazione di
grado della GCN (Meta-GCN, meta_gcn.py) -- vedi `istruzioni seconda parte.md`
§1/§3. Tenuta come file separato invece che un flag dentro MetaSTGAT: sono
due architetture concettualmente distinte da confrontare (con vs senza
meccanismo di compatibilita' appreso per coppia di nodi), non una variazione
minore dello stesso modello.

Tutto il resto dell'architettura (Dual State Encoder, SMK/TMK-Learner,
Meta-LSTM) e' identico a MetaSTGAT: il Meta-LSTM e' gia' "meta" a modo suo e
non tocca in alcun modo il meccanismo spaziale, quindi resta invariato in
entrambe le varianti.
"""

import torch
import torch.nn as nn

from .state_encoder import DualStateEncoder
from .meta_knowledge_learner import SpatialMetaKnowledgeLearner, TemporalMetaKnowledgeLearner
from .meta_gcn import MetaGCNLayer, add_self_loops
from .meta_lstm import MetaLSTM


class MetaSTGNN(nn.Module):
    """
    MetaSTGNN: MetaSTGAT con Meta-GAT sostituito da Meta-GCN.

    Args: identici a MetaSTGAT (vedi metastgat.py) per essere intercambiabile
    da riga di comando (`--model MetaSTGAT|MetaSTGNN`); `num_heads' e'
    accettato solo per uniformita' di interfaccia con `build_model()` e
    inoltrato a MetaGCNLayer, che lo ignora (la GCN non ha teste di
    attenzione).
    """

    def __init__(self,
                 state_dim: int = 32,
                 hidden_dim: int = 64,
                 num_heads: int = 4,            # ignorato da MetaGCNLayer, solo compatibilita'
                 n_actions: int = 8,
                 spatial_meta_dim: int = 28,
                 temporal_meta_dim: int = 72,
                 meta_hidden_dim: int = 64,
                 dropout: float = 0.0,
                 use_tanh_meta: bool = True,
                 num_layers: int = 1):          # 1 o 2, vedi istruzioni seconda parte.md §2.1/§3.1
        super().__init__()
        if num_layers not in (1, 2):
            raise ValueError(f"num_layers deve essere 1 o 2, ricevuto {num_layers}")
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.num_layers = num_layers
        # Self-loop calcolati una volta per ogni edge_index distinto (la
        # topologia e' fissa entro un episodio, cambia solo tra config diverse
        # -- istruzioni seconda parte.md §3.3): cache banale via id() del
        # tensore, evita di rifare lo scan Python ad ogni singolo step.
        self._self_loop_cache = {}

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

        # Meta-GCN CST e CS -- stessa struttura a stack di MetaSTGAT (§2.1),
        # stessa scelta di query fissa/simmetrica descritta li' (MetaGCNLayer
        # ignora comunque `query`, ma manteniamo la stessa forma di forward
        # per coerenza di lettura tra i due file).
        self.meta_gcn_cst = nn.ModuleList([
            MetaGCNLayer(hidden_dim=hidden_dim, num_heads=num_heads,
                         meta_dim=meta_hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.meta_gcn_cs = nn.ModuleList([
            MetaGCNLayer(hidden_dim=hidden_dim, num_heads=num_heads,
                         meta_dim=meta_hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.q_head = nn.Linear(hidden_dim * 2, n_actions)
        self.dropout = nn.Dropout(dropout)

    def _edge_index_with_self_loops(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        key = id(edge_index)
        cached = self._self_loop_cache.get(key)
        if cached is not None and cached.device == edge_index.device:
            return cached
        ei = add_self_loops(edge_index, num_nodes)
        self._self_loop_cache = {key: ei}  # un solo entry: basta l'ultimo edge_index visto
        return ei

    def forward(self,
                states: torch.Tensor,
                edge_index: torch.Tensor,
                spatial_meta_feat: torch.Tensor,
                temporal_meta_feat: torch.Tensor,
                h_prev: torch.Tensor = None,
                c_prev: torch.Tensor = None
                ):
        """Stessa firma/contratto di MetaSTGAT.forward() (vedi metastgat.py),
        cosi' che DQNAgent possa trattare i due modelli in modo intercambiabile
        senza conoscerne il tipo concreto."""
        N = states.size(0)
        device = states.device

        if h_prev is None:
            h_prev = torch.zeros(N, self.hidden_dim, device=device)
        if c_prev is None:
            c_prev = torch.zeros(N, self.hidden_dim, device=device)

        edge_index = self._edge_index_with_self_loops(edge_index, N)

        e_i, e_j = self.encoder(states)

        smk = self.smk_learner(spatial_meta_feat)
        tmk = self.tmk_learner(temporal_meta_feat)

        x_i, c_t = self.meta_lstm(
            e_i=e_i, h_prev=h_prev, c_prev=c_prev, tmk_embedding=tmk
        )

        # Nessuna ELU esterna qui (a differenza dello stack di MetaSTGAT):
        # MetaGCNLayer applica gia' internamente una ReLU in uscita (prassi
        # standard GCN, istruzioni seconda parte.md §3.1 punto 4), che funge
        # gia' da non-linearita' tra un layer e il successivo.
        kv_cst = x_i
        for layer in self.meta_gcn_cst:
            kv_cst = layer(query=e_j, key=kv_cst, value=kv_cst,
                            edge_index=edge_index, meta_embedding=tmk)
        z_cst = kv_cst

        h_cs = e_j
        for layer in self.meta_gcn_cs:
            h_cs = layer(query=h_cs, key=h_cs, value=h_cs,
                         edge_index=edge_index, meta_embedding=smk)
        z_cs = h_cs

        z = torch.cat([z_cst, z_cs], dim=-1)
        q_values = self.q_head(z)

        return q_values, x_i, c_t

    def init_hidden(self, num_nodes: int, device: torch.device = None):
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(num_nodes, self.hidden_dim, device=device)
        c = torch.zeros(num_nodes, self.hidden_dim, device=device)
        return h, c
