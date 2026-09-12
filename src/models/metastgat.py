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

Estensione "seconda parte" (`istruzioni seconda parte.md` §2.1): `num_layers`
permette di impilare 1 o 2 layer indipendenti (pesi propri, non condivisi) sia
per il modulo CST sia per il CS, per confrontare GAT-1L vs GAT-2L a parita' di
resto dell'architettura contro le varianti GCN/SONAR equivalenti
(metastgnn.py, metastsonar.py). Con num_layers=1 il comportamento e'
identico bit-per-bit alla versione precedente di questo file (nessuna
regressione sul modello Pro gia' validato -- verificato con un test di forma,
vedi descrizione_gcn_sonar.md).

Self-loop su edge_index (12/9/2026, decisione definitiva -- vedi
descrizione_gcn_sonar.md §0 e descrizione_modelli.md §4): il Meta-GAT originale
non aggiungeva mai self-loop a edge_index, una deviazione dalla formulazione
standard di Velickovic et al. (2018), che include esplicitamente il nodo nel
proprio vicinato. Corretto qui in via permanente dopo un esperimento
controllato (metastgat_pro_0.2_self_loop, metastgat_pro_0.0_self_loop):
miglioramento del travel time medio del 9.5% (alpha=0.2) e 18.4% (alpha=0.0),
IDENTICO in proporzione tra config di training e di generalizzazione -- non
overfitting alla topologia vista in training, un effetto strutturale reale.
Essendo ora parte dell'architettura base (non un flag di ablation), si applica
a ogni preset (pro/paper/environment/temporal/rl_core/replay_stability) senza
eccezioni: ogni modello MetaSTGAT allenato PRIMA di questa modifica (tutti i
model_id sotto results/ tranne i tre *_self_loop) non e' piu' comparabile ad
armi pari con un modello allenato dopo -- vanno ri-allenati per restare validi
(stesso motivo per cui l'esperimento e' stato trattato come tale invece che
come una correzione silenziosa, finche' non se ne e' avuta la controprova).
Stessa correzione applicata per coerenza a StandardGATLayer
(src/models/stgat.py), che aveva la stessa lacuna strutturale -- nessun
modello STGAT era mai stato allenato in questo progetto, quindi li' la
modifica non ha alcun costo di ri-addestramento.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .state_encoder import DualStateEncoder
from .meta_knowledge_learner import SpatialMetaKnowledgeLearner, TemporalMetaKnowledgeLearner
from .meta_gat import MetaGATLayer
from .meta_gcn import add_self_loops
from .meta_lstm import MetaLSTM


class MetaSTGAT(nn.Module):
    """
    MetaSTGAT: modello completo con meta-learning.

    Args:
        state_dim:      dimensione stato input (d0=32)
        hidden_dim:     dimensione nascosta (d1=D_h, default 64)
        num_heads:      teste attenzione (H, default 4)
        n_actions:      numero di azioni/fasi (default 8)
        spatial_meta_dim:  dimensione feature spaziali per SMK-Learner
        temporal_meta_dim: dimensione feature temporali per TMK-Learner
        meta_hidden_dim:   dimensione nascosta dei meta-learner (default 64)
        dropout:           dropout (default 0.0)
        num_layers:        numero di layer Meta-GAT indipendenti impilati per
                            ciascun modulo (CST e CS), 1 (default, comportamento
                            originale) o 2 (istruzioni seconda parte.md §2.1)
    """

    def __init__(self,
                 state_dim: int = 32,
                 hidden_dim: int = 64,
                 num_heads: int = 4,
                 n_actions: int = 8,
                 spatial_meta_dim: int = 28,    # N_LANES*2 + num_neighbors = 28
                 temporal_meta_dim: int = 72,   # N_LANES + N_LANES*5 = 72
                 meta_hidden_dim: int = 64,
                 dropout: float = 0.0,
                 use_tanh_meta: bool = True,    # Ablation: False = no Tanh sui meta-learner
                 num_layers: int = 1):          # 1 (default) o 2, vedi istruzioni seconda parte.md §2.1
        super().__init__()
        if num_layers not in (1, 2):
            raise ValueError(f"num_layers deve essere 1 o 2, ricevuto {num_layers}")
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.n_actions = n_actions
        self.num_layers = num_layers
        # Self-loop su edge_index, parte permanente dell'architettura dal
        # 12/9/2026 (vedi nota in cima al file) -- cache per evitare di
        # ricalcolarli ad ogni forward (edge_index e' costante per l'intera
        # config), chiave = (id(edge_index), N).
        self._self_loop_cache = {}

        # ── 1. Dual State Encoder (Eq. 3-4) ───────────────────────────────────
        self.encoder = DualStateEncoder(state_dim, hidden_dim)

        # ── 2. Meta-Knowledge Learners (Section 4.3.1) ────────────────────────
        # SMK-Learner: feature spaziali → embedding spaziale
        self.smk_learner = SpatialMetaKnowledgeLearner(
            spatial_dim=spatial_meta_dim,
            hidden_dim=meta_hidden_dim,
            output_dim=meta_hidden_dim,
            use_tanh_output=use_tanh_meta
        )
        # TMK-Learner: feature temporali → embedding temporale
        self.tmk_learner = TemporalMetaKnowledgeLearner(
            temporal_dim=temporal_meta_dim,
            hidden_dim=meta_hidden_dim,
            output_dim=meta_hidden_dim,
            use_tanh_output=use_tanh_meta
        )

        # ── 3. Meta-LSTM (Section 4.3.3, Eq. 20-21) ───────────────────────────
        self.meta_lstm = MetaLSTM(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            meta_dim=meta_hidden_dim
        )

        # ── 4. Meta-GAT CST Module (Section 4.3.2, Eq. 16-17) ────────────────
        # Q=e_j, K=x_i, V=x_i; meta=TMK(i). Con num_layers=2: due layer
        # indipendenti (pesi propri, MAI condivisi tra loro -- a differenza di
        # SONAR, dove invece le L ricorrenze condividono i pesi, vedi
        # istruzioni seconda parte.md §0/§2.1); query e' tenuta fissa a e_j per
        # entrambi i layer, key/value si raffinano da un layer al successivo
        # (scelta dichiarata: l'alternativa "aggiorna anche la query" e'
        # egualmente legittima, non e' ovvio quale sia meglio).
        self.meta_gat_cst = nn.ModuleList([
            MetaGATLayer(hidden_dim=hidden_dim, num_heads=num_heads,
                         meta_dim=meta_hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        # ── 5. Meta-GAT CS Module (Section 4.3.2, Eq. 14-15) ─────────────────
        # Q=K=V=e_j; meta=SMK(i). Con num_layers=2: due layer indipendenti,
        # stack simmetrico standard (Q=K=V=output del layer precedente).
        self.meta_gat_cs = nn.ModuleList([
            MetaGATLayer(hidden_dim=hidden_dim, num_heads=num_heads,
                         meta_dim=meta_hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

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

        edge_index = self._edge_index_with_self_loops(edge_index, N)

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
        # Q = e_j (fissa su tutti i layer), K = V = x_i che si raffina da un
        # layer al successivo; meta = TMK(i) per ogni layer. La non-linearita'
        # ELU si applica solo TRA un layer e il successivo (mai dopo l'ultimo):
        # con num_layers=1 il ciclo esegue un solo layer senza alcuna ELU
        # aggiuntiva, identico bit-per-bit al comportamento precedente questa
        # modifica -- se l'ELU venisse applicata anche dopo l'ultimo layer,
        # num_layers=1 smetterebbe di essere una non-regressione del modello
        # Pro gia' validato.
        kv_cst = x_i
        n_layers_cst = len(self.meta_gat_cst)
        for i, layer in enumerate(self.meta_gat_cst):
            kv_cst = layer(query=e_j, key=kv_cst, value=kv_cst,
                            edge_index=edge_index, meta_embedding=tmk)
            if i < n_layers_cst - 1:
                kv_cst = F.elu(kv_cst)
        z_cst = kv_cst  # (N, D_h)

        # ── 5. Meta-GAT CS Module (Eq. 14-15) ────────────────────────────────
        # Q = K = V = e_j al primo layer, poi Q=K=V=output del layer precedente
        # (stack simmetrico standard); meta = SMK(i) per ogni layer. Stessa
        # convenzione sull'ELU (solo tra i layer) di CST sopra.
        h_cs = e_j
        n_layers_cs = len(self.meta_gat_cs)
        for i, layer in enumerate(self.meta_gat_cs):
            h_cs = layer(query=h_cs, key=h_cs, value=h_cs,
                         edge_index=edge_index, meta_embedding=smk)
            if i < n_layers_cs - 1:
                h_cs = F.elu(h_cs)
        z_cs = h_cs  # (N, D_h)

        # ── 6. Q-value prediction (Eq. 12) ────────────────────────────────────
        z = torch.cat([z_cst, z_cs], dim=-1)  # (N, 2*D_h)
        q_values = self.q_head(z)              # (N, n_actions)

        return q_values, x_i, c_t

    def _edge_index_with_self_loops(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Aggiunge self-loop a edge_index (parte permanente dell'architettura,
        vedi nota in cima al file), con cache per evitare di ricalcolarli ad
        ogni forward."""
        key = (id(edge_index), num_nodes)
        if key not in self._self_loop_cache:
            self._self_loop_cache[key] = add_self_loops(edge_index, num_nodes)
        return self._self_loop_cache[key]

    def init_hidden(self, num_nodes: int, device: torch.device = None):
        """Inizializza gli stati nascosti LSTM a zero."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(num_nodes, self.hidden_dim, device=device)
        c = torch.zeros(num_nodes, self.hidden_dim, device=device)
        return h, c
