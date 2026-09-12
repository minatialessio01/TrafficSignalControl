"""
Meta SONAR: propagazione a lungo raggio ispirata a un'equazione d'onda,
meta-condizionata sullo stesso principio gia' usato da Meta-GAT/Meta-GCN.

Riferimento: Trenta, Gravina, Bacciu, "SONAR: Long-Range Graph Propagation
Through Information Waves" (NeurIPS 2025) -- PDF allegato al repository.
SONAR modella la propagazione dell'informazione su un grafo come un'onda
smorzata con una forza esterna (Eq. 5 del paper):

    Ẍ(t) = -L^a X(t) W - D(X(t)) ⊙ Ẋ(t) + F(X(t))

riscritta come sistema del prim'ordine (velocita' ausiliaria V=Ẋ) e
discretizzata con passo `h` su `L` iterazioni per blocco (Eq. 7-8). Il paper
non ha alcun concetto di "meta-knowledge": qui la resistenza adattiva per
arco (a_uv, l'analogo diretto del peso dinamico di Meta-GAT/Meta-GCN) e'
generata da un embedding SMK(i)/TMK(i) via MetaDense, invece che da una MLP a
pesi fissi condivisi -- vedi `istruzioni seconda parte.md` §4, decisione D3.

Validato contro il codice ufficiale (https://github.com/gravins/SONAR,
graph_transfer_task/models/sonar.py) prima dell'implementazione, per i
dettagli che il paper lascia impliciti:
  - la resistenza e' ricalcolata a ogni iterazione (non fissata all'inizio del
    blocco), dato che dipende dallo stato corrente dei nodi -- adottato qui;
  - l'aggiornamento di X usa la velocita' passata attraverso un'attivazione
    limitata (tanh nel repo ufficiale) invece della velocita' grezza: un
    accorgimento di stabilita' non esplicitato nel paper, adottato qui.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .meta_gat import MetaDense


class MetaSONARLayer(nn.Module):
    """
    Blocco Meta-SONAR: L iterazioni di discretizzazione a pesi CONDIVISI tra
    le iterazioni (a differenza dello stack di Meta-GAT/Meta-GCN, dove i due
    layer hanno pesi indipendenti -- qui la profondità si ottiene con
    ricorrenze a costo di parametri costante, vedi `istruzioni seconda
    parte.md` §0/§2.1).

    Args:
        hidden_dim:    d1 = D_h
        num_heads:     accettato solo per compatibilita' di firma con
                       MetaGATLayer/MetaGCNLayer, ignorato (nessun meccanismo
                       ad attenzione multi-head in SONAR)
        meta_dim:      dimensione dell'embedding meta-knowledge in input
        n_recurrences: L, numero di passi di discretizzazione per blocco
                       (default 2 -- raccomandazione D1 di `istruzioni
                       seconda parte.md`: resta ben sotto il diametro anche
                       della griglia di training piu' piccola, 4x4)
        step_size:     h, passo di discretizzazione (default 0.1)
        use_dissipation: se False, il termine dissipativo D(X) e' nullo
                         (paper Eq. 5, dissipazione facoltativa)
        use_forcing:     se False, la forza esterna F(X) e' nulla
        dropout:       dropout sull'output del blocco
    """

    def __init__(self,
                 hidden_dim: int = 64,
                 num_heads: int = 4,          # ignorato, solo compatibilita' di firma
                 meta_dim: int = 64,
                 n_recurrences: int = 2,
                 step_size: float = 0.1,
                 use_dissipation: bool = True,
                 use_forcing: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_recurrences = n_recurrences
        self.h = step_size
        self.use_dissipation = use_dissipation
        self.use_forcing = use_forcing
        self.dropout_layer = nn.Dropout(dropout)

        # ── Resistenza adattiva a_uv, meta-condizionata (D3: solo questa e'
        # meta-condizionata; dissipazione/forcing restano MLP fisse, vedi
        # sotto) -- analogo diretto del peso dinamico di Meta-GAT/Meta-GCN.
        # MetaDense produce W_res:(N,D_h,1), b_res:(N,1) dall'embedding meta,
        # applicati a (x_u - x_v) per ogni arco (Eq. 2 del paper: il
        # "gradiente" sul grafo).
        self.meta_dense_resistance = MetaDense(meta_dim, hidden_dim, 1)

        # ── Dissipazione e forzante esterna: MLP a pesi FISSI (non
        # meta-condizionate in questo primo design, D3), applicate allo stato
        # corrente del nodo -- sono filtri locali, non meccanismi di
        # comunicazione tra nodi, quindi il collegamento con SMK/TMK e' piu'
        # debole che per la resistenza (che e' l'analogo diretto di Eq. 14-17
        # di MetaSTGAT).
        if use_dissipation:
            self.dissipation_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
            )
        if use_forcing:
            self.forcing_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

        # V^0 = query @ W_V (scelta di design, non nel paper -- vedi
        # `istruzioni seconda parte.md` §4.1: da' al modulo CST un'analogia
        # residua con "Q e K asimmetrici" anche in SONAR, usando `query` come
        # sorgente della velocita' iniziale e `value` come posizione iniziale
        # X^0. Alternativa piu' fedele al paper, se questa risultasse
        # artificiosa: V^0 = value @ W_V, ignorando `query`.)
        self.W_V = nn.Linear(hidden_dim, hidden_dim, bias=False)

    @staticmethod
    def _laplacian_action(X: torch.Tensor, edge_index: torch.Tensor,
                           a_uv: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """(L^a X)_v = Σ_{u in N(v)} a_uv * (X_v - X_u), Eq. 2 del paper.

        Per ogni arco diretto (u=src, v=dst) accumula il contributo su v; con
        un grafo ad archi bidirezionali (il nostro caso, vedi
        CityFlowEnv._build_adjacency) la somma sui due versi di ogni coppia
        (u,v) ricostruisce correttamente la somma su tutto il vicinato N(v).
        """
        src, dst = edge_index[0], edge_index[1]
        diff = X[dst] - X[src]                       # (E, D_h)
        contrib = a_uv * diff                          # (E, D_h), a_uv gia' (E,1) o (E,D_h)
        out = torch.zeros(num_nodes, X.size(-1), device=X.device, dtype=X.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(contrib), contrib)
        return out

    def forward(self,
                query: torch.Tensor,           # (N, d1) -- sorgente di V^0, vedi sopra
                key: torch.Tensor,              # non usato (SONAR non ha un ruolo per "key")
                value: torch.Tensor,            # (N, d1) -- X^0, la posizione iniziale
                edge_index: torch.Tensor,       # (2, E)
                meta_embedding: torch.Tensor,   # SMK(i) o TMK(i): (N, meta_dim)
                ) -> torch.Tensor:
        N = value.size(0)
        src, dst = edge_index[0], edge_index[1]

        X = value
        V = self.W_V(query)

        # Pesi della resistenza generati una volta dall'embedding meta (per
        # nodo destinazione); ricalcolare solo a_uv (che dipende anche dallo
        # stato corrente X, Eq. 2) a ogni iterazione, non i pesi stessi --
        # coerente col fatto che l'embedding SMK/TMK e' fisso per l'intero
        # step ambientale, mentre lo stato del nodo evolve dentro il blocco.
        W_res, b_res = self.meta_dense_resistance(meta_embedding)  # (N,D_h,1), (N,1)

        for _ in range(self.n_recurrences):
            diff_uv = X[src] - X[dst]                                    # (E, D_h)
            a_raw = torch.bmm(diff_uv.unsqueeze(1), W_res[dst]).squeeze(1)  # (E, 1)
            a_uv = F.relu(a_raw + b_res[dst])                              # (E, 1), sempre >= 0

            LaX = self._laplacian_action(X, edge_index, a_uv, N)  # (N, D_h)

            D = self.dissipation_mlp(X) if self.use_dissipation else 0.0
            Fext = self.forcing_mlp(X) if self.use_forcing else 0.0

            V = V - self.h * (LaX + D * V - Fext)
            X = X + self.h * torch.tanh(V)  # tanh su V: accorgimento di stabilita'
                                             # validato sul repo ufficiale, non
                                             # esplicitato nel paper (vedi docstring)

        return self.dropout_layer(X)
