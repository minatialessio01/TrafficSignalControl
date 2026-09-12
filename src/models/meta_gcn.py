"""
Meta Graph Convolutional Network (Meta-GCN).

Sostituisce il meccanismo di attenzione di Meta-GAT (Section 4.3.2 del paper
MetaSTGAT) con la regola di aggregazione classica della GCN (Kipf & Welling,
2017): normalizzazione simmetrica del grado, fissa e dipendente solo dalla
topologia, senza alcun meccanismo di compatibilita' Q*K/softmax appreso per
coppia di nodi. E' la differenza concettuale reale rispetto a Meta-GAT, non un
dettaglio implementativo da nascondere -- vedi `istruzioni seconda parte.md`
§1/§3.1.

Il principio "meta" e' preservato identico a MetaGATLayer: il peso che
trasforma il valore del vicino non e' un parametro fisso condiviso da tutti i
nodi, ma generato da un embedding SMK(i)/TMK(i) via MetaDense (riusata as-is
da meta_gat.py).

Stessa interfaccia di MetaGATLayer per essere un drop-in replacement in
MetaSTGNN (metastgnn.py): `query` e' accettata ma ignorata (la GCN non ha un
meccanismo di compatibilita' che la richieda), tenuta in firma solo per
compatibilita'.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .meta_gat import MetaDense


def add_self_loops(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Aggiunge un self-loop (i,i) per ogni nodo a edge_index, se non gia'
    presente. Pratica standard GCN (Â = A + I): senza self-loop il contributo
    del nodo al proprio stesso output si perderebbe interamente nella
    normalizzazione di grado. Calcolato una volta fuori dal layer (vedi
    `istruzioni seconda parte.md` §3.3) e non ad ogni forward.
    """
    device = edge_index.device
    src, dst = edge_index[0], edge_index[1]
    has_self_loop = (src == dst)
    existing = set(zip(src[has_self_loop].tolist(), dst[has_self_loop].tolist()))
    missing = [i for i in range(num_nodes) if i not in existing]
    if not missing:
        return edge_index
    loops = torch.tensor([missing, missing], dtype=edge_index.dtype, device=device)
    return torch.cat([edge_index, loops], dim=1)


class MetaGCNLayer(nn.Module):
    """
    Singolo layer di Meta-GCN.

    Args:
        hidden_dim: d1 = D_h (dimensione nascosta, deve combaciare con
                    l'output di MetaGATLayer per essere un drop-in replacement)
        num_heads:  accettato solo per compatibilita' di firma con
                    MetaGATLayer/MetaSTGAT (build_model() passa num_heads a
                    entrambi) -- la GCN non ha teste di attenzione, ignorato.
        meta_dim:   dimensione dell'embedding meta-knowledge in input
        dropout:    dropout sull'output (non specificato dal paper originale
                    della GCN, usiamo 0.0 di default per coerenza con
                    MetaGATLayer)
    """

    def __init__(self,
                 hidden_dim: int = 64,
                 num_heads: int = 4,   # non usato, solo compatibilita' di firma
                 meta_dim: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_layer = nn.Dropout(dropout)

        # Genera W: (N, D_h, D_h), b: (N, 1) dall'embedding meta -- stessa
        # classe MetaDense usata da Meta-GAT (istruzioni seconda parte.md §3.1
        # punto 2: "e' gia' pronta, non serve riscriverla").
        self.meta_dense = MetaDense(meta_dim, hidden_dim, hidden_dim)

    def forward(self,
                query: torch.Tensor,           # ignorato, solo compatibilita' di firma
                key: torch.Tensor,              # ignorato (idem)
                value: torch.Tensor,            # X: (N, d1) -- il segnale da propagare
                edge_index: torch.Tensor,       # (2, E), CON self-loop gia' inclusi
                meta_embedding: torch.Tensor,   # SMK(i) o TMK(i): (N, meta_dim)
                ) -> torch.Tensor:
        """
        Args:
            value:      (N, d1) -- stesso ruolo di "value" in MetaGATLayer
            edge_index: (2, E) -- deve gia' includere i self-loop (aggiunti da
                        MetaSTGNN una sola volta, non ad ogni forward)
            meta_embedding: (N, meta_dim)

        Returns:
            out: (N, d1)
        """
        N = value.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Pesi dinamici per nodo destinazione (Eq. "meta" di Meta-GAT, qui
        # applicata alla GCN): W: (N, D_h, D_h), b: (N, 1)
        W, b = self.meta_dense(meta_embedding)

        # Normalizzazione simmetrica del grado (fissa, dipende solo dalla
        # topologia): norm_uv = 1/sqrt(deg(u)*deg(v)). Il grado e' contato sugli
        # archi in ingresso (inclusi i self-loop), coerente con la convenzione
        # standard Â = D^{-1/2} Â D^{-1/2}.
        deg = torch.zeros(N, device=value.device, dtype=value.dtype)
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=value.dtype))
        norm = (deg[src] * deg[dst]).clamp(min=1.0).rsqrt()  # (E,)

        # Trasforma il valore del nodo sorgente con il peso dinamico del nodo
        # DESTINAZIONE (stessa convenzione di MetaGATLayer, dove W[dst]
        # trasforma la query del nodo destinazione, non del vicino).
        V_src = value[src]                                              # (E, d1)
        V_transformed = torch.bmm(V_src.unsqueeze(1), W[dst]).squeeze(1)  # (E, d1)
        weighted = norm.unsqueeze(-1) * V_transformed + b[dst]           # (E, d1)

        out = torch.zeros(N, self.hidden_dim, device=value.device, dtype=value.dtype)
        out.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted), weighted)

        out = F.relu(out)  # la GCN classica ha una non-linearita' in uscita
        out = self.dropout_layer(out)
        return out
