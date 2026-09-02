"""
Meta-Knowledge Learners: SMK-Learner e TMK-Learner.

Corrispondono alla Section 4.3.1 dell'articolo (Fig. 5b):

  SMK(i) = MLP(spatial_features_i)    ← pressione corsie, n_veicoli, distanza
  TMK(i) = MLP(temporal_features_i)   ← queue length, storico stati, posizione

Entrambi sono MLP a 2 strati con attivazione ReLU e inizializzazioni diverse.
Le loro uscite vengono usate da MetaDense per generare i pesi di Meta-GAT
e Meta-LSTM.

Nota: "similar to the previous state encoding, the meta-knowledge learner
is also a two-layer fully connected network with different initializations"
(Section 4.3.1).
"""

import torch
import torch.nn as nn


class MetaKnowledgeLearner(nn.Module):
    """
    MLP a 2 strati per apprendere meta-conoscenza spaziale o temporale.

    Args:
        input_dim:  dimensione delle feature in input
        hidden_dim: dimensione del layer nascosto
        output_dim: dimensione dell'output (meta-embedding)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (batch, input_dim)

        Returns:
            meta_knowledge: (batch, output_dim)
        """
        h = self.activation(self.fc1(features))
        out = self.activation(self.fc2(h))
        return out


class SpatialMetaKnowledgeLearner(MetaKnowledgeLearner):
    """
    SMK-Learner: apprende meta-conoscenza spaziale.

    Feature in input (Section 4.3.1 e 4.3.2):
      - Pressione di ogni corsia (normalizzata)    [N_LANES]
      - Numero di veicoli per corsia (normalizzato) [N_LANES]
      - Distanza dai vicini (normalizzata)          [num_neighbors]

    dim_input = N_LANES * 2 + num_neighbors = 12*2 + 4 = 28
    """

    def __init__(self,
                 spatial_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 64):
        super().__init__(spatial_dim, hidden_dim, output_dim)


class TemporalMetaKnowledgeLearner(MetaKnowledgeLearner):
    """
    TMK-Learner: apprende meta-conoscenza temporale.

    Feature in input (Section 4.3.1 e 4.3.3):
      - Lunghezza coda per corsia (normalizzata)  [N_LANES]
      - Storico stati (ultime k=5 step, n_vec)    [N_LANES * history_len]

    dim_input = N_LANES + N_LANES * history_len = 12 + 12*5 = 72
    """

    def __init__(self,
                 temporal_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 64):
        super().__init__(temporal_dim, hidden_dim, output_dim)
