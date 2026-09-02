"""
State Encoder: MLP a 2 strati per codificare lo stato dell'intersezione.

Corrisponde alle Equazioni 3 e 4 dell'articolo (Section 4.2.1):

  e0 = σ(s_i^t · W11 + b11)
  e_i = σ(e0 · W21 + b21)         → input all'LSTM (temporal branch)

  ê0 = σ(s_i^t · W12 + b12)
  e_j = σ(ê0 · W22 + b22)         → query per la GAT (spatial branch)

Le due MLP hanno la stessa architettura ma inizializzazioni diverse.
Questo consente di catturare sia le correlazioni temporali che spaziali
a partire dallo stesso stato di input.
"""

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """
    Singola MLP a 2 strati con attivazione ReLU.

    Usata sia per il branch temporale (produce e_i) che per quello
    spaziale (produce e_j). Le due istanze hanno pesi distinti.

    Args:
        input_dim:  d0 = dimensione dello stato di input (default: 32 = 12+12+8)
        hidden_dim: d1 = dimensione dello spazio latente (default: 64)
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) o (N, input_dim)

        Returns:
            e: (batch, hidden_dim) — rappresentazione latente
        """
        e0 = self.activation(self.fc1(x))   # W1, b1
        e = self.activation(self.fc2(e0))    # W2, b2
        return e


class DualStateEncoder(nn.Module):
    """
    Due encoder paralleli con inizializzazioni diverse.

    Produce:
      - e_i: branch temporale → input all'LSTM
      - e_j: branch spaziale  → query per la GAT

    Come descritto nelle Eq. 3-4 dell'articolo.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        # Branch temporale (Eq. 3): W11, W21
        self.temporal_encoder = StateEncoder(input_dim, hidden_dim)
        # Branch spaziale (Eq. 4): W12, W22
        self.spatial_encoder = StateEncoder(input_dim, hidden_dim)

    def forward(self, state: torch.Tensor):
        """
        Args:
            state: (batch, input_dim) — stato corrente s_i^t

        Returns:
            e_i: (batch, hidden_dim) — rappresentazione temporale
            e_j: (batch, hidden_dim) — rappresentazione spaziale
        """
        e_i = self.temporal_encoder(state)  # → LSTM
        e_j = self.spatial_encoder(state)   # → GAT query
        return e_i, e_j
