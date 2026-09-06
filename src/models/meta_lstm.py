"""
Meta Long Short-Term Memory (Meta-LSTM).

Implementa la Section 4.3.3 dell'articolo (Equazioni 18-21):

  LSTM standard (Eq. 18-19):
    h_t = LSTM(e_i^t, h_{t-1} | W_Φ, b_Φ)

  Meta-LSTM (Eq. 20-21):
    W_Φ^(i), b_Φ^(i) = MetaDense(3)(TMK(i))    # pesi generati dinamicamente
    x_i^t = Meta-LSTM(e_i^t, h_{t-1} | W_Φ^(i), b_Φ^(i))

L'idea è che ogni intersezione i ha una propria LSTM con pesi unici,
generati dal TMK-Learner a partire dalle feature temporali locali.

Implementazione:
  - Il TMK-Learner produce un embedding TMK(i) per ogni intersezione
  - MetaDense(3) trasforma TMK(i) → W_Φ ∈ R^{2D_h × D_h}, b_Φ ∈ R
  - Questi pesi sostituiscono i pesi standard dell'LSTM

Nota implementativa: un'LSTM con pesi diversi per ogni sample del batch
non è supportata nativamente da nn.LSTM. Utilizziamo un'implementazione
"manuale" dell'LSTM cell che accetta i pesi come argomenti dinamici.
Questo è il pattern standard per le iperreti (hypernetworks).

Dimensioni (dall'articolo):
  W_Φ ∈ R^{2D_h × D_h}  (4 gate × D_h, con input e_i di dim D_h)
  b_Φ ∈ R

Nota: W_Φ^{2D_h × D_h} copre l'input gate e il hidden state gate.
Il totale dei pesi LSTM è 4 × (D_h + D_h) × D_h = 4 × 2D_h × D_h.
Il MetaDense genera i pesi per tutti e 4 i gate contemporaneamente.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaDense3(nn.Module):
    """
    MetaDense(3) per la generazione dei pesi di Meta-LSTM.

    Prende in input TMK(i) e produce:
      W_Φ ∈ R^{4*(D_h + D_h) × D_h}  — pesi per tutti e 4 i gate LSTM
      b_Φ ∈ R^{4*D_h}                 — bias per tutti e 4 i gate

    (Eq. 20: W_Φ^(i) ∈ R^{2D_h × D_h})
    Adattiamo la notazione: "2D_h × D_h" include input (D_h) + hidden (D_h).
    """

    def __init__(self, meta_dim: int, hidden_dim: int):
        """
        Args:
            meta_dim:   dimensione del TMK embedding
            hidden_dim: D_h (hidden size dell'LSTM)
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Layer intermedio
        self.fc1 = nn.Linear(meta_dim, hidden_dim)
        self.activation = nn.ReLU()

        # Genera i pesi per tutti e 4 i gate LSTM:
        # f (forget), i (input), o (output), g (cell)
        # Ogni gate: W_ih (D_h × D_h) + W_hh (D_h × D_h) → 2*D_h*D_h per gate
        # 4 gate totali
        self.fc_weight = nn.Linear(hidden_dim, 4 * 2 * hidden_dim * hidden_dim)
        self.fc_bias = nn.Linear(hidden_dim, 4 * hidden_dim)

    def forward(self, meta_embedding: torch.Tensor):
        """
        Args:
            meta_embedding: (N, meta_dim) — TMK(i) per ogni nodo

        Returns:
            W: (N, 4, 2*D_h, D_h)  — pesi LSTM per tutti i gate
            b: (N, 4*D_h)           — bias LSTM
        """
        N = meta_embedding.size(0)
        h = self.activation(self.fc1(meta_embedding))   # (N, D_h)

        W_flat = self.fc_weight(h)   # (N, 4 * 2 * D_h * D_h)
        b = self.fc_bias(h)          # (N, 4 * D_h)

        D_h = self.hidden_dim
        W = W_flat.view(N, 4, 2 * D_h, D_h)   # (N, 4_gates, 2*D_h, D_h)

        return W, b


class MetaLSTMCell(nn.Module):
    """
    Singola cella LSTM con pesi generati dinamicamente da Meta-LSTM.

    Implementazione "manuale" della cella LSTM (Eq. 19) che accetta i pesi
    come argomenti, anziché parametri fissi del modello.

    Equazioni standard (Eq. 19):
      f_t = σ(W_f · [h_{t-1}, e_i] + b_f)
      i_t = σ(W_i · [h_{t-1}, e_i] + b_i)
      o_t = σ(W_o · [h_{t-1}, e_i] + b_o)
      c_t = f_t ⊗ c_{t-1} + i_t ⊗ φ(W_s · [h_{t-1}, e_i] + b_s)
      h_t = o_t ⊗ φ(c_t)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Args:
            input_dim:  d1 (dimensione di e_i)
            hidden_dim: D_h
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self,
                e_i: torch.Tensor,           # (N, input_dim)
                h_prev: torch.Tensor,         # (N, D_h)
                c_prev: torch.Tensor,         # (N, D_h)
                W: torch.Tensor,              # (N, 4, 2*D_h, D_h) o None
                b: torch.Tensor,              # (N, 4*D_h) o None
                ):
        """
        Args:
            e_i:    (N, input_dim)           — encoding corrente
            h_prev: (N, D_h)                 — hidden state precedente
            c_prev: (N, D_h)                 — cell state precedente
            W:      (N, 4, 2*D_h, D_h)      — pesi dinamici generati dal meta-learner
            b:      (N, 4*D_h)              — bias dinamici

        Returns:
            h_t: (N, D_h)
            c_t: (N, D_h)
        """
        N = e_i.size(0)
        D_h = self.hidden_dim

        # Concatena input e hidden state: [e_i, h_{t-1}] → (N, 2*D_h)
        # Ordine standard: [input, hidden] — coerente con nn.LSTMCell di PyTorch
        # e con la notazione del paper: LSTM(e_i^t, h_{t-1})
        combined = torch.cat([e_i, h_prev], dim=-1)  # (N, 2*D_h)

        # Calcola i pre-gate per tutti e 4 i gate simultaneamente
        # W: (N, 4, 2*D_h, D_h)  →  per ogni gate: (N, 2*D_h, D_h)
        # combined: (N, 2*D_h) → (N, 1, 2*D_h)

        # Gate computation: combined @ W[gate]^T + b[gate]
        # W[:, gate, :, :] ha forma (N, 2*D_h, D_h)
        # Ma vogliamo y = combined @ W^T, quindi usare:
        # y = (W @ combined.unsqueeze(-1)).squeeze(-1)  di dim (N, 4, D_h)

        # W è (N, 4, 2*D_h, D_h) — per ogni gate: (N, 2*D_h, D_h)
        # Vogliamo output (N, 4, D_h):
        # gates_pre = W^T @ combined → (N, 4, D_h, 2*D_h) @ (N, 1, 2*D_h, 1)

        combined_4d = combined.unsqueeze(1).unsqueeze(-1)  # (N, 1, 2*D_h, 1)
        # W: (N, 4, 2*D_h, D_h) — transposta: (N, 4, D_h, 2*D_h)
        W_T = W.transpose(-1, -2)  # (N, 4, D_h, 2*D_h)
        gates_pre = (W_T @ combined_4d).squeeze(-1)  # (N, 4, D_h)

        # Aggiungi il bias
        b_reshaped = b.view(N, 4, D_h)    # (N, 4, D_h)
        gates_pre = gates_pre + b_reshaped  # (N, 4, D_h)

        # Attivazioni per i gate
        f_t = torch.sigmoid(gates_pre[:, 0, :])   # forget gate
        i_t = torch.sigmoid(gates_pre[:, 1, :])   # input gate
        o_t = torch.sigmoid(gates_pre[:, 2, :])   # output gate
        g_t = torch.tanh(gates_pre[:, 3, :])      # cell gate

        # Cell state e hidden state (Eq. 19)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class MetaLSTM(nn.Module):
    """
    Meta-LSTM: LSTM con pesi generati dinamicamente per ogni intersezione.

    Integra MetaDense(3) e MetaLSTMCell per implementare le Eq. 20-21.

    Per ogni intersezione i al timestep t:
      1. TMK(i) fornisce il meta-embedding temporale
      2. MetaDense(3)(TMK(i)) genera W_Φ^(i), b_Φ^(i) — pesi personalizzati
      3. MetaLSTMCell processa e_i^t con i pesi dinamici → x_i^t, c_i^t
    """

    def __init__(self,
                 input_dim: int = 64,   # d1 (= hidden_dim dell'encoder)
                 hidden_dim: int = 64,  # D_h
                 meta_dim: int = 64,    # dimensione TMK embedding
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # MetaDense(3) per generare W_Φ e b_Φ (Eq. 20)
        self.meta_dense3 = MetaDense3(meta_dim, hidden_dim)

        # Cella LSTM con pesi dinamici (Eq. 21)
        self.lstm_cell = MetaLSTMCell(input_dim, hidden_dim)

    def forward(self,
                e_i: torch.Tensor,            # (N, input_dim)
                h_prev: torch.Tensor,          # (N, D_h)
                c_prev: torch.Tensor,          # (N, D_h)
                tmk_embedding: torch.Tensor,   # (N, meta_dim)
                ) -> tuple:
        """
        Singolo step temporale di Meta-LSTM.

        Args:
            e_i:           (N, input_dim)  — encoding corrente e_i^t
            h_prev:        (N, D_h)        — hidden state precedente h_{t-1}
            c_prev:        (N, D_h)        — cell state precedente c_{t-1}
            tmk_embedding: (N, meta_dim)   — TMK(i) per ogni intersezione

        Returns:
            x_i^t: (N, D_h)  — output hidden state (usato come x_i in Meta-GAT)
            c_t:   (N, D_h)  — cell state aggiornato
        """
        # Genera i pesi dinamici (Eq. 20)
        W, b = self.meta_dense3(tmk_embedding)  # (N, 4, 2*D_h, D_h), (N, 4*D_h)

        # Aggiorna lo stato LSTM con i pesi dinamici (Eq. 21)
        x_i, c_t = self.lstm_cell(e_i, h_prev, c_prev, W, b)

        return x_i, c_t

    def init_hidden(self, num_nodes: int, device: torch.device = None):
        """Inizializza gli stati nascosti a zero."""
        if device is None:
            device = next(self.parameters()).device
        h = torch.zeros(num_nodes, self.hidden_dim, device=device)
        c = torch.zeros(num_nodes, self.hidden_dim, device=device)
        return h, c
