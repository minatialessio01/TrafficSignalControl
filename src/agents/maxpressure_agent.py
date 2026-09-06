"""
MaxPressure Agent — baseline classico per il controllo semaforico.

Implementa l'algoritmo MaxPressure (Varaiya, 2013):
  "Max pressure control of a network of signalized intersections"

A ogni passo seleziona la fase che massimizza la pressione totale,
dove la pressione di una fase è la somma dei veicoli in coda sulle
corsie in ingresso attivate da quella fase.

Versione semplificata: usa il vettore di osservazione (n_vec, 12 corsie)
senza accesso diretto alle corsie in uscita.

Mapping corsie → approccio (da Fig. 1 del paper, 3 corsie per direzione):
  lane 0-2 : N approach  (Northbound, veicoli da Sud → Nord)
  lane 3-5 : S approach  (Southbound, veicoli da Nord → Sud)
  lane 6-8 : W approach  (Westbound,  veicoli da Est  → Ovest)
  lane 9-11: E approach  (Eastbound,  veicoli da Ovest→ Est)
"""

from typing import Dict, List

import numpy as np


# Corsie attivate in ogni fase (indici nel vettore n_vec di dim 12).
# Rispecchia la Fig. 1 del paper. Svolta a DX (lane 0,3,6,9), Dritto (lane 1,4,7,10), SX (lane 2,5,8,11).
_PHASE_ACTIVE_LANES: List[List[int]] = [
    [0, 1, 3, 4],      # 0 – NTST : N (DX, Dritto), S (DX, Dritto)
    [2, 5],            # 1 – NLSL : N (SX), S (SX)
    [0, 1, 2],         # 2 – NTNL : N (Tutte)
    [3, 4, 5],         # 3 – STSL : S (Tutte)
    [6, 7, 9, 10],     # 4 – WTET : W (DX, Dritto), E (DX, Dritto)
    [8, 11],           # 5 – WLEL : W (SX), E (SX)
    [9, 10, 11],       # 6 – ETEL : E (Tutte)
    [6, 7, 8],         # 7 – WTWL : W (Tutte)
]


class MaxPressureAgent:
    """
    Agente MaxPressure multi-intersezione.

    Stateless: non ha parametri da allenare. A ogni passo di decisione
    seleziona la fase che massimizza la pressione locale stimata.

    Args:
        n_phases: numero di fasi semaforiche (default: 8, come da paper)
    """

    def __init__(self, n_phases: int = 8):
        self.n_phases = n_phases
        self._phase_lanes = _PHASE_ACTIVE_LANES[:n_phases]

    def reset(self, inter_ids: List[str]) -> None:
        """MaxPressure è stateless: non fa nulla al reset."""
        pass

    def select_actions(self,
                       states: Dict[str, np.ndarray],
                       inter_ids: List[str],
                       env=None,
                       **kwargs) -> Dict[str, int]:
        """
        Seleziona la fase con pressione massima per ogni intersezione.

        Pressione di una fase φ:
            p(φ) = Σ  n_vec[l]   per ogni corsia l attiva in φ

        Args:
            states:    {inter_id → obs_array (dim=20)}
                       Le prime 12 componenti sono i conteggi veicoli (n_vec).
            inter_ids: lista degli id intersezione da controllare.

        Returns:
            {inter_id → phase_index}
        """
        actions: Dict[str, int] = {}
        for iid in inter_ids:
            obs = states[iid]
            n_vec = obs[:12].astype(float)   # conteggi corsie in ingresso

            # 2.5 Calcolo effettivo della pressione: P = n_in - n_out
            if env is not None:
                out_count = env._get_outgoing_vehicles_count(iid)
                out_lanes_count = max(1, len(env._get_outgoing_lanes(iid)))
                avg_out = out_count / out_lanes_count
                
                pressures = [
                    float(np.sum(n_vec[lanes])) - len(lanes) * avg_out
                    for lanes in self._phase_lanes
                ]
            else:
                pressures = [
                    float(np.sum(n_vec[lanes]))
                    for lanes in self._phase_lanes
                ]

            # In caso di parità, manteniamo la fase con indice minore (stabile)
            actions[iid] = int(np.argmax(pressures))

        return actions

    def get_training_state(self) -> dict:
        """Compatibilità con l'interfaccia degli altri agenti."""
        return {
            "type": "MaxPressure",
            "n_phases": self.n_phases,
        }
