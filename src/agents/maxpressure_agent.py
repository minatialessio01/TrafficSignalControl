"""
MaxPressure Agent — baseline classico per il controllo semaforico.

Implementa l'algoritmo MaxPressure (Varaiya, 2013):
  "Max pressure control of a network of signalized intersections"

Fedele all'implementazione di riferimento LibSignal
(https://github.com/DaRL-LibSignal/LibSignal/blob/master/agent/maxpressure.py):
per ogni fase, la pressione e' la somma, su tutti i movimenti (lane-link)
abilitati da quella fase, di (veicoli sulla corsia di provenienza - veicoli
sulla corsia di destinazione). Sceglie la fase con pressione massima.

  pressione(fase) = Σ_{(in, out) ∈ movimenti(fase)}  count(in) - count(out)

Due differenze deliberate rispetto alla versione precedente di questo file:
  1. Pressione per MOVIMENTO (coppia corsia_in -> corsia_out, dal roadnet via
     CityFlowEnv.phase_lanelinks), non piu' un'approssimazione per approccio
     basata sul vettore di stato n_vec.
  2. Conteggio veicoli sulla corsia INTERA, via CityFlowEnv.get_lane_vehicle_count()
     (nativo CityFlow, nessun filtro) — non piu' n_vec/VISION_CUTOFF_M
     dell'osservazione RL. La formula classica di Varaiya non prevede un raggio
     di visibilita' limitato; quello (VISION_CUTOFF_M) e' una scelta di design
     specifica dello stato dei modelli RL di questo progetto, non della pressione.

Nota sul "t_min" di LibSignal: nella loro implementazione l'agente puo' essere
interrogato a un passo temporale piu' fine del tempo minimo di fase, quindi
serve un guardrail esplicito (`if current_phase_time < t_min: mantieni la fase`)
per non switchare troppo spesso. Nel nostro ambiente ogni decisione dura gia'
GREEN_TIME + YELLOW_TIME + RED_TIME = 15s per costruzione (vedi cityflow_env.py):
non esiste un passo piu' fine a cui l'agente venga interrogato, quindi il
vincolo di durata minima e' gia' equivalente e non e' stato reimplementato.
"""

from typing import Dict, List

import numpy as np


class MaxPressureAgent:
    """
    Agente MaxPressure multi-intersezione.

    Stateless: non ha parametri da allenare. A ogni passo di decisione
    seleziona, per ciascuna intersezione, la fase con pressione massima.

    Args:
        n_phases: numero di fasi semaforiche selezionabili (default: 8 — la
                  fase di tutto-rosso, indice 8, non e' mai tra queste).
    """

    def __init__(self, n_phases: int = 8):
        self.n_phases = n_phases

    def reset(self, inter_ids: List[str]) -> None:
        """MaxPressure è stateless: non fa nulla al reset."""
        pass

    def select_actions(self,
                       inter_ids: List[str],
                       env=None,
                       states=None,
                       **kwargs) -> Dict[str, int]:
        """
        Seleziona la fase con pressione massima per ogni intersezione.

        Args:
            inter_ids: lista degli id intersezione da controllare.
            env:       CityFlowEnv corrente — richiesto: serve per i lane-link
                       per fase (env.phase_lanelinks) e il conteggio veicoli
                       per corsia (env.get_lane_vehicle_count()).
            states:    non usato dalla pressione (mantenuto solo per compatibilita'
                       con la firma di chiamata esistente in test.py).

        Returns:
            {inter_id → phase_index}
        """
        if env is None:
            raise ValueError(
                "MaxPressureAgent.select_actions richiede l'ambiente (env): "
                "la pressione si calcola dai lane-link del roadnet e dal "
                "conteggio veicoli per corsia, entrambi esposti da CityFlowEnv."
            )

        lane_counts = env.get_lane_vehicle_count()

        actions: Dict[str, int] = {}
        for iid in inter_ids:
            phase_pairs = env.phase_lanelinks.get(iid, [])[:self.n_phases]
            pressures = [
                sum(lane_counts.get(start, 0) - lane_counts.get(end, 0) for start, end in pairs)
                for pairs in phase_pairs
            ]
            # In caso di parita' (o nessuna fase disponibile), manteniamo l'indice
            # minore (stabile) — coerente con np.argmax su valori uguali.
            actions[iid] = int(np.argmax(pressures)) if pressures else 0

        return actions

    def get_training_state(self) -> dict:
        """Compatibilità con l'interfaccia degli altri agenti."""
        return {
            "type": "MaxPressure",
            "n_phases": self.n_phases,
        }
