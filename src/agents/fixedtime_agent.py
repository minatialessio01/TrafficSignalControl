"""
FixedTime Agent — baseline tradizionale.

Assegna fasi cicliche con durata fissa (come da letteratura).
Serve come lower-bound di confronto (Section 5.3).

Strategia: fase ciclica che cambia ogni singolo step di decisione (15s), la
stessa cadenza di tutti gli altri modelli (RL e MaxPressure) — comparabilità
diretta, nessun fattore di scala arbitrario tra i cicli dei diversi controllori.
"""

from typing import Dict, List


FIXED_GREEN_TIME = 1   # step RL per fase (1 step = 15s simulati, come tutti gli altri modelli)


class FixedTimeAgent:
    """
    Agente con semaforo a tempo fisso.

    Cicla le fasi in ordine (0,1,2,...,7,0,1,...) con durata fissa.
    Non interagisce con l'ambiente — la fase è determinata solo dal tempo.

    Le 8 fasi corrispondono a NTST, NLSL, NTNL, STSL, WTET, WLEL, ETEL, WTWL
    (Fig. 1 del paper). La svolta a destra è controllata dal semaforo come
    tutte le altre manovre (non è più sempre verde).
    """

    def __init__(self, n_phases: int = 8, cycle_time: int = FIXED_GREEN_TIME):
        self.n_phases = n_phases
        self.cycle_time = cycle_time
        self.step_counter: Dict[str, int] = {}

    def reset(self, inter_ids: List[str]):
        """Resetta i contatori all'inizio di ogni episodio."""
        self.step_counter = {iid: 0 for iid in inter_ids}

    def select_actions(self,
                       inter_ids: List[str],
                       current_step: int = 0,
                       **kwargs) -> Dict[str, int]:
        """
        Restituisce la fase corrente basandosi solo sul tempo.

        Args:
            inter_ids:    lista di intersezioni
            current_step: step corrente della simulazione

        Returns:
            {inter_id -> phase_index}
        """
        actions = {}
        for iid in inter_ids:
            # Fase ciclica: cambia ogni cycle_time passi
            phase = (current_step // self.cycle_time) % self.n_phases
            actions[iid] = phase
        return actions

    def get_training_state(self) -> dict:
        return {
            "type": "FixedTime",
            "cycle_time": self.cycle_time,
            "n_phases": self.n_phases
        }
