"""
Metriche di valutazione per MetaSTGAT.

Implementa le due metriche principali dell'articolo (Section 5.4):
  - Travel time (s): tempo medio di viaggio di tutti i veicoli
  - Throughput (veicoli): numero di veicoli che hanno completato il viaggio
"""

from typing import List, Optional
import numpy as np


class EpisodeMetrics:
    """Raccoglie le metriche per un singolo episodio."""

    def __init__(self):
        self.travel_times: List[float] = []
        self.throughputs: List[int] = []
        self.rewards: List[float] = []
        self.losses: List[float] = []

    def update(self, travel_time: float, throughput: int,
               reward: float = 0.0, loss: Optional[float] = None):
        self.travel_times.append(travel_time)
        self.throughputs.append(throughput)
        self.rewards.append(reward)
        if loss is not None:
            self.losses.append(loss)

    @property
    def avg_travel_time(self) -> float:
        """Travel time medio dell'episodio (metrica principale)."""
        return float(np.mean(self.travel_times)) if self.travel_times else 0.0

    @property
    def final_travel_time(self) -> float:
        """Travel time finale (ultimo valore dell'episodio)."""
        return self.travel_times[-1] if self.travel_times else 0.0

    @property
    def final_throughput(self) -> int:
        """Throughput finale (ultimo valore dell'episodio)."""
        return self.throughputs[-1] if self.throughputs else 0

    @property
    def avg_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def total_reward(self) -> float:
        """Reward totale accumulata nell'episodio (somma di tutti gli step)."""
        return float(np.sum(self.rewards)) if self.rewards else 0.0

    @property
    def avg_loss(self) -> float:
        return float(np.mean(self.losses)) if self.losses else 0.0

    def to_dict(self) -> dict:
        return {
            "avg_travel_time": round(self.avg_travel_time, 4),
            "final_travel_time": round(self.final_travel_time, 4),
            "final_throughput": self.final_throughput,
            "avg_reward": round(self.avg_reward, 4),
            "avg_loss": round(self.avg_loss, 6),
        }


class RunningMetrics:
    """
    Statistiche aggregate su più episodi.
    Usato per calcolare la media degli ultimi 10 episodi
    (come nell'articolo: "the average value of the last ten tests").
    """

    def __init__(self, window: int = 10):
        self.window = window
        self.episode_travel_times: List[float] = []
        self.episode_throughputs: List[int] = []

    def add_episode(self, travel_time: float, throughput: int):
        self.episode_travel_times.append(travel_time)
        self.episode_throughputs.append(throughput)

    @property
    def last_n_avg_travel_time(self) -> float:
        """Travel time medio degli ultimi N episodi."""
        if not self.episode_travel_times:
            return 0.0
        window = self.episode_travel_times[-self.window:]
        return float(np.mean(window))

    @property
    def last_n_avg_throughput(self) -> float:
        """Throughput medio degli ultimi N episodi."""
        if not self.episode_throughputs:
            return 0.0
        window = self.episode_throughputs[-self.window:]
        return float(np.mean(window))

    @property
    def best_travel_time(self) -> float:
        return min(self.episode_travel_times) if self.episode_travel_times else float("inf")

    @property
    def best_throughput(self) -> int:
        return max(self.episode_throughputs) if self.episode_throughputs else 0

    def to_dict(self) -> dict:
        return {
            "n_episodes": len(self.episode_travel_times),
            "last_10_avg_travel_time": round(self.last_n_avg_travel_time, 4),
            "last_10_avg_throughput": round(self.last_n_avg_throughput, 4),
            "best_travel_time": round(self.best_travel_time, 4),
            "best_throughput": self.best_throughput,
        }
