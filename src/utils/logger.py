"""
Logger e Checkpoint Manager per il training di MetaSTGAT.

Gestisce:
  - Logging su CSV dei risultati per episodio
  - Salvataggio periodico dei checkpoint (ogni CHECKPOINT_INTERVAL episodi)
  - Salvataggio del training state in JSON (per visualizzare da Windows)
  - Gestione del segnale SIGINT per interruzione pulita (Ctrl+C)
"""

import csv
import json
import os
import signal
import sys
import time
from datetime import datetime
from typing import Optional


CHECKPOINT_INTERVAL = 10   # Salva checkpoint ogni N episodi


class TrainingLogger:
    """
    Gestisce il logging e i checkpoint del training.

    Features:
      - CSV con metriche per ogni episodio
      - JSON con stato corrente del training (leggibile da Windows senza Python)
      - Checkpoint periodici + best model
      - Gestione Ctrl+C: salva e chiude pulito
    """

    def __init__(self, output_dir: str, run_name: str = "run", resume: bool = False):
        self.output_dir = output_dir
        self.run_name = run_name

        os.makedirs(output_dir, exist_ok=True)
        # I checkpoint periodici vivono in una sottocartella dedicata (riordino
        # del 12/9/2026, vedi results/REORGANIZATION.md): best_model.pt/
        # final_model.pth/selected_model.pth restano invece alla radice, sono
        # i "puntatori" cercati da scripts/test.py::_find_checkpoint().
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # File CSV per le metriche
        self.csv_path = os.path.join(output_dir, "training_log.csv")
        self.json_path = os.path.join(output_dir, "training_state.json")
        self.best_path = os.path.join(output_dir, "best_model.pt")

        # Inizializza il CSV (append solo se stiamo riprendendo un training)
        self._csv_file = None
        self._csv_writer = None
        self._init_csv(resume=resume)

        # Timestamp di inizio
        self.start_time = time.time()
        self.episode_start_time = time.time()

        # Flag per interruzione pulita
        self._interrupted = False
        self._setup_signal_handler()

        print(f"[Logger] Output in '{output_dir}/'")

    _FIELDNAMES = [
        "episode", "travel_time", "throughput",
        "total_reward",
        "avg_loss", "epsilon", "buffer_size",
        "wait_max_ns", "wait_max_ew",
        "episode_time_s", "total_time_s"
    ]

    def _init_csv(self, resume: bool = False):
        """Inizializza il file CSV.

        Se resume=True (training ripreso da checkpoint) apre in modalità append,
        altrimenti sovrascrive per evitare di mescolare dati di run diverse.

        Se il CSV esistente ha un header piu' vecchio (es. un run iniziato prima
        dell'aggiunta di wait_max_ns/wait_max_ew), lo migra prima di appendere:
        rilegge le righe esistenti, le riscrive con l'header nuovo (colonne
        mancanti = vuote), cosi' le righe nuove restano allineate sotto lo
        stesso header invece di "sporcare" un CSV con colonne extra non nominate.
        """
        if resume and os.path.exists(self.csv_path):
            self._migrate_csv_header_if_needed()
            mode = "a"   # append: continua dal punto in cui ci si era fermati
        else:
            mode = "w"   # sovrascrittura: nuovo run, CSV pulito
        self._csv_file = open(self.csv_path, mode, newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._FIELDNAMES)
        if mode == "w":
            self._csv_writer.writeheader()

    def _migrate_csv_header_if_needed(self):
        """Riscrive training_log.csv con l'header corrente se quello su disco
        e' piu' vecchio (mancano colonne aggiunte dopo l'inizio di questo run)."""
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            old_fieldnames = reader.fieldnames or []
            if old_fieldnames == self._FIELDNAMES:
                return  # gia' aggiornato, niente da fare
            rows = list(reader)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in self._FIELDNAMES})

    def _setup_signal_handler(self):
        """Cattura Ctrl+C per un'interruzione pulita."""
        def handler(sig, frame):
            print("\n\n[!] Interruzione ricevuta (Ctrl+C).")
            print("    Il training verrà fermato dopo questo episodio.")
            print("    Usa --resume per riprendere dall'ultimo checkpoint.\n")
            self._interrupted = True

        signal.signal(signal.SIGINT, handler)

    @property
    def interrupted(self) -> bool:
        """True se l'utente ha premuto Ctrl+C."""
        return self._interrupted

    def log_episode(self,
                    episode: int,
                    travel_time: float,
                    throughput: int,
                    total_reward: float,
                    avg_loss: float,
                    epsilon: float,
                    buffer_size: int,
                    wait_max_ns: float = 0.0,
                    wait_max_ew: float = 0.0):
        """Registra i risultati di un episodio su CSV e JSON."""
        now = time.time()
        episode_time = now - self.episode_start_time
        total_time = now - self.start_time
        self.episode_start_time = now

        row = {
            "episode": episode,
            "travel_time": int(travel_time),
            "throughput": throughput,
            "total_reward": round(total_reward, 2),
            "avg_loss": round(avg_loss, 6) if avg_loss else 0.0,
            "epsilon": round(epsilon, 6),
            "buffer_size": buffer_size,
            "wait_max_ns": round(wait_max_ns, 1),
            "wait_max_ew": round(wait_max_ew, 1),
            "episode_time_s": round(episode_time, 1),
            "total_time_s": round(total_time, 1),
        }

        if self._csv_writer:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

        # Aggiorna il JSON (leggibile da Windows senza Python)
        state = {
            **row,
            "run_name": self.run_name,
            "timestamp": datetime.now().isoformat(),
            "interrupted": self._interrupted,
        }
        with open(self.json_path, "w") as f:
            json.dump(state, f, indent=2)

        return row

    def print_episode(self,
                      episode: int,
                      total_episodes: int,
                      travel_time: float,
                      throughput: int,
                      loss: Optional[float],
                      epsilon: float,
                      total_reward: float = 0.0,
                      is_best: bool = False,
                      eta_str: str = "",
                      wait_max_ns: float = 0.0,
                      wait_max_ew: float = 0.0):
        """Stampa il riassunto dell'episodio a schermo, inclusa la stima ETA."""
        best_marker = " ★ BEST" if is_best else ""
        loss_str = f"{loss:.5f}" if loss is not None else "  N/A  "
        eta_display = f"  |  ETA: {eta_str}" if eta_str else ""
        print(
            f"  Ep {episode:>4}/{total_episodes}  |  "
            f"TT: {travel_time:>8.1f}s  |  "
            f"TP: {throughput:>5}  |  "
            f"RΣ: {total_reward:>9.1f}  |  "
            f"Loss: {loss_str}  |  "
            f"ε: {epsilon:.4f}"
            f"{eta_display}"
            f"{best_marker}"
        )
        print(
            f"           Attesa max N/S={wait_max_ns:.0f}s  W/E={wait_max_ew:.0f}s"
        )

    def checkpoint_path(self, episode: int) -> str:
        """Restituisce il percorso del checkpoint periodico per l'episodio dato
        (sottocartella checkpoints/, vedi __init__)."""
        return os.path.join(self.checkpoint_dir, f"checkpoint_ep{episode:04d}.pt")

    def should_save_checkpoint(self, episode: int) -> bool:
        """True ogni CHECKPOINT_INTERVAL episodi."""
        return episode % CHECKPOINT_INTERVAL == 0

    def close(self):
        """Chiude i file aperti."""
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None
