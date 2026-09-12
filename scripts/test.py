"""
Script di test e valutazione per MetaSTGAT / STGAT / FixedTime / MaxPressure.

Carica un modello addestrato e lo valuta per N episodi, salvando (struttura
a sottocartelle in <output-dir>, riordino del 12/9/2026, vedi
results/REORGANIZATION.md):
  - Un CSV per episodio in test_csv/: episode, travel_time,
    travel_time_completed_only, throughput, total_vehicles, phase_pct_0..7
  - Un JSON di riepilogo in test_summaries/: avg_travel_time,
    avg_travel_time_completed_only, avg_throughput, total_vehicles
  - Un plot delle percentuali di fase mediato su TUTTE le config di cui esiste
    gia' un CSV in test_csv/ (phase_pct_avg.png, alla radice di <output-dir>,
    ricalcolato e sovrascritto ad ogni chiamata — non un plot per singola
    config)

Nota su travel_time vs travel_time_completed_only:
  - travel_time (metrica principale, usata per ranking/selezione) include anche
    i veicoli ancora in rete a fine episodio, contati col tempo gia' trascorso.
    Un modello che ingolfa la rete e fa passare solo pochi veicoli fortunati non
    ottiene piu' un travel time artificialmente basso.
  - travel_time_completed_only e' la vecchia convenzione "solo veicoli arrivati"
    (stessa del motore CityFlow e del paper originale) — solo informativa, per
    confronto diretto con la letteratura, mai per scegliere tra modelli.

Uso:
  # Test modello avanzato (final_model.pth auto-rilevato)
  python scripts/test.py \\
      --config configs/config_4x4_200m_2k_flat.json \\
      --model-id metastgat_pro \\
      --output-dir results/metastgat_pro

  # Test modello ablation
  python scripts/test.py \\
      --config configs/config_4x4_200m_2k_flat.json \\
      --model-id ablation_environment \\
      --ablation environment \\
      --output-dir results/ablation_environment

  # Test FixedTime (nessun checkpoint)
  python scripts/test.py \\
      --config configs/config_4x4_200m_2k_flat.json \\
      --model FixedTime \\
      --model-id fixedtime \\
      --output-dir results/fixedtime

Note:
  - Checkpoint primario: final_model.pth (ultimo ep). Fallback: best_model.pt,
    poi il checkpoint periodico piu' recente (checkpoints/checkpoint_ep*.pt).
  - Risultati CSV: <output-dir>/test_csv/test_<config_basename>.csv
  - Per confronto tra modelli usare plot_results.py.
"""

import argparse
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.environment.cityflow_env import CityFlowEnv
from src.models.metastgat import MetaSTGAT
from src.models.metastgnn import MetaSTGNN
from src.models.metastsonar import MetaSTSONAR
from src.models.stgat import STGAT
from src.agents.dqn_agent import DQNAgent
from src.agents.fixedtime_agent import FixedTimeAgent
from src.agents.maxpressure_agent import MaxPressureAgent


# Stessi default del training
DEFAULTS = {
    "hidden_dim":    64,
    "num_heads":     4,
    "num_neighbors": 4,
    "history_len":   5,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Valutazione MetaSTGAT / STGAT / FixedTime / MaxPressure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ── Configurazione obbligatoria ──────────────────────────────────────────
    parser.add_argument("--config", required=True,
                        help="File config CityFlow")
    parser.add_argument("--model", default="MetaSTGAT",
                        choices=["MetaSTGAT", "MetaSTGNN", "MetaSTSONAR", "STGAT", "FixedTime", "MaxPressure"],
                        help="Tipo di modello da valutare")
    parser.add_argument("--model-id", default=None,
                        help="Identificativo del modello nei file di output "
                             "(es. 'metastgat_pro', 'ablation_environment'). "
                             "Default: uguale a --model in minuscolo.")

    # ── Checkpoint ───────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint", default=None,
                        help="Percorso esplicito del checkpoint .pt/.pth. "
                             "Se non specificato, cerca final_model.pth poi best_model.pt "
                             "in --output-dir.")

    # ── Output ──────────────────────────────────────────────────────────────
    parser.add_argument("--output-dir", default=None,
                        help="Directory dove cercare il checkpoint e salvare i risultati. "
                             "Default: results/<model-id>")

    # ── Ablation flags (per env) ────────────────────────────────────────
    parser.add_argument("--ablation", default="pro",
                        choices=["pro", "paper", "environment", "temporal",
                                 "rl_core", "replay_stability"],
                        help="Preset ablation: configura CityFlowEnv con i parametri "
                             "corretti durante il test (stessi usati in training)")
    parser.add_argument("--reward-mode", default=None, choices=["custom", "paper"])
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Peso del termine anti-starvation nel reward custom usato in "
                             "training (non influenza le metriche di valutazione qui, che non "
                             "dipendono dal reward — solo per coerenza/tracciabilita' della config).")
    parser.add_argument("--no-vision-cutoff", action="store_true")
    parser.add_argument("--no-wait-vec",      action="store_true")
    parser.add_argument("--no-action-mask",   action="store_true")
    parser.add_argument("--no-tanh-meta",     action="store_true",
                        help="Costruisce il modello senza la Tanh finale nei meta-learner "
                             "(deve corrispondere a come e' stato allenato il checkpoint)")

    # ── Valutazione ─────────────────────────────────────────────────────────
    parser.add_argument("--n-eval", type=int, default=1,
                        help="Numero di episodi di valutazione. Default 1: a epsilon=0, con "
                             "flow/seed CityFlow fissi e laneChange disattivato, ogni episodio "
                             "sulla stessa config e' identico bit per bit — ripeterlo (il paper "
                             "usa 10) non riduce alcuna varianza reale, la simulazione non ha "
                             "nulla di stocastico da mediare. Passa un valore >1 esplicitamente "
                             "solo per confronto diretto col paper o se in futuro si introduce "
                             "una vera fonte di variabilita' (es. piu' flow con seed diversi).")

    # ── Modello ────────────────────────────────────────────────────────────
    parser.add_argument("--hidden-dim",    type=int, default=DEFAULTS["hidden_dim"])
    parser.add_argument("--num-heads",     type=int, default=DEFAULTS["num_heads"])
    parser.add_argument("--num-neighbors", type=int, default=DEFAULTS["num_neighbors"])
    parser.add_argument("--num-layers", type=int, default=1, choices=[1, 2],
                        help="Deve combaciare con --num-layers usato in training per "
                             "--model MetaSTGAT|MetaSTGNN (istruzioni seconda parte.md §2.1).")
    parser.add_argument("--sonar-recurrences", type=int, default=2,
                        help="L, deve combaciare con --sonar-recurrences usato in training "
                             "per --model MetaSTSONAR.")
    parser.add_argument("--sonar-step-size", type=float, default=0.1,
                        help="h, deve combaciare con --sonar-step-size usato in training "
                             "per --model MetaSTSONAR.")
    parser.add_argument("--no-sonar-dissipation", action="store_true")
    parser.add_argument("--no-sonar-forcing", action="store_true")
    parser.add_argument("--device", default=None)

    return parser.parse_args()


def _resolve_env_flags(args):
    """
    Imposta i flag ablation dell'environment in base al preset --ablation.
    I flag espliciti (--no-*) hanno precedenza sul preset.
    """
    preset = args.ablation
    if args.reward_mode is None:
        args.reward_mode = "paper" if preset in ("paper", "environment") else "custom"
    if preset in ("environment", "paper"):
        if not args.no_vision_cutoff:
            args.no_vision_cutoff = True
        if not args.no_wait_vec:
            args.no_wait_vec = True
        if not args.no_action_mask:
            args.no_action_mask = True
    if preset in ("paper", "replay_stability"):
        if not getattr(args, "no_tanh_meta", False):
            args.no_tanh_meta = True
    if not hasattr(args, "no_tanh_meta"):
        args.no_tanh_meta = False


def _find_checkpoint(output_dir):
    """
    Cerca automaticamente il checkpoint in output_dir.
    Priorità: selected_model.pth > final_model.pth > best_model.pt > checkpoint_ep*.pt (più recente)

    selected_model.pth viene scritto da train.py a fine training: è il vincitore
    del confronto automatico tra final_model.pth e il best_model.pt dell'ultimo
    stage di training, valutato su una config di validazione.

    I primi tre sono "puntatori" e restano sempre alla radice di output_dir;
    i checkpoint periodici invece vivono in output_dir/checkpoints/ (riordino
    del 12/9/2026) — cercati anche alla radice per compatibilità con cartelle
    più vecchie non ancora spostate in quella struttura.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return None
    selected = os.path.join(output_dir, "selected_model.pth")
    if os.path.exists(selected):
        return selected
    final = os.path.join(output_dir, "final_model.pth")
    if os.path.exists(final):
        return final
    best = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best):
        print("[WARN] final_model.pth non trovato, uso best_model.pt")
        return best
    for search_dir in (os.path.join(output_dir, "checkpoints"), output_dir):
        if not os.path.isdir(search_dir):
            continue
        checkpoints = sorted(
            f for f in os.listdir(search_dir)
            if f.startswith("checkpoint_ep") and f.endswith(".pt")
        )
        if checkpoints:
            ckpt = os.path.join(search_dir, checkpoints[-1])
            print(f"[WARN] Usando checkpoint periodico: {ckpt}")
            return ckpt
    return None


def evaluate(args):
    # ── Setup ───────────────────────────────────────────────────────────
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_id = args.model_id or args.model.lower()
    output_dir = args.output_dir or f"results/{model_id}"
    os.makedirs(output_dir, exist_ok=True)
    # Sottocartelle per famiglia di file (riordino del 12/9/2026, vedi
    # results/REORGANIZATION.md): create qui una volta per tutte.
    test_csv_dir = os.path.join(output_dir, "test_csv")
    test_summaries_dir = os.path.join(output_dir, "test_summaries")
    raw_distributions_dir = os.path.join(output_dir, "raw_distributions")
    for d in (test_csv_dir, test_summaries_dir, raw_distributions_dir):
        os.makedirs(d, exist_ok=True)

    _resolve_env_flags(args)

    checkpoint_path = args.checkpoint or _find_checkpoint(output_dir)
    if checkpoint_path is None and args.model not in ("FixedTime", "MaxPressure"):
        print("[WARN] Nessun checkpoint trovato. Uso il modello non addestrato.")

    config_basename = os.path.splitext(os.path.basename(args.config))[0]

    print(f"\n{'='*60}")
    print(f"  TEST: {model_id}")
    print(f"  Config:     {config_basename}")
    print(f"  Checkpoint: {checkpoint_path or 'N/A'}")
    print(f"  Ablation:   {args.ablation} | reward={args.reward_mode} | alpha={getattr(args, 'alpha', 0.5)} | "
          f"vision_cutoff={not args.no_vision_cutoff} | "
          f"wait_vec={not args.no_wait_vec} | action_mask={not args.no_action_mask}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")

    # ── Prepara config CityFlow ──────────────────────────────────────────────
    with open(args.config, "r") as f:
        cfg = json.load(f)
    cfg["saveReplay"] = True
    base_dir = cfg.get("dir", "./")
    rel_out_dir = os.path.relpath(output_dir, base_dir)
    cfg["replayLogFile"] = os.path.join(rel_out_dir, "replay.txt").replace("\\", "/")
    cfg["roadnetLogFile"] = os.path.join(rel_out_dir, "roadnet.log").replace("\\", "/")
    temp_config_path = os.path.join(output_dir, f"temp_test_config_{config_basename}.json")
    with open(temp_config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Inizializza ambiente con flag ablation ────────────────────────────────
    env = CityFlowEnv(
        temp_config_path,
        num_neighbors=args.num_neighbors,
        alpha=getattr(args, "alpha", 0.5),
        reward_mode=args.reward_mode,
        use_vision_cutoff=not args.no_vision_cutoff,
        use_wait_vec=not args.no_wait_vec,
        use_action_mask=not args.no_action_mask,
    )
    edge_index = env.get_edge_index().to(device)

    # ── Inizializza agente ────────────────────────────────────────────────────
    if args.model == "FixedTime":
        agent = FixedTimeAgent(n_phases=env.action_space_n)
    elif args.model == "MaxPressure":
        agent = MaxPressureAgent(n_phases=env.action_space_n)
    else:
        if args.model == "MetaSTGAT":
            model = MetaSTGAT(
                state_dim=env.observation_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                n_actions=env.action_space_n,
                spatial_meta_dim=env.spatial_meta_dim,
                temporal_meta_dim=env.temporal_meta_dim,
                meta_hidden_dim=args.hidden_dim,
                use_tanh_meta=not args.no_tanh_meta,
                num_layers=getattr(args, "num_layers", 1),
            )
        elif args.model == "MetaSTGNN":
            model = MetaSTGNN(
                state_dim=env.observation_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                n_actions=env.action_space_n,
                spatial_meta_dim=env.spatial_meta_dim,
                temporal_meta_dim=env.temporal_meta_dim,
                meta_hidden_dim=args.hidden_dim,
                use_tanh_meta=not args.no_tanh_meta,
                num_layers=getattr(args, "num_layers", 1),
            )
        elif args.model == "MetaSTSONAR":
            model = MetaSTSONAR(
                state_dim=env.observation_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                n_actions=env.action_space_n,
                spatial_meta_dim=env.spatial_meta_dim,
                temporal_meta_dim=env.temporal_meta_dim,
                meta_hidden_dim=args.hidden_dim,
                use_tanh_meta=not args.no_tanh_meta,
                n_recurrences=getattr(args, "sonar_recurrences", 2),
                step_size=getattr(args, "sonar_step_size", 0.1),
                use_dissipation=not getattr(args, "no_sonar_dissipation", False),
                use_forcing=not getattr(args, "no_sonar_forcing", False),
            )
        else:
            model = STGAT(
                state_dim=env.observation_dim,
                hidden_dim=args.hidden_dim,
                num_heads=args.num_heads,
                n_actions=env.action_space_n,
            )
        agent = DQNAgent(model=model, n_intersections=env.n_intersections, device=device)
        if checkpoint_path:
            agent.load_checkpoint(checkpoint_path)
            print(f"[OK] Checkpoint caricato: {checkpoint_path}")
        else:
            print("[WARN] Nessun checkpoint caricato — pesi casuali.")
        agent.epsilon = 0.0
        agent.model.eval()

    # ── Loop di valutazione ───────────────────────────────────────────────────
    episode_results = []
    # Distribuzioni grezze per veicolo (non ridotte a statistiche). Accumulate
    # su tutti gli episodi di valutazione (di solito n_eval=1, quindi in
    # pratica un solo episodio). Nota (13/9/2026): nessuno script le legge
    # più dopo la rimozione di compare_distributions.py -- salvate comunque
    # in raw_distributions_<config>.json per un futuro uso diagnostico.
    raw_travel_times = []
    raw_wait_ns = []
    raw_wait_ew = []

    for ep in range(1, args.n_eval + 1):
        obs = env.reset()
        phase_counts = collections.Counter()
        state_history = {iid: [] for iid in env.inter_ids}
        if args.model not in ("FixedTime", "MaxPressure"):
            agent.reset_hidden()
        done = False
        step = 0

        while not done:
            if args.model == "FixedTime":
                actions = agent.select_actions(env.inter_ids, current_step=step)
            elif args.model == "MaxPressure":
                actions = agent.select_actions(states=obs, inter_ids=env.inter_ids, env=env)
            else:
                spatial_meta = None
                temporal_meta = None
                if args.model == "MetaSTGAT":
                    spatial_meta = {
                        iid: env.get_spatial_meta_features(iid)
                        for iid in env.inter_ids
                    }
                    temporal_meta = {
                        iid: env.get_temporal_meta_features(
                            iid, history=state_history[iid],
                            history_len=DEFAULTS["history_len"]
                        )
                        for iid in env.inter_ids
                    }
                with torch.no_grad():
                    actions = agent.select_actions(
                        states=obs,
                        edge_index=edge_index,
                        inter_ids=env.inter_ids,
                        inter_id_to_idx=env.inter_id_to_idx,
                        spatial_meta=spatial_meta,
                        temporal_meta=temporal_meta,
                        invalid_actions=env.get_invalid_actions()
                    )

            for p in actions.values():
                phase_counts[p] += 1
            obs, _, done, _ = env.step(actions)

            if args.model == "MetaSTGAT":
                for iid in env.inter_ids:
                    state_history[iid].append(obs[iid])
                    if len(state_history[iid]) > DEFAULTS["history_len"]:
                        state_history[iid].pop(0)
            step += 1

        # travel_time: metrica robusta (include i veicoli non arrivati) usata per
        # il ranking. travel_time_completed_only: vecchia convenzione "solo
        # arrivati", stessa del paper — solo informativa, mai per il ranking.
        tt = env.get_average_travel_time()
        tt_completed_only = env.get_completed_only_travel_time()
        tp = env.get_throughput()
        total_vehicles = len(env.all_spawned_vehicles)
        total_phase_steps = sum(phase_counts.values()) or 1
        phase_pcts = {
            f"phase_pct_{p}": round(phase_counts.get(p, 0) / total_phase_steps * 100, 2)
            for p in range(8)
        }
        # Metriche di coda (max/std/percentili sul TT dei soli arrivati) ed
        # equita' direzionale (attesa max N/S vs W/E) — vedi
        # CityFlowEnv.get_travel_time_stats()/get_direction_fairness_stats().
        # Utili perche' la media da sola puo' nascondere sia code lunghe su
        # pochi veicoli sia squilibri sistematici tra direzioni (es. arterie).
        tt_stats = env.get_travel_time_stats()
        fairness = env.get_direction_fairness_stats()
        raw_travel_times.extend(env.get_raw_travel_times())
        # Per-veicolo (migliaia di punti), non per-intersezione (16) -- vedi
        # CityFlowEnv.get_raw_vehicle_waits(). max_wait_ns/ew (sopra, "fairness")
        # restano la metrica ufficiale di ranking; questa e' solo per i violin plot.
        veh_wait_ns, veh_wait_ew = env.get_raw_vehicle_waits()
        raw_wait_ns.extend(veh_wait_ns)
        raw_wait_ew.extend(veh_wait_ew)
        row = {"episode": ep, "travel_time": round(tt, 4),
               "travel_time_completed_only": round(tt_completed_only, 4), "throughput": tp,
               "total_vehicles": total_vehicles,
               "tt_max": round(tt_stats["max"], 4), "tt_std": round(tt_stats["std"], 4),
               "tt_p50": round(tt_stats["p50"], 4), "tt_p90": round(tt_stats["p90"], 4),
               "tt_p95": round(tt_stats["p95"], 4), "tt_p99": round(tt_stats["p99"], 4),
               "wait_avg_ns": round(fairness["avg_wait_ns"], 4), "wait_max_ns": round(fairness["max_wait_ns"], 4),
               "wait_max_ns_resolved": round(fairness["max_wait_ns_resolved"], 4),
               "wait_avg_ew": round(fairness["avg_wait_ew"], 4), "wait_max_ew": round(fairness["max_wait_ew"], 4),
               "wait_max_ew_resolved": round(fairness["max_wait_ew_resolved"], 4),
               "wait_worst": round(fairness["worst_wait"], 4),
               **phase_pcts}
        episode_results.append(row)

        pct_str = " | ".join(f"P{p}:{phase_pcts[f'phase_pct_{p}']:.1f}%" for p in range(8))
        print(f"  Ep {ep:>3}/{args.n_eval}: TT={tt:.2f}s (solo arrivati: {tt_completed_only:.2f}s) | "
              f"TP={tp:>4}/{total_vehicles} | {pct_str}")
        # "risolta" = solo stalli conclusi entro la fine dell'episodio (il veicolo si e'
        # mosso di nuovo) -- esclude i casi ancora in corso quando la simulazione finisce,
        # che altrimenti farebbero sembrare "max" un numero potenzialmente troncato (vedi
        # get_direction_fairness_stats()). Le due possono differire parecchio.
        print(f"           TT coda: max={tt_stats['max']:.1f}s p90={tt_stats['p90']:.1f}s "
              f"p99={tt_stats['p99']:.1f}s std={tt_stats['std']:.1f}s | "
              f"Attesa max N/S={fairness['max_wait_ns']:.0f}s (risolta: {fairness['max_wait_ns_resolved']:.0f}s) "
              f"W/E={fairness['max_wait_ew']:.0f}s (risolta: {fairness['max_wait_ew_resolved']:.0f}s)")

    # ── Salvataggio CSV ──────────────────────────────────────────────────────
    csv_path = os.path.join(test_csv_dir, f"test_{config_basename}.csv")
    fieldnames = (["episode", "travel_time", "travel_time_completed_only", "throughput", "total_vehicles",
                   "tt_max", "tt_std", "tt_p50", "tt_p90", "tt_p95", "tt_p99",
                   "wait_avg_ns", "wait_max_ns", "wait_max_ns_resolved",
                   "wait_avg_ew", "wait_max_ew", "wait_max_ew_resolved", "wait_worst"]
                  + [f"phase_pct_{p}" for p in range(8)])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_results)

    avg_tt = float(np.mean([r["travel_time"] for r in episode_results]))
    avg_tt_completed_only = float(np.mean([r["travel_time_completed_only"] for r in episode_results]))
    avg_tp = float(np.mean([r["throughput"]  for r in episode_results]))
    # I veicoli totali spawnati dipendono solo dal flow della config (deterministico):
    # identici episodio per episodio, la mediana e' solo per robustezza.
    total_vehicles_spawned = int(round(np.median([r["total_vehicles"] for r in episode_results])))

    # Coda del travel time: ricalcolata sul pool di tutti i travel time grezzi
    # di tutti gli episodi di valutazione, non come media delle statistiche
    # per-episodio. Bug corretto il 13/9/2026: con n_eval=1 (il default e finora
    # l'unico uso reale di questo script, la simulazione e' deterministica --
    # vedi help di --n-eval) le due formulazioni coincidono esattamente, ma per
    # n_eval>1 la media di percentili per-episodio non e' il percentile del
    # pool complessivo (es. una "p99" media di 10 p99 separati non e' il vero
    # 99-esimo percentile su tutti i veicoli insieme) -- qui usiamo il pool
    # grezzo (raw_travel_times, gia' raccolto per ogni episodio) per calcolarle
    # una sola volta, correttamente, indipendentemente da n_eval.
    if raw_travel_times:
        _tt_arr = np.array(raw_travel_times, dtype=np.float64)
        avg_tt_max = float(_tt_arr.max())
        avg_tt_std = float(_tt_arr.std())
        avg_tt_p90 = float(np.percentile(_tt_arr, 90))
        avg_tt_p95 = float(np.percentile(_tt_arr, 95))
        avg_tt_p99 = float(np.percentile(_tt_arr, 99))
    else:
        avg_tt_max = avg_tt_std = avg_tt_p90 = avg_tt_p95 = avg_tt_p99 = 0.0
    avg_wait_avg_ns = float(np.mean([r["wait_avg_ns"] for r in episode_results]))
    avg_wait_max_ns = float(np.mean([r["wait_max_ns"] for r in episode_results]))
    avg_wait_max_ns_resolved = float(np.mean([r["wait_max_ns_resolved"] for r in episode_results]))
    avg_wait_avg_ew = float(np.mean([r["wait_avg_ew"] for r in episode_results]))
    avg_wait_max_ew = float(np.mean([r["wait_max_ew"] for r in episode_results]))
    avg_wait_max_ew_resolved = float(np.mean([r["wait_max_ew_resolved"] for r in episode_results]))
    avg_wait_worst = float(np.mean([r["wait_worst"] for r in episode_results]))

    print("\n" + "─"*60)
    print(f"  RISULTATI FINALI — {model_id} su {config_basename}")
    print(f"  Travel Time medio:  {avg_tt:.2f} s  (solo arrivati: {avg_tt_completed_only:.2f} s)")
    print(f"  Throughput medio:   {avg_tp:.1f} / {total_vehicles_spawned} veicoli")
    print(f"  TT coda:            max={avg_tt_max:.1f}s  p90={avg_tt_p90:.1f}s  p95={avg_tt_p95:.1f}s  "
          f"p99={avg_tt_p99:.1f}s  std={avg_tt_std:.1f}s")
    print(f"  Equita' direzionale: N/S - media={avg_wait_avg_ns:.1f}s max={avg_wait_max_ns:.1f}s "
          f"(risolta={avg_wait_max_ns_resolved:.1f}s)  |  "
          f"W/E - media={avg_wait_avg_ew:.1f}s max={avg_wait_max_ew:.1f}s "
          f"(risolta={avg_wait_max_ew_resolved:.1f}s)  |  peggiore={avg_wait_worst:.1f}s")
    print(f"  CSV:                {csv_path}")
    print("\u2500"*60 + "\n")

    # ── JSON di riepilogo ────────────────────────────────────────────────────
    summary = {
        "model_id":        model_id,
        "model":           args.model,
        "config":          args.config,
        "config_basename": config_basename,
        "checkpoint":      checkpoint_path,
        "ablation":        args.ablation,
        "alpha":           getattr(args, "alpha", 0.5),
        "n_eval":          args.n_eval,
        "avg_travel_time": round(avg_tt, 4),
        "avg_travel_time_completed_only": round(avg_tt_completed_only, 4),
        "avg_throughput":  round(avg_tp, 1),
        "total_vehicles":  total_vehicles_spawned,
        "tt_max": round(avg_tt_max, 4), "tt_std": round(avg_tt_std, 4),
        "tt_p90": round(avg_tt_p90, 4), "tt_p95": round(avg_tt_p95, 4), "tt_p99": round(avg_tt_p99, 4),
        "wait_avg_ns": round(avg_wait_avg_ns, 4), "wait_max_ns": round(avg_wait_max_ns, 4),
        "wait_max_ns_resolved": round(avg_wait_max_ns_resolved, 4),
        "wait_avg_ew": round(avg_wait_avg_ew, 4), "wait_max_ew": round(avg_wait_max_ew, 4),
        "wait_max_ew_resolved": round(avg_wait_max_ew_resolved, 4),
        "wait_worst": round(avg_wait_worst, 4),
    }
    summary_path = os.path.join(test_summaries_dir, f"test_summary_{config_basename}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    # ── Distribuzioni grezze per veicolo (nessun consumatore, vedi nota sopra) ──
    raw_path = os.path.join(raw_distributions_dir, f"raw_distributions_{config_basename}.json")
    with open(raw_path, "w") as f:
        json.dump({
            "travel_times": raw_travel_times,   # un valore per veicolo (arrivato o no, get_raw_travel_times())
            "wait_ns": raw_wait_ns,             # un valore per veicolo (arrivato o no, get_raw_vehicle_waits())
            "wait_ew": raw_wait_ew,
        }, f)

    # ── Plot percentuali di fase, mediato su tutte le config gia' testate ────
    _update_phase_pct_avg(model_id, output_dir, test_csv_dir)

    # ── Pulizia ──────────────────────────────────────────────────────────────
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    temp_roadnet_log = os.path.join(output_dir, "roadnet.log")
    if os.path.exists(temp_roadnet_log):
        os.rename(temp_roadnet_log, os.path.join(output_dir, "roadnet_log.json"))

    return summary


def _update_phase_pct_avg(model_id, output_dir, test_csv_dir):
    """Ricalcola e sovrascrive un UNICO plot delle percentuali di fase,
    mediato su tutte le config di cui esiste gia' un CSV in test_csv_dir
    (non un plot per singola config -- riordino del 12/9/2026, vedi
    results/REORGANIZATION.md e results/plots/ per la convenzione analoga
    usata nel riordino dei risultati preesistenti).

    Chiamata ad ogni test: se questa chiamata copre solo la config X ma
    test_csv_dir contiene gia' i CSV di altre config testate in precedenza
    per lo stesso model_id, il plot risultante media su TUTTE, non solo su X
    (stessa filosofia di scripts/plot_results.py::plot_phase_percentages(),
    solo che qui l'output e' dentro la cartella del modello invece che in un
    plots/ condiviso).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib non trovato. Installa con: pip install matplotlib")
        return

    csv_paths = sorted(
        os.path.join(test_csv_dir, f) for f in os.listdir(test_csv_dir)
        if f.startswith("test_") and f.endswith(".csv")
    )
    if not csv_paths:
        return

    per_config_means = []  # una entry per config: {phase_pct_p: media episodi}
    for path in csv_paths:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        phase_cols = [c for c in rows[0].keys() if c.startswith("phase_pct_")]
        if not phase_cols:
            continue
        cfg_means = {
            col: float(np.mean([float(r[col]) for r in rows if r.get(col) not in (None, "")]))
            for col in phase_cols
        }
        per_config_means.append(cfg_means)
    if not per_config_means:
        return

    phases = list(range(8))
    avg_pcts = [
        float(np.mean([d[f"phase_pct_{p}"] for d in per_config_means if f"phase_pct_{p}" in d]))
        for p in phases
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(phases, avg_pcts, color="steelblue", edgecolor="black")
    ax.set_xlabel("Fase Semaforica (Indice)")
    ax.set_ylabel("Percentuale di Scelta (%)")
    ax.set_title(f"Distribuzione media fasi (media su {len(per_config_means)} config) — {model_id}")
    ax.set_xticks(phases)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, pct in zip(bars, avg_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "phase_pct_avg.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Plot fasi (media su {len(per_config_means)} config) salvato in: {plot_path}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
