"""
Graph builder: costruisce il grafo delle intersezioni dal roadnet CityFlow.

Usato per creare gli oggetti Data/Batch di PyTorch Geometric.
"""

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


def load_roadnet(roadnet_path: str) -> dict:
    """Carica il file roadnet.json."""
    with open(roadnet_path, "r") as f:
        return json.load(f)


def get_intersection_positions(roadnet: dict) -> Dict[str, Tuple[float, float]]:
    """Restituisce un dict {intersection_id -> (x, y)}."""
    positions = {}
    for inter in roadnet.get("intersections", []):
        if not inter.get("virtual", True):
            pt = inter.get("point", {})
            positions[inter["id"]] = (pt.get("x", 0.0), pt.get("y", 0.0))
    return positions


def build_graph(
    inter_ids: List[str],
    adjacency: Dict[str, List[str]],
    node_features: Optional[np.ndarray] = None,
    edge_features: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
    inter_id_to_idx: Optional[Dict[str, int]] = None,
) -> Data:
    """
    Costruisce un oggetto Data di PyTorch Geometric.

    Args:
        inter_ids: lista ordinata degli ID delle intersezioni
        adjacency: dict {id -> [vicini]}
        node_features: array (N, F) di feature per ogni nodo
        edge_features: dict {(src_id, dst_id) -> array} (opzionale)
        inter_id_to_idx: mappa id -> indice (se None, costruita da inter_ids)

    Returns:
        torch_geometric.data.Data con:
            x: (N, F) node features
            edge_index: (2, E) connessioni
            edge_attr: (E, EF) edge features (se fornite)
    """
    if inter_id_to_idx is None:
        inter_id_to_idx = {iid: idx for idx, iid in enumerate(inter_ids)}

    n = len(inter_ids)

    # ── Node features ──────────────────────────────────────────────────────────
    if node_features is not None:
        x = torch.tensor(node_features, dtype=torch.float32)
    else:
        x = torch.zeros((n, 1), dtype=torch.float32)

    # ── Edge index ─────────────────────────────────────────────────────────────
    src_list, dst_list = [], []
    edge_attr_list = []

    for iid in inter_ids:
        i = inter_id_to_idx[iid]
        for jid in adjacency.get(iid, []):
            j = inter_id_to_idx.get(jid, -1)
            if j < 0:
                continue
            src_list.append(i)
            dst_list.append(j)

            if edge_features:
                ef = edge_features.get((iid, jid))
                if ef is not None:
                    edge_attr_list.append(ef)

    edge_index = torch.tensor(
        [src_list, dst_list], dtype=torch.long
    )  # (2, E)

    # ── Edge attributes ────────────────────────────────────────────────────────
    if edge_attr_list:
        edge_attr = torch.tensor(
            np.stack(edge_attr_list, axis=0), dtype=torch.float32
        )  # (E, EF)
    else:
        edge_attr = None

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def build_batch_graph(
    inter_ids: List[str],
    adjacency: Dict[str, List[str]],
    node_features_batch: np.ndarray,
    inter_id_to_idx: Dict[str, int],
) -> "torch_geometric.data.Batch":
    """
    Costruisce un Batch PyG da un array (B, N, F) di feature.

    Args:
        node_features_batch: array (B, N, F), B = batch size
        Restituisce un Batch con B grafi identici (stessa topologia,
        feature diverse).
    """
    from torch_geometric.data import Batch

    B = node_features_batch.shape[0]
    graphs = []
    for b in range(B):
        g = build_graph(
            inter_ids=inter_ids,
            adjacency=adjacency,
            node_features=node_features_batch[b],
            inter_id_to_idx=inter_id_to_idx,
        )
        graphs.append(g)

    return Batch.from_data_list(graphs)
