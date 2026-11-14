"""Collate instance graphs + destroyed states into one batched graph (PyG-style offsets)."""
from __future__ import annotations

import numpy as np
import torch


def collate(items, device="cpu"):
    """items: list of (sample_dict, list_of_states). Returns dict of tensors."""
    xs, xc, stop_idx, class_idx, ntype = [], [], [], [], []
    ei, rel, ea = [], [], []
    cand_T, cand_xi, stop_node, stop_cand = [], [], [], []
    pair_P, pair_D, pair_c, pair_g, pair_y, chi = [], [], [], [], [], []
    co_src, co_dst, co_eidx, co_y = [], [], [], []
    off = 0
    eoff = 0
    coff = 0
    goff = 0
    for sample, states in items:
        g = sample["graph"]
        n, C = g["n"], g["C"]
        Nn = 2 * n + C
        xs.append(g["x_stop"]); xc.append(g["x_class"])
        stop_idx.append(np.arange(off, off + 2 * n)); class_idx.append(np.arange(off + 2 * n, off + Nn))
        ntype.append(np.concatenate([np.zeros(n), np.ones(n), np.full(C, 2)]).astype(np.int64))
        ei.append(g["edge_index"] + off); rel.append(g["edge_rel"]); ea.append(g["edge_attr"])
        if sample.get("co_edges") is not None and len(sample["co_edges"]):
            ce = sample["co_edges"]
            co_src.append(g["edge_index"][0, ce] + off); co_dst.append(g["edge_index"][1, ce] + off)
            co_eidx.append(ce + eoff); co_y.append(sample["co_y"])
        for st in states:
            nc = len(st["cand"])
            cand_T.append(off + 2 * n + st["cls"])
            cand_xi.append(st["xi"])
            for c, stops in enumerate(st["stops"]):
                if stops:
                    stop_node.append(np.asarray(stops, dtype=np.int64) + off)
                    stop_cand.append(np.full(len(stops), coff + c, dtype=np.int64))
            R = len(st["removed"])
            for a in range(R):
                cs = np.flatnonzero(st["mask"][a])
                if len(cs) == 0:
                    continue
                r = int(st["removed"][a])
                pair_P.append(np.full(len(cs), off + r)); pair_D.append(np.full(len(cs), off + n + r))
                pair_c.append(cs + coff); pair_g.append(np.full(len(cs), goff))
                pair_y.append((cs == st["target"][a]).astype(np.float32)) if "target" in st else pair_y.append(np.zeros(len(cs), np.float32))
                chi.append(st["chi"][a, cs])
                goff += 1
            coff += nc
        off += Nn
        eoff += g["edge_index"].shape[1]

    def T(x, dt=torch.float32):
        return torch.as_tensor(np.concatenate(x) if len(x) else np.zeros(0), dtype=dt, device=device)

    out = {
        "x_stop": T(xs), "x_class": T(xc), "stop_idx": T(stop_idx, torch.long),
        "class_idx": T(class_idx, torch.long), "ntype": T(ntype, torch.long),
        "ei": torch.as_tensor(np.concatenate(ei, axis=1), dtype=torch.long, device=device),
        "rel": T(rel, torch.long), "eattr": T(ea),
        "cand_T": T(cand_T, torch.long), "cand_xi": T(cand_xi), "n_cand": coff,
        "stop_node": T(stop_node, torch.long), "stop_cand": T(stop_cand, torch.long),
        "pair_P": T(pair_P, torch.long), "pair_D": T(pair_D, torch.long), "pair_c": T(pair_c, torch.long),
        "pair_g": T(pair_g, torch.long), "pair_y": T(pair_y), "chi": T(chi), "n_groups": goff,
        "co_src": T(co_src, torch.long), "co_dst": T(co_dst, torch.long), "co_eidx": T(co_eidx, torch.long),
        "co_y": T(co_y),
    }
    if out["x_stop"].dim() == 1:
        out["x_stop"] = out["x_stop"].view(0, -1)
    return out
