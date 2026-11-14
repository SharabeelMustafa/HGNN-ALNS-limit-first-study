"""HGNN (Final Methodology Section 7).

Relation-aware dynamic (GATv2-style) attention with edge features, softmax taken jointly
over all incoming relations (7.2)-(7.3); pre-LN residual blocks with type-specific
LayerNorm and GELU FFN (7.4)-(7.5); route-conditioned scoring head (7.6)-(7.7); co-routing
head (7.8).  Ablation switches: hetero (False = homogeneous GATv2, B5 / A-HGNN-1),
act ('gelu'|'relu', A-HGNN-5), norm ('layer'|'batch', A-HGNN-6), edge_feats (A-HGNN-3),
route_state (A-HGNN-4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax as seg_softmax

from ..graph import N_REL, F_STOP, F_CLASS, F_EDGE, F_XI, F_CHI

F_PAD = max(F_STOP, F_CLASS)


def _act(name):
    return nn.GELU() if name == "gelu" else nn.ReLU()


class Typed(nn.Module):
    """Apply a separate module per node type."""

    def __init__(self, make, n_types):
        super().__init__()
        self.mods = nn.ModuleList([make() for _ in range(n_types)])

    def forward(self, x, ntype, out_dim=None):
        if len(self.mods) == 1:
            return self.mods[0](x)
        out = None
        for t, m in enumerate(self.mods):
            idx = (ntype == t).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            y = m(x[idx])
            if out is None:
                out = x.new_zeros((x.shape[0], y.shape[1]))
            out[idx] = y
        return out


class HetLayer(nn.Module):
    def __init__(self, d, heads, n_rel, n_types, act, norm, dropout, edge_feats):
        super().__init__()
        self.d, self.h, self.dh = d, heads, d // heads
        self.n_rel = n_rel
        self.edge_feats = edge_feats
        mk_norm = (lambda: nn.LayerNorm(d)) if norm == "layer" else (lambda: nn.BatchNorm1d(d))
        self.ln1 = Typed(mk_norm, n_types)
        self.ln2 = Typed(mk_norm, n_types)
        self.w_src = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_rel)])
        self.w_dst = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_rel)])
        self.w_e = nn.Linear(d, d, bias=False)
        self.r = nn.Parameter(torch.zeros(n_rel, d))
        self.a = nn.Parameter(torch.randn(n_rel, heads, self.dh) * 0.1)
        self.v = nn.ModuleList([nn.Linear(d, d) for _ in range(n_rel)])
        self.u = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(n_rel)])
        self.w_o = Typed(lambda: nn.Linear(d, d), n_types)
        self.ffn = Typed(lambda: nn.Sequential(nn.Linear(d, 2 * d), _act(act), nn.Linear(2 * d, d)), n_types)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, ntype, ei, rel, e):
        N = h.shape[0]
        x = self.ln1(h, ntype)
        scores, msgs, dsts = [], [], []
        for p in range(self.n_rel):
            idx = (rel == p).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            s, t = ei[0, idx], ei[1, idx]
            z = self.w_src[p](x[s]) + self.w_dst[p](x[t]) + self.r[p]
            m = self.v[p](x[s])
            if self.edge_feats:
                z = z + self.w_e(e[idx])
                m = m + self.u[p](e[idx])
            z = F.leaky_relu(z, 0.2).view(-1, self.h, self.dh)
            scores.append((z * self.a[p]).sum(-1))
            msgs.append(m.view(-1, self.h, self.dh))
            dsts.append(t)
        sc = torch.cat(scores)
        mg = torch.cat(msgs)
        dst = torch.cat(dsts)
        alpha = seg_softmax(sc, dst, num_nodes=N)
        out = h.new_zeros((N, self.h, self.dh)).index_add_(0, dst, alpha.unsqueeze(-1) * mg).view(N, self.d)
        u = h + self.drop(self.w_o(out, ntype))
        return u + self.drop(self.ffn(self.ln2(u, ntype), ntype))


class HGNN(nn.Module):
    def __init__(self, d=128, heads=8, layers=3, hetero=True, act="gelu", norm="layer", dropout=0.1,
                 edge_feats=True, route_state=True):
        super().__init__()
        self.hetero = hetero
        self.route_state = route_state
        n_types = 3 if hetero else 1
        n_rel = N_REL if hetero else 1
        self.n_rel = n_rel
        if hetero:
            self.in_stop = nn.Linear(F_STOP, d)
            self.in_class = nn.Linear(F_CLASS, d)
        else:
            self.in_any = nn.Linear(F_PAD, d)
        self.edge_enc = nn.ModuleList([nn.Linear(F_EDGE, d) for _ in range(n_rel)])
        self.layers = nn.ModuleList([HetLayer(d, heads, n_rel, n_types, act, norm, dropout, edge_feats)
                                     for _ in range(layers)])
        self.out_ln = nn.LayerNorm(d)
        # route head
        self.w_g = nn.Linear(d, d)
        self.w_beta = nn.Linear(d, 1)
        self.w_s = nn.Linear(F_XI, d)
        self.score = nn.Sequential(nn.Linear(3 * d + F_CHI, d), _act(act), nn.Linear(d, 1))
        # co-routing head
        self.co = nn.Sequential(nn.Linear(3 * d, d), _act(act), nn.Linear(d, 1))

    # ---------------------------------------------------------------- encoder
    def encode(self, x_stop, x_class, stop_idx, class_idx, ntype, ei, rel, eattr):
        N = ntype.shape[0]
        if self.hetero:
            h = x_stop.new_zeros((N, self.in_stop.out_features))
            h[stop_idx] = self.in_stop(x_stop)
            h[class_idx] = self.in_class(x_class)
            nt = ntype
            r = rel
        else:
            xp = x_stop.new_zeros((N, F_PAD))
            xp[stop_idx, :F_STOP] = x_stop
            xp[class_idx, :F_CLASS] = x_class
            h = self.in_any(xp)
            nt = torch.zeros_like(ntype)
            r = torch.zeros_like(rel)
        e = torch.zeros((ei.shape[1], self.edge_enc[0].out_features), device=h.device)
        for p in range(self.n_rel):
            idx = (r == p).nonzero(as_tuple=True)[0]
            if idx.numel():
                e[idx] = self.edge_enc[p](eattr[idx])
        for layer in self.layers:
            h = layer(h, nt, ei, r, e)
        return self.out_ln(h), e

    # ---------------------------------------------------------------- heads
    def route_embed(self, h, cand_T, cand_xi, stop_node, stop_cand, n_cand):
        g = h[cand_T]
        if stop_node.numel():
            hs = h[stop_node]
            beta = seg_softmax(self.w_beta(hs).squeeze(-1), stop_cand, num_nodes=n_cand)
            g = g.index_add(0, stop_cand, beta.unsqueeze(-1) * self.w_g(hs))
        if self.route_state:
            g = g + self.w_s(cand_xi)
        return g

    def pair_scores(self, h, g, pair_P, pair_D, pair_c, chi):
        z = torch.cat([h[pair_P], h[pair_D], g[pair_c], chi], dim=-1)
        return self.score(z).squeeze(-1)

    def co_logits(self, h, e, src, dst, eidx):
        return self.co(torch.cat([h[src], h[dst], e[eidx]], dim=-1)).squeeze(-1)
