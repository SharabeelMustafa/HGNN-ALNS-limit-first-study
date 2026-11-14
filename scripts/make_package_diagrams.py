"""Methodology diagrams for the research package (PNG 300 dpi + PDF + SVG). Boxes and arrows only; every label matches the code."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
OUT = os.path.abspath(os.path.join(REPO, "..", "Research_Package", "10_figures"))
INK, MUTED = "#1f1f1e", "#52514e"
F = {"data": ("#f3f2ee", "#8a8984"), "classic": ("#e4f5ee", "#1baf7a"), "learn": ("#e3eefb", "#2a78d6"), "math": ("#fdf1d9", "#c98a00"),
     "oracle": ("#fbe6f3", "#d6409f"), "check": ("#ffffff", "#52514e"), "bad": ("#fde9e2", "#eb6834")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})


def canvas(w, h):
    S = 1.34                       # drawing scale: boxes get room for 8 pt labels
    fig, ax = plt.subplots(figsize=(w * S, h * S))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100 * h / w); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, kind="data", fs=8, bold=False, align="center"):
    fc, ec = F[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.2", fc=fc, ec=ec, lw=1.1, zorder=2))
    ax.text(x + (w / 2 if align == "center" else 1.2), y + h / 2, text, ha=align if align != "center" else "center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", zorder=3, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p, q, text=None, rad=0.0, color=MUTED, ls="-", tpos=0.5, toff=(0, 1.2), fs=7):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, linestyle=ls, shrinkA=0, shrinkB=0,
                                                   connectionstyle=f"arc3,rad={rad}"), zorder=1)
    if text:
        ax.text(p[0] + (q[0] - p[0]) * tpos + toff[0], p[1] + (q[1] - p[1]) * tpos + toff[1], text, ha="center", va="center", fontsize=fs, color=MUTED)


def R(b): return (b[0] + b[2], b[1] + b[3] / 2)
def L(b): return (b[0], b[1] + b[3] / 2)
def T(b): return (b[0] + b[2] / 2, b[1] + b[3])
def B(b): return (b[0] + b[2] / 2, b[1])


def lane(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.5", fc="#fafaf8", ec="#c9c8c3", lw=0.9, ls="--", zorder=0))
    ax.text(x + 1.2, y + h - 2.0, title, ha="left", va="center", fontsize=8.5, color=MUTED, fontweight="bold")


def legend(ax, x, y, items):
    for i, (kind, lab) in enumerate(items):
        fc, ec = F[kind]
        ax.add_patch(FancyBboxPatch((x, y - 0.9), 2.2, 1.8, boxstyle="round,pad=0,rounding_size=0.4", fc=fc, ec=ec, lw=1.0))
        ax.text(x + 3.0, y, lab, va="center", ha="left", fontsize=7, color=MUTED)
        x += 3.6 + 1.02 * len(lab)


def save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def d1_overview():
    fig, ax = canvas(7.2, 3.9)
    lane(ax, 1, 31, 98, 22, "Offline (once per problem family)")
    lane(ax, 1, 3, 98, 25, "Online (per instance)")
    a = box(ax, 3, 35, 15, 11, "Generated\ntraining\ninstances")
    b = box(ax, 22, 35, 17, 11, "Teacher runs:\nclassical ALNS,\n3 runs with\n4x iterations", "classic")
    c = box(ax, 43, 35, 17, 11, "Target $y_{rr'}$:\nshare of runs in\nwhich r and r'\nshare a route")
    d = box(ax, 64, 35, 15, 11, "Train HGNN\n(cross-entropy)", "learn")
    e = box(ax, 83, 35, 14, 11, "Trained\nmodel", "learn")
    for p, q in ((a, b), (b, c), (c, d), (d, e)):
        arrow(ax, R(p), L(q))
    f = box(ax, 3, 8, 13, 12, "Instance\n(requests,\nfleet)")
    g = box(ax, 19, 8, 14, 12, "Heterogeneous\ngraph\n(3 node types)", "learn")
    h = box(ax, 36, 8, 13, 12, "HGNN,\none forward\npass", "learn")
    i = box(ax, 52, 8, 15, 12, "ALNS with\nsoft hint (prior\non insertion cost)", "classic")
    j = box(ax, 70, 8, 13, 12, "Set partitioning\nover the\nroute pool", "math")
    k = box(ax, 86, 8, 11, 12, "Checked\nsolution", "check")
    arrow(ax, R(f), L(g)); arrow(ax, R(g), L(h)); arrow(ax, R(h), L(i)); arrow(ax, R(i), L(j))
    ax.text(50.5, 5.2, r"passes $\hat C$", ha="center", fontsize=7, color=MUTED); ax.text(68.5, 5.2, "passes routes", ha="center", fontsize=7, color=MUTED); arrow(ax, R(j), L(k))
    arrow(ax, (e[0] + e[2] / 2, e[1]), (h[0] + h[2] / 2 + 3, h[1] + h[3]), rad=-0.18, color=F["learn"][1], ls="--")
    legend(ax, 3, 0.9, [("classic", "classical search"), ("learn", "learned part"), ("math", "integer program"), ("data", "data")])
    save(fig, "D1_pipeline_overview")


def d2_alns():
    fig, ax = canvas(6.2, 5.3)
    a = box(ax, 30, 79.5, 40, 5.5, "Initial solution: regret-2 insertion", "classic")
    b = box(ax, 30, 69, 40, 7, "Choose removal and insertion operator\n(roulette wheel)", "classic", fs=7.5)
    c = box(ax, 30, 59.5, 40, 7, "Remove q requests\n(random, Shaw, worst, route, string)", "classic", fs=7.5)
    d = box(ax, 30, 50, 40, 7.5, "Re-insert (greedy, regret-2, regret-3)\nexact feasibility test for every insertion", "classic", fs=7.5)
    e = box(ax, 30, 41, 40, 5.5, "Accept? (simulated annealing, true cost)", "classic", fs=7.5)
    f = box(ax, 30, 32, 40, 5.5, "Update best solution and operator weights", "classic", fs=7.5)
    g = box(ax, 30, 23, 40, 5.5, "Budget used up?", "check")
    h = box(ax, 30, 12, 40, 6.5, "Set partitioning over the route pool", "math")
    i = box(ax, 30, 2.5, 40, 5.5, "Independent feasibility check, final solution", "check")
    for p, q in ((a, b), (b, c), (c, d), (d, e), (e, f), (f, g)):
        arrow(ax, B(p), T(q))
    arrow(ax, B(g), T(h), "yes", toff=(3, 0)); arrow(ax, B(h), T(i))
    ax.plot([30, 20, 20], [25.75, 25.75, 72.5], color=MUTED, lw=1.1, zorder=1)
    arrow(ax, (20, 72.5), (30, 72.5))
    ax.text(12, 49, "no:\nnext\niteration", ha="center", va="center", fontsize=7.5, color=MUTED)
    s = box(ax, 76, 50, 22, 7.5, "Soft hint:\nprior from $\\hat C$ added\nto the insertion cost", "learn", fs=7.2)
    arrow(ax, L(s), R(d), color=F["learn"][1], ls="--")
    p = box(ax, 76, 30.5, 22, 8.5, "Route pool:\nroutes of every\naccepted solution", "math", fs=7.2)
    arrow(ax, R(f), L(p), color=F["math"][1]); arrow(ax, (87, 30.5), (70, 15.25), rad=-0.25, color=F["math"][1])
    ax.set_xlim(4, 100)
    v = box(ax, 76, 68, 22, 9.5, "Li & Lim only:\nfirst 40% of the\nbudget minimises\nthe number of vehicles", "data", fs=7)
    save(fig, "D2_alns_workflow")


def d3_hgnn():
    fig, ax = canvas(7.2, 3.3)
    a1 = box(ax, 2, 32, 15, 7, "Pickup nodes\n15 features", "data", fs=7.2)
    a2 = box(ax, 2, 22.5, 15, 7, "Delivery nodes\n15 features", "data", fs=7.2)
    a3 = box(ax, 2, 13, 15, 7, "Vehicle-class nodes\n12 features", "data", fs=7.2)
    a4 = box(ax, 2, 3.5, 15, 7, "Edges, 8 relations\n4 features", "data", fs=7.2)
    b = box(ax, 21, 13, 13, 26, "Linear layer\nper node type\n\nwidth 128", "learn", fs=7.2)
    b2 = box(ax, 21, 3.5, 13, 7, "Linear per relation", "learn", fs=7)
    c = box(ax, 39, 8, 26, 31, "", "learn")
    ax.text(52, 36, "3 x attention layer", ha="center", fontsize=8, fontweight="bold", color=INK)
    ax.text(52, 23.5, "pre-layer-norm per node type\n\nattention over all incoming\nrelations (8 heads, dynamic,\nrelation-specific weights,\nedge features in keys and messages)\n\nresidual + feed-forward (GELU)", ha="center", va="center", fontsize=6.9, color=INK, linespacing=1.25)
    d = box(ax, 69, 22, 12, 12, "Node\nembeddings\n$h_v$", "learn", fs=7.5)
    e = box(ax, 69, 5, 12, 11, "Pair head:\nMLP($h_u$, $h_v$, $\\epsilon_e$)\n+ sigmoid", "learn", fs=7)
    f = box(ax, 85, 5, 13, 11, "Co-routing\nmatrix $\\hat C$\n(n x n)", "data", fs=7.5)
    for a in (a1, a2, a3):
        arrow(ax, R(a), (21, a[1] + a[3] / 2))
    arrow(ax, R(a4), L(b2)); arrow(ax, R(b), (39, 26)); arrow(ax, R(b2), (39, 11)); arrow(ax, (65, 28), L(d)); arrow(ax, B(d), T(e)); arrow(ax, R(e), L(f))
    ax.text(91.5, 27, "Ablations:\none node type and\none relation\n(homogeneous);\n0 layers\n(no message passing)", ha="center", va="center", fontsize=6.6, color=MUTED)
    save(fig, "D3_hgnn_architecture")


def d4_graph():
    fig, ax = canvas(7.2, 3.6)
    cp, cd, cv = "#2a78d6", "#1baf7a", "#eda100"
    P = {"p1": (12, 34), "p2": (26, 40), "p3": (30, 24)}
    Dn = {"d1": (14, 16), "d2": (40, 32), "d3": (36, 10)}
    V = {"v1": (4, 44), "v2": (46, 44)}
    def node(xy, col, lab):
        ax.add_patch(Circle(xy, 2.3, fc=col, ec="white", lw=1.2, zorder=3)); ax.text(xy[0], xy[1], lab, ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=4)
    def edge(p, q, col, ls="-", rad=0.0):
        ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color=col, lw=0.9, linestyle=ls, shrinkA=6.5, shrinkB=6.5, connectionstyle=f"arc3,rad={rad}"), zorder=1)
    for r in "123":
        edge(P["p" + r], Dn["d" + r], INK, rad=0.0)
    for a, b in (("p1", "p2"), ("p2", "p3"), ("p3", "d1"), ("d1", "d3"), ("p2", "d2"), ("d2", "d3"), ("p1", "p3")):
        pa = P.get(a) or Dn.get(a); pb = P.get(b) or Dn.get(b)
        edge(pa, pb, "#8a8984", rad=0.15)
    for v, ps in (("v1", ("p1", "p2")), ("v2", ("p2", "p3"))):
        for p in ps:
            edge(V[v], P[p], cv, ls="--", rad=0.0)
    for k, xy in P.items(): node(xy, cp, "P" + k[1])
    for k, xy in Dn.items(): node(xy, cd, "D" + k[1])
    for k, xy in V.items(): node(xy, cv, "V" + k[1])
    ax.text(25, 3, "Example with 3 requests and 2 vehicle classes", ha="center", fontsize=7.2, color=MUTED)
    x0 = 55
    ax.text(x0, 46, "Node types", fontsize=8, fontweight="bold", color=INK)
    for i, (col, lab) in enumerate(((cp, "Pickup (one per request)"), (cd, "Delivery (one per request)"), (cv, "Vehicle class (vehicles with equal depots, capacity, costs)"))):
        ax.add_patch(Circle((x0 + 1.5, 42 - 3.6 * i), 1.3, fc=col, ec="white")); ax.text(x0 + 4, 42 - 3.6 * i, lab, va="center", fontsize=7, color=INK)
    ax.text(x0, 29.5, "Edge types (8 relations, both directions counted)", fontsize=8, fontweight="bold", color=INK)
    rows = [(INK, "-", "pair: pickup to its delivery, and back"),
            ("#8a8984", "-", "successor: to the 16 nearest stops that are reachable\nin time (4 relations: P-P, P-D, D-P, D-D)"),
            (cv, "--", "compatible: vehicle class to pickup, and back")]
    y = 25.5
    for col, ls, lab in rows:
        ax.plot([x0, x0 + 4], [y, y], color=col, ls=ls, lw=1.3); ax.text(x0 + 5.5, y, lab, va="center", fontsize=7, color=INK, linespacing=1.2)
        y -= 5.2 if "\n" in lab else 3.8
    ax.text(x0, 9.5, "Features (scaled per instance)", fontsize=8, fontweight="bold", color=INK)
    ax.text(x0, 4.3, "stops: position, time window, service time, demand,\n   ride-time limit, distance to partner and depot\nvehicle classes: capacity, costs, shift, depot, fleet share\nedges: travel time, distance, time slack, waiting or margin", va="center", fontsize=6.8, color=INK, linespacing=1.3)
    save(fig, "D4_graph_construction")


def d5_teacher():
    fig, ax = canvas(7.2, 2.7)
    a = box(ax, 2, 20, 14, 11, "Generator\n(Li & Lim style,\nH-DARP style)")
    b = box(ax, 19, 20, 14, 11, "300 training +\n40 validation\ninstances\nper family")
    c = box(ax, 36, 20, 15, 11, "Classical ALNS\n3 seeds x\n4x iterations\n(2,040 runs)", "classic", fs=7.5)
    d = box(ax, 54, 20, 15, 11, "Soft target\n$y_{rr'}$ in [0, 1]\nper request pair")
    e = box(ax, 72, 20, 12, 11, "HGNN\ntraining,\nGPU", "learn")
    f = box(ax, 87, 20, 11, 11, "Model with\nlowest\nvalidation loss", "learn", fs=7.2)
    for p, q in ((a, b), (b, c), (c, d), (d, e), (e, f)):
        arrow(ax, R(p), L(q))
    g = box(ax, 36, 4, 15, 10, "Heterogeneous\ngraph of each\ninstance", "learn", fs=7.5)
    arrow(ax, B(b), (g[0] + 2, g[1] + g[3]), rad=0.25); arrow(ax, R(g), (e[0] + 3, e[1]), rad=0.2)
    ax.text(77, 9, "AdamW, cosine schedule, batch 16,\nearly stopping, checkpoint each epoch", ha="center", va="center", fontsize=6.8, color=MUTED)
    ax.text(43.5, 34.2, "teacher is stronger than the search it will guide", ha="center", fontsize=6.8, color=MUTED)
    save(fig, "D5_teacher_and_training_pipeline")


def d6_hint():
    fig, ax = canvas(7.2, 3.0)
    a = box(ax, 2, 25, 17, 11, "Solution after\nremoval: routes\n$R_1 \\ldots R_K$, removed\nrequests", "classic", fs=7.3)
    b = box(ax, 2, 7, 17, 11, "Predicted\nco-routing $\\hat C$\n(once per instance)", "learn", fs=7.3)
    c = box(ax, 25, 14, 22, 15, "Affinity of request r\nfor route k:\nshare of the predicted\nroute-mates of r that\nare in route k,  A(r, k)", "learn", fs=7.3)
    d = box(ax, 52, 14, 17, 15, "Prior\n$\\pi(k \\mid r)$ =\nsoftmax over routes\nof $\\beta$ A(r, k)", "learn", fs=7.3)
    e = box(ax, 74, 22, 24, 13, "Biased cost =\nexact insertion cost\n$-\\ \\lambda\\, d_{avg}\\, \\log \\pi(k \\mid r)$", "classic", fs=7.3)
    f = box(ax, 74, 4, 24, 12, "Insertion operator picks by\nbiased cost; all routes stay\ncandidates; acceptance uses\nthe true cost", "classic", fs=7)
    arrow(ax, R(a), (25, 24), rad=-0.1); arrow(ax, R(b), (25, 19), rad=0.1); arrow(ax, R(c), L(d)); arrow(ax, R(d), (74, 26), rad=-0.1); arrow(ax, B(e), T(f))
    ax.text(36, 2.2, "$\\lambda$ = 0 gives classical ALNS exactly.  No pruning.\nNo network call inside the loop.", ha="center", fontsize=7, color=MUTED)
    save(fig, "D6_soft_hint_in_alns")


def d7_sp():
    fig, ax = canvas(7.2, 2.8)
    a = box(ax, 2, 13, 17, 14, "ALNS iterations\n\nroutes of every\naccepted solution", "classic", fs=7.5)
    b = box(ax, 24, 13, 17, 14, "Route pool\n(route, vehicle class,\ncost)\n100 to 5,000 routes", "math", fs=7.3)
    c = box(ax, 46, 9, 26, 22, "Integer program (HiGHS)\n\nminimise total route cost\n\neach request in exactly one\nchosen route\n\nat most $m_g$ routes per\nvehicle class g", "math", fs=7.2)
    d = box(ax, 77, 20, 21, 11, "Better than the ALNS\nincumbent? Keep it.", "check", fs=7.3)
    e = box(ax, 77, 4, 21, 11, "Otherwise keep the\nALNS incumbent\n(never worse)", "check", fs=7.3)
    arrow(ax, R(a), L(b)); arrow(ax, R(b), L(c)); arrow(ax, (72, 24), L(d)); arrow(ax, (72, 14), L(e))
    ax.text(33, 6, "the incumbent's routes are in the pool", ha="center", fontsize=6.8, color=MUTED)
    save(fig, "D7_set_partitioning")


def d8_experiments():
    fig, ax = canvas(7.2, 4.9)
    rows = [("E04 Main experiment: first method vs classical ALNS\n140 benchmark instances, equal time", "bad", "not better;\nworse on 2 of 3 sets"),
            ("E05-E07 Diagnostics: equal iterations, components,\nlabel accuracy, first oracle", "check", "causes found;\noracle helps"),
            ("E08-E11 Limit-first pilots on 50 tuning instances:\nwhat to remove, teacher strength, use of advice,\nper-instance settings", "oracle", "only a stronger teacher,\nused as prior, helps"),
            ("E12-E13 Teacher runs (2,040) and training;\noffline gate G3", "learn", "prediction no better\nthan the old model"),
            ("E14 Search with the trained model; gate G4", "learn", "no gain"),
            ("E15-E18 Confirmation on the benchmarks:\nset partitioning, oracle, trained model", "classic", "set partitioning never\nworse; oracle helps;\nmodel does not")]
    y = 64
    prev = None
    for text, kind, out in rows:
        h = 9.2 if text.count("\n") >= 2 else 7.4
        y -= h + 1.6
        b = box(ax, 3, y, 56, h, text, kind, fs=7.3)
        o = box(ax, 68, y, 29, h, out, "data", fs=7.2)
        arrow(ax, R(b), L(o))
        if prev is not None:
            arrow(ax, B(prev), T(b))
        prev = b
    ax.text(31, 65.2, "Experiment", ha="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.text(82.5, 65.2, "Outcome", ha="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.text(50, 0.8, "Tuning and all gates use generated instances only; the benchmarks are used for E04-E07 and E15-E18.", ha="center", fontsize=6.9, color=MUTED)
    save(fig, "D8_experimental_workflow")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for f in (d1_overview, d2_alns, d3_hgnn, d4_graph, d5_teacher, d6_hint, d7_sp, d8_experiments):
        f()
    print(sorted(x for x in os.listdir(OUT) if x.endswith(".png")))
