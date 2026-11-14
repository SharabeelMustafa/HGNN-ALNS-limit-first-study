"""Locate and load benchmark instances from data/raw (layout created by scripts/download_data.sh)."""
from __future__ import annotations

import glob
import os

from .lilim import read_lilim
from .darp import read_malheiros, read_cordeau

DATA_ROOT = os.environ.get(
    "HGALNS_DATA", os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw"))

LILIM_DIRS = {100: "pdp_100", 200: "pdp_200", 400: "pdp_400", 600: "pdp_600",
              800: "pdptw800", 1000: "pdptw1000"}


def list_instances(family: str, size: int | None = None, root: str = DATA_ROOT) -> list[str]:
    root = os.path.abspath(root)
    if family == "lilim":
        sizes = [size] if size else sorted(LILIM_DIRS)
        out = []
        for s in sizes:
            if s == 100:
                out += glob.glob(os.path.join(root, "lilim", "pdp_100", "*.txt"))
            else:
                k = s // 100   # files like LC1_2_1.txt (200), LR1_4_3.txt (400), ... _10_ (1000)
                out += [p for p in glob.glob(os.path.join(root, "lilim", "L*_%d_*.txt" % k))]
        return sorted(set(out))
    if family == "malheiros":
        return sorted(glob.glob(os.path.join(root, "malheiros", "instances", "*.txt")))
    if family in ("gen-lilim", "gen-hdarp"):   # generated validation-for-tuning instances
        fam = family.split("-")[1]
        return sorted(glob.glob(os.path.join(root, "..", "labels", fam, f"{size or 'valopt'}_*.pkl")))
    if family == "cordeau":
        return sorted(p for p in glob.glob(os.path.join(root, "cordeau", "*")) if os.path.isfile(p))
    raise ValueError(family)


def load_instance(path: str):
    p = path.replace("\\", "/")
    if p.endswith(".pkl"):              # generated instance (labels file)
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)["instance"]
    if "/malheiros/" in p:
        return read_malheiros(path)
    if "/cordeau/" in p:
        return read_cordeau(path)
    if "/lilim/" in p or p.endswith(".txt"):
        return read_lilim(path)
    raise ValueError(f"cannot infer family of {path}")
