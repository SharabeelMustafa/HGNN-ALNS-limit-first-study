#!/usr/bin/env bash
# One-time setup on a RunPod CPU pod (Final Methodology Section 11.7). Run from /workspace/<repo>.
set -euo pipefail
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
./.venv/bin/pip install -q numpy numba scipy pandas pyyaml matplotlib torch_geometric
bash scripts/download_data.sh data/raw
mkdir -p results logs
lscpu > results/hardware_lscpu.txt
nproc > results/hardware_nproc.txt
./.venv/bin/python -c "import torch, numba; print('torch', torch.__version__, 'numba', numba.__version__)" | tee results/hardware_versions.txt
echo "Physical cores: $(lscpu -p=core | grep -v '^#' | sort -u | wc -l)  (use that minus 1 as --workers)"
