#!/usr/bin/env bash
# Download all public benchmark datasets used in the Final Proposed Methodology (Section 5.2).
# Links verified November 2026. Usage: bash download_data.sh [target_dir]
set -euo pipefail
T=${1:-data/raw}
mkdir -p "$T"/{lilim,sartori,detti,malheiros,cordeau}
cd "$T"

# Li & Lim PDPTW (SINTEF) - 354 instances
B=https://www.sintef.no/contentassets/1338af68996841d3922bc8e87adc430c
for f in pdp_100 pdp_200 pdp_400 pdp_600 pdptw800 pdptw1000; do
  wget -q --show-progress -O lilim/$f.zip "$B/$f.zip"
  unzip -q -o lilim/$f.zip -d lilim/
done
# Li & Lim best-known solutions: https://www.sintef.no/projectweb/top/pdptw/<N>-customers/  (N = 100,200,400,600,800,1000)

# Detti et al. (2017) real-world MD-H-DARP - 24 instances
D=http://www.dii.unisi.it/~detti/darp-instances
for f in SmallInstances.zip BigInstances.zip ReadMe.txt; do wget -q --show-progress -O detti/$f "$D/$f"; done
unzip -q -o detti/SmallInstances.zip -d detti/small
unzip -q -o detti/BigInstances.zip -d detti/big

# Malheiros et al. (2021) MD-H-DARP - 24 instances; Karimi et al. reference solutions
git clone -q --depth 1 https://github.com/igormalheiros/Instances-MDHDARP malheiros/instances
git clone -q --depth 1 https://github.com/mohammadkarimi-poly/HDARP-solution malheiros/karimi_solutions

if [ "${WITH_SARTORI:-0}" = "1" ]; then
# Sartori & Buriol (2020) PDPTW - BKS, validator, format docs (instances: see below)
git clone -q --depth 1 https://github.com/cssartori/pdptw-instances sartori/repo
# Instance files: full Mendeley Data archive (public API, redirects to S3; may be large)
wget -q --show-progress -O sartori/sartori_buriol_all.zip \
  "https://data.mendeley.com/public-api/zip/wr2ct4r22f/download/2" \
  || echo "warn: Mendeley download failed - get it manually from https://data.mendeley.com/datasets/wr2ct4r22f/2"
[ -f sartori/sartori_buriol_all.zip ] && unzip -q -o sartori/sartori_buriol_all.zip -d sartori/ || true

fi

# Cordeau (2006) / Ropke et al. (2007) DARP a/b instances (support: feasibility tests, generator template)
C=http://neumann.hec.ca/chairedistributique/data/darp/branch-and-cut
for i in a2-16 a2-20 a2-24 a3-18 a3-24 a3-30 a3-36 a4-16 a4-24 a4-32 a4-40 a4-48 \
         a5-40 a5-50 a5-60 a6-48 a6-60 a6-72 a7-56 a7-70 a7-84 a8-64 a8-80 a8-96; do
  wget -q -O cordeau/$i "$C/$i" || echo "warn: $i failed"
done
echo "Done. Counts:"; find lilim -name '*.txt' | wc -l; ls malheiros/instances/*.txt | wc -l
