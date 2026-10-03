#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PACKAGE_DIR"

# Requires run_cpu_exact.sh to have completed first.
python src/compare_target.py

# The following two steps additionally require prepare_full_source.sh.
if [[ -d data/full_database/MASTER ]]; then
  python src/run_sensitivities.py
  python src/open_world_compare.py
else
  echo "Skipping threshold sensitivities and open-world analysis: run prepare_full_source.sh first."
fi

