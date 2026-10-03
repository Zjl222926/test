#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PACKAGE_DIR"

python src/inspect_inputs.py
python src/run_main_methods.py
python src/run_tpot.py --folds 0,1,2,3 --minutes 2 --population 12 --generations 2
python src/fit_final_and_predict.py
python src/verify_reproduction.py

