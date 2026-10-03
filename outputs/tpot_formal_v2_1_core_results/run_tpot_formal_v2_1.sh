#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PACKAGE_DIR"
mkdir -p logs outputs/tpot_formal_v2_1

python src/run_tpot_formal_v2.py \
  --mode all \
  --seeds 20260920,20260921,20260922,20260923,20260924 \
  --folds 0,1,2,3 \
  --population 30 \
  --generations 10 \
  --n-jobs 2 \
  --max-eval-minutes 10 \
  2>&1 | tee logs/tpot_formal_v2_1.log
