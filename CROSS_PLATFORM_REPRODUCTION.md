# Cross-platform reproduction record

This file documents an independent cross-platform reproduction of the
main CPU pipeline (`run_cpu_exact.sh`) and explains the expected
difference from the frozen reference results.

## Environment

| Item | Reference run | Independent run (2026-10-04) |
|---|---|---|
| OS | formal Linux server | Ubuntu 22.04 (8 vCPU / 16 GB cloud VM) |
| Python | 3.13.6 | 3.13.15 (conda-forge) |
| Pinned packages | requirements-exact.txt | same versions |
| Setup notes | — | `setuptools<81` required (pkg_resources removed in setuptools >=81); `.sh` files must have LF endings |

## Result: per-method agreement

Out-of-fold (OOF) predictions across the 6 methods in
`main_oof_predictions.csv` (702 method x sample rows, 117 samples x 4
grouped folds) were compared label-by-label against the frozen
reference outputs:

| Method | Macro-F1 (independent) | Macro-F1 (reference) | OOF label agreement |
|---|---:|---:|---|
| Pearson R2 (KDE) | 0.4578 | 0.4578 | 117/117 identical |
| KS (ECDF) | 0.5909 | 0.5909 | 117/117 identical |
| Wasserstein-1 | 0.3824 | 0.3824 | 117/117 identical |
| Random Forest | 0.5935 | 0.5935 | 117/117 identical |
| XGBoost | 0.5286 | 0.530 | **112/117 (5 flipped)** |
| LightGBM | 0.5295 | 0.530 | 117/117 identical |

**Overall: 697/702 OOF predictions (99.3%) reproduced exactly.** The 5
discordant predictions all occurred in XGBoost, consistent with the
documented floating-point reduction-order nondeterminism of
histogram-based gradient boosting across CPU architectures.

TPOT results are expected to differ more across platforms because the
search budget is wall-clock based (2 min/fold); see README Step 1.
TPOT numbers must be reported as fixed-budget exploratory results.

## Why `verify_reproduction.py` asserts exact equality

The automatic check is intentionally strict: it is designed to certify
same-platform reruns of the frozen configuration (bitwise-stable
inputs, folds and seeds). On a different CPU, histogram-based
boosters may flip a handful of borderline samples; this is not an
error in the data or the pipeline. When running on new hardware,
compare the aggregate metrics (Macro-F1 / balanced accuracy) against
the reference table in README Step 1 and report the label-flip count
as part of the reproduction record.
