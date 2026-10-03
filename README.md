# Borneo Detrital Zircon ML

Reproduction and application of automated machine learning (AutoML / TPOT) classification of detrital zircon U–Pb age distributions for Borneo (SE Asia) provenance analysis.

This repository reproduces the methodology of Fekete et al. (2025) and applies it to a curated database of **117 detrital sedimentary samples from Borneo** (11,637 model U–Pb ages), classifying samples into three geologically defined provenance domains, then predicting the affinity of 140 stratigraphic TARGET samples.

## What is reproduced

- **E1** — SW Borneo / Schwaner affinity (30 samples, 5 studies)
- **E2** — Malay–Thai Peninsula / Western Sundaland affinity (53 samples, 9 studies)
- **E3** — Palawan–PCT affinity (34 samples, 7 studies)

Methods: KDE feature extraction (bandwidth 20 Ma, 0–4600 Ma, 10 Ma grid), baseline metrics (Pearson R², KS, Wasserstein), Random Forest / XGBoost / LightGBM, and TPOT AutoML — all under **4-fold DOI/study-grouped cross-validation** (no study crosses the train/test split). TARGET samples never participate in training, tuning, or model selection.

## Datasets

All input data are included in this repository (43 MB total; the largest single file is 21 MB):

| Path | Content |
|---|---|
| `data/training/` | Frozen 117-sample lists, fixed CV folds, KDE/age-array model inputs |
| `data/target/` | Grain-level inputs for 140 TARGET_STRATIGRAPHIC samples |
| `data/audits/` | Geological audit basis for E1 / E1–E3 / E4 endpoint decisions |
| `data/source_support/` | Original supplementary tables marking author-accepted E1 river sands |
| `source_archives/` | Frozen full database ZIP (`Borneo_DZ_database_FINAL_v1`, ~260k grains) for full-source rebuild and open-world tests |
| `reference_outputs/` | Frozen formal results used by automatic verification |
| `outputs/tpot_formal_v2*/` | Frozen formal TPOT result tables (part of the record) |

File integrity can be checked against `SHA256SUMS.txt`.

## Step 0: Installation

Requirements: Linux (Ubuntu 22.04+ recommended), Python 3.13, 8–16 CPU cores, 32 GB RAM. A GPU is **not** required for the main results.

```bash
# Option A: Conda
conda env create -f environment.yml
conda activate borneo-dz-repro

# Option B: venv
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-exact.txt
```

Pinned versions used for the formal runs: Python 3.13.6, NumPy 2.4.6, Pandas 3.0.3, SciPy 1.18.0, scikit-learn 1.6.1, XGBoost 3.4.0, LightGBM 4.7.0, TPOT 1.1.0.

## Step 1: Reproduce the main (CPU) results

```bash
mkdir -p logs
chmod +x *.sh
bash run_cpu_exact.sh 2>&1 | tee logs/cpu_exact.log
```

This runs, in order: input audit → baseline/RF/XGBoost/LightGBM grouped CV → TPOT fixed-budget (2 min/fold) → final RF fit on all 117 samples + prediction of 140 TARGET samples → automatic comparison against the frozen reference results.

Expected overall Macro-F1 / Balanced Accuracy:

| Method | Macro-F1 | Balanced Acc. |
|---|---:|---:|
| Pearson R² | 0.458 | 0.458 |
| KS | 0.591 | 0.590 |
| Wasserstein | 0.382 | 0.426 |
| Random Forest | 0.594 | 0.602 |
| XGBoost | 0.530 | 0.551 |
| LightGBM | 0.530 | 0.551 |
| TPOT (fixed budget, exploratory) | 0.442 | 0.450 |

On success, `outputs/reproduction_verification.json` is written with `"status": "PASS"`.

> **Note on TPOT:** the search budget is fixed (2 min/fold). Pipelines chosen may differ slightly across operating systems and CPU load; TPOT results should be reported as *fixed-budget exploratory* results. Do not extend the budget or cherry-pick runs.

## Step 2 (optional): GPU backend sensitivity

```bash
bash run_gpu_optional.sh 2>&1 | tee logs/gpu_optional.log
```

Tests whether switching XGBoost/LightGBM to CUDA backends changes classifications or runtime. This is a sensitivity check only — it does not replace the formal CPU numbers.

## Step 3 (optional): Full-source rebuild and extended analysis

```bash
bash prepare_full_source.sh 2>&1 | tee logs/prepare_full_source.log   # rebuild training inputs from the frozen database ZIP
bash run_extended_analysis.sh 2>&1 | tee logs/extended_analysis.log   # grain-threshold sensitivity + open-world affinity tests
```

Rebuilt inputs are written to `outputs/rebuilt_training/` and never overwrite the frozen `data/training/` inputs.

## Key outputs

```text
outputs/method_comparison_cpu/main_method_comparison.csv   method comparison table
outputs/method_comparison_cpu/main_fold_metrics.csv        per-fold metrics
outputs/method_comparison_cpu/main_confusion_long.csv      confusion matrices
outputs/target_rf/TARGET_predictions_key.csv               RF predictions for 140 TARGET samples
outputs/open_world_affinity/07_potential_omitted_affinity_target_list.csv
outputs/reproduction_verification.json                     PASS/FAIL check
```

## Repository structure

```text
borneo-zircon-ml/
├── data/                  # frozen inputs, folds, audits (tracked)
├── src/                   # all analysis code (Python)
├── outputs/tpot_formal_v2*/  # frozen formal TPOT results (tracked)
├── reference_outputs/     # frozen reference results for verification
├── source_archives/       # frozen full database ZIP
├── run_cpu_exact.sh       # Step 1 entry point
├── run_gpu_optional.sh    # Step 2 entry point
├── run_extended_analysis.sh / prepare_full_source.sh   # Step 3 entry points
├── environment.yml / requirements-exact.txt
├── SHA256SUMS.txt
└── README_zh.md           # detailed Chinese operating manual (口径、常见错误、运行记录要求)
```

Generated run products (`logs/`, new folders under `outputs/`) are git-ignored; the frozen formal results remain tracked.

## License

- Code: [MIT](LICENSE)
- Data: compiled from peer-reviewed detrital zircon publications; please cite the original sources (see `data/audits/`) when reusing.

## Citation

If you use this repository, please cite:

> Fekete, J.W., Sharman, G.R., Huang, X., 2025. Classifying detrital zircon U-Pb age distributions using automated machine learning. *Applied Computing and Geosciences* 26, 100251. https://doi.org/10.1016/j.acags.2025.100251

And the associated manuscript (in preparation):

```bibtex
@article{borneo-zircon-ml-2026,
  title   = {Machine-learning classification of detrital zircon U-Pb age
             distributions constrains provenance domains of Borneo},
  author  = {},
  journal = {in preparation},
  year    = {2026},
  note    = {Code and data: https://github.com/<your-account>/borneo-zircon-ml}
}
```

## Contact

Open an [Issue](../../issues) for questions about reproduction or data endpoints.
