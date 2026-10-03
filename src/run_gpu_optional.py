"""Optional GPU sensitivity for XGBoost and LightGBM.

This does not replace the frozen CPU results used in the manuscript.  It uses
the same samples, features, folds, weights, hyperparameters and seed, changing
only the computational backend.  GPU results are written to a separate folder.
"""

from __future__ import annotations

import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from run_main_methods import LABELS, SEED, inputs, summarize


PACKAGE = Path(__file__).resolve().parents[1]
OUT = PACKAGE / "outputs" / "gpu_sensitivity"


def factory(name: str):
    if name == "XGBoost_GPU":
        return XGBClassifier(
            n_estimators=400, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
            reg_lambda=1.0, objective="multi:softprob", eval_metric="mlogloss",
            random_state=SEED, n_jobs=2, tree_method="hist", device="cuda",
        )
    if name == "LightGBM_GPU":
        return LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, colsample_bytree=0.8, subsample=0.8,
            subsample_freq=1, reg_lambda=1.0, random_state=SEED,
            n_jobs=2, verbosity=-1, device_type="cuda",
        )
    raise ValueError(name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index, x, _ = inputs()
    y_text = index["E_label"].to_numpy()
    y = index["E_label"].map({label: i for i, label in enumerate(LABELS)}).to_numpy(int)
    folds = index["test_fold"].to_numpy(int)
    predictions, timings, failures = [], [], []

    for name in ("XGBoost_GPU", "LightGBM_GPU"):
        try:
            for fold in range(4):
                train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                assert not set(index.iloc[train].cv_group) & set(index.iloc[test].cv_group)
                counts = np.bincount(y[train], minlength=3)
                weights = len(train) / (3 * counts)
                model = factory(name)
                started = time.monotonic()
                model.fit(x[train], y[train], sample_weight=weights[y[train]])
                pred = model.predict(x[test]).astype(int)
                timings.append({"method": name, "fold": fold, "seconds": time.monotonic() - started})
                for row_index, predicted in zip(test, pred):
                    row = index.iloc[row_index]
                    predictions.append({
                        "method": name, "fold": fold, "row_index": int(row_index),
                        "global_sample_uid": row.global_sample_uid, "cv_group": row.cv_group,
                        "true_label": y_text[row_index], "predicted_label": LABELS[predicted],
                    })
        except Exception as exc:
            failures.append({"method": name, "error_type": type(exc).__name__,
                             "error": str(exc), "traceback": traceback.format_exc()})
            predictions = [row for row in predictions if row["method"] != name]

    pred = pd.DataFrame(predictions)
    if not pred.empty:
        summary, fold_metrics, confusion, study_metrics = summarize(pred)
        pred.to_csv(OUT / "gpu_oof_predictions.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(OUT / "gpu_method_comparison.csv", index=False, encoding="utf-8-sig")
        fold_metrics.to_csv(OUT / "gpu_fold_metrics.csv", index=False, encoding="utf-8-sig")
        confusion.to_csv(OUT / "gpu_confusion_long.csv", index=False, encoding="utf-8-sig")
        study_metrics.to_csv(OUT / "gpu_study_metrics.csv", index=False, encoding="utf-8-sig")
        print(summary[["method", "macro_f1", "balanced_accuracy", "E1_recall", "E2_recall", "E3_recall"]].to_string(index=False))
    pd.DataFrame(timings).to_csv(OUT / "gpu_runtime_seconds.csv", index=False, encoding="utf-8-sig")
    (OUT / "gpu_run_metadata.json").write_text(json.dumps({
        "status": "optional computational-backend sensitivity; not manuscript primary result",
        "seed": SEED, "python": platform.python_version(), "failures": failures,
        "successful_methods": sorted(pred.method.unique().tolist()) if not pred.empty else [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        print(json.dumps(failures, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

