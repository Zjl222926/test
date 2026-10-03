"""TPOT 1.1 on identical outer DOI folds, with DOI-grouped inner search CV."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tpot import TPOTClassifier

from run_main_methods import LABELS, OUT, ROOT, SEED, inputs, summarize


class FixedGroupedCV:
    """TPOT-compatible CV object with precomputed DOI-disjoint inner splits."""

    def __init__(self, splits, n_rows):
        self.splits = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in splits]
        self.n_rows = n_rows

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.splits)

    def split(self, X, y=None, groups=None):
        assert len(X) == self.n_rows
        for train, test in self.splits:
            yield train, test


def run_fold(fold: int, minutes: float, population: int, generations: int):
    index, x, _ = inputs()
    y = index["E_label"].map({v: i for i, v in enumerate(LABELS)}).to_numpy(int)
    outer = index["test_fold"].to_numpy(int)
    train, test = np.flatnonzero(outer != fold), np.flatnonzero(outer == fold)
    assert not set(index.iloc[train]["cv_group"]) & set(index.iloc[test]["cv_group"])
    groups = index.iloc[train]["cv_group"].to_numpy()
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED + fold)
    splits = list(inner.split(x[train], y[train], groups))
    # Crucial: TPOT.fit accepts no groups argument, so supply precomputed group-safe
    # positional split arrays for this particular outer training subset.
    for tr, te in splits:
        assert not set(groups[tr]) & set(groups[te])
        assert set(y[train][tr]) == {0, 1, 2}
    tpot = TPOTClassifier(
        search_space="linear", scorers=["balanced_accuracy"], scorers_weights=[1],
        cv=FixedGroupedCV(splits, len(train)), population_size=population, generations=generations,
        max_time_mins=minutes, max_eval_time_mins=min(0.5, minutes / 2),
        n_jobs=2, processes=False, preprocessing=False, validation_strategy="none",
        random_state=SEED + fold, verbose=1,
    )
    started = time.monotonic()
    tpot.fit(x[train], y[train])
    elapsed = time.monotonic() - started
    predicted = tpot.predict(x[test]).astype(int)
    rows = []
    for i, p in zip(test, predicted):
        entry = index.iloc[i]
        rows.append({"method": "TPOT", "fold": fold, "row_index": int(i), "global_sample_uid": entry.global_sample_uid,
                     "cv_group": entry.cv_group, "true_label": entry.E_label, "predicted_label": LABELS[p]})
    pipeline = getattr(tpot, "fitted_pipeline_", None)
    descriptor = {"fold": fold, "elapsed_seconds": elapsed, "search_minutes_cap": minutes,
                  "population_size": population, "generations_cap": generations,
                  "inner_cv": "precomputed StratifiedGroupKFold(3) by DOI/study within outer training fold",
                  "inner_train_test_groups_disjoint": True,
                  "pipeline_repr": repr(pipeline)[:12000],
                  "scoring": "balanced_accuracy"}
    return rows, descriptor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--minutes", type=float, default=2.0)
    parser.add_argument("--population", type=int, default=12)
    parser.add_argument("--generations", type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    requested = [int(value) for value in args.folds.split(",")]
    all_rows, all_descriptors, errors = [], [], []
    for fold in requested:
        print(f"TPOT outer fold {fold} starting", flush=True)
        try:
            rows, descriptor = run_fold(fold, args.minutes, args.population, args.generations)
            all_rows.extend(rows)
            all_descriptors.append(descriptor)
            print(f"TPOT outer fold {fold} complete: {len(rows)} predictions, {descriptor['elapsed_seconds']:.1f}s", flush=True)
        except Exception as exc:
            errors.append({"fold": fold, "type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
            print(f"TPOT outer fold {fold} failed: {type(exc).__name__}: {exc}", flush=True)
        pd.DataFrame(all_rows).to_csv(OUT / "tpot_oof_predictions_partial.csv", index=False, encoding="utf-8-sig")
        (OUT / "tpot_run_log.json").write_text(json.dumps({"completed_folds": [r["fold"] for r in all_descriptors],
            "descriptors": all_descriptors, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    if set(requested) == {0, 1, 2, 3} and len(all_descriptors) == 4:
        pred = pd.DataFrame(all_rows)
        assert len(pred) == 117
        summary, folds, confusion, study = summarize(pred)
        pred.to_csv(OUT / "tpot_oof_predictions.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(OUT / "tpot_method_comparison.csv", index=False, encoding="utf-8-sig")
        folds.to_csv(OUT / "tpot_fold_metrics.csv", index=False, encoding="utf-8-sig")
        confusion.to_csv(OUT / "tpot_confusion_long.csv", index=False, encoding="utf-8-sig")
        study.to_csv(OUT / "tpot_study_metrics.csv", index=False, encoding="utf-8-sig")
        print(summary[["method", "macro_f1", "balanced_accuracy", "E1_recall", "E2_recall", "E3_recall", "fold_sd_macro_f1"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
