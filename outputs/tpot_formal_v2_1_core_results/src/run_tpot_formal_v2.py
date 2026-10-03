"""Formal TPOT v2 experiment for the frozen Borneo E1-E3 training domain.

This script is intentionally separate from the historical two-minute TPOT run.
It never edits the frozen database, labels, folds, or earlier outputs.

Formal performance is estimated with the frozen four outer DOI/study folds.
Within every outer-training subset, TPOT uses precomputed three-fold
StratifiedGroupKFold splits so that a study never crosses inner train/test.
The evolutionary budget is defined by population size and generations, not by
wall-clock time. Five predeclared search seeds quantify AutoML search variance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import tpot as tpot_package
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from tpot import TPOTClassifier

from run_main_methods import LABELS, SEED as DATA_SEED, inputs, summarize


PACKAGE = Path(os.environ.get("BORNEO_REPRO_PACKAGE", Path(__file__).resolve().parents[1])).resolve()
OUT = PACKAGE / "outputs" / "tpot_formal_v2_1"
TARGET_X = PACKAGE / "outputs" / "target_rf" / "TARGET_X_KDE_bw20_grid10.csv"
TARGET_INDEX = PACKAGE / "outputs" / "target_rf" / "TARGET_feature_row_index.csv"
DEFAULT_SEEDS = (20260920, 20260921, 20260922, 20260923, 20260924)
PROTOCOL_ID = "TPOT_FORMAL_V2.1_2026-09-29"


class FixedGroupedCV:
    """TPOT-compatible wrapper around precomputed group-disjoint splits."""

    def __init__(self, splits, n_rows):
        self.splits = [(np.asarray(a, dtype=int), np.asarray(b, dtype=int)) for a, b in splits]
        self.n_rows = int(n_rows)

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(self.splits)

    def split(self, X, y=None, groups=None):
        if len(X) != self.n_rows:
            raise ValueError(f"CV expected {self.n_rows} rows, received {len(X)}")
        yield from self.splits


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_data():
    index, x, _ = inputs()
    encoding = {label: i for i, label in enumerate(LABELS)}
    y = index["E_label"].map(encoding).to_numpy(int)
    assert len(index) == len(x) == 117
    assert index["E_label"].value_counts().to_dict() == {"E2": 53, "E3": 34, "E1": 30}
    assert index["cv_group"].nunique() == 21
    assert set(index["test_fold"]) == {0, 1, 2, 3}
    return index, np.asarray(x, dtype=np.float32), y


def inner_splits(index, x, y, outer_train, outer_fold):
    """Class-complete, group-disjoint inner splits shared by all search seeds.

    With this frozen manifest, StratifiedGroupKFold without shuffling gives
    three validation folds that each contain E1, E2, and E3 for every outer
    training subset. The earlier shuffled V2 trial could create class-missing
    validation folds and is therefore not used for the formal experiment.
    """
    groups = index.iloc[outer_train]["cv_group"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=False)
    splits = list(splitter.split(x[outer_train], y[outer_train], groups))
    for train, test in splits:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Inner study leakage")
        if set(y[outer_train][train]) != {0, 1, 2}:
            raise AssertionError("An inner training fold lacks a class")
        if set(y[outer_train][test]) != {0, 1, 2}:
            raise AssertionError("An inner validation fold lacks a class")
    return splits, groups


def full_data_splits(index, x, y):
    """Use the frozen four study folds for final full-data search selection."""
    fold_ids = index["test_fold"].to_numpy(int)
    groups = index["cv_group"].to_numpy()
    splits = []
    for fold in range(4):
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Full-data grouped split leakage")
        splits.append((train, test))
    return splits


def make_tpot(cv, seed, population, generations, n_jobs, max_eval_minutes, checkpoint):
    checkpoint.mkdir(parents=True, exist_ok=True)
    return TPOTClassifier(
        search_space="linear",
        scorers=["balanced_accuracy"],
        scorers_weights=[1],
        cv=cv,
        population_size=population,
        generations=generations,
        max_time_mins=None,
        max_eval_time_mins=max_eval_minutes,
        n_jobs=n_jobs,
        processes=False,
        preprocessing=False,
        validation_strategy="none",
        periodic_checkpoint_folder=str(checkpoint),
        random_state=seed,
        verbose=2,
    )


def save_search_artifacts(model, search_dir: Path):
    pipeline = model.fitted_pipeline_
    joblib.dump(pipeline, search_dir / "selected_pipeline.joblib", compress=3)
    (search_dir / "selected_pipeline_repr.txt").write_text(repr(pipeline), encoding="utf-8")
    try:
        evaluated = model.make_evaluated_individuals()
        joblib.dump(evaluated, search_dir / "evaluated_individuals.joblib", compress=3)
        safe = evaluated.copy()
        for column in safe.columns:
            if safe[column].dtype == object:
                safe[column] = safe[column].map(lambda value: repr(value)[:4000])
        safe.to_csv(search_dir / "evaluated_individuals.csv", index=True, encoding="utf-8-sig")
    except Exception as exc:
        (search_dir / "evaluated_individuals_error.txt").write_text(
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}", encoding="utf-8"
        )
    return pipeline


def grouped_pipeline_score(pipeline, x, y, splits):
    truth, predicted = [], []
    for train, test in splits:
        candidate = clone(pipeline)
        candidate.fit(x[train], y[train])
        truth.extend(y[test].tolist())
        predicted.extend(candidate.predict(x[test]).astype(int).tolist())
    return {
        "selection_grouped_balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "selection_grouped_macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
    }


def common_metadata(args):
    return {
        "protocol_id": PROTOCOL_ID,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "tpot": tpot_package.__version__,
        "population_size": args.population,
        "generations": args.generations,
        "search_space": "linear",
        "optimization_metric": "balanced_accuracy",
        "max_time_mins": None,
        "max_eval_time_mins": args.max_eval_minutes,
        "n_jobs": args.n_jobs,
        "processes": False,
        "preprocessing": False,
        "validation_strategy": "none",
        "class_or_sample_weights": "not supported by TPOTClassifier.fit in TPOT 1.1.0; limitation recorded",
        "target_used_for_search_or_selection": False,
    }


def run_outer_search(seed, fold, args, index, x, y):
    search_dir = OUT / "outer_searches" / f"seed_{seed}" / f"fold_{fold}"
    completed = search_dir / "completed.json"
    if completed.exists() and not args.overwrite:
        print(f"SKIP completed outer seed={seed} fold={fold}", flush=True)
        return
    search_dir.mkdir(parents=True, exist_ok=True)
    outer_fold = index["test_fold"].to_numpy(int)
    train = np.flatnonzero(outer_fold != fold)
    test = np.flatnonzero(outer_fold == fold)
    if set(index.iloc[train]["cv_group"]) & set(index.iloc[test]["cv_group"]):
        raise AssertionError("Outer study leakage")
    splits, groups = inner_splits(index, x, y, train, fold)
    split_record = []
    for inner_fold, (inner_train, inner_test) in enumerate(splits):
        split_record.append({
            "inner_fold": inner_fold,
            "train_groups": sorted(set(groups[inner_train])),
            "test_groups": sorted(set(groups[inner_test])),
            "n_train": len(inner_train),
            "n_test": len(inner_test),
            "train_class_counts_E1_E2_E3": np.bincount(y[train][inner_train], minlength=3).tolist(),
            "test_class_counts_E1_E2_E3": np.bincount(y[train][inner_test], minlength=3).tolist(),
        })
    (search_dir / "inner_group_splits.json").write_text(
        json.dumps(split_record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    model = make_tpot(
        FixedGroupedCV(splits, len(train)), seed, args.population, args.generations,
        args.n_jobs, args.max_eval_minutes, search_dir / "checkpoints"
    )
    started = time.time()
    try:
        model.fit(x[train], y[train])
        prediction = model.predict(x[test]).astype(int)
        pipeline = save_search_artifacts(model, search_dir)
        rows = []
        for row_index, predicted in zip(test, prediction):
            row = index.iloc[row_index]
            rows.append({
                "method": "TPOT_FORMAL_V2", "search_seed": seed, "fold": fold,
                "row_index": int(row_index), "global_sample_uid": row.global_sample_uid,
                "cv_group": row.cv_group, "true_label": row.E_label,
                "predicted_label": LABELS[int(predicted)],
            })
        pd.DataFrame(rows).to_csv(search_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
        metadata = {
            **common_metadata(args), "status": "COMPLETE", "search_seed": seed,
            "outer_fold": fold, "elapsed_seconds": time.time() - started,
            "n_outer_train": len(train), "n_outer_test": len(test),
            "inner_cv": "precomputed non-shuffled StratifiedGroupKFold(n_splits=3); every validation fold contains E1/E2/E3",
            "pipeline_repr_length": len(repr(pipeline)),
        }
        completed.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"COMPLETE outer seed={seed} fold={fold} elapsed={metadata['elapsed_seconds']:.1f}s", flush=True)
    except Exception as exc:
        error = {**common_metadata(args), "status": "FAILED", "search_seed": seed,
                 "outer_fold": fold, "elapsed_seconds": time.time() - started,
                 "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
        (search_dir / "failed.json").write_text(json.dumps(error, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"FAILED outer seed={seed} fold={fold}: {type(exc).__name__}: {exc}", flush=True)


def run_full_search(seed, args, index, x, y):
    search_dir = OUT / "full_data_searches" / f"seed_{seed}"
    completed = search_dir / "completed.json"
    if completed.exists() and not args.overwrite:
        print(f"SKIP completed full-data seed={seed}", flush=True)
        return
    search_dir.mkdir(parents=True, exist_ok=True)
    splits = full_data_splits(index, x, y)
    model = make_tpot(
        FixedGroupedCV(splits, len(index)), seed, args.population, args.generations,
        args.n_jobs, args.max_eval_minutes, search_dir / "checkpoints"
    )
    started = time.time()
    try:
        model.fit(x, y)
        pipeline = save_search_artifacts(model, search_dir)
        scores = grouped_pipeline_score(pipeline, x, y, splits)
        target_rows = None
        if TARGET_X.exists() and TARGET_INDEX.exists():
            target_x = pd.read_csv(TARGET_X).to_numpy(np.float32)
            target_rows = pd.read_csv(TARGET_INDEX)
            target_prediction = pipeline.predict(target_x).astype(int)
            target_rows["TPOT_FORMAL_V2_predicted_class"] = [LABELS[i] for i in target_prediction]
            if hasattr(pipeline, "predict_proba"):
                probability = pipeline.predict_proba(target_x)
                for position, model_class in enumerate(pipeline.classes_.astype(int)):
                    target_rows[f"TPOT_FORMAL_V2_probability_{LABELS[model_class]}"] = probability[:, position]
            target_rows.to_csv(search_dir / "target_predictions.csv", index=False, encoding="utf-8-sig")
        metadata = {
            **common_metadata(args), **scores, "status": "COMPLETE", "search_seed": seed,
            "elapsed_seconds": time.time() - started,
            "final_selection_cv": "frozen four study-disjoint folds; training data only",
            "pipeline_repr_length": len(repr(pipeline)),
            "target_predictions_written": target_rows is not None,
        }
        completed.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"COMPLETE full-data seed={seed} BA={scores['selection_grouped_balanced_accuracy']:.4f}", flush=True)
    except Exception as exc:
        error = {**common_metadata(args), "status": "FAILED", "search_seed": seed,
                 "elapsed_seconds": time.time() - started, "error_type": type(exc).__name__,
                 "error": str(exc), "traceback": traceback.format_exc()}
        (search_dir / "failed.json").write_text(json.dumps(error, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"FAILED full-data seed={seed}: {type(exc).__name__}: {exc}", flush=True)


def summarize_outer(seeds):
    prediction_frames = []
    for seed in seeds:
        for fold in range(4):
            path = OUT / "outer_searches" / f"seed_{seed}" / f"fold_{fold}" / "oof_predictions.csv"
            if path.exists():
                prediction_frames.append(pd.read_csv(path))
    if not prediction_frames:
        print("No completed outer predictions available for summary", flush=True)
        return
    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(OUT / "all_outer_oof_predictions.csv", index=False, encoding="utf-8-sig")
    overall, folds, confusion, studies = [], [], [], []
    complete_seeds = []
    for seed, current in predictions.groupby("search_seed"):
        if len(current) != 117 or current["global_sample_uid"].nunique() != 117 or set(current["fold"]) != {0, 1, 2, 3}:
            continue
        complete_seeds.append(int(seed))
        summary, fold_summary, cm, study = summarize(current)
        for frame in (summary, fold_summary, cm, study):
            frame.insert(1, "search_seed", int(seed))
        overall.append(summary); folds.append(fold_summary); confusion.append(cm); studies.append(study)
    if not complete_seeds:
        print("No seed has all four outer folds complete", flush=True)
        return
    overall_df = pd.concat(overall, ignore_index=True)
    overall_df.to_csv(OUT / "formal_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    pd.concat(folds, ignore_index=True).to_csv(OUT / "formal_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(confusion, ignore_index=True).to_csv(OUT / "formal_confusion_by_seed.csv", index=False, encoding="utf-8-sig")
    pd.concat(studies, ignore_index=True).to_csv(OUT / "formal_study_metrics.csv", index=False, encoding="utf-8-sig")
    numeric = [column for column in overall_df.columns if column not in {"method", "search_seed"} and pd.api.types.is_numeric_dtype(overall_df[column])]
    aggregate_rows = []
    for column in numeric:
        aggregate_rows.append({"metric": column, "mean": overall_df[column].mean(),
                               "sd_across_search_seeds": overall_df[column].std(ddof=1),
                               "min": overall_df[column].min(), "max": overall_df[column].max(),
                               "n_complete_seeds": len(complete_seeds)})
    pd.DataFrame(aggregate_rows).to_csv(OUT / "formal_metric_aggregate.csv", index=False, encoding="utf-8-sig")
    print(f"Outer summary complete for seeds {complete_seeds}", flush=True)
    print(overall_df[["search_seed", "macro_f1", "balanced_accuracy", "E1_recall", "E2_recall", "E3_recall", "fold_sd_macro_f1"]].to_string(index=False), flush=True)


def select_final_pipeline(seeds):
    rows = []
    for seed in seeds:
        metadata_path = OUT / "full_data_searches" / f"seed_{seed}" / "completed.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            rows.append(metadata)
    if not rows:
        print("No completed full-data searches available for selection", flush=True)
        return
    table = pd.DataFrame(rows).sort_values(
        ["selection_grouped_balanced_accuracy", "selection_grouped_macro_f1", "pipeline_repr_length", "search_seed"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    table["selected_final"] = False
    table.loc[0, "selected_final"] = True
    table.to_csv(OUT / "full_data_search_comparison.csv", index=False, encoding="utf-8-sig")
    winner = int(table.loc[0, "search_seed"])
    source = OUT / "full_data_searches" / f"seed_{winner}"
    selected = OUT / "selected_final_pipeline"
    selected.mkdir(parents=True, exist_ok=True)
    for name in ("selected_pipeline.joblib", "selected_pipeline_repr.txt", "target_predictions.csv", "completed.json"):
        if (source / name).exists():
            shutil.copy2(source / name, selected / name)
    decision = {
        "protocol_id": PROTOCOL_ID,
        "selected_seed": winner,
        "selection_rule": "highest training-only grouped balanced accuracy; then macro-F1; then shorter repr; then lower seed",
        "target_used_for_selection": False,
        "selection_grouped_balanced_accuracy": float(table.loc[0, "selection_grouped_balanced_accuracy"]),
        "selection_grouped_macro_f1": float(table.loc[0, "selection_grouped_macro_f1"]),
    }
    (selected / "selection_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


def write_protocol_record(args, seeds):
    OUT.mkdir(parents=True, exist_ok=True)
    index_path = PACKAGE / "data" / "training" / "model_inputs" / "model_index.csv"
    feature_path = PACKAGE / "data" / "training" / "model_inputs" / "X_KDE_bw20_grid10.csv"
    record = {
        **common_metadata(args),
        "seeds": seeds,
        "outer_folds": [0, 1, 2, 3],
        "outer_group": "frozen cv_group (DOI/study)",
        "outer_performance_reporting": "one 117-sample OOF result per seed; report mean and SD across five predeclared seeds",
        "final_full_data_searches": "one search per seed; select using training-only grouped CV rule",
        "training_rows": 117,
        "model_index_sha256": sha256(index_path),
        "KDE_feature_sha256": sha256(feature_path),
        "command_line": sys.argv,
    }
    (OUT / "formal_protocol_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("outer", "full", "summarize", "all"), default="all")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--folds", default="0,1,2,3")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--max-eval-minutes", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    seeds = parse_ints(args.seeds)
    folds = parse_ints(args.folds)
    if not seeds or not set(folds).issubset({0, 1, 2, 3}):
        raise ValueError("Invalid seeds or folds")
    index, x, y = load_data()
    write_protocol_record(args, seeds)
    if args.mode in {"outer", "all"}:
        for seed in seeds:
            for fold in folds:
                run_outer_search(seed, fold, args, index, x, y)
        summarize_outer(seeds)
    if args.mode in {"full", "all"}:
        for seed in seeds:
            run_full_search(seed, args, index, x, y)
        select_final_pipeline(seeds)
    if args.mode == "summarize":
        summarize_outer(seeds)
        select_final_pipeline(seeds)


if __name__ == "__main__":
    main()
